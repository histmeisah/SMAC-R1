# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import os
import logging
import subprocess
import traceback
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil  # 添加这个import

logger = logging.getLogger(__name__)

# Define constants for code templates
PREFIX_CODE = '''
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race, Difficulty
from sc2.ids.ability_id import AbilityId
from sc2.ids.effect_id import EffectId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
import math
import random

class MarineBot(BotAI):
'''

def get_post_code(map_name):
    return f'''
if __name__ == '__main__':
    bot = MarineBot()
    result = run_game(maps.get("{map_name}"), [Bot(Race.Random, bot), Computer(Race.Random, Difficulty.VeryHard)], realtime=False)
    print(result)
'''

def extract_code(response):
    """Extract code and strategy from LLM response"""
    try:
        strategy = None
        strategy_patterns = [
            r'(?i)<strategy>\s*(.*?)\s*</strategy>',
            r'(?i)strategy\s*\n(.*?)(?=\n\s*(?:<code>|```python))',
            r'(?i)^(.*?)(?=\n\s*(?:<code>|```python))'
        ]
        
        for pattern in strategy_patterns:
            strategy_match = re.search(pattern, response, re.DOTALL)
            if strategy_match:
                strategy = strategy_match.group(1).strip()
                break
        
        code = None
        class_patterns = [
            r'class\s+BattleBot\s*\([^)]*\):\s*(.*?)(?=\s*if\s+__name__\s*==|\s*$)',
            r'class\s+\w+\s*\([^)]*\):\s*(.*?)(?=\s*if\s+__name__\s*==|\s*$)'
        ]
        
        for pattern in class_patterns:
            class_match = re.search(pattern, response, re.DOTALL)
            if class_match:
                class_code = class_match.group(1).strip()
                if 'def on_step' in class_code:
                    code = class_code
                    break
        
        if not code:
            code_patterns = [
                r'def\s+on_step[^}]*?}',
                r'async\s+def\s+on_step[^}]*?}',
                r'```python\s*(.*?)\s*```',
                r'<code>\s*```python\s*(.*?)\s*```\s*</code>',
                r'<code>\s*(.*?)\s*</code>'
            ]
            
            for pattern in code_patterns:
                code_match = re.search(pattern, response, re.DOTALL)
                if code_match:
                    code = code_match.group(1).strip() if '```python' in pattern else code_match.group(0).strip()
                    break
        
        if not code:
            logger.error("No code block found in response")
            logger.debug(f"Full response: {response}")
            return None, None
        
        code = code.replace('```python', '').replace('```', '').strip()
        
        if not code or len(code.strip()) < 10:
            logger.error("Extracted code is too short or empty")
            return None, None
        
        if 'def on_step' not in code:
            if 'self.' in code and ('move' in code or 'attack' in code):
                code = 'async def on_step(self, iteration: int):\n' + code
            else:
                logger.error("Code does not contain on_step function")
                return None, None
        
        required_elements = ['self.units', 'UnitTypeId', 'move', 'attack']
        if not any(element in code for element in required_elements):
            logger.error("Code missing essential StarCraft II elements")
            return None, None
        
        return code, strategy
        
    except Exception as e:
        error_msg = f"Code extraction failed: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return None, None

class ResourceMonitor:
    """监控系统资源使用情况"""
    def __init__(self, warning_threshold=80, critical_threshold=90):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold

    def check_resources(self):
        """检查系统资源状态"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            status = 'normal'
            if cpu_percent > self.critical_threshold or memory_percent > self.critical_threshold:
                status = 'critical'
            elif cpu_percent > self.warning_threshold or memory_percent > self.warning_threshold:
                status = 'warning'
                
            return {
                'status': status,
                'cpu': cpu_percent,
                'memory': memory_percent,
                'memory_available': memory.available / (1024 * 1024 * 1024)  # GB
            }
        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return {'status': 'unknown', 'error': str(e)}

class GameProcessManager:
    def __init__(self, max_concurrent_games=8):
        """优化的进程管理器初始化"""
        try:
            cpu_count = os.cpu_count()
            total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
            
            # 动态计算最大并发数
            self.max_games = min(
                max_concurrent_games,
                cpu_count // 4,  # CPU限制
                int(total_memory / 4),  # 内存限制（假设每个游戏约需4GB）
                16  # 硬上限
            )
        except Exception as e:
            logger.warning(f"Failed to optimize concurrent games, using default: {e}")
            self.max_games = min(max_concurrent_games, 8)

        self.current_games = {}
        self.process_semaphore = threading.Semaphore(self.max_games)
        self.launch_interval = 3  # 增加启动间隔
        self.last_launch_time = 0
        self.launch_lock = threading.Lock()
        self.resource_monitor = ResourceMonitor()
        self.process_cleanup_timeout = 10

    def _find_best_cpu(self):
        """找到负载最低的CPU"""
        try:
            cpu_loads = psutil.cpu_percent(percpu=True)
            return cpu_loads.index(min(cpu_loads))
        except:
            return 0

    def _kill_game(self, pid):
        """Safely terminate game process"""
        try:
            if pid in self.current_games:
                process = self.current_games[pid]
                
                # 首先尝试正常终止
                try:
                    import psutil
                    parent = psutil.Process(pid)
                    children = parent.children(recursive=True)
                    
                    # 先终止子进程
                    for child in children:
                        try:
                            child.terminate()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                            
                    # 等待子进程终止
                    gone, alive = psutil.wait_procs(children, timeout=3)
                    
                    # 强制结束还在运行的子进程
                    for p in alive:
                        try:
                            p.kill()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                            
                    # 终止父进程
                    parent.terminate()
                    parent.wait(timeout=3)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
                    logger.debug(f"Process {pid} already terminated or access denied: {e}")
                    # 如果psutil方法失败，使用原始方法
                    try:
                        process.terminate()
                        process.wait(timeout=3)
                    except:
                        process.kill()
                        process.wait(timeout=1)
                
                del self.current_games[pid]
        except Exception as e:
            logger.error(f"Error killing game process {pid}: {e}")

    def run_game(self, temp_file):
        """优化的游戏运行函数"""
        process = None
        try:
            with self.process_semaphore:
                # 验证生成的Python文件
                try:
                    with open(temp_file, 'r') as f:
                        content = f.read()
                        # 使用ast模块验证语法
                        import ast
                        ast.parse(content)
                except Exception as e:
                    logger.error(f"Invalid Python file {temp_file}: {e}")
                    return {'result': 'error', 'error': 'syntax_error'}

                # 检查系统资源
                resources = self.resource_monitor.check_resources()
                if resources['status'] == 'critical':
                    logger.warning("System resources critical, waiting...")
                    time.sleep(5)
                    resources = self.resource_monitor.check_resources()
                    if resources['status'] == 'critical':
                        return {'result': 'resource_limit'}

                # 控制启动间隔
                with self.launch_lock:
                    current_time = time.time()
                    if current_time - self.last_launch_time < self.launch_interval:
                        time.sleep(self.launch_interval - (current_time - self.last_launch_time))
                    self.last_launch_time = time.time()

                # 简化进程启动
                process = subprocess.Popen(
                    f'python {temp_file}',
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    universal_newlines=True,
                    env=os.environ.copy()  # 继承当前环境变量
                )

                pid = process.pid
                self.current_games[pid] = process

                try:
                    output, _ = process.communicate(timeout=300)
                    
                    # 更严谨的游戏结果判断
                    output_lower = output.lower()
                    if 'traceback' in output_lower or 'error' in output_lower:
                        error_lines = [line for line in output.split('\n') if 'error' in line.lower() or 'traceback' in line.lower()]
                        error_msg = error_lines[0] if error_lines else "Unknown error"
                        logger.error(f"Game error: {error_msg}")
                        return {'result': 'error', 'error': error_msg, 'details': output}
                    
                    # 更详细的游戏结果分类
                    result_mapping = {
                        'Result.Victory': {'result': 'success', 'outcome': 'v'},
                        'Result.Defeat': {'result': 'success', 'outcome': 'd'},
                        'Result.Tie': {'result': 'success', 'outcome': 't'},
                        'Result.Crash': {'result': 'error', 'error': 'game_crash'},
                        'Result.Disconnect': {'result': 'error', 'error': 'disconnected'}
                    }
                    
                    for key, value in result_mapping.items():
                        if key in output:
                            return value
                    
                    logger.error(f"Unexpected game output: {output}")
                    return {'result': 'error', 'error': 'invalid_output', 'details': output}

                except subprocess.TimeoutExpired:
                    logger.warning(f"Game process {pid} timed out")
                    return {'result': 'timeout'}
                finally:
                    if pid in self.current_games:
                        self._kill_game(pid)

        except Exception as e:
            logger.error(f"Game process error: {str(e)}")
            if process and process.pid in self.current_games:
                self._kill_game(process.pid)
            return {'result': 'error'}

    def __del__(self):
        """Ensure all processes are cleaned up when object is destroyed"""
        for pid in list(self.current_games.keys()):
            self._kill_game(pid)

def kill_process_tree(pid):
    """Kill a process and all its children"""
    try:
        import psutil
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.kill()
        parent.kill()
    except Exception as e:
        logger.error(f"Error killing process tree: {e}")

def run_parallel_games(code, map_name, num_games=10):
    """优化的并行游戏运行函数"""
    results = {
        'wins': 0, 'losses': 0, 'ties': 0,
        'errors': 0, 'timeouts': 0,
        'total_games': num_games
    }
    
    try:
        # 计算最优worker数量
        cpu_count = os.cpu_count()
        memory_available = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
        optimal_workers = min(
            cpu_count // 4,  # CPU限制
            int(memory_available / 4),  # 内存限制
            num_games,
            16  # 硬上限
        )

        # 初始化游戏管理器
        game_manager = GameProcessManager(max_concurrent_games=optimal_workers)
        
        # 创建临时文件
        temp_file = f'res-temp-{os.getpid()}-{int(time.time()*1000)}.py'
        total_code = PREFIX_CODE + '\n    ' + code.replace('\n', '\n    ') + get_post_code(map_name)
        
        with open(temp_file, 'w') as writer:
            writer.write(total_code)

        # 批次处理
        batch_size = optimal_workers
        for i in range(0, num_games, batch_size):
            current_batch = min(batch_size, num_games - i)
            
            with ThreadPoolExecutor(max_workers=current_batch) as executor:
                futures = [executor.submit(game_manager.run_game, temp_file) 
                          for _ in range(current_batch)]

                for future in as_completed(futures):
                    try:
                        data = future.result()
                        if data['result'] == 'error':
                            results['errors'] += 1
                        elif data['result'] == 'timeout':
                            results['timeouts'] += 1
                        elif data['result'] == 'resource_limit':
                            results['errors'] += 1
                        elif data['result'] == 'success':
                            results['wins'] += 1
                            if data['outcome'] == 'v':
                                results['wins'] += 1
                            elif data['outcome'] == 'd':
                                results['losses'] += 1
                            else:
                                results['ties'] += 1
                    except Exception as e:
                        logger.error(f"Error processing game result: {str(e)}")
                        results['errors'] += 1

            # 批次间休息
            if i + batch_size < num_games:
                time.sleep(2)

        logger.info(f"Games completed - Wins: {results['wins']}, Losses: {results['losses']}, Ties: {results['ties']}, Errors: {results['errors']}, Timeouts: {results['timeouts']}")

    except Exception as e:
        logger.error(f"Parallel games execution failed: {str(e)}")
    finally:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass

    return results

def compute_score(solution_str, ground_truth, method='strict', format_score=-1.0, num_games=10):
    """Compute reward score for code generation task.
    
    Args:
        solution_str: The solution code string from LLM
        ground_truth: The map name to test on
        method: Scoring method (currently unused)
        format_score: Score to return if code cannot be parsed (-1.0 by default)
        num_games: Number of games to run for evaluation (10 by default)
        
    Returns:
        float: Score in range [-1.0, 1.0] where:
            -1.0: Code cannot be parsed
            -0.5: Code execution error or timeout
            [0.0, 1.0]: Win rate from game results
    """
    try:
        # Extract code from solution
        code, strategy = extract_code(solution_str)
        if not code:
            logger.error("Code extraction failed")
            return format_score
            
        # 添加代码长度限制
        if len(code) > 10000:  
            logger.error("Code too long")
            return format_score
            
        # 运行游戏前验证语法
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            logger.error(f"Syntax error in code: {e}")
            return format_score
            
        # Run parallel games with timeout
        try:
            results = run_parallel_games(code, ground_truth, num_games=num_games)
        except Exception as e:
            logger.error(f"Game execution failed: {e}")
            return -0.5
        
        # 验证结果的合理性
        if results['total_games'] == 0:
            logger.error("No games were run")
            return -0.5
            
        # 如果有任何错误或超时，返回-0.5
        if results['errors'] > 0 or results['timeouts'] > 0:
            return -0.5
            
        # 计算胜率
        valid_games = results['wins'] + results['losses'] + results['ties']
        if valid_games == 0:
            return -0.5
            
        win_rate = results['wins'] / valid_games
        logger.info(f"Final score (win rate): {win_rate}")
        return win_rate

    except Exception as e:
        logger.error(f"Score computation failed: {e}")
        return format_score 