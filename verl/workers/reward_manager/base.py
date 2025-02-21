import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from verl.utils.reward_score import _default_compute_score
from verl import DataProto
import logging
import time
import torch

logger = logging.getLogger(__name__)

class BaseRewardManager:
    """Base class for reward managers with data logging capability."""
    
    def __init__(self, tokenizer, num_examine: int, compute_score=None, config=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score
        self.config = config
        
        # 初始化数据保存相关配置
        self.enable_logging = config.trainer.get('save_rewards', False) if config else False
        
        if self.enable_logging:
            # 使用rank作为文件名的一部分，避免多进程冲突
            self.rank = int(os.environ.get('RANK', '0'))
            self.save_dir = os.path.join(config.trainer.default_local_dir, 
                                       config.trainer.get('reward_logs_dir', 'reward_logs'))
            
            # 减小保存频率，默认改为1000
            self.save_frequency = config.trainer.get('reward_save_frequency', 100)
            
            # 限制buffer大小
            self.max_buffer_size = config.trainer.get('max_buffer_size', 100)
            self.data_buffer: List[Dict] = []
            self.total_samples = 0
            
            # 创建保存目录
            os.makedirs(self.save_dir, exist_ok=True)
            
            # 创建新的日志文件，包含rank信息
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 每个rank的reward结果文件
            self.current_log_file = os.path.join(
                self.save_dir, 
                f"reward_log_rank{self.rank}_{timestamp}.jsonl"
            )
            
            # prompt模板文件（只由rank 0保存）
            self.prompt_template_file = os.path.join(
                self.save_dir,
                f"prompt_template_{timestamp}.json"
            )
            
            # 添加文件锁，避免多进程写入冲突
            self.file_lock = False
            
            logger.info(f"Reward manager initialized on rank {self.rank}")
            
            # rank 0负责保存prompt模板
            if self.rank == 0:
                self.save_prompt_template()
            
        self.total_games_run = 0
        self.total_games_failed = 0
        self.error_count = 0
        self.max_consecutive_errors = 3
        
    def save_prompt_template(self):
        """保存prompt模板（只在rank 0执行）"""
        if not hasattr(self, 'saved_template') and hasattr(self.config, 'prompt_template'):
            try:
                with open(self.prompt_template_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'prompt_template': self.config.prompt_template,
                        'system_prompt': getattr(self.config, 'system_prompt', ''),
                        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
                    }, f, ensure_ascii=False, indent=2)
                self.saved_template = True
                logger.info(f"Prompt template saved to {self.prompt_template_file}")
            except Exception as e:
                logger.error(f"Failed to save prompt template: {e}")
            
    def save_data_buffer(self):
        """将数据缓冲区的内容保存到文件"""
        if not self.enable_logging or not self.data_buffer:
            return
            
        # 如果文件被锁，等待一段时间后重试
        max_retries = 3
        retry_count = 0
        
        while self.file_lock and retry_count < max_retries:
            time.sleep(0.1)
            retry_count += 1
            
        if self.file_lock:
            logger.warning("Failed to acquire file lock for saving rewards")
            return
            
        try:
            self.file_lock = True
            with open(self.current_log_file, 'a', encoding='utf-8') as f:
                for item in self.data_buffer:
                    # 保存response和必要的上下文信息
                    minimal_item = {
                        'response': item['response'],
                        'ground_truth': item['ground_truth'],
                        'score': float(item['score']),
                        'input_context': item.get('input_context', {}),  # 保存特定的输入参数
                    }
                    f.write(json.dumps(minimal_item, ensure_ascii=False) + '\n')
            
            # 清空缓冲区
            self.data_buffer = []
            
        except Exception as e:
            logger.error(f"Error saving reward buffer: {e}")
        finally:
            self.file_lock = False
        
    def log_sample(self, prompt_str: str, response_str: str, 
                  ground_truth: str, data_source: str, 
                  score: float, full_sequence: str, 
                  step: Optional[int] = None,
                  games_run: int = 0,
                  games_failed: int = 0):
        """记录单个样本数据
        
        Args:
            prompt_str: 提示词
            response_str: 模型回复
            ground_truth: 真实答案（对于SC2任务是地图名）
            data_source: 数据来源
            score: reward分数
            full_sequence: 完整序列
            step: 当前训练的step数
            games_run: 实际运行的游戏数
            games_failed: 失败的游戏数
        """
        if not self.enable_logging:
            return
            
        self.total_games_run += games_run
        self.total_games_failed += games_failed
        
        if self.total_games_run % 100 == 0:
            logger.info(f"Total games run: {self.total_games_run}, Failed: {self.total_games_failed}")
            
        # 提取特定的输入上下文（比如地图信息等）
        try:
            input_context = {
                'data_source': data_source,
                'map_name': ground_truth,  # 对于SC2任务，ground_truth就是地图名
                'step': step,  # 添加step信息
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        except:
            input_context = {}
            
        self.data_buffer.append({
            'response': response_str,
            'ground_truth': ground_truth,
            'score': float(score),
            'input_context': input_context,
            'step': step  # 在顶层也保存step信息，方便查询
        })
        
        self.total_samples += 1
        
        # 当buffer达到最大大小或达到保存频率时保存
        if (len(self.data_buffer) >= self.max_buffer_size or 
            self.total_samples % self.save_frequency == 0):
            self.save_data_buffer()
            
    def __call__(self, data: DataProto):
        if self.error_count >= self.max_consecutive_errors:
            logger.error("Too many consecutive errors, skipping reward computation")
            self.error_count = 0  # 重置计数
            return torch.zeros_like(data.batch['responses'], dtype=torch.float32)
            
        try:
            # ... 现有代码 ...
            self.error_count = 0  # 成功后重置
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error in reward computation: {e}")
            return torch.zeros_like(data.batch['responses'], dtype=torch.float32)
            
    def __del__(self):
        """确保在对象销毁时保存所有剩余数据"""
        try:
            self.save_data_buffer()
        except:
            pass 