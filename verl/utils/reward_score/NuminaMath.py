import re
from typing import Tuple, Optional

# 配置字典
REWARD_CONFIG = {
    'use_format_reward': True,  # 是否使用格式奖励
    'format_reward': 1.0,       # 格式奖励分值
    'answer_reward': 1.0,       # 答案奖励分值
    'debug': True              # 是否打印调试信息
}

def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    """从模型响应中提取最终答案。"""
    if "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None, solution_str

    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str

def validate_response_structure(processed_str: str) -> bool:
    """验证响应结构的完整性。"""
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        if count != expected_count:
            return False

    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        return False

    return True

def normalize_answer(text: str) -> str:
    """标准化答案格式"""
    try:
        # 移除所有空白字符
        text = ''.join(text.split())
        
        # 移除LaTeX的\boxed{}
        boxed_pattern = r'\\boxed{([^}]*)}'
        match = re.search(boxed_pattern, text)
        if match:
            text = match.group(1)
            
        # 移除所有空格和$符号
        text = text.replace(' ', '').replace('$', '')
        
        return text
    except Exception:
        return text

def calculate_answer_score(model_content: str, standard_content: str) -> float:
    """计算答案分数
    
    Args:
        model_content: 模型生成的答案
        standard_content: 标准答案
        
    Returns:
        float: 1.0表示正确，-1.0表示错误
    """
    try:
        # 标准化答案
        model_norm = normalize_answer(model_content)
        standard_norm = normalize_answer(standard_content)
        
        # 直接比较标准化后的字符串
        if model_norm == standard_norm:
            return 1.0
        return -1.0
        
    except Exception:
        return -1.0

def compute_score(solution_str: str, ground_truth: str) -> float:
    """计算模型响应的得分。"""
    # 提取模型答案
    answer_text, processed_str = extract_solution(solution_str)
    
    if REWARD_CONFIG['use_format_reward']:
        # 验证响应结构
        format_correct = validate_response_structure(processed_str)
        format_score = REWARD_CONFIG['format_reward'] if format_correct else -REWARD_CONFIG['format_reward']
        
        # 如果格式错误且没有答案，直接返回最低分
        if not format_correct and not answer_text:
            return -(REWARD_CONFIG['format_reward'] + REWARD_CONFIG['answer_reward'])
    
    # 验证答案内容
    if not answer_text:
        return -1.0
        
    # 计算答案分数
    answer_score = calculate_answer_score(answer_text, ground_truth)
    
    if REWARD_CONFIG['debug']:
        print(f"\n答案验证:")
        print(f"模型答案: {answer_text}")
        print(f"标准答案: {ground_truth}")
        print(f"答案得分: {answer_score}")
        if REWARD_CONFIG['use_format_reward']:
            print(f"格式得分: {format_score}")
            print(f"总分: {format_score + answer_score}")
    
    # 返回最终分数
    if REWARD_CONFIG['use_format_reward']:
        return format_score + answer_score
    return answer_score  # 直接返回1.0或-1.0

def compute_score_with_debug(solution_str: str,
                           ground_truth: str,
                           mode: str = 'full',
                           format_reward: float = 1.0,
                           answer_reward: float = 2.0,
                           debug: bool = False) -> float:
    """带调试信息的评分函数版本"""
    if not debug:
        return compute_score(solution_str, ground_truth)
        
    print("\n" + "="*80)
    print(" 处理新样本 ".center(80, '='))
    
    # 提取模型答案
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[模型响应]\n{processed_str}")
    
    if mode == 'full':
        # 验证响应结构
        format_correct = validate_response_structure(processed_str)
        format_score = format_reward if format_correct else -abs(format_reward)
        print(f"\n  格式验证: {'通过' if format_correct else '失败'}")
        print(f"  格式得分: {format_score}")
        
        if not format_correct and not answer_text:
            print("\n[内容验证] 由于格式错误且无答案而跳过")
            total_score = -(format_reward + answer_reward)
        else:
            # 验证答案内容
            if answer_text:
                model_content = normalize_answer(answer_text)
                standard_content = normalize_answer(ground_truth)
                
                print(f"\n[内容验证]")
                print(f"  期望答案: {ground_truth}")
                print(f"  模型答案: {answer_text}")
                print(f"  标准化后 - 期望: {standard_content}")
                print(f"  标准化后 - 模型: {model_content}")
                
                answer_score = calculate_answer_score(model_content, standard_content) * answer_reward
                print(f"  答案验证: {'正确' if answer_score > 0 else '错误'}")
            else:
                answer_score = -answer_reward
                print("\n[内容验证] 未找到答案")
                
            total_score = format_score + answer_score
            
    else:  # answer_only mode
        if not answer_text:
            print("\n[内容验证] 未找到答案")
            total_score = -answer_reward
        else:
            model_content = normalize_answer(answer_text)
            standard_content = normalize_answer(ground_truth)
            
            print(f"\n[内容验证]")
            print(f"  期望答案: {ground_truth}")
            print(f"  模型答案: {answer_text}")
            print(f"  标准化后 - 期望: {standard_content}")
            print(f"  标准化后 - 模型: {model_content}")
            
            total_score = calculate_answer_score(model_content, standard_content) * answer_reward
            print(f"  答案验证: {'正确' if total_score > 0 else '错误'}")
    
    print("\n" + "-"*80)
    print(f" 最终得分: {total_score} ".center(80, '-'))
    print("="*80 + "\n")
    
    return total_score