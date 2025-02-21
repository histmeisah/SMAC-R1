import re
from typing import Union, Optional
from .prime_math.grader import math_equal
from .prime_math.math_normalize import normalize_answer

def extract_answer(text: str) -> Optional[str]:
    """从文本中提取答案"""
    if not text:
        return None
        
    # 尝试从<answer>标签中提取
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
        
    # 尝试从\boxed{}中提取
    boxed_pattern = r'\\boxed{([^}]*)}'
    boxed_match = re.search(boxed_pattern, text)
    if boxed_match:
        return boxed_match.group(1).strip()
        
    # 尝试提取最后一个数值表达式
    number_pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    numbers = re.findall(number_pattern, text)
    if numbers:
        return numbers[-1]
        
    return None

def compute_score(solution_str: str, ground_truth: str) -> float:
    """计算模型响应的得分
    
    Args:
        solution_str: 模型的完整输出
        ground_truth: 标准答案
        
    Returns:
        float: 1.0表示正确，-1.0表示错误
    """
    # 提取模型答案
    model_answer = extract_answer(solution_str)
    if not model_answer:
        return -1.0
    
    # 使用prime_math中的工具进行答案比较
    try:
        # 标准化处理
        model_norm = normalize_answer(model_answer)
        truth_norm = normalize_answer(ground_truth)
        
        if not model_norm or not truth_norm:
            return -1.0
            
        # 使用math_equal进行更智能的比较
        if math_equal(model_norm, truth_norm, 
                     include_percentage=True,  # 支持百分比
                     tolerance=1e-4,           # 数值近似容差
                     timeout=10.0):           # 符号计算超时
            return 1.0
            
    except Exception as e:
        print(f"Error in answer comparison: {e}")
        
    return -1.0 