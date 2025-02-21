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

def check_format(response_str):
    """Check if the response follows the required format.
    
    Args:
        response_str: The response string to check
        
    Returns:
        dict: A dictionary containing format check results and scores
    """
    # Initialize format check results
    format_check = {
        'has_response_tags': False,
        'sections_complete': False,
        'sections_ordered': False,
        'sections_no_duplicate': False,
        'word_count_sufficient': False,
        'total_score': 0.0
    }
    
    # Check for response tags
    if re.search(r'<RESPONSE>.*</RESPONSE>', response_str, re.DOTALL):
        format_check['has_response_tags'] = True
        # Extract content within RESPONSE tags
        response_content = re.search(r'<RESPONSE>(.*)</RESPONSE>', response_str, re.DOTALL)
        if response_content:
            response_str = response_content.group(1)
    
    # Define expected sections in order
    expected_sections = [
        '**问题分析**',
        '**科研思路**',
        '**技术挑战**',
        '**关键思路**',
        '**具体方法**',
        '**方法意义**',
        '**总结**'
    ]
    
    # Find all section positions
    section_positions = {}
    for section in expected_sections:
        positions = [m.start() for m in re.finditer(re.escape(section), response_str)]
        section_positions[section] = positions
    
    # Check if all sections exist exactly once
    sections_complete = True
    sections_no_duplicate = True
    for section, positions in section_positions.items():
        if len(positions) == 0:
            sections_complete = False
        if len(positions) > 1:
            sections_no_duplicate = False
    
    # Check section order
    sections_ordered = True
    last_pos = -1
    for section in expected_sections:
        positions = section_positions[section]
        if positions:
            if positions[0] < last_pos:
                sections_ordered = False
                break
            last_pos = positions[0]
    
    format_check['sections_complete'] = sections_complete
    format_check['sections_ordered'] = sections_ordered
    format_check['sections_no_duplicate'] = sections_no_duplicate
    
    # Check word count for methods section
    methods_pattern = r'\*\*具体方法\*\*：(.*?)(?=\*\*|$)'
    methods_match = re.search(methods_pattern, response_str, re.DOTALL)
    if methods_match:
        methods_text = methods_match.group(1)
        word_count = len(methods_text)  # Simple character count for Chinese
        format_check['word_count_sufficient'] = word_count >= 500
    
    # Calculate total score with adjusted weights
    criteria_weights = {
        'has_response_tags': 0.1,        # 基础格式要求
        'sections_complete': 0.3,        # 所有部分都存在
        'sections_ordered': 0.3,         # 顺序正确
        'sections_no_duplicate': 0.2,    # 没有重复
        'word_count_sufficient': 0.1     # 字数要求
    }
    
    format_check['total_score'] = sum(
        criteria_weights[key] * float(value) 
        for key, value in format_check.items() 
        if key in criteria_weights
    )
    
    return format_check

def compute_score(response_str, ground_truth=None, method='strict'):
    """Compute the format compliance score for the response.
    
    Args:
        response_str: The response to evaluate
        ground_truth: Not used for format checking, included for API compatibility
        method: The scoring method ('strict' or 'flexible')
        
    Returns:
        float: The format compliance score between 0 and 1
    """
    format_results = check_format(response_str)
    return format_results['total_score']
