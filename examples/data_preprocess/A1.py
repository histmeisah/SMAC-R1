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
"""
Preprocess A1 dataset for research innovation tasks.
"""

import os
import json
import datasets
import argparse
from verl.utils.hdfs_io import copy, makedirs

# Format instruction template
FORMAT_INSTRUCTION = '''3. **Response**：  
   - 回答要从实际需求出发，提出一个可行的、具体的、创新思路。  
   - 逻辑清晰，内容详实，尤其是科研方法部分须不少于500字，不能出现文章中的具体方法名称或实验结果。  
   - 具体结构如下：
     - **问题分析**：
      - 在收到 **Instruction** 后，模型根据用户提出的科研问题进行推理、分析和思考，得到最终的回答思路，你可以模仿Chain-of-Thought的形式展示从问题到回答的关键逻辑与思考步骤。该部分的语言可以详细展开，突出对核心问题、关键挑战以及可行方向的分析。
      - 不能直接引用论文中的具体方法名称或实验结果，但可以在思考中暗示或推演出基于论文思路的创新解决方案。
     - **科研思路**：
       - **技术挑战**：阐述当前研究领域中面临的主要技术挑战。这些挑战可能包括数据获取的困难、现有技术的局限性、理论模型的不足等。请详细描述这些挑战对研究进展的具体影响。
       - **关键思路**：说明这个研究的核心理念或哲学是什么。这种理念应该是高度抽象的，能够为整个研究提供指导思想。请确保描述清晰明确，能够让读者快速理解研究的核心思想。尽量以讨论的口吻进行阐述，但不要使用"本文"等字眼。例如，可以这样表达："一个可能的解决方案是……"，或者"一个值得探讨的方向是……"。请不要在这个部分具体展开描述具体方法，而是一个高度凝练的想法，请注意和方法以及问题保持一致。
       - **具体方法**：要以建议的方式给出描述，详细展开说明可行且具有创新性的解决策略，详细展开的每一点也尽量详细，必要时可以用公式说明。请注意不要出现文章中的方法名称，不要出现实验结果。
       - **方法意义**：阐述提出的方法对该研究领域有什么价值或优势，避免讨论实验结果。  
     - **总结**：对整个问题和思路进行简要概括，不要写"本文"等字眼，请注意你是在根据用户的问题给出建议，应该在用户 Instruction 的语境下总结这个问题和方法。

示例格式：
<RESPONSE>
**问题分析**：...  
**科研思路**：  
   - **技术挑战**：...  
   - **关键思路**：...  
   - **具体方法**：...  
   - **方法意义**：...  
**总结**：...  
</RESPONSE>'''

def load_jsonl(file_path):
    """Load data from jsonl file."""
    try:
        # First try to load as a single JSON array
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            data = json.loads(content)
            if isinstance(data, list):
                print(f"Successfully loaded {len(data)} items from {file_path} as JSON array")
                return data
    except json.JSONDecodeError:
        # If that fails, try loading as JSONL
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            current_item = {}
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                # Handle array brackets
                if line == '[' or line == ']':
                    continue
                    
                # Remove trailing comma if present
                if line.endswith(','):
                    line = line[:-1]
                    
                # Handle object brackets
                if line == '{' or line == '}' or line == '},':
                    if current_item:  # If we have a complete item
                        data.append(current_item)
                        current_item = {}
                    continue
                
                try:
                    # Try to parse as a complete JSON object
                    if line.startswith('{'):
                        item = json.loads(line.rstrip(','))
                        data.append(item)
                        continue
                        
                    # Handle key-value pairs
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().strip('"')
                        value = value.strip().strip(',').strip()
                        if value.startswith('"') and value.endswith('"'):
                            value = json.loads(value)
                        current_item[key] = value
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {i}: {e}")
                    print(f"Problematic line content: {line[:100]}...")
                    continue
                
        if current_item:  # Add the last item if exists
            data.append(current_item)
            
        if not data:
            raise ValueError(f"No valid JSON objects found in {file_path}")
            
        print(f"Successfully loaded {len(data)} items from {file_path} as JSONL")
        
    return data

def make_map_fn(split):
    """Create mapping function for dataset processing."""
    def process_fn(example, idx):
        # Get the input question and reference output
        question = example['input']
        reference = example['output']
        
        # Combine format instruction with the question
        formatted_question = FORMAT_INSTRUCTION + "\n\n" + question
        
        data = {
            "data_source": "A1_innovation",
            "prompt": [{
                "role": "user",
                "content": formatted_question
            }],
            "ability": "research_innovation",
            "reward_model": {
                "style": "rule",
                "ground_truth": reference
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/ma-user/modelarts/work/weiyu/python/verl_project/verl/examples/data_preprocess/data/A1')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', 
                      default='/home/ma-user/modelarts/work/weiyu/python/verl_project/rlhf_data/A1_data/LlamaFactory_ICLR_generate_innovation_v3_train.jsonl')
    parser.add_argument('--test_size', type=float, default=0.1,
                      help='Proportion of the dataset to include in the test split')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for dataset splitting')
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.data_path}")
    
    # Check if file exists
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file not found: {args.data_path}")
        
    # Load the jsonl data
    raw_data = load_jsonl(args.data_path)
    
    # Print some statistics
    print(f"Total dataset size: {len(raw_data)}")
    if raw_data:
        print("First item example:")
        print(json.dumps(raw_data[0], indent=2, ensure_ascii=False)[:500] + "...")
    
    # Convert to datasets format
    full_dataset = datasets.Dataset.from_list(raw_data)
    
    # Split the dataset
    splits = full_dataset.train_test_split(
        test_size=args.test_size,
        seed=args.seed,
        shuffle=True
    )
    train_dataset = splits['train']
    test_dataset = splits['test']
    
    print(f"Split sizes - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Process the datasets
    processed_train = train_dataset.map(
        function=make_map_fn('train'), 
        with_indices=True
    )
    processed_test = test_dataset.map(
        function=make_map_fn('test'), 
        with_indices=True
    )
    
    # Create output directories if they don't exist
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    # Save to parquet format
    train_path = os.path.join(local_dir, 'train.parquet')
    test_path = os.path.join(local_dir, 'test.parquet')
    
    processed_train.to_parquet(train_path)
    processed_test.to_parquet(test_path)
    
    print(f"Saved processed training dataset to: {train_path}")
    print(f"Saved processed test dataset to: {test_path}")
    
    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied data to HDFS: {args.hdfs_dir}")
