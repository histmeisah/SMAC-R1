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
Preprocess the StarCraft II instruction dataset to parquet format
"""

import json
import os
import argparse
from verl.utils.hdfs_io import copy, makedirs
import pandas as pd

def load_sc2_dataset(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['samples']

def process_sample(sample, idx, split='train'):
    """Process a single sample into the required format."""
    return {
        "data_source": "llm_smac",
        "prompt": [{
            "role": "user",
            "content": sample['instruction'] + "\n\n" + sample['input']
        }],
        "ability": "sc2_micro",
        "reward_model": {
            "style": "rule",
            "ground_truth": sample['map_name']
        },
        "extra_info": {
            'split': split,
            'index': idx,
            'map_name': sample['map_name']
        }
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/home/ma-user/modelarts/work/weiyu/python/verl_project/verl/examples/data_preprocess/data/sc2_instruction')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--input_json', default='/home/ma-user/modelarts/work/weiyu/test_project/LLM-SMAC/smac_instruction_dataset.json')
    parser.add_argument('--train_ratio', type=float, default=0.8)

    args = parser.parse_args()

    # Expand the local_dir path
    local_dir = os.path.expanduser(args.local_dir)
    
    # Load the dataset
    samples = load_sc2_dataset(args.input_json)
    
    # Calculate split points
    total_copies = 100
    train_copies = int(total_copies * args.train_ratio)
    
    # Process training data
    train_data = []
    for i in range(train_copies):
        for sample in samples:
            train_data.append(process_sample(sample, len(train_data), 'train'))

    # Process test data
    test_data = []
    for i in range(total_copies - train_copies):
        for sample in samples:
            test_data.append(process_sample(sample, len(test_data), 'test'))

    # Convert to DataFrames
    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Save as parquet
    train_output_path = os.path.join(local_dir, 'train.parquet')
    test_output_path = os.path.join(local_dir, 'test.parquet')
    
    train_df.to_parquet(train_output_path)
    test_df.to_parquet(test_output_path)

    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
