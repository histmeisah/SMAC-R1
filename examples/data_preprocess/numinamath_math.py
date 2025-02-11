import os
import json
from datasets import Dataset
import argparse
from verl.utils.hdfs_io import copy, makedirs

def make_prefix(dp, template_type):
    question = dp['input'][0]['content']
    if template_type == 'base':
        prefix = f"""The user asks a math problem, and the Assistant solves it. The assistant first thinks about the solution process step by step in the mind and then provides the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> solution steps here </think><answer> final answer here </answer>. After thinking, when you finally reach a conclusion, clearly state the answer within <answer> </answer> tags.\n\nUser: {question}\nAssistant: <think>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are a helpful math assistant. The assistant first thinks about the solution process step by step and then provides the final answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively. After thinking, clearly state the answer within <answer> </answer> tags.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
<think>"""
    return prefix

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./NuminaMath_data')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--data_path', default='/home/ma-user/modelarts/work/weiyu/python/rlhf_project/NuminaMath-CoT/ppo_data/stratified_dataset_10000per_level.jsonl')
    parser.add_argument('--train_size', type=float, default=0.9)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--template_type', type=str, default='qwen-instruct')
    
    args = parser.parse_args()
    
    data_source = 'numina_math'

    # 加载JSONL数据集
    def gen_from_jsonl(path):
        with open(path) as f:
            for line in f:
                if line.strip():  # 跳过空行
                    yield json.loads(line)
    
    raw_dataset = Dataset.from_generator(gen_from_jsonl, gen_kwargs={'path': args.data_path})
    print(f"Total dataset size: {len(raw_dataset)}")

    # 修改数据集分割逻辑
    total_size = len(raw_dataset)
    train_size = int(total_size * args.train_size)  # 使用比例计算训练集大小
    test_size = total_size - train_size             # 剩余作为测试集

    # 分割数据集
    train_dataset = raw_dataset.select(range(train_size))
    test_dataset = raw_dataset.select(range(train_size, total_size))

    def make_map_fn(split):
        def process_fn(example, idx):
            # 生成带模板的问题
            question = make_prefix(example, template_type=args.template_type)
            
            # 直接使用原始答案格式
            solution = example['standard_answer']
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx
                }
            }
            return data
        return process_fn

    # 处理数据集
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    # 验证数据
    def validate_data(dataset, split):
        """验证数据集格式"""
        print(f"\nValidating {split} dataset:")
        print(f"Dataset size: {len(dataset)}")
        
        # 检查第一个样本
        sample = dataset[0]
        print("\nSample structure:")
        for key, value in sample.items():
            print(f"{key}:")
            print(f"  Type: {type(value)}")
            print(f"  Content: {value}")
            
        # 验证必需字段
        required_fields = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
        for item in dataset:
            missing_fields = [field for field in required_fields if field not in item]
            if missing_fields:
                print(f"Warning: Missing fields {missing_fields}")
                
        return len(dataset) > 0

    assert validate_data(train_dataset, 'train'), "Training dataset is empty!"
    assert validate_data(test_dataset, 'test'), "Test dataset is empty!"

    # 创建本地目录
    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    # 保存为parquet格式
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    # 如果指定了HDFS目录，则复制到HDFS
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        
    print(f"Processed {train_size} training examples and {test_size} test examples")
    print(f"Data saved to {local_dir}")

def _default_compute_score(data_source, solution_str, ground_truth):
    print(f"Computing score for data source: {data_source}")
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        return math.compute_score(solution_str, ground_truth)
    elif data_source == 'numina_math':
        return NuminaMath.compute_score(solution_str, ground_truth)
    else:
        print(f"Available data sources: openai/gsm8k, lighteval/MATH, numina_math")
        raise NotImplementedError(f"Unsupported data source: {data_source}")