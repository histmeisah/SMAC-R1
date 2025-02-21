# Copyright 2024 PRIME team and/or its affiliates
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

import asyncio
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import torch

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from .base import BaseRewardManager


async def single_compute_score(evaluation_func, completion, reference, task, executor, timeout=300.):
    loop = asyncio.get_running_loop()
    try:
        # Ensure process_completion is called properly
        tasks = [
            asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    partial(evaluation_func, task, completion, reference)  # Ensure synchronous
                ),
                timeout=timeout)
        ]
        return await asyncio.gather(*tasks)
    except asyncio.TimeoutError:
        print(f"Timeout occurred for completion: {completion}")
        return None  # Default value for timed-out rows
    except Exception as e:
        print(f"Error processing completion: {completion[:10]}, Error: {e}")
        return None  # Default value for failed rows


async def parallel_compute_score_async(evaluation_func, completions, references, tasks, num_processes=64):
    scores = []
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Create tasks for all rows
        tasks_async = [
            single_compute_score(evaluation_func, completion, reference, task, executor, timeout=300.)
            for completion, reference, task in zip(completions, references, tasks)
        ]
        # to prevent very occasional starvation caused by some anomalous programs ( like infinite loop ), the exceptions in async programs will instantly halt the evaluation, and all summoned processes will be killed.
        try:
            results = await asyncio.gather(*tasks_async, return_exceptions=False)
        except:
            for pid, proc in executor._processes.items():
                try:
                    proc.kill()
                except Exception as kill_err:
                    print('shut down failed: ' + str(kill_err))
            raise

    # Process results
    for result, completion, reference, task in zip(results, completions, references, tasks):
        if isinstance(result, Exception) or result is None:
            # Handle failed or timed-out tasks
            scores.append(0.0)
        elif isinstance(result[0], (int, float, bool)):
            scores.append(float(result[0]))
        else:
            scores.append(float(result[0][0]))
    return scores


class PrimeRewardManager(BaseRewardManager):
    """The Prime reward manager with data logging capability."""

    def __init__(self, tokenizer, num_examine, compute_score=None, config=None) -> None:
        super().__init__(tokenizer, num_examine, compute_score, config)

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""
        # Get current step from data meta info if available
        current_step = data.meta_info.get('global_steps', None)

        # If there is rm score, we directly return rm score
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        
        # 解码序列
        prompt_strs = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        response_strs = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']

        try:
            scores = asyncio.run(
                parallel_compute_score_async(self.compute_score,
                                          sequences_str,
                                          ground_truth,
                                          data_sources,
                                          num_processes=64))
        except (asyncio.TimeoutError, Exception) as e:
            print(f'Error in reward computing: {e}. Setting all scores to 0.')
            scores = [0. for _ in range(len(sequences_str))]

        for i in range(len(data)):
            data_source = data_sources[i]
            score = scores[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = score

            # 记录样本数据，添加step信息
            self.log_sample(
                prompt_str=prompt_strs[i],
                response_str=response_strs[i],
                ground_truth=ground_truth[i],
                data_source=data_source,
                score=score,
                full_sequence=sequences_str[i],
                step=current_step  # 添加step信息
            )

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str[i])

        return reward_tensor
