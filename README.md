
## 代码说明
代码位于weiyu_branch 下
### 核心修改
代码基于verl 进行修改实现，核心修改部分为```verl/verl/utils/reward_score/reward_compute_llm_smac.py```, 实现了能够从LLM的输出解析出代码，在星际争霸2 中运行计算得到胜率，如果使用自己的框架可以直接迁移这部分即可。

### 其他修改

修改了reward manager ```verl/verl/workers/reward_manager```，实现了保存reward log以及模型和星际争霸2游戏的日志。
### 环境安装

python==3.10

pip install -r requirements.txt

### 模型与游戏

基础模型与星际争霸2 linux版本：

通过网盘分享的文件：qwen2.5-dpo等2个文件
链接: https://pan.baidu.com/s/1HM_DRzigYHWiSnmxdh-UDA 提取码: urey 
--来自百度网盘超级会员v4的分享

### 训练数据

位于：/verl/examples/data_preprocess/data/sc2_instruction/train.parquet   ；  verl/examples/data_preprocess/data/sc2_instruction/test.parquet

原始实现是
## 训练bash脚本
```bash
set -x

# Kill all existing ray processes
ray stop
sleep 2
pkill -9 ray
sleep 2
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export SC2PATH="/home/ma-user/modelarts/work/weiyu/python/StarCraftII"
# Add verl to PYTHONPATH
export PYTHONPATH="/home/ma-user/modelarts/work/weiyu/python/:${PYTHONPATH}"

# Set experiment name and paths
EXPERIMENT_NAME="LLM_smac_grpo_refine_${TIMESTAMP}"
CHECKPOINT_ROOT="/home/ma-user/modelarts/work/weiyu/python/verl_project/checkpoints"
HDFS_DIR="${CHECKPOINT_ROOT}/hdfs/${EXPERIMENT_NAME}"
LOCAL_DIR="${CHECKPOINT_ROOT}/local/${EXPERIMENT_NAME}"

# Create checkpoint directories if they don't exist
mkdir -p "${HDFS_DIR}"
mkdir -p "${LOCAL_DIR}"

# 添加日志相关的环境变量
export LOG_DIR="${LOCAL_DIR}/logs"
mkdir -p "${LOG_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/ma-user/modelarts/work/weiyu/python/verl_project/verl/examples/data_preprocess/data/sc2_instruction/train.parquet \
    data.val_files=/home/ma-user/modelarts/work/weiyu/python/verl_project/verl/examples/data_preprocess/data/sc2_instruction/test.parquet \
    data.train_batch_size=32 \
    data.val_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    actor_rollout_ref.model.path=/home/ma-user/modelarts/work/weiyu/test_project/qwen2.5-dpo \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_llm_smac' \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=1 \
    trainer.total_epochs=100 \
    trainer.default_hdfs_dir="${HDFS_DIR}" \
    trainer.default_local_dir="${LOCAL_DIR}" \
    trainer.resume_mode="disable" \
    +trainer.log_dir="${LOG_DIR}" \
    +trainer.log_level="INFO" \
    +trainer.print_samples=True \
    +trainer.num_examine=2 \
    +trainer.save_rewards=true \
    +trainer.reward_logs_dir="reward_logs" \
    +trainer.reward_save_frequency=100 \
    trainer.val_generations_to_log_to_wandb=5 \
    2>&1 | tee "${LOG_DIR}/training.log"  # 将输出同时写入到日志文件

# single gpu test

# Checkpoint related parameters explanation:
# trainer.default_hdfs_dir - HDFS checkpoint directory
# trainer.default_local_dir - Local checkpoint directory
# trainer.save_freq=5 - Save checkpoint every 5 steps
```
