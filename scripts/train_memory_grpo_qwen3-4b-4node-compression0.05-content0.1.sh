python memory_server.py --server_url http://localhost:8001/v1 \
  > server_outputs.log 2>&1 &

#!/usr/bin/env bash
set -xeuo pipefail

compression_ratio_weight=0.05
function_content_reward_weight=0.1
sub_sample_question_ratio=1.0

# Project and experiment configuration
project_name='Mem-alpha'
exp_name="qwen3-4b-4node-compression${compression_ratio_weight}-content${function_content_reward_weight}"

# Memory-specific configurations
export WANDB_API_KEY="your-wandb-api-key-here"
export WANDB_PROJECT="${project_name}"
export WANDB_NAME="${exp_name}"
export BASE_MODEL='Qwen/Qwen3-4B'
export EXPERIMENT_NAME="${exp_name}"

# Data paths
export DATA_DIR='/home/wangyu/data/memalpha'
export ROLLOUT_DATA_DIR="${DATA_DIR}/${exp_name}"
TRAIN_DATA_DIR='/path/to/current_directory/data/memalpha/'
TEST_DATA_DIR='/path/to/current_directory/data/memalpha/'

# Algorithm parameters
adv_estimator=grpo
use_kl_loss=true
kl_loss_coef=0.001
kl_loss_type=low_var_kl
use_kl_in_reward=False
kl_coef=0.0

# Memory-specific parameters
max_prompt_length=4096
max_response_length=2048
max_start_length=2048
max_obs_length=500
enable_thinking=false
max_turns=5
respond_url="http://127.0.0.1:5000/batch_process"
analyze_function_url="http://127.0.0.1:5000/analyze_function"
use_memory_mode=true
do_search=true
customized_grpo_rollout_n=8

# Batch sizes (adjusted for 4 nodes)
train_batch_size=32
val_batch_size=32

# Ray configuration
RAY_ADDRESS=${RAY_ADDRESS:-""}
WORKING_DIR="/home/wangyu/work/Mem-alpha"
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-${PET_NNODES:-4}}

# Model and checkpoint paths
MODEL_PATH=${MODEL_PATH:-"${BASE_MODEL}"}
CKPTS_DIR=${CKPTS_DIR:-"/home/wangyu/ckpt/${exp_name}"}

# Training parameters
temperature=1.0
top_p=1.0
top_k=-1
offload=true

# Set important environment variables
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# Distributed Ray setup
NODE_RANK=${PET_NODE_RANK:-0}
MASTER_ADDR=${PET_MASTER_ADDR:-"localhost"}

# Resolve hostname to IP address if needed
if [ "$MASTER_ADDR" != "localhost" ]; then
    # Try to resolve hostname to IP address
    RESOLVED_IP=$(getent hosts "$MASTER_ADDR" | awk '{ print $1 }' | head -1)
    if [ -n "$RESOLVED_IP" ]; then
        echo "Resolved $MASTER_ADDR to $RESOLVED_IP"
        MASTER_ADDR="$RESOLVED_IP"
    else
        echo "Warning: Could not resolve $MASTER_ADDR, using as-is"
    fi
fi

# Set RAY_ADDRESS after hostname resolution
if [ -z "$RAY_ADDRESS" ]; then
    RAY_ADDRESS="http://${MASTER_ADDR}:8265"
fi

if [ "$NODE_RANK" -eq 0 ]; then
    echo "Starting Ray head on master node (rank $NODE_RANK)"
    # Stop any existing Ray processes
    ray stop || true
    # Start Ray head
    ray start --head --dashboard-host=0.0.0.0 --dashboard-port=8265
    echo "Ray head started at $MASTER_ADDR:8265"

    # sleep 10 seconds:
    sleep 10

    # Submit the job only from master node
    ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
        --working-dir "${WORKING_DIR}" \
        -- python3 -m verl.trainer.main_ppo \
        data.train_files="${TRAIN_DATA_DIR}/train.parquet" \
        data.val_files="${TEST_DATA_DIR}/test.parquet" \
        data.train_data_num=null \
        data.val_data_num=null \
        data.train_batch_size=${train_batch_size} \
        data.val_batch_size=${val_batch_size} \
        data.max_prompt_length=${max_prompt_length} \
        data.max_response_length=${max_response_length} \
        data.max_start_length=${max_start_length} \
        data.max_obs_length=${max_obs_length} \
        data.shuffle_train_dataloader=True \
        algorithm.adv_estimator=${adv_estimator} \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.enable_gradient_checkpointing=true \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
        actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
        actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
        actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.top_p=${top_p} \
        actor_rollout_ref.rollout.top_k="${top_k}" \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
        trainer.logger=['console','wandb'] \
        trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
        trainer.val_only=false \
        trainer.val_before_train=false \
        trainer.validation_data_dir="${ROLLOUT_DATA_DIR}/validation"\
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes="${NNODES}" \
        trainer.save_freq=1 \
        trainer.test_freq=50 \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.total_epochs=5 \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.resume_mode=auto \
        reward_model.compression_ratio_weight=${compression_ratio_weight} \
        reward_model.function_content_reward_weight=${function_content_reward_weight} \
        enable_thinking=${enable_thinking} \
        max_turns=${max_turns} \
        respond_url="${respond_url}" \
        analyze_function_url="${analyze_function_url}" \
        sub_sample_question_ratio=${sub_sample_question_ratio} \
        use_memory_mode=${use_memory_mode} \
        do_search=${do_search} \
        customized_grpo_rollout_n=${customized_grpo_rollout_n}

    echo "Job submitted successfully. Keeping Ray head alive..."

    # Keep master node alive to maintain Ray head
    while true; do
        echo "$(date): Master node is alive, Ray head running"
        sleep 60
    done

else
    echo "Starting Ray worker on node rank $NODE_RANK"
    # Stop any existing Ray processes
    ray stop || true
    # Connect to Ray head
    ray start --address="${MASTER_ADDR}:6379"
    echo "Ray worker connected to $MASTER_ADDR:6379"

    # Keep worker running
    echo "Ray worker is running. Waiting for jobs..."
    while true; do
        echo "$(date): Worker node is alive, Ray worker running"
        sleep 60
        # Check if Ray is still running
        if ! ray status >/dev/null 2>&1; then
            echo "$(date): Ray connection lost. Attempting to reconnect..."
            ray start --address="${MASTER_ADDR}:6379"
        fi
    done
fi
