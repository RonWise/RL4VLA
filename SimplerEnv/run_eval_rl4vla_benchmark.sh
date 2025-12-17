
# docker compose -f docker/docker-compose.yml run --rm \
#   -e WANDB_DIR=/workspace/results/test1 \
#   -e cuda=0 \
#   -e vla_path="models/openvla-7b-fixed" \
#   -e vla_unnorm_key="bridge_orig" \
#   -e lora="checkpoints/lora23" \
#   -e num_envs=32 \
#   -e buffer_inferbatch=1 \
#   rl4vla bash SimplerEnv/run_eval.sh

vla_path="${vla_path:-models/openvla-7b-fixed}"
vla_unnorm_key="${vla_unnorm_key:-bridge_orig}"
vla_load_path="${lora:-}"
num_envs="${num_envs:-64}"
buffer_inferbatch="${buffer_inferbatch:-32}"
wandb_dir="${WANDB_DIR:-/workspace/SimplerEnv/results}"

# Export WANDB_DIR for wandb to use
export WANDB_DIR="${wandb_dir}"

# start evaluation
for seed in 0 1 2 ; do
    for env_id in \
       "PutOnPlateInScene25VisionImage-v1" #\
    #   "PutOnPlateInScene25VisionTexture03-v1" \
    #   "PutOnPlateInScene25VisionTexture05-v1" \
    #   "PutOnPlateInScene25VisionWhole03-v1" \
    #   "PutOnPlateInScene25VisionWhole05-v1" \
    #   "PutOnPlateInScene25Carrot-v1" \
    #   "PutOnPlateInScene25Plate-v1" \
    #   "PutOnPlateInScene25Instruct-v1" \
    #   "PutOnPlateInScene25MultiCarrot-v1" \
    #   "PutOnPlateInScene25MultiPlate-v1" \
    #   "PutOnPlateInScene25Position-v1" \
    #   "PutOnPlateInScene25EEPose-v1" \
    #   "PutOnPlateInScene25PositionChangeTo-v1" ; \
    do
      CUDA_VISIBLE_DEVICES=${cuda:-0} XLA_PYTHON_CLIENT_PREALLOCATE=false \
      python SimplerEnv/simpler_env/train_ms3_ppo.py \
        --vla_path="${vla_path}" --vla_unnorm_key="${vla_unnorm_key}" \
        --vla_load_path="${vla_load_path}" \
        --env_id="${env_id}" \
        --seed=${seed} \
        --num_envs=${num_envs} \
        --buffer_inferbatch=${buffer_inferbatch} \
        --no_wandb --only_render
    done
done