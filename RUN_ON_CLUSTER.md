# Running Evaluation Scripts on Cluster

This document describes how to run evaluation and statistics collection scripts on a cluster environment.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with CUDA support
- Access to the cluster with appropriate GPU allocation
- Models directory mounted (default: `models/` symlink or directory)
- Checkpoints directory available (if using LoRA fine-tuned models)

---

## Docker Container Setup

The project includes a Makefile with convenient commands for managing the Docker container. This section describes how to prepare and use the Docker environment.

### Available Make Commands

View all available commands:

```bash
make help
```

### Building the Docker Image

**First-time setup** - Build the Docker image from source:

```bash
make build
```

This will:

- Build the `rl4vla:latest` image using `docker/Dockerfile`
- Install all dependencies (OpenVLA, ManiSkill, SimplerEnv)
- Set up the Python environment with required packages

**Using pre-built image from Docker Hub** (optional):

```bash
export DOCKERHUB_IMAGE=roman2021/rl4vla:latest
make build
```

### Starting the Container

Start the container in detached mode:

```bash
make up
```

This will:

- Start the container with name `rl4vla-container`
- Mount all necessary volumes (project code, models, checkpoints, cache directories)
- Configure GPU access via NVIDIA runtime
- Set up environment variables for wandb, ManiSkill, etc.

### Accessing the Container

Open an interactive shell in the running container:

```bash
make shell
```

### Container Management

**View running containers:**

```bash
make ps
```

**View container logs:**

```bash
make logs
```

**Follow container logs (real-time):**

```bash
make logs-follow
```

**Restart the container:**

```bash
make restart
```

**Stop the container:**

```bash
make down
```

**Rebuild image and restart** (after code changes):

```bash
make rebuild
```

**Clean up** (stop and remove container, keep image):

```bash
make clean
```

### Quick Start Workflow

1. **Build the image** (first time only):

   ```bash
   make build
   ```

2. **Start the container**:

   ```bash
   make up
   ```

3. **Access the container**:

   ```bash
   make shell
   ```

4. **Inside the container**, you can now run evaluation scripts (see sections below)

5. **Stop the container** when done:
   ```bash
   make down
   ```

### Volume Mounts

The container automatically mounts:

- Project directory: `..:/workspace` (entire repo)
- Models: `../models:/workspace/models`
- Checkpoints: `../checkpoints:/workspace/checkpoints`
- Results: `../SimplerEnv/results:/workspace/SimplerEnv/results`
- HuggingFace cache: `~/.cache/huggingface:/root/.cache/huggingface`
- ManiSkill assets: `~/.maniskill:/workspace/.maniskill`
- Sapien cache: `~/.sapien:/root/.sapien`

### Running Commands in Container

You can run commands directly without entering the shell:

```bash
docker exec -it rl4vla-container <command>
```

Or use docker compose:

```bash
docker compose -f docker/docker-compose.yml run --rm rl4vla <command>
```

---

## Scripts Overview

1. **`run_eval_rl4vla_benchmark.sh`** - Evaluates models on RL4VLA benchmark environments
2. **`run_eval_simplerenv_envs.sh`** - Evaluates models on SimplerEnv environments
3. **`calc_statistics.py`** - Collects and aggregates statistics from wandb runs

---

## 1. Running `run_eval_rl4vla_benchmark.sh`

This script evaluates models on the RL4VLA benchmark suite, which includes various vision and generalization tasks.

### Location

`SimplerEnv/run_eval_rl4vla_benchmark.sh`

### Environment Variables

The script accepts the following environment variables (with defaults shown):

- `vla_path` (default: `models/openvla-7b-fixed`) - Path to the base VLA model
- `vla_unnorm_key` (default: `bridge_orig`) - Dataset key for action normalization
- `lora` (default: empty) - Path to LoRA checkpoint directory (e.g., `checkpoints/lora23`)
- `num_envs` (default: `64`) - Number of parallel environments
- `buffer_inferbatch` (default: `32`) - Batch size for inference
- `cuda` (default: `0`) - CUDA device ID
- `WANDB_DIR` (default: `/workspace/SimplerEnv/results`) - Directory for wandb logs

### Usage Examples

#### Using Docker Compose (Recommended)

```bash
docker compose -f docker/docker-compose.yml run --rm \
  -e WANDB_DIR=/workspace/results/benchmark_eval \
  -e cuda=0 \
  -e vla_path="models/openvla-7b-fixed" \
  -e vla_unnorm_key="bridge_orig" \
  -e lora="checkpoints/lora23" \
  -e num_envs=32 \
  -e buffer_inferbatch=1 \
  rl4vla bash SimplerEnv/run_eval_rl4vla_benchmark.sh
```

#### Direct Execution (if running inside container)

```bash
export WANDB_DIR=/workspace/SimplerEnv/results/benchmark_eval
export vla_path="models/openvla-7b-fixed"
export vla_unnorm_key="bridge_orig"
export lora="checkpoints/lora23"
export num_envs=32
export buffer_inferbatch=1
export cuda=0

bash SimplerEnv/run_eval_rl4vla_benchmark.sh
```

### What It Does

The script runs evaluation across:

- **Seeds**: 0, 1, 2 (3 runs per environment)
- **Environments**: Currently configured for `PutOnPlateInScene25VisionImage-v1` (see script for full list of available environments)

Each evaluation run:

- Uses `train_ms3_ppo.py` with `--only_render` flag (evaluation mode)
- Disables wandb logging (`--no_wandb`)
- Saves results to the specified `WANDB_DIR`

### Notes

- The script iterates over all seed/environment combinations automatically
- Results are saved in wandb offline format in `WANDB_DIR/wandb/`
- Each run generates visualization and statistics files

---

## 2. Running `run_eval_simplerenv_envs.sh`

This script evaluates models on SimplerEnv task suite, which includes simpler manipulation tasks.

### Location

`SimplerEnv/run_eval_simplerenv_envs.sh`

### Environment Variables

Same as `run_eval_rl4vla_benchmark.sh`, but with different defaults:

- `vla_path` (default: `models/openvla-7b-fixed`)
- `vla_unnorm_key` (default: `fractal20220817_data`) - Different normalization key
- `lora` (default: empty)
- `num_envs` (default: `64`)
- `buffer_inferbatch` (default: `32`)
- `cuda` (default: `0`)
- `WANDB_DIR` (default: `/workspace/SimplerEnv/results`)

### Usage Examples

#### Using Docker Compose

```bash
docker compose -f docker/docker-compose.yml run --rm \
  -e WANDB_DIR=/workspace/results/simplerenv_eval \
  -e cuda=0 \
  -e vla_path="models/openvla-7b-fixed" \
  -e vla_unnorm_key="fractal20220817_data" \
  -e lora="checkpoints/lora23" \
  -e num_envs=32 \
  -e buffer_inferbatch=1 \
  rl4vla bash SimplerEnv/run_eval_simplerenv_envs.sh
```

#### Direct Execution

```bash
export WANDB_DIR=/workspace/SimplerEnv/results/simplerenv_eval
export vla_path="models/openvla-7b-fixed"
export vla_unnorm_key="fractal20220817_data"
export lora="checkpoints/lora23"
export num_envs=32
export buffer_inferbatch=1
export cuda=0

bash SimplerEnv/run_eval_simplerenv_envs.sh
```

### What It Does

The script runs evaluation across:

- **Seeds**: 0, 1, 2 (3 runs per environment)
- **Environments**:
  - `PutCarrotOnPlateInScene-v1`
  - `PutSpoonOnTableClothInScene-v1`
  - `StackGreenCubeOnYellowCubeBakedTexInScene-v1`
  - `PutEggplantInBasketScene-v1`

### Notes

- Same evaluation workflow as benchmark script
- Uses different action normalization key (`fractal20220817_data` vs `bridge_orig`)
- Results saved in wandb offline format

---

## 3. Running `calc_statistics.py`

This script aggregates statistics from wandb evaluation runs and saves them in YAML format.

### Location

`SimplerEnv/scripts/calc_statistics.py`

### Command Line Arguments

- `--wandb_dir` (optional) - Path to WANDB_DIR (directory containing `wandb` subdirectory)
  - Default: `SimplerEnv/wandb` (for backward compatibility)
  - Example: `SimplerEnv/results` or `/workspace/SimplerEnv/results`
- `--output_dir` (optional) - Directory to save statistics file
  - Default: `{wandb_dir}/stats` if `--wandb_dir` is specified, otherwise `SimplerEnv/scripts/stats`
  - Example: `SimplerEnv/results/stats` or `/workspace/SimplerEnv/results/stats`

### Usage Examples

#### Using Docker Compose

**Default paths** (reads from `SimplerEnv/wandb`, saves to `SimplerEnv/scripts/stats`):

```bash
docker compose -f docker/docker-compose.yml run --rm rl4vla \
  python SimplerEnv/scripts/calc_statistics.py
```

**Custom wandb directory** (saves to `{wandb_dir}/stats`):

```bash
docker compose -f docker/docker-compose.yml run --rm rl4vla \
  python SimplerEnv/scripts/calc_statistics.py \
    --wandb_dir /workspace/SimplerEnv/results
```

**Explicit output directory**:

```bash
docker compose -f docker/docker-compose.yml run --rm rl4vla \
  python SimplerEnv/scripts/calc_statistics.py \
    --wandb_dir /workspace/SimplerEnv/results \
    --output_dir /workspace/SimplerEnv/results/stats
```

**Only change output directory** (data from default folder):

```bash
docker compose -f docker/docker-compose.yml run --rm rl4vla \
  python SimplerEnv/scripts/calc_statistics.py \
    --output_dir /workspace/SimplerEnv/results/stats
```

#### Direct Execution

```bash
python SimplerEnv/scripts/calc_statistics.py \
  --wandb_dir /workspace/SimplerEnv/results \
  --output_dir /workspace/SimplerEnv/results/stats
```

### What It Does

1. **Collects old statistics** from `SimplerEnv/scripts/stats/stats-*.yaml` files
2. **Scans wandb runs** in the specified directory for `offline-run-*` folders
3. **Extracts statistics** from:
   - Training visualization stats: `{run}/glob/vis_0_train/stats.yaml`
   - Test visualization stats: `{run}/glob/vis_0_test/stats.yaml`
4. **Aggregates data** by:
   - Model path (from `vla_load_path` in config)
   - Environment ID
   - Seed
   - Train/Test split
5. **Saves results** to `stats-{timestamp}.yaml` in the output directory

### Output Format

The script generates a YAML file with structure:

```yaml
{model_path}:
  {env_name}:
    train:
      {seed}:
        {statistics}
        path: {run_path}
    test:
      {seed}:
        {statistics}
        path: {run_path}
```

### Notes

- The script merges statistics from both old YAML files and new wandb runs
- If a wandb path doesn't exist, it will skip wandb data collection with a warning
- Statistics are timestamped for easy tracking

---

## Cluster-Specific Considerations

### GPU Allocation

When running on a cluster with job schedulers (SLURM, PBS, etc.):

```bash
# Example SLURM script
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Set GPU device
export cuda=0

# Run evaluation
bash SimplerEnv/run_eval_simplerenv_envs.sh
```

### Resource Requirements

- **GPU Memory**: At least 24GB VRAM recommended for base models
- **CPU**: 4+ cores recommended for parallel environments
- **RAM**: 16GB+ recommended
- **Disk Space**: Ensure sufficient space for wandb logs and statistics

### Parallel Execution

To run multiple evaluations in parallel on different GPUs:

```bash
# Terminal 1
export cuda=0
bash SimplerEnv/run_eval_simplerenv_envs.sh

# Terminal 2
export cuda=1
bash SimplerEnv/run_eval_rl4vla_benchmark.sh
```

### Monitoring

Check evaluation progress:

```bash
# Monitor wandb directory
watch -n 5 'ls -lh /workspace/SimplerEnv/results/wandb/offline-run-*/glob/vis_0_test/'

# Check GPU usage
nvidia-smi
```

---

## Complete Workflow Example

1. **Run SimplerEnv evaluations**:

```bash
docker compose -f docker/docker-compose.yml run --rm \
  -e WANDB_DIR=/workspace/results/simplerenv_eval \
  -e cuda=0 \
  -e vla_path="models/openvla-7b-fixed" \
  -e lora="checkpoints/lora23" \
  rl4vla bash SimplerEnv/run_eval_simplerenv_envs.sh
```

2. **Run benchmark evaluations**:

```bash
docker compose -f docker/docker-compose.yml run --rm \
  -e WANDB_DIR=/workspace/results/benchmark_eval \
  -e cuda=0 \
  -e vla_path="models/openvla-7b-fixed" \
  -e lora="checkpoints/lora23" \
  rl4vla bash SimplerEnv/run_eval_rl4vla_benchmark.sh
```

3. **Collect statistics**:

```bash
docker compose -f docker/docker-compose.yml run --rm rl4vla \
  python SimplerEnv/scripts/calc_statistics.py \
    --wandb_dir /workspace/results/simplerenv_eval \
    --output_dir /workspace/results/stats
```

---

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `num_envs` or `buffer_inferbatch`
2. **Model not found**: Check `vla_path` points to correct model directory
3. **LoRA checkpoint not found**: Verify `lora` path is correct
4. **Wandb directory not found**: Ensure `WANDB_DIR` is set correctly and evaluation has run
5. **Statistics file empty**: Check that wandb runs completed successfully and contain stats files

### Debugging

Enable verbose output:

```bash
set -x  # Enable bash debugging
bash SimplerEnv/run_eval_simplerenv_envs.sh
```

Check wandb run structure:

```bash
ls -R /workspace/SimplerEnv/results/wandb/offline-run-*/
```
