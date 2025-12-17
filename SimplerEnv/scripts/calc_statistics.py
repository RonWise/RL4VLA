#!/usr/bin/env python3
"""
Script to collect statistics from wandb runs.

Usage examples:

1. With default paths (SimplerEnv/wandb -> SimplerEnv/scripts/stats):
   docker compose -f docker/docker-compose.yml run --rm rl4vla \
     python SimplerEnv/scripts/calc_statistics.py

2. With custom wandb directory (statistics will be saved to {wandb_dir}/stats):
   docker compose -f docker/docker-compose.yml run --rm rl4vla \
     python SimplerEnv/scripts/calc_statistics.py \
       --wandb_dir /workspace/SimplerEnv/results

3. With explicit output directory:
   docker compose -f docker/docker-compose.yml run --rm rl4vla \
     python SimplerEnv/scripts/calc_statistics.py \
       --wandb_dir /workspace/SimplerEnv/results \
       --output_dir /workspace/SimplerEnv/results/stats

4. Only change output directory (data from default folder):
   docker compose -f docker/docker-compose.yml run --rm rl4vla \
     python SimplerEnv/scripts/calc_statistics.py \
       --output_dir /workspace/SimplerEnv/results/stats
"""

import argparse
from datetime import datetime
from pathlib import Path

import yaml


def main(wandb_dir=None, output_dir=None):
    stats = {}

    # old stats
    sold_path = Path(__file__).parent.parent / "scripts" / "stats"
    solds = sold_path.glob("stats-*.yaml")
    solds = sorted(list(solds), key=lambda x: x.name)
    for s in solds:
        print(f"{s.name}")
    for sold in solds:
        cfg = yaml.safe_load(sold.read_text())
        for load_path, envs in cfg.items():
            if load_path not in stats:
                stats[load_path] = {}
            for env_name, seeds in envs.items():
                if env_name not in stats[load_path]:
                    stats[load_path][env_name] = {}
                for seed, stat in seeds.items():
                    if seed not in stats[load_path][env_name]:
                        stats[load_path][env_name][seed] = {}
                    stats[load_path][env_name][seed].update(stat)

    # wandb
    if wandb_dir is None:
        # Default: use wandb folder (backward compatibility)
        wandb_path = Path(__file__).parent.parent / "wandb" / "wandb"
    else:
        # Custom path: should point to WANDB_DIR/wandb
        wandb_path = Path(wandb_dir) / "wandb"
    
    if not wandb_path.exists():
        print(f"⚠️  Wandb path does not exist: {wandb_path}")
        print(f"   Skipping wandb data collection...")
    else:
        runs = wandb_path.glob("offline-run-*")

        for run in runs:
            cfg = yaml.safe_load((run / "glob" / "config.yaml").read_text())

            load_path = "/".join(cfg["vla_load_path"].split("/")[-3:])
            env_name = cfg["env_id"]
            seed = cfg["seed"]

            if load_path not in stats:
                stats[load_path] = {}
            if env_name not in stats[load_path]:
                stats[load_path][env_name] = {}

            train_vis_dir = run / "glob" / "vis_0_train" / "stats.yaml"
            if train_vis_dir.exists():
                train_stats = yaml.safe_load(train_vis_dir.read_text())
                if "stats" in train_stats:
                    if "train" not in stats[load_path][env_name]:
                        stats[load_path][env_name]["train"] = {}
                    stats[load_path][env_name]["train"][seed] = train_stats["stats"]
                    stats[load_path][env_name]["train"][seed]["path"] = str(run)

            test_vis_dir = run / "glob" / "vis_0_test" / "stats.yaml"
            if test_vis_dir.exists():
                test_stats = yaml.safe_load(test_vis_dir.read_text())
                if "stats" in test_stats:
                    if "test" not in stats[load_path][env_name]:
                        stats[load_path][env_name]["test"] = {}
                    stats[load_path][env_name]["test"][seed] = test_stats["stats"]
                    stats[load_path][env_name]["test"][seed]["path"] = str(run)

    # save stats
    tt = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        # Default: if wandb_dir is specified, save in the same directory, otherwise use scripts/stats
        if wandb_dir is not None:
            # Save in the same directory as wandb_dir
            output_path = Path(wandb_dir) / "stats"
        else:
            # Default: save in SimplerEnv/scripts/stats
            output_path = Path(__file__).parent.parent / "scripts" / "stats"
    else:
        # Custom output directory explicitly specified
        output_path = Path(output_dir)
    
    save_path = output_path / f"stats-{tt}.yaml"

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(stats, f, default_flow_style=False)
    
    print(f"✅ Statistics saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate statistics from wandb runs")
    parser.add_argument(
        "--wandb_dir",
        type=str,
        default=None,
        help="Path to WANDB_DIR (directory containing 'wandb' subdirectory). "
             "Default: SimplerEnv/wandb (for backward compatibility). "
             "Example: SimplerEnv/results or /workspace/SimplerEnv/results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save statistics file. "
             "Default: {wandb_dir}/stats if --wandb_dir is specified, "
             "otherwise SimplerEnv/scripts/stats. "
             "Example: SimplerEnv/results/stats or /workspace/SimplerEnv/results/stats"
    )
    args = parser.parse_args()
    main(wandb_dir=args.wandb_dir, output_dir=args.output_dir)
