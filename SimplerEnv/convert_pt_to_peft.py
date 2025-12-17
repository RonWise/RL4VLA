#!/usr/bin/env python3
"""
Скрипт для конвертации checkpoint .pt файла из rl-research в формат PEFT.

Использование:
    python convert_pt_to_peft.py \
        --checkpoint /path/to/rl-research/runs/checkpoints/last23.pt \
        --vla_path models/openvla-7b-fixed \
        --output_dir ./converted_lora \
        [--lora_rank 16] \
        [--lora_alpha 32] \
        [--unnorm_key bridge_orig]
"""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from prismatic.extern.hf.modeling_prismatic import \
    OpenVLAForActionPredictionWithValueHead
from prismatic.extern.hf.processing_prismatic import (PrismaticImageProcessor,
                                                      PrismaticProcessor)
from transformers import AutoTokenizer


def convert_pt_to_peft(
    checkpoint_path: str,
    vla_path: str,
    output_dir: str,
    lora_rank: int = None,
    lora_alpha: int = None,
    lora_dropout: float = 0.0,
    unnorm_key: str = "bridge_orig",
    device: str = "cuda:0",
):
    """
    Конвертирует checkpoint .pt файл из rl-research в формат PEFT.
    
    Args:
        checkpoint_path: Путь к .pt файлу checkpoint
        vla_path: Путь к базовой модели OpenVLA
        output_dir: Директория для сохранения PEFT модели
        lora_rank: Rank LoRA (если None, будет извлечен из checkpoint)
        lora_alpha: Alpha LoRA (если None, будет извлечен из checkpoint)
        lora_dropout: Dropout LoRA
        unnorm_key: Ключ для нормализации действий
        device: Устройство для загрузки модели
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Загрузка checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Извлечение LoRA весов
    if "actor_lora_state_dict" not in checkpoint:
        raise ValueError("Checkpoint не содержит 'actor_lora_state_dict'!")
    
    lora_state_dict = checkpoint["actor_lora_state_dict"]
    print(f"✅ Найдено {len(lora_state_dict)} LoRA весов")
    
    # Определение конфигурации LoRA из checkpoint или параметров
    config = checkpoint.get("config", {})
    if lora_rank is None:
        lora_rank = config.get("lora_rank", 32)
    if lora_alpha is None:
        lora_alpha = config.get("lora_alpha", min(lora_rank, 16))
    
    print(f"📋 LoRA конфигурация: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}")
    
    # Загрузка базовой модели
    print(f"🔄 Загрузка базовой модели: {vla_path}")
    device_obj = torch.device(device)
    
    image_processor = PrismaticImageProcessor.from_pretrained(vla_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(vla_path, trust_remote_code=True, padding_side="left")
    processor = PrismaticProcessor.from_pretrained(
        vla_path,
        image_processor=image_processor,
        tokenizer=tokenizer,
        trust_remote_code=True
    )
    
    vla = OpenVLAForActionPredictionWithValueHead.from_pretrained(
        vla_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device,
        vh_mode="a0",
    )
    
    # Создание PEFT модели с правильной конфигурацией
    print("🔧 Создание PEFT модели...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "proj", "qkv", "fc1", "fc2",  # vision
            "q", "kv", "fc3",  # project
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head",  # llm
        ],
        init_lora_weights="gaussian"
    )
    
    peft_model = get_peft_model(vla, lora_config)
    peft_model.print_trainable_parameters()
    
    # Загрузка LoRA весов
    print("📥 Загрузка LoRA весов из checkpoint...")
    # Ключи в checkpoint уже имеют правильный формат: base_model.model.language_model.model.layers...
    # PEFT модель ожидает такой же формат, так что используем ключи как есть
    # Загружаем веса напрямую
    missing_keys, unexpected_keys = peft_model.load_state_dict(lora_state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠️  Пропущенные ключи ({len(missing_keys)}): {missing_keys[:5]}...")
    if unexpected_keys:
        print(f"⚠️  Неожиданные ключи ({len(unexpected_keys)}): {unexpected_keys[:5]}...")
    
    if not missing_keys and not unexpected_keys:
        print("✅ Все LoRA веса успешно загружены!")
    else:
        print("⚠️  Некоторые ключи не совпали, но это может быть нормально")
    
    # Сохранение в формате PEFT
    print(f"💾 Сохранение PEFT модели в: {output_dir}")
    peft_model.save_pretrained(str(output_dir))
    
    # Сохранение dataset_statistics.json
    dataset_stats_saved = False
    if "norm_stats" in config:
        norm_stats = config["norm_stats"]
        if unnorm_key in norm_stats:
            dataset_stats = {unnorm_key: norm_stats[unnorm_key]}
            with open(output_dir / "dataset_statistics.json", "w") as f:
                json.dump(dataset_stats, f, indent=2)
            print(f"✅ Сохранен dataset_statistics.json с ключом '{unnorm_key}'")
            dataset_stats_saved = True
        else:
            available_keys = list(norm_stats.keys())
            print(f"⚠️  Ключ '{unnorm_key}' не найден в norm_stats")
            print(f"   Доступные ключи: {available_keys}")
            # Сохраняем все доступные ключи
            if norm_stats:
                with open(output_dir / "dataset_statistics.json", "w") as f:
                    json.dump(norm_stats, f, indent=2)
                print(f"✅ Сохранен dataset_statistics.json со всеми доступными ключами")
                print(f"   💡 Если нужен другой ключ, используйте --unnorm_key с одним из: {available_keys}")
                dataset_stats_saved = True
    else:
        print("⚠️  norm_stats не найден в checkpoint")
    
    if not dataset_stats_saved:
        print("⚠️  dataset_statistics.json не будет создан")
    
    print(f"\n✅ Конвертация завершена!")
    print(f"📁 Результат сохранен в: {output_dir}")
    print(f"\n💡 Использование в run_eval.sh:")
    print(f"   vla_load_path=\"{output_dir}\"")


def main():
    parser = argparse.ArgumentParser(
        description="Конвертация checkpoint .pt из rl-research в формат PEFT"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Путь к .pt файлу checkpoint из rl-research"
    )
    parser.add_argument(
        "--vla_path",
        type=str,
        required=True,
        help="Путь к базовой модели OpenVLA"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Директория для сохранения PEFT модели"
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=None,
        help="LoRA rank (если не указан, будет извлечен из checkpoint)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=None,
        help="LoRA alpha (если не указан, будет извлечен из checkpoint)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout"
    )
    parser.add_argument(
        "--unnorm_key",
        type=str,
        default="bridge_orig",
        help="Ключ для нормализации действий"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Устройство для загрузки модели (cuda:0, cpu, etc.)"
    )
    
    args = parser.parse_args()
    
    convert_pt_to_peft(
        checkpoint_path=args.checkpoint,
        vla_path=args.vla_path,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        unnorm_key=args.unnorm_key,
        device=args.device,
    )


if __name__ == "__main__":
    main()

