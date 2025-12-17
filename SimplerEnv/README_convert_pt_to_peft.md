# Конвертация checkpoint .pt в формат PEFT

Скрипт `convert_pt_to_peft.py` позволяет конвертировать checkpoint файлы из rl-research (формат .pt) в формат PEFT, который используется в RL4VLA.

## Использование

### Базовый пример:

```bash
python SimplerEnv/convert_pt_to_peft.py \
    --checkpoint /home/work/WORK/MIPT/repo/rl-research/runs/checkpoints/last23.pt \
    --vla_path models/openvla-7b-fixed \
    --output_dir ./converted_lora
```

### С указанием параметров LoRA:

```bash
python SimplerEnv/convert_pt_to_peft.py \
    --checkpoint /home/work/WORK/MIPT/repo/rl-research/runs/checkpoints/last23.pt \
    --vla_path models/openvla-7b-fixed \
    --output_dir ./converted_lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --unnorm_key fractal20220817_data
```

### Параметры:

- `--checkpoint` (обязательно): Путь к .pt файлу checkpoint из rl-research
- `--vla_path` (обязательно): Путь к базовой модели OpenVLA
- `--output_dir` (обязательно): Директория для сохранения PEFT модели
- `--lora_rank` (опционально): LoRA rank (если не указан, извлекается из checkpoint)
- `--lora_alpha` (опционально): LoRA alpha (если не указан, извлекается из checkpoint)
- `--lora_dropout` (опционально): LoRA dropout (по умолчанию 0.0)
- `--unnorm_key` (опционально): Ключ для нормализации действий (по умолчанию "bridge_orig")
- `--device` (опционально): Устройство для загрузки модели (по умолчанию "cuda:0")

## Использование результата в run_eval.sh

После конвертации используйте полученную директорию в `run_eval.sh`:

```bash
vla_load_path="./converted_lora"
```

## Что создается

Скрипт создает в `output_dir`:
- `adapter_model.bin` (или `adapter_model.safetensors`) - LoRA веса
- `adapter_config.json` - конфигурация LoRA
- `dataset_statistics.json` - статистики для нормализации действий (если доступны)

## Запуск в контейнере

Скрипт можно запустить в Docker контейнере:

```bash
docker compose -f docker/docker-compose.yml run --rm rl4vla \
    python SimplerEnv/convert_pt_to_peft.py \
    --checkpoint /workspace/../rl-research/runs/checkpoints/last23.pt \
    --vla_path models/openvla-7b-fixed \
    --output_dir ./converted_lora
```
