#!/usr/bin/env bash
set -euo pipefail

# =========================
# 你需要改的路径
# =========================
DATA_ROOT="/home/haotian/RLoc/dataset/human_held_device_wifi_indoor_localization_dataset-main/Conference"
OUTPUT_DIR="/home/haotian/RLoc/reproduction/results/Table4/S3/leave_one_user_out/epoch100"

# =========================
# 训练超参数（按需改）
# =========================
EPOCHS=1
BATCH=128
WORKERS=2
LR=1e-3

# 要跑的 seeds（按需改）
SEEDS=(0 1 2 3 4)

# 使用的两张 GPU（如果你只有 0 和 1，就这样）
GPUS=(0 1)

# 可选：每次启动前打印一下
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUT=${OUTPUT_DIR}"
echo "SEEDS=${SEEDS[*]}"
echo "GPUS=${GPUS[*]}"
echo "EPOCHS=${EPOCHS} BATCH=${BATCH} WORKERS=${WORKERS} LR=${LR}"
echo "========================="

mkdir -p "${OUTPUT_DIR}"

# =========================
# 核心逻辑：两个 GPU 一组并行跑
# =========================
i=0
n=${#SEEDS[@]}
while [ $i -lt $n ]; do
  s0=${SEEDS[$i]}
  echo "[LAUNCH] seed=${s0} on GPU=${GPUS[0]}"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} \
    python train.py --data_root "${DATA_ROOT}" --output_dir "${OUTPUT_DIR}" \
      --gpu 0 --seed "${s0}" --epochs "${EPOCHS}" --batch_size "${BATCH}" \
      --num_workers "${WORKERS}" --lr "${LR}" &

  # 第二张卡如果还有 seed 就启动
  j=$((i+1))
  if [ $j -lt $n ]; then
    s1=${SEEDS[$j]}
    echo "[LAUNCH] seed=${s1} on GPU=${GPUS[1]}"
    CUDA_VISIBLE_DEVICES=${GPUS[1]} \
      python train.py --data_root "${DATA_ROOT}" --output_dir "${OUTPUT_DIR}" \
        --gpu 0 --seed "${s1}" --epochs "${EPOCHS}" --batch_size "${BATCH}" \
        --num_workers "${WORKERS}" --lr "${LR}" &
  fi

  # 等这一组跑完，再跑下一组
  wait
  i=$((i+2))
done

echo "✅ All seeds finished."
