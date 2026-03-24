#!/bin/bash
#SBATCH --job-name=curvphormer
#SBATCH --output=./log/curvphormer_%j.out
#SBATCH --error=./log/curvphormer_%j.out
#SBATCH --partition=Star
#SBATCH --nodelist=fuse2
#SBATCH --gres=gpu:H100:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --time=7:00:00

# 创建log目录
mkdir -p ./log

# 记录作业信息
echo "========================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Node: ${SLURMD_NODENAME}"
echo "========================================"

# 激活环境并运行
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Jing
cd /home/jingyj/HYX

# 运行Python脚本
python -u /home/jingyj/HYX/0325gpt_improve01.py

# 记录结束信息
echo "========================================"
echo "End time: $(date)"
echo "========================================"
