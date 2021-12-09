#!/usr/bin/env bash
#SBATCH -n 1
#SBATCH --mem=0
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --output=/tmp-network/user/zaemyung-kim/slurm_logs/silver-discourse-%j.log

n_parts=$1
ind=$2
first_sent_path=$3
second_sent_path=$4

set -e

source /tmp-network/user/zaemyung-kim/projects/discourse_style/env/bin/activate

# Usage: python prepare_silver_dt.py n_parts split_ind first_sent_path second_sent_path
python prepare_silver_dt.py ${n_parts} ${ind} ${first_sent_path} ${second_sent_path}
