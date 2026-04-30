# 数据评分命令

 # 第 1 步：448 结果切分
 SCORES_FILE=outputs/gsm8k_p_scores_manual_shell_1_K32_448.jsonl \
KEEP_SCORES_OUT=outputs/gsm8k_p_scores_448_p1_keep.jsonl \
REFINE_UIDS_OUT=outputs/gsm8k_p_scores_448_p_lt_1_need_1024_uids.jsonl \
./run_split_p_for_refine.sh

生成：

outputs/gsm8k_p_scores_448_p1_keep.jsonl
含义：448 下 p=1 的样本，最终直接保留。

outputs/gsm8k_p_scores_448_p_lt_1_need_1024_uids.jsonl
含义：448 下 p<1 的 uid，用 1024 重评。

# 第 2 步：1024 重评 p<1 的样本
CONDA_ENV_NAME=grpo_b \
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=$HOME/.cache/huggingface \
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
ADAPTER_PATH= \
SPLIT=train \
RESUME=1 \中断续跑
MAX_SAMPLES=0 \
K=32 \
MAX_NEW_TOKENS=1024 \
GENERATION_BATCH_SIZE=16 \
PROMPT_BATCH_SIZE=2 \
PROMPT_STYLE=short \
USE_4BIT=1 \
UID_FILE=outputs/gsm8k_p_scores_448_p_lt_1_need_1024_uids.jsonl \
OUT_FILE=outputs/gsm8k_p_scores_1024_nontruncated.jsonl \
TRUNCATION_OUT=outputs/gsm8k_p_scores_1024_truncated_need_2048_uids.jsonl \
SUMMARY_FILE=outputs/gsm8k_p_summary_1024.json \
bash run_gsm8k_p_filter.sh

生成：

outputs/gsm8k_p_scores_1024_nontruncated.jsonl
含义：1024 下没有截断的完整评分结果。

outputs/gsm8k_p_scores_1024_truncated_need_2048_uids.jsonl
含义：1024 下仍然截断的 uid，下一轮 2048 用。

中断续跑：

RESUME=1 \
UID_FILE=outputs/gsm8k_p_scores_448_p_lt_1_need_1024_uids.jsonl \
OUT_FILE=outputs/gsm8k_p_scores_1024_nontruncated.jsonl \
TRUNCATION_OUT=outputs/gsm8k_p_scores_1024_truncated_need_2048_uids.jsonl \
...
bash run_gsm8k_p_filter.sh
第 3 步：检查 1024 是否还有截断

wc -l outputs/gsm8k_p_scores_1024_truncated_need_2048_uids.jsonl
如果是 0，可以合并最终文件。

如果不是 0，继续 2048。

第 4 步：2048 重评 1024 仍截断的样本

CONDA_ENV_NAME=grpo_b \
CUDA_VISIBLE_DEVICES=0 \
HF_HOME=$HOME/.cache/huggingface \
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
ADAPTER_PATH= \
SPLIT=train \
MAX_SAMPLES=0 \
K=32 \
MAX_NEW_TOKENS=2048 \
GENERATION_BATCH_SIZE=8 \
PROMPT_BATCH_SIZE=1 \
PROMPT_STYLE=short \
USE_4BIT=1 \
UID_FILE=outputs/gsm8k_p_scores_1024_truncated_need_2048_uids.jsonl \
OUT_FILE=outputs/gsm8k_p_scores_2048_nontruncated.jsonl \
TRUNCATION_OUT=outputs/gsm8k_p_scores_2048_truncated_need_3072_uids.jsonl \
SUMMARY_FILE=outputs/gsm8k_p_summary_2048.json \
bash run_gsm8k_p_filter.sh
继续检查：

wc -l outputs/gsm8k_p_scores_2048_truncated_need_3072_uids.jsonl
如果还有，就继续按同样方式跑 3072。

第 5 步：最终合并
如果 1024 已经没有截断：

SCORE_FILES="outputs/gsm8k_p_scores_448_p1_keep.jsonl outputs/gsm8k_p_scores_1024_nontruncated.jsonl" \
OUT_FILE=outputs/gsm8k_p_scores_final_no_truncation.jsonl \
EXPECTED_COUNT=7473 \
./run_merge_score_files.sh
如果跑到了 2048：

SCORE_FILES="outputs/gsm8k_p_scores_448_p1_keep.jsonl outputs/gsm8k_p_scores_1024_nontruncated.jsonl outputs/gsm8k_p_scores_2048_nontruncated.jsonl" \
OUT_FILE=outputs/gsm8k_p_scores_final_no_truncation.jsonl \
EXPECTED_COUNT=7473 \
./run_merge_score_files.sh
最终文件：

outputs/gsm8k_p_scores_final_no_truncation.jsonl
含义：

全量 7473 条
每条都有 p
最终纳入的评分结果都来自“没有截断”的那一轮