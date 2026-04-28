标准 baseline 版本：Qwen2.5-1.5B-Instruct + LoRA + TRL GRPOTrainer + GSM8K

保留文件：
- dataset_utils.py
- reward_utils.py
- model_utils.py
- evaluate.py
- train_grpo.py
- run_train.sh
- run_lm_eval_gsm8k.sh
- run_infer.sh
- requirements.txt
- readme.txt

删除或忽略：
- grpo.py
- selector.py
- __pycache__/
- logs/
- hf_cache/
- .vscode/
- grpo_qwen_math_qlora/
- grpo_qwen_math_qlora_test/

安装：
1) pip install -U -r requirements.txt
2) accelerate config

训练：
bash run_train.sh

评测：
bash run_eval.sh

单条推理：
bash run_infer.sh

训练日志画图：
1) 先安装 matplotlib
   pip install matplotlib
2) 列出日志里可画的指标
   python plot_train_metrics.py --log_file logs/train_20260318_234552.log --list_metrics
3) 画默认常用指标
   python plot_train_metrics.py --log_file logs/train_20260318_234552.log
4) 只画指定指标
   python plot_train_metrics.py --log_file logs/train_20260318_234552.log --metrics reward rewards/gsm8k_correctness_reward/mean entropy

这套 baseline 的关键设置：
- TRL 的 GRPOTrainer
- conversational prompt（system + user），由 tokenizer.apply_chat_template 统一格式化
- reward = correctness + 0.1 * format
- correctness 用最终答案数字比较
- beta = 0.0
- epsilon = 0.2
- loss_type = dapo
- num_generations = 4
- LoRA target_modules = q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj
- 默认低资源配置：train_samples=1000, max_steps=500, 4bit + LoRA

重要约束：
- effective batch size = per_device_train_batch_size * gradient_accumulation_steps
- 这个值必须能被 num_generations 整除
- 默认脚本里 1 * 4 = 4，正好能被 num_generations=4 整除

如果你显存更大，最直接的升级方式：
- train_samples 改成 -1（全量 train）
- max_steps 提到 1000
- num_generations 提到 8
- 同时保证 effective batch size 仍然能被 num_generations 整除

bash run_smoke_test.sh

python train_grpo.py --train_samples 8 --max_steps 2 --use_4bit

bash run_train.sh --foreground 训练前台模式

# 1. 你的最终 GRPO LoRA
bash run_lm_eval_gsm8k.sh \
  --task gsm8k_cot \
  ./grpo_qwen25_15b_gsm8k_lora_grpo_baseline_2500_256_advantage_8X1

# 2. 如果你要精确看某个 checkpoint
bash run_lm_eval_gsm8k.sh \
  --task gsm8k_cot \
  ./grpo_qwen25_15b_gsm8k_lora_grpo_baseline_2500_256_advantage_8X1/checkpoint-2500

# 3. 用完全同协议评 base model，做对比
bash run_lm_eval_gsm8k.sh \
  --task gsm8k_cot \
  --no_adapter


跑评估的命令行
  bash run_lm_eval_gsm8k.sh ./grpo_qwen25_15b_gsm8k_lora_pvar_uid1_0_to_5188/checkpoint-300
  bash run_lm_eval_gsm8k.sh --max_gen_toks 512 ./grpo_qwen25_15b_gsm8k_lora_pvar_uid1_0_to_5188/checkpoint-300
  bash run_lm_eval_gsm8k.sh --max_gen_toks 512 ./grpo_qwen25_15b_gsm8k_lora_grpo_baseline/checkpoint-1000

跑模型的命令

MAX_COMPLETION_LENGTH=256 \
bash run_train.sh \
  --mode foreground \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_path /path/to/gsm8k-train.arrow \
  --scores_file outputs/gsm8k_p_scores_manual_shell_1.jsonl \ 
  排序/筛选后的样本文件。训练会按这个 jsonl 里的 uid 顺序取 GSM8K 原始题目，并可用 uid1 范围筛选。
  --start_uid1 0 \ 起始 uid1
  --end_uid1 5188 \ 终止 uid1
  --output_dir ./my_grpo_run \  输出文件目录
  --max_steps 1000 \ 训练最大步数
  --batch_size 8 \
  --grad_acc 1 \
  --num_generations 4 \
  --init_adapter_path /path/to/lora_adapter \
  --prompt_style short \
  /path/to/checkpoint-500

# 测试
MAX_COMPLETION_LENGTH=1024 bash run_train.sh \
  --mode foreground \
  --output_dir ./memtest_1024_g16 \
  --max_steps 5 \
  --batch_size 16 \
  --grad_acc 1 \
  --num_generations 16 \
  --prompt_style short

从某一个点开始
bash run_train.sh \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_path /home/nhlling/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-train.arrow \
  --scores_file outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149_sorted.jsonl \
  --start_uid1 2287 \、
  --end_uid1 5188 \
  --output_dir ./grpo_qwen25_15b_gsm8k_lora_from_uid1_2287 \
  ./grpo_qwen25_15b_gsm8k_lora_from_uid1_2287/checkpoint-2600

从头开始
  bash run_train.sh \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_path /home/nhlling/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-train.arrow \
  --scores_file outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149_sorted.jsonl \
  --start_uid1 2287 \
  --output_dir ./grpo_qwen25_15b_gsm8k_lora_from_uid1_2287

MAX_COMPLETION_LENGTH=1024 \
bash run_train.sh \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_path /home/ling/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-train.arrow \
  --scores_file outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149_pvar_sorted.jsonl \
  --start_uid1 0 \
  --end_uid1 495 \
  --output_dir ./grpo_qwen25_15b_gsm8k_lora_uid1_st_0_end_495_base_1st


  bash run_train.sh \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset_path /home/ling/.cache/huggingface/datasets/openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866/gsm8k-train.arrow \
  --output_dir ./grpo_qwen25_15b_gsm8k_lora_grpo_baseline 

  权重打包
  tar -cf /tmp/grpo_qwen25_15b_gsm8k_lora_grpo_baseline.tar grpo_qwen25_15b_gsm8k_lora_grpo_baseline
  生成 /tmp/grpo_qwen25_15b_gsm8k_lora_grpo_baseline.tar
  U盘复制进文件夹 
  cd /home/nhlling/rl_project
  ls -lh grpo_qwen25_15b_gsm8k_lora_grpo_baseline.tar

  # 样例
  tar -cf /tmp/hf_cache.tar hf_cache
  cd /home/nhlling/rl_project

# 创建目标目录并解压
 mkdir -p /home/nhlling/GRPO-B
 tar -xvf grpo_qwen25_15b_gsm8k_lora_grpo_baseline.tar -C /home/nhlling/GRPO-B/

 # 样例
tar -xvf hf_cache.tar -C /mnt/c/Users/NHLling/Desktop/GRPO-B


 

# 数据抽样 命令行代码
INPUT=outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149.jsonl \ 
P_BIN_EDGES="0,0.25,0.5,0.75,1" \  如果不传参 会按 各个P的百分比抽样 传参就是 在各个区间取分层左闭右开
SAMPLE_PERCENT=10 \ 抽样百分比
bash run_sample_by_p_distribution.sh \
  outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149.jsonl

# 样例
SAMPLED_OUTPUT=outputs/my_sampled.jsonl \
REMAINING_OUTPUT=outputs/my_remaining.jsonl \
SAMPLED_SUMMARY=outputs/my_sampled_summary.json \
REMAINING_SUMMARY=outputs/my_remaining_summary.json \
INPUT=outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149.jsonl \
P_BIN_EDGES="0,0.25,0.5,0.75,1" \
SAMPLE_PERCENT=10 \
bash run_sample_by_p_distribution.sh \
outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149.jsonl

# 直接用
SAMPLED_OUTPUT=outputs/my_sampled_test1.jsonl \
REMAINING_OUTPUT=outputs/my_remaining_test1.jsonl \
SAMPLED_SUMMARY=outputs/my_sampled_summary_test1.json \
REMAINING_SUMMARY=outputs/my_remaining_summary_test1.json \
INPUT=outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149.jsonl \
SAMPLE_PERCENT=10 \
bash run_sample_by_p_distribution.sh \
outputs/gsm8k_deff_scores_k8_t448_tau0.2_20260412_212149.jsonl

# 将代码上传到github
git add .
git commit -m "update"
git push