import argparse  # 导入命令行参数解析模块
import os  # 导入操作系统接口模块，用于目录创建等操作
import random  # 导入随机数模块，用于设置随机种子
from tqdm import tqdm  # 导入 tqdm 用于进度条显示

import torch  # 导入 PyTorch 框架
from peft import LoraConfig, PeftModel, get_peft_model  # 导入 LoRA 配置和模型包装方法
from trl import GRPOConfig  # 导入 GRPO 训练配置

from dataset_utils import build_grpo_dataset  # 导入用于构建 GRPO 训练数据集的函数
from instrumented_grpo_trainer import InstrumentedGRPOTrainer  # 导入带 advantage 监控的 GRPO 训练器
from model_utils import load_model_for_training, load_tokenizer, resolve_cached_model_path  # 导入模型和分词器加载函数
from reward_utils import final_answer_format_reward, gsm8k_correctness_reward  # 导入奖励函数


def seed_everything(seed: int = 42):  # 统一设置各类随机种子以增强实验可复现性
    random.seed(seed)  # 设置 Python 标准库的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch CPU 侧随机种子
    torch.cuda.manual_seed_all(seed)  # 设置所有 CUDA 设备的随机种子


def parse_args():  # 定义并解析训练脚本所需的命令行参数
    parser = argparse.ArgumentParser()  # 创建参数解析器对象
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")  # 指定要训练的基础模型名称
    parser.add_argument(  # 添加输出目录相关参数
        "--output_dir",  # 指定模型和检查点的保存路径参数名
        type=str,  # 将输出目录参数解析为字符串
        default="./grpo_qwen25_15b_gsm8k_lora_grpo_baseline",  # 设置默认输出目录
    )  # 结束输出目录参数定义
    parser.add_argument("--seed", type=int, default=42)  # 设置随机种子参数
    parser.add_argument("--train_split", type=str, default="train")  # 设置数据集切分名称参数
    parser.add_argument("--dataset_path", type=str, default=None)  # 设置显式本地训练数据集 arrow 文件路径
    parser.add_argument("--train_samples", type=int, default=1000)  # 设置训练样本数量上限参数
    parser.add_argument("--train_scores_file", type=str, default=None)  # 设置按排序分数文件构造训练集的输入文件
    parser.add_argument("--min_uid1", type=int, default=None)  # 设置从排序文件中的哪个 uid1 开始取样
    parser.add_argument("--max_uid1", type=int, default=None)  # 设置从排序文件中取样时的结束 uid1（包含）
    parser.add_argument("--prompt_style", type=str, default="short", choices=["short", "fewshot"])  # 设置训练时使用的提示词风格
    parser.add_argument("--max_steps", type=int, default=500)  # 设置最大训练步数参数
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)  # 设置每张设备上的训练批大小参数
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)  # 设置梯度累积步数参数
    parser.add_argument("--learning_rate", type=float, default=1e-5)  # 设置学习率参数
    parser.add_argument("--weight_decay", type=float, default=0.0)  # 设置权重衰减参数
    parser.add_argument("--warmup_ratio", type=float, default=0.03)  # 设置学习率预热比例参数
    parser.add_argument("--num_generations", type=int, default=4)  # 设置每个提示生成的样本数参数
    parser.add_argument("--max_completion_length", type=int, default=448)  # 设置最大生成长度参数
    parser.add_argument("--temperature", type=float, default=1.0)  # 设置采样温度参数
    parser.add_argument("--top_p", type=float, default=0.95)  # 设置 top-p 采样参数
    parser.add_argument("--beta", type=float, default=0.0)  # 设置 GRPO 中的 beta 参数
    parser.add_argument("--epsilon", type=float, default=0.2)  # 设置 GRPO 中的 epsilon 参数
    parser.add_argument("--logging_steps", type=int, default=1)  # 设置日志打印间隔参数
    parser.add_argument("--save_steps", type=int, default=100)  # 设置保存检查点的步数间隔参数
    parser.add_argument("--save_total_limit", type=int, default=2)  # 设置最多保留的检查点数量参数
    parser.add_argument("--lora_r", type=int, default=16)  # 设置 LoRA 的秩参数
    parser.add_argument("--lora_alpha", type=int, default=32)  # 设置 LoRA 的 alpha 参数
    parser.add_argument("--lora_dropout", type=float, default=0.05)  # 设置 LoRA 的 dropout 参数
    parser.add_argument("--init_adapter_path", type=str, default=None)  # 设置用于初始化训练的已有 LoRA adapter 路径
    parser.add_argument("--use_4bit", action="store_true")  # 设置是否启用 4bit 量化加载模型
    parser.add_argument("--report_to", type=str, default="none")  # 设置日志上报目标参数
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)  # 设置从指定检查点续训的参数
    return parser.parse_args()  # 解析并返回命令行参数


def main():  # 定义训练脚本主流程
    args = parse_args()  # 解析用户传入的训练参数
    seed_everything(args.seed)  # 设置全局随机种子
    os.makedirs(args.output_dir, exist_ok=True)  # 创建输出目录，如果已存在则忽略错误

    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps  # 计算有效批大小
    if effective_batch_size % args.num_generations != 0:  # 检查有效批大小是否能被生成数整除
        raise ValueError(  # 如果不能整除则抛出异常提示配置不合法
            "effective batch size must be divisible by num_generations. "  # 错误信息第一部分
            f"Got per_device_train_batch_size={args.per_device_train_batch_size}, "  # 错误信息中追加单卡批大小
            f"gradient_accumulation_steps={args.gradient_accumulation_steps}, "  # 错误信息中追加梯度累积步数
            f"num_generations={args.num_generations}."  # 错误信息中追加生成数量
        )  # 结束异常抛出

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()  # 根据硬件能力判断是否使用 bf16
    use_fp16 = torch.cuda.is_available() and not use_bf16  # 在有 CUDA 且不支持 bf16 时退回到 fp16

    print("模型初始化，加载数据中...")  # 打印数据加载开始提示
    train_dataset = build_grpo_dataset(  # 构建用于 GRPO 训练的数据集
        split=args.train_split,  # 指定使用的数据集切分
        dataset_path=args.dataset_path,  # 如果给定显式本地数据集路径，则优先从该路径加载
        max_samples=args.train_samples if args.train_samples > 0 else None,  # 若样本数大于 0 则限制样本数量，否则使用全部
        seed=args.seed,  # 传入随机种子以控制采样顺序
        selected_rows_path=args.train_scores_file,  # 如果给定排序分数文件，则从该文件按顺序构造训练集
        min_uid1=args.min_uid1,  # 如果给定起始 uid1，则仅保留从该 uid1 开始到结尾的数据
        max_uid1=args.max_uid1,  # 如果给定结束 uid1，则仅保留到该 uid1 为止的数据
        prompt_style=args.prompt_style,  # 传入提示词风格以控制训练时的 prompt 模板
    )  # 结束训练数据集构建调用

    tokenizer = load_tokenizer(args.model_name)  # 加载与模型对应的分词器
    print("初始化模型中...")  # 打印模型初始化提示
    model = load_model_for_training(  # 加载用于训练的基础模型
        model_name=args.model_name,  # 指定基础模型名称
        use_4bit=args.use_4bit,  # 根据参数决定是否启用 4bit 量化
    )  # 结束模型加载调用

    if args.init_adapter_path is not None:  # 如果给定已有 LoRA adapter，则以它作为训练初始化
        print(f"加载初始 LoRA adapter: {args.init_adapter_path}")  # 打印初始化来源
        model = PeftModel.from_pretrained(model, args.init_adapter_path, is_trainable=True)  # 加载已有 LoRA 并保持可训练
    else:  # 如果未提供已有 adapter，则创建新的 LoRA 配置
        lora_config = LoraConfig(  # 创建 LoRA 配置对象
            task_type="CAUSAL_LM",  # 指定任务类型为因果语言模型
            r=args.lora_r,  # 设置 LoRA 的低秩维度
            lora_alpha=args.lora_alpha,  # 设置 LoRA 的缩放系数
            lora_dropout=args.lora_dropout,  # 设置 LoRA 的 dropout 概率
            bias="none",  # 指定不对 bias 参数应用 LoRA
            target_modules=[  # 定义要应用 LoRA 的模块
                "q_proj",  # 对注意力中的 query 投影层应用 LoRA
                "k_proj",  # 对注意力中的 key 投影层应用 LoRA
                "v_proj",  # 对注意力中的 value 投影层应用 LoRA
                "o_proj",  # 对注意力输出投影层应用 LoRA
                "up_proj",  # 对前馈网络上投影层应用 LoRA
                "down_proj",  # 对前馈网络下投影层应用 LoRA
                "gate_proj",  # 对前馈网络门控投影层应用 LoRA
            ],  # 结束目标模块列表定义
        )  # 结束 LoRA 配置对象创建
        model = get_peft_model(model, lora_config)  # 将基础模型包装为 PEFT LoRA 模型
    resolved_base_model_path = resolve_cached_model_path(args.model_name)
    for peft_config in model.peft_config.values():
        peft_config.base_model_name_or_path = resolved_base_model_path
    model.print_trainable_parameters()  # 打印当前可训练参数统计信息

    training_args = GRPOConfig(  # 创建 GRPO 训练配置对象
        output_dir=args.output_dir,  # 指定训练输出目录
        max_steps=args.max_steps,  # 设置最大训练步数
        learning_rate=args.learning_rate,  # 设置优化器学习率
        weight_decay=args.weight_decay,  # 设置权重衰减系数
        warmup_ratio=args.warmup_ratio,  # 设置学习率预热比例
        lr_scheduler_type="cosine",  # 使用 cosine 学习率调度器
        per_device_train_batch_size=args.per_device_train_batch_size,  # 设置每设备训练批大小
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # 设置梯度累积步数
        logging_steps=args.logging_steps,  # 设置日志记录间隔
        save_strategy="steps",  # 指定按步数保存模型
        save_steps=args.save_steps,  # 设置保存间隔步数
        save_total_limit=args.save_total_limit,  # 限制保留的检查点数量
        report_to=args.report_to,  # 设置训练日志上报目标
        seed=args.seed,  # 设置训练配置中的随机种子
        bf16=use_bf16,  # 根据硬件能力决定是否启用 bf16
        fp16=use_fp16,  # 根据硬件能力决定是否启用 fp16
        gradient_checkpointing=True,  # 开启梯度检查点以节省显存
        max_grad_norm=1.0,  # 设置梯度裁剪的最大范数
        remove_unused_columns=False,  # 保留数据集中未显式使用的列
        num_generations=args.num_generations,  # 设置每个 prompt 生成的候选数量
        max_completion_length=args.max_completion_length,  # 设置最大生成长度
        temperature=args.temperature,  # 设置采样温度
        top_p=args.top_p,  # 设置 nucleus sampling 的 top-p 值
        repetition_penalty=1.0,  # 设置重复惩罚系数
        beta=args.beta,  # 设置 GRPO 损失中的 beta 参数
        epsilon=args.epsilon,  # 设置 GRPO 损失中的 epsilon 参数
        num_iterations=1,  # 设置每次更新的迭代次数
        scale_rewards="group",  # 指定按组对奖励进行缩放
        loss_type="dapo",  # 指定使用 dapo 类型损失
        reward_weights=[1.0, 0.1],  # 设置两个奖励函数的权重
        log_completions=False,  # 关闭 completion 日志记录
    )  # 结束训练配置创建

    trainer = InstrumentedGRPOTrainer(  # 创建带额外监控的 GRPO 训练器实例
        model=model,  # 传入待训练的模型
        args=training_args,  # 传入训练配置参数
        train_dataset=train_dataset,  # 传入训练数据集
        processing_class=tokenizer,  # 传入用于处理文本的分词器
        reward_funcs=[gsm8k_correctness_reward, final_answer_format_reward],  # 指定训练时使用的奖励函数列表
    )  # 结束训练器创建

    print("开始训练...")  # 打印训练开始提示

    # 使用 TQDM 添加进度条  # 说明下面的循环使用 tqdm 显示训练进度
    for step, batch in tqdm(enumerate(trainer.get_train_dataloader()), total=len(trainer.get_train_dataloader())):  # 手动遍历训练数据加载器并显示进度
        # 确保批次的格式正确  # 说明下面会对 batch 的数据结构做兼容处理
        if isinstance(batch, list):  # 如果当前 batch 是列表形式
            batch = {k: v for k, v in zip(trainer.train_dataset.column_names, batch)}  # 将列表 batch 转换为按列名索引的字典

        # 确保批次中包含 'input_ids' 和 'attention_mask'  # 说明下面仅在关键字段存在时执行训练步
        if 'input_ids' in batch and 'attention_mask' in batch:  # 检查 batch 中是否包含模型前向所需的关键张量
            trainer.training_step(batch, step)  # 手动执行一次训练步骤

    if args.resume_from_checkpoint is not None:  # 如果用户指定了续训检查点
        print(f"从 checkpoint 续训: {args.resume_from_checkpoint}")  # 打印续训来源
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)  # 从指定检查点继续训练
    else:  # 如果没有指定续训检查点
        trainer.train()     # 训练过程通过 trainer.train() 方法进行
    trainer.save_model(args.output_dir)  # 保存训练后的模型权重
    tokenizer.save_pretrained(args.output_dir)  # 保存分词器配置和词表
    print(f"模型已保存到 {args.output_dir}")  # 打印模型保存完成提示


if __name__ == "__main__":  # 仅在当前文件作为主程序运行时执行主函数
    main()  # 调用主函数启动训练流程
