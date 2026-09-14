import re  # 导入正则表达式模块，用于文本模式匹配和替换
from typing import Any, List, Optional  # 导入类型注解，便于描述函数参数和返回值类型

LM_EVAL_STRICT_ANSWER_RE = re.compile(
    r"The answer is (\-?[0-9\.\,]+)\.",
    flags=re.IGNORECASE,
)


def _completion_to_text(completion: Any) -> str:  # 将不同结构的 completion 统一转换为纯文本
    if isinstance(completion, str):  # 如果输入本身就是字符串，直接返回
        return completion  # 返回原始字符串内容

    if isinstance(completion, list):  # 如果 completion 是列表，则逐项拼接为文本
        parts: List[str] = []  # 初始化一个字符串列表，用来收集每个片段
        for item in completion:  # 遍历 completion 列表中的每个元素
            if isinstance(item, dict) and "content" in item:  # 如果元素是包含 content 字段的字典
                parts.append(str(item["content"]))  # 取出 content 字段并转成字符串后加入列表
            else:  # 如果元素不是带 content 的字典
                parts.append(str(item))  # 直接将元素转成字符串后加入列表
        return "\n".join(parts).strip()  # 用换行符拼接所有片段并去掉首尾空白

    return str(completion)  # 其他类型统一转成字符串返回


def _strip_boxed(text: str) -> str:  # 清理答案文本中的 LaTeX boxed 和美元符号
    text = text.strip()  # 去掉文本首尾空白字符
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)  # 将 \boxed{...} 替换成其中的内容
    text = text.replace("$", "")  # 移除 LaTeX 数学环境常见的美元符号
    return text.strip()  # 再次去掉首尾空白后返回


def extract_final_answer(text: str) -> str:  # 从完整输出文本中提取最终答案
    if not text:  # 如果输入文本为空
        return ""  # 返回空字符串表示没有答案

    match = LM_EVAL_STRICT_ANSWER_RE.search(_strip_boxed(text))  # lm-eval gsm8k_cot 默认严格提取 The answer is <number>.
    if not match:
        return ""
    return match.group(1).strip()


def extract_gold_answer_target(text: str) -> str:
    if not text:
        return ""
    # 对齐 lm-eval 的 doc_to_target: answer.split('####')[-1].strip()
    return _strip_boxed(text).split("####")[-1].strip()


def extract_last_number(text: str) -> Optional[str]:  # 从文本中提取最后一个数字字符串
    if not text:  # 如果输入为空
        return None  # 返回 None 表示没有可提取的数字

    text = text.replace(",", "")  # 移除千分位逗号，避免影响数字提取
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)  # 匹配文本中所有整数或小数形式的数字
    if not numbers:  # 如果没有找到任何数字
        return None  # 返回 None 表示提取失败
    return numbers[-1]  # 返回最后一个数字，通常视为最终数值答案


def normalize_text_answer(text: str) -> str:  # 对文本答案做归一化，方便后续比较
    text = _strip_boxed(text)  # 先清理 boxed 和美元符号等格式内容
    text = text.strip().lower()  # 去掉首尾空白并统一转成小写
    text = text.replace(",", "")  # 去掉逗号，减少格式差异的影响
    text = re.sub(r"\s+", " ", text)  # 将连续空白折叠为单个空格
    text = text.rstrip(".")  # 去掉末尾句号，避免标点导致误判
    return text  # 返回归一化后的文本答案


def score_prediction_against_answer(prediction_text: str, gold_answer: str) -> float:  # 比较预测答案和标准答案并返回分数
    pred_final = extract_final_answer(prediction_text)  # 从预测文本中提取最终答案
    gold_final = extract_gold_answer_target(gold_answer)

    if not pred_final or not gold_final:
        return 0.0

    pred_text = normalize_text_answer(pred_final)  # 对齐 lm-eval exact_match 中对逗号、美元符、句号的忽略规则
    gold_text = normalize_text_answer(gold_final)
    return float(pred_text == gold_text)  # 数字完全一致记为 1.0，否则记为 0.0


def gsm8k_correctness_reward(completions, answer, **kwargs) -> List[float]:  # 计算一批样本在 GSM8K 任务上的正确性奖励
    rewards: List[float] = []  # 初始化奖励列表
    for completion, gold in zip(completions, answer):  # 同时遍历模型输出和对应标准答案
        text = _completion_to_text(completion)  # 将当前 completion 转换为文本
        rewards.append(score_prediction_against_answer(text, gold))  # 计算当前样本得分并加入奖励列表
    return rewards  # 返回整批样本的奖励结果


def final_answer_format_reward(completions, **kwargs) -> List[float]:  # 检查输出是否符合 Final Answer 格式并给予奖励
    rewards: List[float] = []  # 初始化格式奖励列表
    for completion in completions:  # 遍历所有模型输出
        text = _completion_to_text(completion)  # 将当前 completion 转为文本
        has_format = LM_EVAL_STRICT_ANSWER_RE.search(_strip_boxed(text))  # 检查文本中是否存在 lm-eval 的严格答案格式
        rewards.append(1.0 if has_format else 0.0)  # 命中格式给 1.0，否则给 0.0
    return rewards  # 返回整批样本的格式奖励结果


class CompletionTokenLengthReward:
    def __init__(
        self,
        max_length: int,
        good_min: int,
        good_max: int,
        short_min: int = 0,
        max_penalty: float = 1.0,
    ):
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        if short_min < 0:
            raise ValueError("short_min must be non-negative.")
        if good_min < short_min:
            raise ValueError("good_min must be greater than or equal to short_min.")
        if good_max < good_min:
            raise ValueError("good_max must be greater than or equal to good_min.")
        if good_max >= max_length:
            raise ValueError("good_max must be smaller than max_length.")
        if max_penalty < 0:
            raise ValueError("max_penalty must be non-negative.")

        self.max_length = max_length
        self.good_min = good_min
        self.good_max = good_max
        self.short_min = short_min
        self.max_penalty = max_penalty
        self.__name__ = "completion_token_length_reward"

    def __call__(self, completions, completion_ids=None, **kwargs) -> List[float]:
        rewards: List[float] = []
        if completion_ids is None:
            # Fallback for older trainer APIs. Character length is only a rough proxy,
            # but keeps the reward callable safe if token ids are unavailable.
            lengths = [len(_completion_to_text(completion)) for completion in completions]
            good_min = self.good_min * 4
            good_max = self.good_max * 4
            short_min = self.short_min * 4
            max_length = self.max_length * 4
        else:
            lengths = [len(ids) for ids in completion_ids]
            good_min = self.good_min
            good_max = self.good_max
            short_min = self.short_min
            max_length = self.max_length

        short_penalty_span = max(good_min - short_min, 1)
        long_penalty_span = max(max_length - good_max, 1)
        for length in lengths:
            if length < good_min:
                penalty_ratio = min(max((good_min - length) / short_penalty_span, 0.0), 1.0)
                rewards.append(-self.max_penalty * penalty_ratio)
                continue
            if length <= good_max:
                rewards.append(0.0)
                continue
            penalty_ratio = min(max((length - good_max) / long_penalty_span, 0.0), 1.0)
            rewards.append(-self.max_penalty * penalty_ratio)
        return rewards
