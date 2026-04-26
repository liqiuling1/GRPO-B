import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _log(message: str) -> None:
    print(message, flush=True)


def _candidate_hf_homes() -> List[Path]:
    repo_root = Path(__file__).resolve().parent
    homes: List[Path] = []
    seen: set[Path] = set()

    for raw_path in [
        os.environ.get("HF_HOME"),
        str(repo_root / "hf_cache"),
        str(Path.home() / ".cache" / "huggingface"),
    ]:
        if not raw_path:
            continue
        path = Path(raw_path).expanduser().resolve()
        if path not in seen:
            homes.append(path)
            seen.add(path)
    return homes


def resolve_cached_model_path(model_name_or_path: str) -> str:
    if os.path.isdir(model_name_or_path):
        return model_name_or_path

    repo_name = f"models--{model_name_or_path.replace('/', '--')}"
    for hf_home in _candidate_hf_homes():
        repo_dir = hf_home / "hub" / repo_name
        if not repo_dir.is_dir():
            continue

        ref_path = repo_dir / "refs" / "main"
        if ref_path.is_file():
            revision = ref_path.read_text(encoding="utf-8").strip()
            snapshot_dir = repo_dir / "snapshots" / revision
            if snapshot_dir.is_dir():
                return str(snapshot_dir)

        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.is_dir():
            candidates = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
            if candidates:
                return str(candidates[-1])

    return model_name_or_path


def get_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def build_bnb_config(use_4bit: bool = True) -> Optional[BitsAndBytesConfig]:
    if not use_4bit:
        return None

    compute_dtype = get_compute_dtype()
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def load_tokenizer(model_name_or_path: str):
    resolved_path = resolve_cached_model_path(model_name_or_path)
    local_only = os.path.isdir(resolved_path)
    tokenizer = AutoTokenizer.from_pretrained(
        resolved_path,
        trust_remote_code=True,
        local_files_only=local_only,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


def load_model_for_training(
    model_name: str,
    use_4bit: bool = True,
):
    compute_dtype = get_compute_dtype()
    quantization_config = build_bnb_config(use_4bit=use_4bit)
    resolved_model_name = resolve_cached_model_path(model_name)
    local_only = os.path.isdir(resolved_model_name)

    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        local_files_only=local_only,
    )

    model.config.use_cache = False

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    return model


@torch.no_grad()
def load_model_for_inference(
    base_model_name: str,
    adapter_path: Optional[str] = None,
    use_4bit: bool = True,
):
    _log("Loading tokenizer...")
    tokenizer = load_tokenizer(adapter_path or base_model_name)
    compute_dtype = get_compute_dtype()
    quantization_config = build_bnb_config(use_4bit=use_4bit)
    resolved_base_model_name = resolve_cached_model_path(base_model_name)
    local_only = os.path.isdir(resolved_base_model_name)

    _log("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        resolved_base_model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=compute_dtype,
        quantization_config=quantization_config,
        local_files_only=local_only,
    )

    if adapter_path:
        _log("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    _log("Model ready.")
    return model, tokenizer


def _get_input_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_inputs(
    tokenizer,
    model,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
):
    if messages is None:
        if prompt is None:
            raise ValueError("Either `prompt` or `messages` must be provided.")
        messages = [{"role": "user", "content": prompt}]

    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(rendered_prompt, return_tensors="pt")
    device = _get_input_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    return inputs


def truncate_on_stop_strings(text: str, stop_strings: Optional[List[str]] = None) -> str:
    if not stop_strings:
        return text.strip()

    truncated = text
    for stop_string in stop_strings:
        if not stop_string:
            continue
        stop_index = truncated.find(stop_string)
        if stop_index != -1:
            truncated = truncated[:stop_index]
    return truncated.strip()


@torch.no_grad()
def generate_n_responses(
    model,
    tokenizer,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    n: int = 1,
    max_new_tokens: int = 160,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 0,
    do_sample: Optional[bool] = None,
    repetition_penalty: float = 1.0,
    stop_strings: Optional[List[str]] = None,
):
    if do_sample is None:
        do_sample = temperature > 0.0

    inputs = _prepare_inputs(
        tokenizer=tokenizer,
        model=model,
        prompt=prompt,
        messages=messages,
    )
    prompt_length = inputs["input_ids"].shape[1]

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "num_return_sequences": n,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "use_cache": True,
    }

    if do_sample:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    outputs = model.generate(**inputs, **generation_kwargs)

    responses: List[str] = []
    for output_ids in outputs:
        completion_ids = output_ids[prompt_length:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        responses.append(truncate_on_stop_strings(text, stop_strings=stop_strings))

    return responses
