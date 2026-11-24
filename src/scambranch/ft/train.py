"""LoRA fine-tuning on verbalizer first-token CE (lazy peft/transformers imports)."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from scambranch.eval.constants import (
    DETECTOR_INSTRUCTION,
    LABEL_FRAUD,
    LABEL_LEGITIMATE,
    VIEW_FULL,
)
from scambranch.eval.data import Conversation, select_view_turns
from scambranch.eval.detectors import (
    document_verbalizers,
    pairwise_fraud_score,
    resolve_model_spec,
)
from scambranch.eval.resources import pick_device, pick_dtype
from scambranch.ft.data import (
    ARM_BRANCHED_CONSISTENCY,
    SelectionResult,
    assert_no_split_leakage,
)


@dataclass
class TrainConfig:
    model: str
    arm: str
    seed: int
    out_dir: Path
    run_dir: Path
    n: int | None = None
    target_n: int = 432
    lambda_consistency: float = 1.0
    epochs: int = 4
    lr: float = 1e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_length: int | None = None  # None → fit longest; left-truncate if forced
    device: str | None = None
    dtype: str | None = "bf16"
    grad_accum: int = 1
    log_every: int = 10
    early_stop_patience: int = 2
    dry_run: bool = False


@dataclass
class TruncationReport:
    occurred: bool = False
    n_truncated: int = 0
    max_prompt_tokens: int = 0
    max_length_used: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _apply_chat(tokenizer: Any, user_content: str, enable_thinking: bool | None) -> str:
    messages = [{"role": "user", "content": user_content}]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if enable_thinking is False:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        try:
            return tokenizer.apply_chat_template(messages, **kwargs)
        except Exception:
            return f"{user_content}\n\nAnswer:"


def build_prompt_for_conversation(
    conv: Conversation,
    tokenizer: Any,
    *,
    enable_thinking: bool | None,
) -> str:
    turns = select_view_turns(conv.turns, VIEW_FULL, conv.branch_after_turn)
    from scambranch.eval.data import build_detector_input, render_transcript, assert_no_private_leak

    detector_input = build_detector_input(conv.conversation_id, turns)
    transcript = render_transcript(detector_input["turns"])
    user_content = f"{DETECTOR_INSTRUCTION}\n\nConversation:\n{transcript}"
    assert_no_private_leak(user_content, detector_input)
    return _apply_chat(tokenizer, user_content, enable_thinking)


def _label_index(conv: Conversation) -> int:
    """0 = fraud, 1 = legitimate (matches verbalizer order for 2-way CE)."""
    if conv.label in {"fraud_attempt", "fraud"}:
        return 0
    return 1


def discover_lora_targets(model: Any) -> list[str]:
    """Attention + MLP projection Linear module leaf names."""
    wanted_suffixes = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "c_attn",
        "c_proj",
        "c_fc",
    )
    names: set[str] = set()
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in wanted_suffixes:
            names.add(leaf)
    if not names:
        # Fallback common set.
        names = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    return sorted(names)


def _left_truncate_ids(ids: list[int], max_length: int) -> tuple[list[int], bool]:
    if len(ids) <= max_length:
        return ids, False
    return ids[-max_length:], True


def prepare_examples(
    conversations: Sequence[Conversation],
    tokenizer: Any,
    *,
    enable_thinking: bool | None,
    fraud_token_id: int,
    legit_token_id: int,
    max_length: int | None,
) -> tuple[list[dict[str, Any]], TruncationReport, int]:
    """Tokenize FULL-view prompts; optionally left-truncate (never drops answer token)."""
    report = TruncationReport()
    raw_lens: list[int] = []
    prompts: list[str] = []
    for conv in conversations:
        prompt = build_prompt_for_conversation(conv, tokenizer, enable_thinking=enable_thinking)
        prompts.append(prompt)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        raw_lens.append(len(ids))
    report.max_prompt_tokens = max(raw_lens) if raw_lens else 0
    # Fit longest by default; only truncate if an explicit max_length is smaller.
    used_max = int(max_length) if max_length is not None else report.max_prompt_tokens
    if used_max <= 0:
        used_max = report.max_prompt_tokens or 1
    report.max_length_used = used_max

    examples: list[dict[str, Any]] = []
    for conv, prompt, raw_len in zip(conversations, prompts, raw_lens):
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        truncated = False
        if len(ids) > used_max:
            # Left truncate: keep the end of the prompt (answer position + late turns).
            # Spec: never cut the caller's scam ask if avoidable — prefer fitting longest.
            ids, truncated = _left_truncate_ids(ids, used_max)
            report.occurred = True
            report.n_truncated += 1
            if len(report.examples) < 5:
                report.examples.append(
                    {
                        "conversation_id": conv.conversation_id,
                        "raw_tokens": raw_len,
                        "kept_tokens": len(ids),
                        "branch_after_turn": conv.branch_after_turn,
                    }
                )
        examples.append(
            {
                "conversation_id": conv.conversation_id,
                "history_key": conv.history_key,
                "input_ids": ids,
                "label_index": _label_index(conv),
                "fraud_token_id": fraud_token_id,
                "legit_token_id": legit_token_id,
                "truncated": truncated,
            }
        )
    return examples, report, used_max


def _verbalizer_ce_loss(
    logits_last: Any,
    label_index: int,
    fraud_token_id: int,
    legit_token_id: int,
) -> tuple[Any, float]:
    """2-way CE over verbalizer first-token logits. Returns (loss, p_fraud)."""
    import torch
    import torch.nn.functional as F

    pair = torch.stack(
        [logits_last[fraud_token_id], logits_last[legit_token_id]],
        dim=0,
    )
    target = torch.tensor(label_index, device=pair.device, dtype=torch.long)
    loss = F.cross_entropy(pair.unsqueeze(0), target.unsqueeze(0))
    p_fraud = float(torch.softmax(pair, dim=0)[0].item())
    return loss, p_fraud


def _p_fraud_from_logits(logits_last: Any, fraud_token_id: int, legit_token_id: int) -> float:
    import torch

    logit_f = float(logits_last[fraud_token_id].float().item())
    logit_l = float(logits_last[legit_token_id].float().item())
    return pairwise_fraud_score(logit_f, logit_l)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def evaluate_dev_ce(
    model: Any,
    examples: list[dict[str, Any]],
    device: str,
) -> dict[str, float]:
    import torch

    model.eval()
    losses: list[float] = []
    correct = 0
    with torch.inference_mode():
        for ex in examples:
            ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device=device)
            out = model(input_ids=ids)
            logits_last = out.logits[0, -1]
            loss, p_fraud = _verbalizer_ce_loss(
                logits_last,
                ex["label_index"],
                ex["fraud_token_id"],
                ex["legit_token_id"],
            )
            losses.append(float(loss.item()))
            pred = 0 if p_fraud >= 0.5 else 1
            if pred == ex["label_index"]:
                correct += 1
    model.train()
    return {
        "n": float(len(examples)),
        "mean_ce": float(sum(losses) / len(losses)) if losses else float("nan"),
        "acc_at_0_5": float(correct / len(examples)) if examples else float("nan"),
    }


def train_lora(
    config: TrainConfig,
    selection: SelectionResult,
    *,
    dev_conversations: Sequence[Conversation] | None = None,
) -> dict[str, Any]:
    """Train one LoRA adapter. peft/transformers imported lazily."""
    assert_no_split_leakage(selection.conversations)
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    short_id, spec = resolve_model_spec(config.model)
    run_config = {
        "model": config.model,
        "short_id": short_id,
        "hf_id": spec["hf_id"],
        "arm": selection.arm,
        "seed": selection.seed,
        "n": selection.n,
        "lambda_consistency": config.lambda_consistency if selection.arm == ARM_BRANCHED_CONSISTENCY else 0.0,
        "epochs": config.epochs,
        "lr": config.lr,
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "conversation_ids": [c.conversation_id for c in selection.conversations],
        "selection_summary": selection.to_config(),
        "dry_run": config.dry_run,
    }
    _atomic_json(out_dir / "run_config.json", run_config)

    if config.dry_run:
        return {"dry_run": True, "run_config": run_config, "out_dir": str(out_dir)}

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    device = pick_device(config.device)
    dtype = pick_dtype(device, config.dtype)
    if device == "cuda" and config.dtype is None:
        dtype = torch.bfloat16

    torch.manual_seed(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"], trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True, "trust_remote_code": True}
    try:
        model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], dtype=dtype, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], torch_dtype=dtype, **load_kwargs)
    model.to(device)

    # Probe verbalizers with the same chat template as eval.
    dummy_user = f"{DETECTOR_INSTRUCTION}\n\nConversation:\nCaller: hello"
    dummy_prompt = _apply_chat(tokenizer, dummy_user, spec.get("enable_thinking"))
    vdoc = document_verbalizers(tokenizer, dummy_prompt)
    fraud_tid = int(vdoc["chosen"]["verbalizers"][LABEL_FRAUD]["first_token_id"])
    legit_tid = int(vdoc["chosen"]["verbalizers"][LABEL_LEGITIMATE]["first_token_id"])
    if fraud_tid == legit_tid:
        raise RuntimeError("verbalizer first tokens collide")

    train_examples, trunc_report, used_max = prepare_examples(
        selection.conversations,
        tokenizer,
        enable_thinking=spec.get("enable_thinking"),
        fraud_token_id=fraud_tid,
        legit_token_id=legit_tid,
        max_length=config.max_length,
    )
    run_config["max_length"] = used_max
    run_config["truncation"] = trunc_report.to_dict()
    run_config["verbalizer_tokenization"] = vdoc
    _atomic_json(out_dir / "run_config.json", run_config)
    if trunc_report.occurred:
        print(
            f"WARNING: left-truncation occurred for {trunc_report.n_truncated} chats "
            f"(max_prompt={trunc_report.max_prompt_tokens}, max_length={used_max})",
            flush=True,
        )

    dev_examples: list[dict[str, Any]] = []
    if dev_conversations:
        assert_no_split_leakage(dev_conversations, allowed_split="dev")
        # Cap DEV for early stopping speed.
        capped = list(dev_conversations)[:256]
        dev_examples, _, _ = prepare_examples(
            capped,
            tokenizer,
            enable_thinking=spec.get("enable_thinking"),
            fraud_token_id=fraud_tid,
            legit_token_id=legit_tid,
            max_length=used_max,
        )

    targets = discover_lora_targets(model)
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=targets,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.lr,
    )

    # Every arm uses batches of 4 so optimizer steps match across arms.
    # Branched arms batch whole sibling groups (lambda=0 for plain branched);
    # unbranched batches 4 random chats.
    use_consistency = bool(
        selection.arm == ARM_BRANCHED_CONSISTENCY and selection.sibling_groups
    )
    batches: list[list[dict[str, Any]]] = []
    if selection.sibling_groups:
        id_to_ex = {ex["conversation_id"]: ex for ex in train_examples}
        for grp in selection.sibling_groups:
            batch = [id_to_ex[c.conversation_id] for c in grp if c.conversation_id in id_to_ex]
            if len(batch) == 4:
                batches.append(batch)
    else:
        shuffled = list(train_examples)
        random.Random(config.seed).shuffle(shuffled)
        batches = [shuffled[i : i + 4] for i in range(0, len(shuffled), 4)]
    torch.manual_seed(config.seed)

    best_dev = float("inf")
    patience_left = config.early_stop_patience
    global_step = 0
    t_start = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        order = torch.randperm(len(batches)).tolist()
        epoch_batches = [batches[i] for i in order]

        epoch_loss = 0.0
        epoch_ce = 0.0
        epoch_var = 0.0
        n_steps = 0
        optimizer.zero_grad(set_to_none=True)

        for batch in epoch_batches:
            ce_terms = []
            p_frauds = []
            for ex in batch:
                ids = torch.tensor([ex["input_ids"]], dtype=torch.long, device=device)
                out = model(input_ids=ids)
                logits_last = out.logits[0, -1]
                ce, p_fraud = _verbalizer_ce_loss(
                    logits_last,
                    ex["label_index"],
                    ex["fraud_token_id"],
                    ex["legit_token_id"],
                )
                ce_terms.append(ce)
                p_frauds.append(
                    torch.softmax(
                        torch.stack(
                            [logits_last[ex["fraud_token_id"]], logits_last[ex["legit_token_id"]]]
                        ),
                        dim=0,
                    )[0]
                )

            ce_loss = torch.stack(ce_terms).mean()
            var_loss = torch.tensor(0.0, device=device)
            if use_consistency and len(p_frauds) >= 2:
                stacked = torch.stack(p_frauds)
                var_loss = stacked.var(unbiased=False)
            loss = ce_loss + (
                config.lambda_consistency * var_loss if use_consistency else 0.0
            )
            (loss / max(config.grad_accum, 1)).backward()
            if (n_steps + 1) % max(config.grad_accum, 1) == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += float(loss.detach().item())
            epoch_ce += float(ce_loss.detach().item())
            epoch_var += float(var_loss.detach().item()) if use_consistency else 0.0
            n_steps += 1
            global_step += 1
            if global_step % config.log_every == 0:
                row = {
                    "event": "step",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": float(loss.detach().item()),
                    "ce": float(ce_loss.detach().item()),
                    "var": float(var_loss.detach().item()) if use_consistency else 0.0,
                }
                _append_jsonl(log_path, row)

        # Flush leftover grads.
        if n_steps % max(config.grad_accum, 1) != 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        mean_loss = epoch_loss / max(n_steps, 1)
        mean_ce = epoch_ce / max(n_steps, 1)
        mean_var = epoch_var / max(n_steps, 1)
        dev_stats = evaluate_dev_ce(model, dev_examples, device) if dev_examples else {}
        row = {
            "event": "epoch",
            "epoch": epoch,
            "train_loss": mean_loss,
            "train_ce": mean_ce,
            "train_var": mean_var,
            "dev": dev_stats,
            "elapsed_s": time.perf_counter() - t_start,
        }
        _append_jsonl(log_path, row)
        print(json.dumps(row), flush=True)

        # Early stop on DEV CE.
        if dev_stats and math.isfinite(dev_stats.get("mean_ce", float("nan"))):
            if dev_stats["mean_ce"] < best_dev - 1e-4:
                best_dev = dev_stats["mean_ce"]
                patience_left = config.early_stop_patience
                model.save_pretrained(out_dir / "adapter")
                tokenizer.save_pretrained(out_dir / "adapter")
                _append_jsonl(log_path, {"event": "checkpoint", "epoch": epoch, "dev_ce": best_dev})
            else:
                patience_left -= 1
                if patience_left <= 0:
                    _append_jsonl(log_path, {"event": "early_stop", "epoch": epoch})
                    break
        else:
            model.save_pretrained(out_dir / "adapter")
            tokenizer.save_pretrained(out_dir / "adapter")

    # Ensure adapter exists even if no DEV.
    adapter_dir = out_dir / "adapter"
    if not adapter_dir.exists():
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

    result = {
        "out_dir": str(out_dir),
        "adapter_dir": str(adapter_dir),
        "truncation": trunc_report.to_dict(),
        "best_dev_ce": best_dev if math.isfinite(best_dev) else None,
        "run_config": run_config,
    }
    _atomic_json(out_dir / "train_result.json", result)
    return result
