"""veRL 0.8 的窄范围运行时兼容。"""

import importlib.abc
import importlib.util
import sys


def _install_attention_fallback():
    from verl.utils import attention_utils
    from verl.utils import npu_flash_attn_utils as fallback

    functions = (
        fallback.index_first_axis,
        fallback.pad_input,
        fallback.rearrange,
        fallback.unpad_input,
    )
    # veRL 0.8 在 CUDA 上硬导入 FA2；上游提供 torch fallback 后删除此 hook。
    attention_utils._get_attention_functions = lambda: functions


class _PostImportLoader(importlib.abc.Loader):
    def __init__(self, loader, callback):
        self.loader = loader
        self.callback = callback

    def create_module(self, spec):
        if hasattr(self.loader, "create_module"):
            return self.loader.create_module(spec)
        return None

    def exec_module(self, module):
        self.loader.exec_module(module)
        self.callback()


class _DeferredCompatFinder(importlib.abc.MetaPathFinder):
    callbacks = {
        "verl.utils.attention_utils": _install_attention_fallback,
        "verl.trainer.ppo.ray_trainer": lambda: _install_trace_actor_update(),
    }

    def find_spec(self, fullname, path=None, target=None):
        callback = self.callbacks.get(fullname)
        if callback is None:
            return None
        # Resolve through the normal import chain without recursively calling this finder.
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = _PostImportLoader(spec.loader, callback)
        return spec


def _install_trace_actor_update():
    """Install mutually exclusive optional Actor advantage transforms."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    if getattr(RayPPOTrainer._update_actor, "_shopping_trace", False):
        return

    from shopping_grpo.training.grpo.trace import (
        apply_trace_advantages,
        build_trace_score_batch,
    )
    from shopping_grpo.training.grpo.evidence_credit import (
        apply_evidence_credit_advantages,
    )
    from shopping_grpo.training.grpo.turn_credit import (
        apply_turn_credit_advantages,
    )

    original_update_actor = RayPPOTrainer._update_actor

    def update_actor_with_trace(self, batch):
        config = self.config.get("shopping_trace", {})
        enabled = bool(config.get("enable", False))
        evidence_config = self.config.get("shopping_evidence_credit", {})
        evidence_enabled = bool(evidence_config.get("enable", False))
        turn_config = self.config.get("shopping_turn_credit", {})
        turn_enabled = bool(turn_config.get("enable", False))
        if sum((enabled, evidence_enabled, turn_enabled)) > 1:
            raise RuntimeError("TRACE, evidence credit and Rubric turn credit are mutually exclusive")
        if turn_enabled:
            batch.non_tensor_batch.pop("trace_target", None)
            turn_metrics = apply_turn_credit_advantages(
                batch,
                strength=float(turn_config["strength"]),
                max_turn_delta=float(turn_config["max_turn_delta"]),
            )
            output = original_update_actor(self, batch)
            output.meta_info["metrics"].update(turn_metrics)
            return output
        if evidence_enabled:
            batch.non_tensor_batch.pop("trace_target", None)
            evidence_metrics = apply_evidence_credit_advantages(
                batch,
                strength=float(evidence_config["strength"]),
                min_multiplier=float(evidence_config["min_multiplier"]),
                max_multiplier=float(evidence_config["max_multiplier"]),
                preserve_advantage_mass=bool(
                    evidence_config.get("preserve_advantage_mass", False)
                ),
            )
            output = original_update_actor(self, batch)
            output.meta_info["metrics"].update(evidence_metrics)
            return output
        if not enabled:
            batch.non_tensor_batch.pop("trace_target", None)
            return original_update_actor(self, batch)
        if not self.ref_in_actor:
            raise RuntimeError("shopping TRACE requires LoRA so the frozen base model is available")

        score_batch, state_counts = build_trace_score_batch(
            batch,
            self.tokenizer,
            max_sequence_length=int(config["max_sequence_length"]),
        )
        reference = self._compute_ref_log_prob(score_batch)
        target_mask = score_batch.batch["response_mask"]
        mean_log_probs = (
            reference.batch["ref_log_prob"] * target_mask
        ).sum(dim=-1) / target_mask.sum(dim=-1)
        batch.non_tensor_batch.pop("trace_target", None)
        trace_metrics = apply_trace_advantages(
            batch,
            mean_log_probs,
            state_counts,
            epsilon=float(config["epsilon"]),
            horizon=int(config["horizon"]),
            discount=float(config["discount"]),
            terminal_weight=float(config["terminal_weight"]),
            outcome_weight=float(config["outcome_weight"]),
            turn_weight=float(config["turn_weight"]),
        )
        trace_metrics["trace/target_log_prob_mean"] = float(mean_log_probs.mean())
        output = original_update_actor(self, batch)
        output.meta_info["metrics"].update(trace_metrics)
        return output

    update_actor_with_trace._shopping_trace = True
    RayPPOTrainer._update_actor = update_actor_with_trace


def install_torch_padding_fallback():
    """在 Ray 分配 GPU 后加载 veRL 兼容补丁，避免提前缓存 CUDA 映射。"""
    callbacks = _DeferredCompatFinder.callbacks
    for module_name, callback in callbacks.items():
        if module_name in sys.modules:
            callback()
    if not any(isinstance(finder, _DeferredCompatFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _DeferredCompatFinder())
