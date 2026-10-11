"""Collective optimizer decisions for the manually managed FP16 scaler."""

import torch

from .muon import HybridMuon, unwrap_optimizer


def _backoff_scaler(scaler):
    if scaler.is_enabled():
        # This explicit update also clears the unscale stage on ranks whose
        # local gradients were finite but whose peer/update transaction failed.
        scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
        state = scaler.state_dict()
        state["_growth_tracker"] = 0
        scaler.load_state_dict(state)


def optimizer_attempt(model, optimizer, scaler, accelerator, *, nonfinite_loss, max_norm):
    """Return (committed, preclip_norm); every rank makes the same decision.

    Called only after a full accumulation window. A skip never calls the
    optimizer (including AdamW decay), and resets the scaler's growth history.
    The caller owns gradient cleanup, scheduler/EMA advancement, and counters.
    """
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.grad is not None]
    if not parameters:
        raise RuntimeError("Optimizer attempt has no gradients")
    # Computing the norm before clipping also catches a norm overflow when all
    # individual gradient entries are finite. Never count a zeroed overflow as a step.
    norm = torch.nn.utils.get_total_norm([p.grad for p in parameters], foreach=True)
    valid = torch.isfinite(norm).to(torch.int32)
    if nonfinite_loss:
        valid.zero_()
    finite_ranks = accelerator.reduce(valid, reduction="sum")
    committed = finite_ranks.item() == accelerator.num_processes
    base = unwrap_optimizer(optimizer)
    if committed:
        if max_norm > 0:
            torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, norm, foreach=True)
        if isinstance(base, HybridMuon):
            base.stage_updates()
            committed = base.reconcile_staged(accelerator)
        if committed:
            scaler.step(optimizer)
            scaler.update()
    if not committed:
        if isinstance(base, HybridMuon):
            base.discard_staged()
            base.last_step_diagnostics["committed"] = False
        _backoff_scaler(scaler)
    return committed, norm
