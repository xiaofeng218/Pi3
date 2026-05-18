"""
Minimal patch to inject memory logging into the training loop.

Add this to trainers/base_trainer_accelerate.py OR import and call at key points.

Quick-start: Add these lines to base_trainer_accelerate.py `train_one_epoch` method:

    from debug.memory_profiler import snapshot
    _mem_log = []

    # Inside the training loop:
    if it % 40 == 0:
        s = snapshot(f"step_{self.global_step}")
        _mem_log.append(s)
        print(f"[MEM] step={self.global_step:5d}  rss={s['rss_mb']:.0f}MB  "
              f"gpu0_alloc={s.get('gpu0_alloc', 0):.0f}MB  "
              f"gpu0_peak={s.get('gpu0_peak', 0):.0f}MB", flush=True)

Or for deeper profiling, replace the training step with:

    from debug.memory_profiler import snapshot
    snaps = []
    snaps.append(snapshot("pre_forward"))
    forward_output = self.forward_batch(batch, mode='train')
    snaps.append(snapshot("post_forward"))
    batch_output = self.calculate_loss(forward_output, batch, mode='train')
    loss = batch_output.loss
    snaps.append(snapshot("post_loss"))
    self.accelerator.backward(loss)
    snaps.append(snapshot("post_backward"))
    self.optimizer.step()
    self.optimizer.zero_grad()
    snaps.append(snapshot("post_optimizer"))
    # Print deltas
    for i in range(1, len(snaps)):
        rss_delta = snaps[i]['rss_mb'] - snaps[i-1]['rss_mb']
        print(f"[MEM] {snaps[i-1]['label']} -> {snaps[i]['label']}: "
              f"ΔRSS={rss_delta:+.0f}MB, peak_gpu0={snaps[i].get('gpu0_peak', 0):.0f}MB", flush=True)
"""

# This file is documentation, not executable code.
print(__doc__)
