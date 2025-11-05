# utils/train_debut_utils.py
import contextlib
import time
import torch

class StepTimers:
    """Accumulate timing information across phases of each training step."""
    def __init__(self):
        self.t_load = self.t_fwd = self.t_bwd = self.t_opt = 0.0
        self.last = time.time()

    def mark_load(self):  self._mark('t_load')
    def mark_fwd(self):   self._mark('t_fwd')
    def mark_bwd(self):   self._mark('t_bwd')
    def mark_opt(self):   self._mark('t_opt')

    def _mark(self, name):
        now = time.time()
        setattr(self, name, getattr(self, name) + (now - self.last))
        self.last = now

    def consume(self):
        vals = (self.t_load, self.t_fwd, self.t_bwd, self.t_opt)
        self.t_load = self.t_fwd = self.t_bwd = self.t_opt = 0.0
        return vals


def print_device_summary(model, device):
    """Print device, memory, and model parameter info for debugging."""
    try:
        p = next(model.parameters())
        print(f"[device] model on {p.device}; requested {device}")
    except StopIteration:
        print("[device] model has no parameters?")
    if device.type == "cuda" and torch.cuda.is_available():
        print(f"[cuda] name: {torch.cuda.get_device_name(0)}")
        print(f"[cuda] capability: {torch.cuda.get_device_capability(0)}")
        print(f"[cuda] memory: {torch.cuda.memory_allocated()/1e6:.1f} MB "
              f"(reserved {torch.cuda.memory_reserved()/1e6:.1f} MB)")


@contextlib.contextmanager
def maybe_sync_cuda(device):
    """Ensure CUDA ops complete before and after timed regions."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    yield
    if device.type == "cuda":
        torch.cuda.synchronize()
