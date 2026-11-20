"""Hot reload of forward-pass candidates into a running engine (--enable-hot-reload).

A candidate is a directory::

    patch.py   def apply(model_runner, draft_model_runner=None) -> undo callable
    kernels/   optional kernel sources (Triton / FlyDSL / HIP)

``apply`` may monkeypatch any forward of the model, its layers or the attention /
MoE backends. ``draft_model_runner`` (the MTP / EAGLE draft, if any) is passed
only when ``apply`` declares that parameter; class-level patches reach the draft
anyway. ``Patches`` records attribute patches so that undo restores the exact
original objects.

POST /hot_reload {candidate_path} undoes the applied candidate, imports the new
one under a module name holding its content hash (kept, so A -> B -> A reuses the
module and its already-compiled kernels), applies it, flushes the caches and
state pools, resets the Dynamo caches and recaptures every CUDA graph of the
target and the draft, so a do-nothing reload serves what startup served. Any
exception rolls back to the baseline. POST /hot_reload/teardown goes back to the
baseline.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import inspect
import logging
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# content hash -> candidate module. Never evicted: dropping a module would drop the
# kernels it has compiled, and a re-submission would compile them again.
_MODULES: Dict[str, Any] = {}
# Patches not undone yet; a candidate whose apply raises is rolled back through them.
_LIVE_PATCHES: List[Patches] = []
# Dirs that never change what a candidate runs.
_HASH_SKIP_DIRS = {"__pycache__", ".git"}


def _own_attr(obj, name):
    d = vars(obj)
    if name in d:
        return True, d[name]
    if isinstance(obj, torch.nn.Module):
        for registry in ("_parameters", "_buffers", "_modules"):
            if name in d.get(registry, {}):
                return True, d[registry][name]
    return False, None


class Patches:
    """Attribute patches that undo() restores to the exact original objects."""

    def __init__(self):
        self._saved = []
        _LIVE_PATCHES.append(self)

    def set(self, obj, name: str, value) -> None:
        # Only obj's own attribute is saved, so undo never shadows an inherited one.
        self._saved.append((obj, name, *_own_attr(obj, name)))
        setattr(obj, name, value)

    def undo(self) -> None:
        while self._saved:
            obj, name, had, old = self._saved.pop()
            if had:
                setattr(obj, name, old)
            else:
                delattr(obj, name)
        if self in _LIVE_PATCHES:
            _LIVE_PATCHES.remove(self)


def kernels_dir(candidate_file: str) -> Path:
    """The candidate's kernels/ dir; call with the patch.py ``__file__``."""
    return Path(candidate_file).resolve().parent / "kernels"


def import_kernel_module(candidate_file: str, filename: str):
    """Import kernels/<filename> once per content (Triton needs a real source file)."""
    path = kernels_dir(candidate_file) / filename
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    name = f"sglang_hot_reload_kernel_{path.stem}_{digest}"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[name] = module
    return sys.modules[name]


def load_hip_extension(name: str, sources: List[str], **kwargs):
    """torch.utils.cpp_extension.load with the sources' hash in the module name:
    a loaded .so cannot be unloaded, so changed sources must get a new name."""
    from torch.utils.cpp_extension import load

    h = hashlib.sha256()
    for src in sources:
        h.update(Path(src).read_bytes())
    return load(name=f"{name}_{h.hexdigest()[:12]}", sources=sources, **kwargs)


def candidate_hash(path: Path) -> str:
    """Content hash of the whole candidate dir.

    Every file counts (relative path + contents, sorted walk), not only patch.py
    and kernels/: patch.py may read any file of the dir, so a change anywhere must
    give a new module instead of a noop or a cached one. Only __pycache__ and .git
    are skipped.
    """
    if not (path / "patch.py").is_file():
        raise FileNotFoundError(f"{path / 'patch.py'}: not a candidate dir")
    h = hashlib.sha256()
    for dirpath, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in _HASH_SKIP_DIRS)
        for name in sorted(files):
            f = Path(dirpath) / name
            rel = f.relative_to(path).as_posix()
            h.update(rel.encode() + b"\0")
            h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()[:16]


def load_candidate(path: Path):
    """Return (module, content hash, cache_hit)."""
    digest = candidate_hash(path)
    if digest in _MODULES:
        return _MODULES[digest], digest, True
    name = f"sglang_hot_reload_candidate_{digest}"
    spec = importlib.util.spec_from_file_location(name, path / "patch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    _MODULES[digest] = module
    return module, digest, False


@contextmanager
def _count_triton_compiles(stats: Dict[str, Any]):
    """Count Triton JIT compiles (in-memory cache misses) and their time."""
    try:
        from triton import knobs
    except ImportError:
        yield
        return
    rt = knobs.runtime
    prev_pre, prev_post = rt.jit_cache_hook, rt.jit_post_compile_hook
    starts = {}

    def pre(*args, **kwargs):
        starts[kwargs["key"]] = time.perf_counter()
        return prev_pre(*args, **kwargs) if prev_pre else None

    def post(*args, **kwargs):
        dt = time.perf_counter() - starts.pop(kwargs["key"], time.perf_counter())
        stats["triton_compiles"].append([kwargs["fn"].name, round(dt, 4)])
        return prev_post(*args, **kwargs) if prev_post else None

    rt.jit_cache_hook, rt.jit_post_compile_hook = pre, post
    try:
        yield
    finally:
        rt.jit_cache_hook, rt.jit_post_compile_hook = prev_pre, prev_post


class HotReloader:
    """Scheduler-side state of /hot_reload: the applied candidate and its undo."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.target_runner = scheduler.tp_worker.model_runner
        # EAGLEWorkerV2 -> EagleDraftWorker (owns the draft decode / extend graphs)
        self.draft_worker = getattr(scheduler.draft_worker, "_draft_worker", None)
        self.draft_runner = getattr(self.draft_worker, "draft_runner", None)
        self.applied: Optional[str] = None
        self.module = None
        self.undo_fn: Optional[Callable[[], None]] = None
        # The graphs the baseline captured at startup; a reload recaptures all of them.
        t, eager = self.target_runner, self.target_runner.eager_runner
        self.graphs = {
            "target_prefill": t.prefill_cuda_graph_runner not in (None, eager),
            "target_decode": t.decode_cuda_graph_runner not in (None, eager),
        }
        if self.draft_worker is not None:
            self.graphs.update(
                draft_prefill=self.draft_runner.prefill_cuda_graph_runner is not None,
                draft_decode=self.draft_worker.cuda_graph_runner is not None,
                draft_extend=self.draft_worker.cuda_graph_runner_for_draft_extend
                is not None,
            )

    def handle(self, candidate_path: Optional[str], mem_floor_bytes: int) -> dict:
        t0 = time.perf_counter()
        res = dict(
            ok=False,
            error=None,
            candidate_hash=None,
            cache_hit=False,
            noop=False,
            import_s=0.0,
            apply_s=0.0,
            recapture_s=0.0,
            warmup_s=0.0,
            reload_s=0.0,
            graphs_recaptured=[],
            triton_compiles=[],
            mem_headroom_bytes=None,
        )
        if not self.scheduler.is_fully_idle():
            res["error"] = "engine is not idle (requests running or waiting)"
        else:
            with _count_triton_compiles(res):
                self._reload(candidate_path, mem_floor_bytes, res)
        res["applied"] = self.applied
        if self.module is not None and hasattr(self.module, "status"):
            try:
                res["candidate_status"] = self.module.status()
            except Exception:
                res["candidate_status"] = traceback.format_exc()
        res["reload_s"] = time.perf_counter() - t0
        return res

    def _reload(self, candidate_path, mem_floor_bytes, res) -> None:
        touched = False
        try:
            module = digest = None
            if candidate_path is not None:
                t = time.perf_counter()
                module, digest, res["cache_hit"] = load_candidate(Path(candidate_path))
                res["import_s"] = time.perf_counter() - t
                res["candidate_hash"] = digest
            if digest == self.applied:  # same candidate, or teardown at baseline
                res["ok"] = res["noop"] = True
                return
            touched = True
            self._undo()
            if module is not None:
                t = time.perf_counter()
                self.module = module
                kwargs = {}
                if "draft_model_runner" in inspect.signature(module.apply).parameters:
                    kwargs["draft_model_runner"] = self.draft_runner
                undo = module.apply(self.target_runner, **kwargs)
                if not callable(undo):
                    raise TypeError("apply() must return an undo callable")
                self.applied, self.undo_fn = digest, undo
                res["apply_s"] = time.perf_counter() - t
            res["graphs_recaptured"], res["recapture_s"] = self._reset_and_recapture()
            res["mem_headroom_bytes"] = self._mem_headroom(mem_floor_bytes)
            if res["mem_headroom_bytes"] < 0:
                raise RuntimeError(
                    f"memory headroom {res['mem_headroom_bytes']} B is below the floor"
                )
            res["ok"] = True
        except Exception:
            res["error"] = traceback.format_exc()
            logger.error(f"Hot reload failed, rolling back:\n{res['error']}")
            if touched:
                self.rollback(res)

    def rollback(self, res: dict) -> None:
        """Back to the baseline after a failure; the engine keeps serving it."""
        try:
            self._undo()
            # A forward aborted by an exception can leave per-forward state behind.
            for runner in (self.target_runner, self.draft_runner):
                for m in runner.model.modules() if runner is not None else ():
                    if hasattr(m, "reset_aborted_forward"):
                        m.reset_aborted_forward()
            self._reset_and_recapture()
            res["rolled_back"] = True
        except Exception:
            res["error"] += "\nRollback failed:\n" + traceback.format_exc()
            res["rolled_back"] = False

    def _undo(self) -> None:
        undo, self.undo_fn = self.undo_fn, None
        self.applied = self.module = None
        try:
            if undo is not None:
                undo()
        finally:
            while _LIVE_PATCHES:  # whatever a failed apply / undo left behind
                _LIVE_PATCHES[-1].undo()

    def _reset_and_recapture(self):
        torch.cuda.synchronize()
        # Request state written by the previous code must not leak into the next.
        if not self.scheduler.flush_cache(empty_cache=False):
            raise RuntimeError("flush_cache refused")
        mamba_pool = getattr(self.scheduler.req_to_token_pool, "mamba_pool", None)
        if mamba_pool is not None:
            num_slots = mamba_pool.mamba_cache.temporal.shape[1]
            mamba_pool.clear_slots(
                torch.arange(num_slots, device=mamba_pool.mamba_cache.temporal.device)
            )
        for runner in (self.target_runner, self.draft_runner):
            pool = getattr(runner, "token_to_kv_pool", None)
            if hasattr(pool, "clear_request_state"):
                pool.clear_request_state()
        t = time.perf_counter()
        # Recapture with the compile state of startup. Dynamo tries the newest cache
        # entry of a torch.compile'd function first: at startup the largest prefill
        # graph is captured while only the static-shape entry exists, a recapture
        # would replay the dynamic-shape entry added later (e.g. HyperConnection
        # mixing: Triton template mm instead of hipBLASLt, ~10 % slower prefill).
        # Recompiling is served by the Inductor caches.
        torch._dynamo.reset()
        g = self.graphs
        self.target_runner.recapture_cuda_graphs(
            prefill=g["target_prefill"], decode=g["target_decode"]
        )
        if self.draft_worker is not None:
            self.draft_worker.recapture_cuda_graphs(prefill=g["draft_prefill"])
        torch.cuda.synchronize()
        gc.collect()
        names = [name for name, captured in g.items() if captured]
        return names, time.perf_counter() - t

    @staticmethod
    def _mem_headroom(mem_floor_bytes: int) -> int:
        free, _ = torch.cuda.mem_get_info()
        reserve = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        return free + reserve - mem_floor_bytes
