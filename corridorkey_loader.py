"""Resolve CorridorKey-Engine on ``sys.path`` and expose ``create_engine``.

Place ``CorridorKey.pth`` under ``CorridorKeyModule/checkpoints/`` inside the
CorridorKey-Engine checkout (same layout as the upstream project).

Override the root with the ``CORRIDORKEY_ENGINE_ROOT`` environment variable or
with the add-on preference *CorridorKey engine root* (if set).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

_MODULE_DIR = Path(__file__).resolve().parent


def debug_print(msg: str) -> None:
    """Log to stderr so messages show when Blender is started from a terminal."""
    if os.environ.get("CORRIDORKEY_VERBOSE", "1").lower() in ("0", "false", "no"):
        return
    print(f"[CorridorKey {time.monotonic():.3f}s] {msg}", file=sys.stderr, flush=True)


def resolve_default_torch_device() -> str:
    """Prefer GPU: CUDA, then Apple MPS, else CPU.

    CorridorKey's ``create_engine(device=None)`` defaults to **cpu** inside the engine,
    which on macOS is often tens of minutes per frame at 2048² — set ``mps`` here.

    Override with add-on preference *Device* or ``CORRIDORKEY_DEVICE`` (e.g. ``cpu``, ``mps``).
    """
    override = os.environ.get("CORRIDORKEY_DEVICE", "").strip()
    if override:
        debug_print(f"device: CORRIDORKEY_DEVICE={override!r}")
        return override
    try:
        import torch

        if torch.cuda.is_available():
            debug_print("device: auto -> cuda")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            debug_print("device: auto -> mps (Apple GPU — empty Device preference uses this)")
            return "mps"
    except Exception as exc:
        debug_print(f"device: auto failed ({exc!r}) -> cpu")
    debug_print("device: auto -> cpu (slow for large img_size; set Device to mps if on Apple Silicon)")
    return "cpu"


def apply_torch_embedded_safety() -> None:
    """Reduce OpenMP/MKL vs Blender thread contention (can look like an infinite hang)."""
    try:
        import torch

        n = int(os.environ.get("CORRIDORKEY_TORCH_THREADS", "1"))
        n = max(1, n)
        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
        debug_print(
            f"torch {torch.__version__}: set_num_threads({n}), "
            f"set_num_interop_threads(1) — CORRIDORKEY_TORCH_THREADS to change"
        )
    except Exception as exc:
        debug_print(f"torch embedded safety skipped: {exc!r}")


_engine_cache: dict[str, Any] = {}
_create_engine_fn: Callable[..., Any] | None = None


def default_engine_root() -> Path:
    """``CorridorKey-Engine`` next to this add-on's parent directory (sibling of the repo folder)."""
    return _MODULE_DIR.parent / "CorridorKey-Engine"


def resolve_engine_root() -> Path:
    env = os.environ.get("CORRIDORKEY_ENGINE_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    try:
        import bpy

        mod = __package__ or __name__.split(".")[0]
        prefs = bpy.context.preferences.addons[mod].preferences
        custom = getattr(prefs, "corridorkey_engine_root", "")
        if isinstance(custom, str) and custom.strip():
            return Path(bpy.path.abspath(custom)).resolve()
    except Exception:
        pass
    return default_engine_root()


def ensure_corridorkey_importable() -> Path:
    root = resolve_engine_root()
    debug_print(f"engine root -> {root}")
    if not (root / "CorridorKeyModule").is_dir():
        raise FileNotFoundError(
            f"CorridorKey-Engine not found or incomplete at {root}. "
            "Expected a directory named CorridorKeyModule inside. "
            "Clone CorridorKey-Engine, set CORRIDORKEY_ENGINE_ROOT or the add-on preference, "
            f"and place CorridorKey.pth in {root / 'CorridorKeyModule' / 'checkpoints'}."
        )
    root_str = str(root)
    if root_str not in sys.path:
        debug_print(f"sys.path.insert(0, {root_str!r})")
        sys.path.insert(0, root_str)
    else:
        debug_print("engine root already on sys.path")
    return root


def get_create_engine() -> Callable[..., Any]:
    global _create_engine_fn
    if _create_engine_fn is None:
        debug_print("importing CorridorKeyModule.engine_factory (first time, can be slow)…")
        t0 = time.perf_counter()
        ensure_corridorkey_importable()
        from CorridorKeyModule.engine_factory import create_engine

        _create_engine_fn = create_engine
        debug_print(f"engine_factory imported in {time.perf_counter() - t0:.2f}s")
    return _create_engine_fn


def get_engine(
    *,
    backend: str | None = None,
    device: str | None = None,
    img_size: int | None = None,
    cache_key_extra: str = "",
) -> Any:
    """Return a cached engine instance (keyed by backend, device, img_size)."""
    create_engine = get_create_engine()
    apply_torch_embedded_safety()
    from CorridorKeyModule.constants import DEFAULT_IMG_SIZE

    try:
        import bpy

        mod = __package__ or __name__.split(".")[0]
        prefs = bpy.context.preferences.addons[mod].preferences
        if backend is None:
            pb = getattr(prefs, "corridorkey_backend", None)
        else:
            pb = backend
        if pb in (None, "", "AUTO"):
            backend_resolved = None
        else:
            backend_resolved = pb
        if device is None:
            pd = getattr(prefs, "corridorkey_device", None)
        else:
            pd = device
        device_resolved = (pd or "").strip() or None
        if img_size is None:
            img_size = int(getattr(prefs, "corridorkey_img_size", DEFAULT_IMG_SIZE) or DEFAULT_IMG_SIZE)
    except Exception:
        backend_resolved = backend if backend not in (None, "", "AUTO") else None
        device_resolved = (device or "").strip() or None if device is not None else None
        if img_size is None:
            img_size = DEFAULT_IMG_SIZE

    if img_size is None:
        img_size = DEFAULT_IMG_SIZE

    if device_resolved is None:
        device_resolved = resolve_default_torch_device()

    from CorridorKeyModule.engine_factory import resolve_backend

    eff_backend = resolve_backend(backend_resolved)
    debug_print(
        f"CorridorKey backend (effective): {eff_backend!r} — "
        "on Apple Silicon, AUTO may pick mlx if corridorkey_mlx and .safetensors exist; "
        "if inference misbehaves, set add-on Backend to Torch."
    )

    key = f"{backend_resolved!s}:{device_resolved!s}:{int(img_size)}:{cache_key_extra}"
    if key not in _engine_cache:
        debug_print(
            f"create_engine(backend={backend_resolved!r}, device={device_resolved!r}, "
            f"img_size={int(img_size)}) — loading weights can take minutes; UI may look frozen…"
        )
        t0 = time.perf_counter()
        _engine_cache[key] = create_engine(
            backend=backend_resolved, device=device_resolved, img_size=int(img_size)
        )
        eng = _engine_cache[key]
        dev = getattr(eng, "device", None)
        if dev is not None:
            debug_print(f"engine device (torch): {dev}")
        debug_print(f"create_engine finished in {time.perf_counter() - t0:.2f}s (key={key!r})")
    else:
        debug_print(f"reusing cached engine (key={key!r})")
    return _engine_cache[key]


def clear_engine_cache() -> None:
    _engine_cache.clear()
