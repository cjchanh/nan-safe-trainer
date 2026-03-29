"""Metal/GPU memory governance for MLX workloads.

Import this before loading MLX models to enforce memory limits
and prevent OOM crashes on Apple Silicon.
"""

from __future__ import annotations

import contextlib

PROFILES = {
    "training": {
        "memory_limit_gb": 80,
        "cache_limit_gb": 4,
    },
    "inference": {
        "memory_limit_gb": 60,
        "cache_limit_gb": 2,
    },
    "light": {
        "memory_limit_gb": 30,
        "cache_limit_gb": 1,
    },
}

_LAST_INIT: tuple[str, int, int] | None = None


def init_metal(
    profile: str = "inference",
    memory_limit_gb: float | None = None,
    cache_limit_gb: float | None = None,
) -> dict[str, int | str]:
    import mlx.core as mx

    metal = getattr(mx, "metal", None)
    if metal is None:
        raise RuntimeError("MLX Metal backend is unavailable in this environment.")
    set_memory_limit = getattr(mx, "set_memory_limit", None) or metal.set_memory_limit
    set_cache_limit = getattr(mx, "set_cache_limit", None) or metal.set_cache_limit
    device_info = getattr(mx, "device_info", None) or metal.device_info

    selected = PROFILES.get(profile, PROFILES["inference"])
    memory_limit_gb = memory_limit_gb or selected["memory_limit_gb"]
    cache_limit_gb = cache_limit_gb or selected["cache_limit_gb"]

    memory_limit_bytes = int(memory_limit_gb * 1024**3)
    cache_limit_bytes = int(cache_limit_gb * 1024**3)

    global _LAST_INIT
    init_key = (profile, memory_limit_bytes, cache_limit_bytes)
    if _LAST_INIT == init_key:
        return {
            "profile": profile,
            "memory_limit_bytes": memory_limit_bytes,
            "cache_limit_bytes": cache_limit_bytes,
        }

    set_memory_limit(memory_limit_bytes)
    set_cache_limit(cache_limit_bytes)

    info: dict[str, int] = {}
    with contextlib.suppress(Exception):
        info = device_info() or {}
    recommended = int(info.get("recommended_max_working_set_size", 0) or 0)

    print(
        "[metal_init] "
        f"profile={profile} "
        f"memory_limit={memory_limit_bytes // 1024**3}GB "
        f"cache_limit={cache_limit_bytes // 1024**3}GB "
        f"device_rec={recommended // 1024**3}GB",
        flush=True,
    )

    _LAST_INIT = init_key
    return {
        "profile": profile,
        "memory_limit_bytes": memory_limit_bytes,
        "cache_limit_bytes": cache_limit_bytes,
        "recommended_max_working_set_size": recommended,
    }
