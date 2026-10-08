"""索引缓存回归测试。"""

from __future__ import annotations

import numpy as np

from rag_eval import RunConfig
from rag_eval import index as index_module


def test_cached_vectors_reject_dimension_mismatch(tmp_path, monkeypatch) -> None:
    """分块数相同也不能复用维度不同的旧向量。"""
    monkeypatch.setattr(index_module, "CACHE_DIR", tmp_path)
    cfg = RunConfig(use_cache=True)
    digest = "dimension-regression"
    path = tmp_path / f"{index_module.cache_key(cfg, digest)}.npy"
    np.save(path, np.ones((1, 3), dtype=np.float32))

    assert index_module.load_cached_vectors(cfg, digest, 1, expected_dim=4) is None
    accepted = index_module.load_cached_vectors(cfg, digest, 1, expected_dim=3)
    assert accepted is not None and accepted.shape == (1, 3)
