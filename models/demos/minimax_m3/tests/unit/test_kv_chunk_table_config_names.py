# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""The M3 prefill KV chunk table must name its configs like blaze's decode table (the KV manager pairs the
two tables by config name and id), and the migrated index_k must carry the decode peer's bf8 chunk size."""

import ttnn
from models.demos.minimax_m3.tt.attention.kv_cache import MIGRATION_INDEX_K_DTYPE
from models.demos.minimax_m3.tt.runners.kv_chunk_table import _chunk_size_bytes, config_name


def test_config_names_match_blaze_zero_padding():
    # blaze kv_chunk_migration_helpers: width = max(2, len(str(n - 1))), names f"{i:0{width}d}".
    assert [config_name(i, 9) for i in range(9)] == [f"{i:02d}" for i in range(9)]
    assert config_name(7, 120) == "007"
    names = [config_name(i, 9) for i in range(9)]
    assert sorted(names) == names, "sorted names must keep ids in position order"


def test_migrated_index_k_chunk_is_blaze_bf8():
    # blaze m3_migration_spec: index_k is bfloat8_b, head_dim 128 -> 4 tiles x 1088 B.
    assert MIGRATION_INDEX_K_DTYPE == ttnn.bfloat8_b
    assert _chunk_size_bytes(MIGRATION_INDEX_K_DTYPE, 128) == 4352
