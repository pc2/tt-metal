# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import Optional

import torch

import ttnn
from models.demos.common.prefill.adapter import KvCaches

# DRAM ND-shard geometry for the packed prefill KV cache — M3-local (decoupled from the DeepSeek substrate
# so they can diverge). The sequence is tiled into NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK-token blocks that
# round-robin across BH_NUM_DRAM_BANKS DRAM banks. The address-table builder must match both values.
NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK = 32
BH_NUM_DRAM_BANKS = 8

# The dtype the migration peer stores index_k in. Blaze's M3 decode indexer (DsaIndexerDirectKCacheUpdate) only
# supports bfp8 tiles, and the migration worker copies raw chunk bytes, so the migrated index_k must be bf8 even when
# prefill keeps its own copy in bf16 (M3_INDEX_CACHE_BF16=1).
MIGRATION_INDEX_K_DTYPE = ttnn.bfloat8_b


@dataclass
class MiniMaxKVCache(KvCaches):
    """M3's on-device prefill KV cache: three persistent, user-major packed device caches, each per-chip
    shape ``[num_users*num_layers, 1, seq_local, head_dim]`` on the DRAM ND-shard substrate, written in
    place by ``ttnn.experimental.deepseek_prefill.update_padded_kv_cache(slot_idx, layer_idx, ...)``:

      * ``k`` / ``v``  — GQA K/V. Under TP=cols each chip holds one head (heads sharded on the TP cols);
                         the sequence is SP-sharded block-cyclic on the ``sp`` rows.
      * ``index_k``    — MSA lightning-indexer key (one shared head, REPLICATED across the TP cols); only
                         the MSA layers populate it — dense layers leave their slots zeroed.
      * ``index_k_migration`` — optional bf8 copy of ``index_k``, same layout, written alongside it. Allocated
                         only when migration is on and ``index_k`` is not already bf8; the chunk table points
                         at it so the peer receives its own dtype. Prefill itself never reads it.

    Batch dim is user-major (``slot = user_id * num_layers + layer_idx``) so each user's layers stay
    contiguous, matching ``update_padded_kv_cache``'s indexing. The adapter allocates this once and the
    engine holds it as an opaque handle, passing it back into every runtime call that touches it.
    """

    k: ttnn.Tensor
    v: ttnn.Tensor
    index_k: ttnn.Tensor
    num_users: int
    num_layers: int
    max_seq_len: int
    sp: int
    index_k_migration: Optional[ttnn.Tensor] = None

    @property
    def migration_index_k(self) -> ttnn.Tensor:
        """The index_k tensor the migration peer reads: the bf8 copy when there is one, else ``index_k``."""
        return self.index_k if self.index_k_migration is None else self.index_k_migration

    def deallocate(self) -> None:
        """Free the device caches (e.g. to re-allocate at a different ``max_seq_len`` while the
        model stays resident). The handle is dead afterwards; do not pass it into the runtime again."""
        for t in (self.k, self.v, self.index_k, self.index_k_migration):
            if t is not None:
                ttnn.deallocate(t)


def allocate_kv_caches(
    mesh_device,
    *,
    num_layers,
    max_seq_len,
    sp_axis=0,
    num_users=1,
    head_dim=128,
    cache_dtype=ttnn.bfloat8_b,
    migration=False,
) -> MiniMaxKVCache:
    """Allocate the three external prefill KV caches (K, V, index_k). See :class:`MiniMaxKVCache`.

    Deliberately NOT ``init_kvpe_cache`` (that is MLA-specific and allocates a single cache): this owns
    the M3 GQA triple and the user-major packing. It reuses the same DRAM NdShard spec (same bank grid +
    32-token contiguous shard) so ``update_padded_kv_cache`` can write into these tensors unchanged.

    Args:
        num_layers: layers per user (full model = 60). All three caches allocate all layers; only the MSA
            layers will write ``index_k`` (dense slots stay zeroed — capacity is cheap, packing stays simple).
        max_seq_len: per-user cache capacity in tokens, a multiple of ``sp``. ``seq_local = max_seq_len // sp``.
        sp_axis: mesh axis the sequence is sharded over (rows).
        num_users: independent user slots sharing the cache (1 for bring-up).
        head_dim: per-head width (128 for M3 main K/V and the index head alike).
        cache_dtype: on-device cache dtype (bf8 matches the DeepSeek substrate + the device golden check).
        migration: the KV cache will be migrated to a decode peer. If index_k is not
            ``MIGRATION_INDEX_K_DTYPE``, a same-layout copy in that dtype is allocated for the peer.
    """
    sp = mesh_device.shape[sp_axis]
    assert max_seq_len % sp == 0, f"max_seq_len ({max_seq_len}) must be divisible by sp ({sp})"
    seq_local = max_seq_len // sp

    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, head_dim],
        grid=ttnn.CoreRangeSet(core_ranges),
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    mem_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM, nd_shard_spec=nd_shard_spec)

    def _alloc(dtype=cache_dtype):
        # Per-chip cache is one head ([.., 1, ..]); WHICH head a chip holds (or whether index_k is
        # replicated across cols) is decided at write time by how the input chunk is mesh-mapped, not
        # here. Allocated zeroed + ReplicateTensorToMesh: every chip gets the same empty buffer; content
        # diverges on the first update_padded_kv_cache write.
        return ttnn.from_torch(
            torch.zeros(num_users * num_layers, 1, seq_local, head_dim),
            dtype=dtype,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

    # index_k feeds the indexer's HARD top-16 block selection (not a smooth softmax like K/V), so bf8's
    # ~2-3 mantissa bits perturb the block scores enough to flip many picks -> chunked vs one-shot
    # selection diverges (~7/16 overlap) -> residual drift compounding over MSA layers. Cache it in bf16
    # (M3_INDEX_CACHE_BF16=1) to keep selection stable; it's tiny (1 head) and only the indexer reads it.
    index_dtype = ttnn.bfloat16 if os.getenv("M3_INDEX_CACHE_BF16") == "1" else cache_dtype

    index_k_migration = (
        _alloc(MIGRATION_INDEX_K_DTYPE) if migration and index_dtype != MIGRATION_INDEX_K_DTYPE else None
    )

    return MiniMaxKVCache(
        k=_alloc(),
        v=_alloc(),
        index_k=_alloc(index_dtype),
        num_users=num_users,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        sp=sp,
        index_k_migration=index_k_migration,
    )


def _write_one(cache, tensor, *, slot_idx, layer_idx, num_layers, kv_actual, sp_axis):
    """Write one SP-sharded chunk tensor into a packed cache via update_padded_kv_cache.

    The op requires TILE layout and input.dtype == cache.dtype, so cast a copy to the cache's dtype when
    needed (the original stays live for the attention op that follows). At ``kv_actual % 32 == 0`` chunk
    boundaries the per-device write offset is contiguous (block-cyclic degenerates to a reshape).
    """
    src = tensor if tensor.dtype == cache.dtype else ttnn.typecast(tensor, cache.dtype)
    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
        cache,
        src,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=num_layers,
        kv_actual_global=kv_actual,
        cluster_axis=sp_axis,
    )
    if src is not tensor:
        src.deallocate(True)


def write_kv_chunk(kv_cache: MiniMaxKVCache, tt_k, tt_v, *, slot_idx, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-RoPE K and raw V into the packed cache (every layer type).

    tt_k / tt_v are the per-device SP shards [1, n_kv_local, s_local, head_dim] (heads TP-sharded on the
    cols, sequence SP-sharded on the ``sp_axis`` rows) — exactly the per-chip cache layout, so they write
    in place. ``kv_actual`` is the cumulative valid prefix before this chunk (0 for non-chunked).
    """
    _write_one(
        kv_cache.k,
        tt_k,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=kv_cache.num_layers,
        kv_actual=kv_actual,
        sp_axis=sp_axis,
    )
    _write_one(
        kv_cache.v,
        tt_v,
        slot_idx=slot_idx,
        layer_idx=layer_idx,
        num_layers=kv_cache.num_layers,
        kv_actual=kv_actual,
        sp_axis=sp_axis,
    )


def write_index_k_chunk(kv_cache: MiniMaxKVCache, tt_index_k, *, slot_idx, layer_idx, kv_actual, sp_axis):
    """Write this chunk's post-norm/post-RoPE MSA index_k (MSA layers only).

    tt_index_k is the single shared index head [1, 1, s_local, head_dim], SP-sharded on the rows and
    REPLICATED across the TP cols (so each col writes the same data into its replicated cache slot).
    Also writes the migration copy when there is one (``_write_one`` casts to its dtype).
    """
    for cache in (kv_cache.index_k, kv_cache.index_k_migration):
        if cache is None:
            continue
        _write_one(
            cache,
            tt_index_k,
            slot_idx=slot_idx,
            layer_idx=layer_idx,
            num_layers=kv_cache.num_layers,
            kv_actual=kv_actual,
            sp_axis=sp_axis,
        )
