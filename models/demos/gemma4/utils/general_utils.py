# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
General utilities for the Gemma4 demo.
"""

import ttnn


def device_tensors(tensor):
    """Return the tensor of every device in mesh order.

    Device storage is local to each rank, a host gather exposes the complete mesh.
    Every rank must enter the gather in the same order.
    """
    if ttnn.using_distributed_env() and ttnn.is_tensor_storage_on_device(tensor):
        tensor = tensor.cpu()
    return ttnn.get_device_tensors(tensor)


def get_cache_file_name(tensor_cache_path, name):
    """Build a tensor-cache path prefix for ``ttnn.as_tensor``.

    Always return the path when a cache root is configured so a writable CI
    pass can populate new names (e.g. DRAM-sharded ``*.ws``); later read-only
    CI then hits those cached tensorbins.
    """
    return f"{tensor_cache_path}/{name}" if tensor_cache_path else None


def cast_host_for_ttnn(torch_tensor, ttnn_dtype):
    """Cast a host torch tensor to the torch dtype matching ``ttnn_dtype``.

    ``ttnn.as_tensor``/``from_torch`` run a C++ ``to_dtype`` conversion whenever the
    source dtype differs from the requested one. That conversion queries tile metadata
    on a ROW_MAJOR host intermediate and emits the #18536 "extract tile information out
    of a ROW MAJOR layout" warning. Matching the host dtype up front lets ``to_dtype``
    short-circuit, so no warning is emitted.

    Only float targets representable in torch are handled; block formats (bfloat8_b /
    bfloat4_b) have no torch equivalent and are returned unchanged.
    """
    import ttnn

    try:
        import torch
    except ImportError:
        return torch_tensor

    mapping = {ttnn.bfloat16: torch.bfloat16, ttnn.float32: torch.float32}
    target = mapping.get(ttnn_dtype)
    if target is not None and torch_tensor.dtype != target:
        return torch_tensor.to(target)
    return torch_tensor
