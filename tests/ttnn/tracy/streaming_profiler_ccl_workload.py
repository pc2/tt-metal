# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""A ttnn CCL op in a loop on every chip of the box for a fixed time, optionally with matmuls between the ops."""

from __future__ import annotations

import argparse
import time

import torch
import ttnn

FABRICS = {
    "ring": (ttnn.FabricConfig.FABRIC_1D_RING, ttnn.Topology.Ring),
    "2d": (ttnn.FabricConfig.FABRIC_2D, ttnn.Topology.Linear),
}
OPS_PER_SYNC = 20


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--op", choices=["all_gather", "all_reduce"], required=True)
    parser.add_argument("--fabric", choices=list(FABRICS), required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--load", type=int, default=0, help="matmuls before each CCL op")
    args = parser.parse_args()

    fabric, topology = FABRICS[args.fabric]
    rows, cols = sorted(tuple(ttnn._ttnn.multi_device.SystemMeshDescriptor().shape()), reverse=True)
    ttnn.set_fabric_config(fabric)
    mesh = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(rows, cols))

    def to_mesh(t: torch.Tensor, mapper) -> ttnn.Tensor:
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            mesh_mapper=mapper,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    x = to_mesh(
        torch.randn([rows, cols, 512, 2048]).bfloat16(),
        ttnn.ShardTensor2dMesh(mesh, dims=(0, 1), mesh_shape=(rows, cols)),
    )
    tensors = [x]
    if args.load:
        a = to_mesh(torch.randn([1, 1, 2048, 4096]).bfloat16(), ttnn.ReplicateTensorToMesh(mesh))
        b = to_mesh(torch.randn([1, 1, 4096, 4096]).bfloat16(), ttnn.ReplicateTensorToMesh(mesh))
        tensors += [a, b]

    ops = 0
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        for _ in range(OPS_PER_SYNC):
            for _ in range(args.load):
                ttnn.matmul(a, b)
            if args.op == "all_gather":
                ttnn.all_gather(x, dim=3, cluster_axis=0, topology=topology)
            else:
                ttnn.all_reduce(x, cluster_axis=0, topology=topology)
        ttnn.synchronize_device(mesh)
        ops += OPS_PER_SYNC
    print(f"{args.op} over {args.fabric} fabric, load {args.load}: {ops} ops in {args.seconds:.0f} s")
    for t in tensors:
        ttnn.deallocate(t)
    ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
