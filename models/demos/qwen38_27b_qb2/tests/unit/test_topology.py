# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reject unsupported meshes before loading weights or constructing the decoder."""

from unittest.mock import Mock

import pytest

import ttnn
from models.demos.qwen38_27b_qb2.tt import model
from models.demos.qwen38_27b_qb2.tt.decoder_tp import Qwen38TPDecoder, validate_qb2_mesh


@pytest.mark.parametrize("entrypoint", ["model", "decoder"])
@pytest.mark.parametrize("unsupported", ["architecture", "product", "shape", "device_count"])
def test_rejects_unsupported_mesh_before_loading_weights(monkeypatch, expect_error, entrypoint, unsupported):
    mesh = Mock(shape=(2, 2) if unsupported == "shape" else (1, 4))
    mesh.arch.return_value = ttnn.Arch.WORMHOLE_B0 if unsupported == "architecture" else ttnn.Arch.BLACKHOLE
    mesh.get_num_devices.return_value = 3 if unsupported == "device_count" else 4
    monkeypatch.setattr(
        ttnn.cluster,
        "get_cluster_type",
        lambda: ttnn.cluster.ClusterType.BLACKHOLE_GALAXY
        if unsupported == "product"
        else ttnn.cluster.ClusterType.P300_X2,
    )
    precision_loader = Mock()
    checkpoint_loader = Mock()
    monkeypatch.setattr(model, "load_precision", precision_loader)
    monkeypatch.setattr(model, "checkpoint_path", checkpoint_loader)
    with expect_error(ValueError, "requires four Blackhole devices"):
        if entrypoint == "model":
            model.Qwen38Model(mesh)
        else:
            Qwen38TPDecoder.from_state_dict(None, hf_config=None, layer_idx=0, mesh_device=mesh)
    precision_loader.assert_not_called()
    checkpoint_loader.assert_not_called()


@pytest.mark.parametrize("cluster_type", ["P300_X2", "P150", "P150_X4"])
def test_accepts_four_blackhole_devices_in_a_ring(monkeypatch, cluster_type):
    mesh = Mock(shape=(1, 4))
    mesh.arch.return_value = ttnn.Arch.BLACKHOLE
    mesh.get_num_devices.return_value = 4
    monkeypatch.setattr(ttnn.cluster, "get_cluster_type", lambda: getattr(ttnn.cluster.ClusterType, cluster_type))
    validate_qb2_mesh(mesh)
