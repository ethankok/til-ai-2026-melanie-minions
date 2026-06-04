"""TDD test: streaming memmap BC dataset round-trips identically to the npz path.

No env/collection needed — synthetic data only.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
TRAINING_AE = REPO_ROOT / "training" / "ae"
sys.path.insert(0, str(TRAINING_AE))

from train_bc import BCDataset  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers to build synthetic data
# ---------------------------------------------------------------------------

N_SAMPLES = 8
N_FRAMES = 4

# Per-sample shapes (match the spec exactly)
AGENT_VIEW_SHAPE = (100, 7, 5)   # 4-frames * 25 channels, 7, 5
BASE_VIEW_SHAPE = (100, 7, 7)
BELIEF_SHAPE = (11, 16, 16)
SCALAR_SHAPE = (68,)
ACTION_MASK_SHAPE = (6,)


def _make_synthetic(rng: np.random.Generator):
    """Return a dict of synthetic arrays with the correct shapes."""
    return dict(
        agent_views=rng.random((N_SAMPLES, *AGENT_VIEW_SHAPE)).astype(np.float32) * 4,
        base_views=rng.random((N_SAMPLES, *BASE_VIEW_SHAPE)).astype(np.float32) * 4,
        beliefs=rng.random((N_SAMPLES, *BELIEF_SHAPE)).astype(np.float32),
        scalars=rng.random((N_SAMPLES, *SCALAR_SHAPE)).astype(np.float32),
        action_masks=rng.integers(0, 2, (N_SAMPLES, *ACTION_MASK_SHAPE)).astype(np.float32),
        actions=rng.integers(0, 6, (N_SAMPLES,)).astype(np.int64),
        n_frames=np.asarray(N_FRAMES, dtype=np.int32),
        with_belief=np.asarray(1, dtype=np.int32),
    )


def _write_npz(data: dict, out_path: Path) -> None:
    """Write as legacy npz (same keys as collect_bc.py)."""
    np.savez_compressed(
        out_path,
        agent_views=data["agent_views"],
        base_views=data["base_views"],
        beliefs=data["beliefs"],
        scalars=data["scalars"],
        action_masks=data["action_masks"],
        actions=data["actions"],
        n_frames=data["n_frames"],
        with_belief=data["with_belief"],
    )


def _write_memmap_dir(data: dict, out_dir: Path) -> None:
    """Write as a memmap directory — mirroring the format that collect_bc streaming produces."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = N_SAMPLES

    field_specs = {
        "agent_views":  (AGENT_VIEW_SHAPE,  "float16"),
        "base_views":   (BASE_VIEW_SHAPE,   "float16"),
        "beliefs":      (BELIEF_SHAPE,      "float16"),
        "scalars":      (SCALAR_SHAPE,      "float16"),
        "action_masks": (ACTION_MASK_SHAPE, "uint8"),
        "actions":      ((),               "int8"),
    }

    fields_meta = {}
    for name, (shape, dtype) in field_specs.items():
        full_shape = (n, *shape) if shape else (n,)
        mm = np.lib.format.open_memmap(
            str(out_dir / f"{name}.npy"),
            mode="w+",
            dtype=dtype,
            shape=full_shape,
        )
        src = data[name]
        if name == "actions":
            mm[:] = src.astype(np.int8)
        elif name == "action_masks":
            mm[:] = src.astype(np.uint8)
        else:
            mm[:] = src.astype(np.float16)
        mm.flush()
        fields_meta[name] = {
            "shape": list(shape) if shape else [],
            "store_dtype": dtype,
        }

    meta = {
        "n_frames": int(data["n_frames"]),
        "with_belief": bool(int(data["with_belief"])),
        "n_samples": n,
        "fields": fields_meta,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBCDatasetRoundTrip:
    """BCDataset(npz) and BCDataset(dir) must agree within float16 tolerance."""

    @pytest.fixture(scope="class")
    def datasets(self, tmp_path_factory):
        rng = np.random.default_rng(42)
        data = _make_synthetic(rng)

        npz_path = tmp_path_factory.mktemp("bc_npz") / "bc.npz"
        dir_path = tmp_path_factory.mktemp("bc_dir")

        _write_npz(data, npz_path)
        _write_memmap_dir(data, dir_path)

        ds_npz = BCDataset(npz_path)
        ds_dir = BCDataset(dir_path)
        return ds_npz, ds_dir, data

    def test_len_equal(self, datasets):
        ds_npz, ds_dir, _ = datasets
        assert len(ds_npz) == len(ds_dir) == N_SAMPLES

    def test_n_frames_equal(self, datasets):
        ds_npz, ds_dir, _ = datasets
        assert ds_npz.n_frames == ds_dir.n_frames == N_FRAMES

    def test_has_belief_equal(self, datasets):
        ds_npz, ds_dir, _ = datasets
        assert ds_npz.has_belief == ds_dir.has_belief == True  # noqa: E712

    def test_getitem_tuple_length(self, datasets):
        ds_npz, ds_dir, _ = datasets
        item_npz = ds_npz[0]
        item_dir = ds_dir[0]
        assert len(item_npz) == len(item_dir), (
            f"tuple lengths differ: npz={len(item_npz)}, dir={len(item_dir)}"
        )

    def test_actions_exact(self, datasets):
        ds_npz, ds_dir, _ = datasets
        for i in range(N_SAMPLES):
            item_npz = ds_npz[i]
            item_dir = ds_dir[i]
            action_npz = item_npz[-1]   # last element is action
            action_dir = item_dir[-1]
            assert action_npz.item() == action_dir.item(), (
                f"action mismatch at i={i}: npz={action_npz}, dir={action_dir}"
            )

    def test_action_dtype_long(self, datasets):
        ds_npz, ds_dir, _ = datasets
        item_npz = ds_npz[0]
        item_dir = ds_dir[0]
        assert item_npz[-1].dtype == torch.long, f"npz action dtype={item_npz[-1].dtype}"
        assert item_dir[-1].dtype == torch.long, f"dir action dtype={item_dir[-1].dtype}"

    def test_float_fields_within_float16_tolerance(self, datasets):
        ds_npz, ds_dir, _ = datasets
        # has_belief=True → tuple is (agent_v, base_v, scalars, mask, belief, action)
        for i in range(N_SAMPLES):
            item_npz = ds_npz[i]
            item_dir = ds_dir[i]
            for field_idx, name in enumerate(
                ["agent_views", "base_views", "scalars", "action_masks", "beliefs"]
            ):
                t_npz = item_npz[field_idx]
                t_dir = item_dir[field_idx]
                assert t_npz.shape == t_dir.shape, (
                    f"shape mismatch [{name}] i={i}: npz={t_npz.shape} dir={t_dir.shape}"
                )
                assert torch.allclose(t_npz, t_dir, atol=1e-3), (
                    f"values differ [{name}] i={i}: max_diff="
                    f"{(t_npz - t_dir).abs().max():.6f}"
                )

    def test_float_fields_are_float32(self, datasets):
        """BCDataset must return float32 tensors regardless of disk dtype."""
        ds_npz, ds_dir, _ = datasets
        for i in range(2):
            item_npz = ds_npz[i]
            item_dir = ds_dir[i]
            for field_idx, name in enumerate(
                ["agent_views", "base_views", "scalars", "action_masks", "beliefs"]
            ):
                assert item_npz[field_idx].dtype == torch.float32, (
                    f"npz [{name}] dtype={item_npz[field_idx].dtype}"
                )
                assert item_dir[field_idx].dtype == torch.float32, (
                    f"dir [{name}] dtype={item_dir[field_idx].dtype}"
                )

    def test_disk_dtypes_are_compact(self, datasets, tmp_path_factory):
        """The .npy files on disk must use the compact dtypes (float16/uint8/int8)."""
        rng = np.random.default_rng(99)
        data = _make_synthetic(rng)
        dir_path = tmp_path_factory.mktemp("bc_dtype_check")
        _write_memmap_dir(data, dir_path)

        assert np.load(str(dir_path / "agent_views.npy"), mmap_mode="r").dtype == np.float16
        assert np.load(str(dir_path / "base_views.npy"),  mmap_mode="r").dtype == np.float16
        assert np.load(str(dir_path / "beliefs.npy"),     mmap_mode="r").dtype == np.float16
        assert np.load(str(dir_path / "scalars.npy"),     mmap_mode="r").dtype == np.float16
        assert np.load(str(dir_path / "action_masks.npy"), mmap_mode="r").dtype == np.uint8
        assert np.load(str(dir_path / "actions.npy"),     mmap_mode="r").dtype == np.int8

    def test_n_samples_truncation(self, tmp_path_factory):
        """BCDataset(dir).__len__ is bounded by meta['n_samples'], not the memmap size."""
        rng = np.random.default_rng(7)
        data = _make_synthetic(rng)
        dir_path = tmp_path_factory.mktemp("bc_trunc")
        _write_memmap_dir(data, dir_path)

        # Artificially lower n_samples in meta to simulate a truncated run
        meta_path = dir_path / "meta.json"
        meta = json.loads(meta_path.read_text())
        meta["n_samples"] = 5
        meta_path.write_text(json.dumps(meta, indent=2))

        ds = BCDataset(dir_path)
        assert len(ds) == 5, f"expected len 5 after truncation, got {len(ds)}"

    def test_no_belief_dir(self, tmp_path_factory):
        """When with_belief=False, dir dataset returns 5-tuple (no belief tensor)."""
        rng = np.random.default_rng(13)
        data = _make_synthetic(rng)
        # Remove beliefs
        del data["beliefs"]
        data["with_belief"] = np.asarray(0, dtype=np.int32)

        dir_path = tmp_path_factory.mktemp("bc_nobelief")
        dir_path.mkdir(exist_ok=True)
        n = N_SAMPLES
        field_specs = {
            "agent_views":  (AGENT_VIEW_SHAPE,  "float16"),
            "base_views":   (BASE_VIEW_SHAPE,   "float16"),
            "scalars":      (SCALAR_SHAPE,      "float16"),
            "action_masks": (ACTION_MASK_SHAPE, "uint8"),
            "actions":      ((),               "int8"),
        }
        fields_meta = {}
        for name, (shape, dtype) in field_specs.items():
            full_shape = (n, *shape) if shape else (n,)
            mm = np.lib.format.open_memmap(
                str(dir_path / f"{name}.npy"), mode="w+", dtype=dtype, shape=full_shape
            )
            src = data[name]
            if name == "actions":
                mm[:] = src.astype(np.int8)
            elif name == "action_masks":
                mm[:] = src.astype(np.uint8)
            else:
                mm[:] = src.astype(np.float16)
            mm.flush()
            fields_meta[name] = {"shape": list(shape) if shape else [], "store_dtype": dtype}
        meta = {
            "n_frames": N_FRAMES,
            "with_belief": False,
            "n_samples": n,
            "fields": fields_meta,
        }
        (dir_path / "meta.json").write_text(json.dumps(meta, indent=2))

        ds = BCDataset(dir_path)
        assert not ds.has_belief
        item = ds[0]
        assert len(item) == 5, f"expected 5-tuple without belief, got {len(item)}"
        assert item[-1].dtype == torch.long


class TestStreamDirOverwriteGuard:
    """C2: collect_dataset_streaming must refuse to clobber an existing dir.

    Tests the guard logic directly (no env / collection) by writing a minimal
    memmap dir and calling the guard-path through the function's early-exit.
    """

    def _write_minimal_dir(self, dir_path: Path) -> None:
        """Write a single .npy + meta.json into dir_path to simulate an existing run."""
        dir_path.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        data = _make_synthetic(rng)
        _write_memmap_dir(data, dir_path)

    def test_raises_without_overwrite(self, tmp_path):
        """Second write attempt without --overwrite must raise SystemExit."""
        # We import here to avoid polluting the module namespace at collection time.
        import sys as _sys
        sys.path.insert(0, str(TRAINING_AE))
        from collect_bc import collect_dataset_streaming  # noqa: E402

        stream_dir = tmp_path / "stream"
        self._write_minimal_dir(stream_dir)

        with pytest.raises(SystemExit) as exc_info:
            collect_dataset_streaming(
                games=1,
                stream_dir=stream_dir,
                overwrite=False,
            )
        assert "already has data" in str(exc_info.value)

    def test_succeeds_with_overwrite_and_no_stale_fields(self, tmp_path):
        """--overwrite clears old files then writes a fresh run; no stale meta fields survive."""
        import sys as _sys
        sys.path.insert(0, str(TRAINING_AE))
        from collect_bc import collect_dataset_streaming  # noqa: E402

        stream_dir = tmp_path / "stream2"
        self._write_minimal_dir(stream_dir)

        # Inject a stale extra key into meta to verify it is removed on overwrite.
        meta_path = stream_dir / "meta.json"
        stale_meta = json.loads(meta_path.read_text())
        stale_meta["stale_key"] = "should_disappear"
        meta_path.write_text(json.dumps(stale_meta))

        # Run a real 1-game collection with --overwrite.
        collect_dataset_streaming(
            games=1,
            stream_dir=stream_dir,
            overwrite=True,
        )

        new_meta = json.loads(meta_path.read_text())
        assert "stale_key" not in new_meta, "stale meta key survived --overwrite"
        assert "n_samples" in new_meta
        assert "skipped" in new_meta
