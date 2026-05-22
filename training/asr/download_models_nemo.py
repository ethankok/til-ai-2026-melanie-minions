"""Download NeMo ASR weights for offline use inside the eval container.

Run this on the GCP Workbench instance once before `til build asr <tag>`.
The eval container has no internet access, so the `.nemo` file must be on
local disk (baked into the image via `COPY models /workspace/models/asr`)
before the build.

Usage:
    python training/asr/download_models_nemo.py \
        --model nvidia/parakeet-unified-en-0.6b \
        --out asr/models

Idempotent: if the target file already exists and `--force` is not set, the
script exits without re-downloading.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def _resolve_default_filename(model_id: str) -> str:
    """Map an HF model id to the on-disk `.nemo` filename we expect at runtime."""
    short = model_id.split("/")[-1]
    return f"{short}.nemo"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="nvidia/parakeet-tdt-0.6b-v2",
        help="HuggingFace / NGC model id to download.",
    )
    parser.add_argument(
        "--out",
        default="asr/models",
        help="Output directory. The .nemo file lands inside this directory.",
    )
    parser.add_argument(
        "--filename",
        default=None,
        help="Override the on-disk filename. Defaults to <model-shortname>.nemo.",
    )
    parser.add_argument(
        "--hf-filename",
        default=None,
        help="Filename to fetch from the Hugging Face repo. Defaults to "
             "<model-shortname>.nemo, independent of --filename.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the target file already exists.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_filename = args.hf_filename or _resolve_default_filename(args.model)
    target_filename = args.filename or source_filename
    target_path = out_dir / target_filename
    if target_path.exists() and not args.force:
        print(f"{target_path} already exists. Pass --force to re-download.")
        return 0

    print(f"Downloading {args.model}/{source_filename} ...", flush=True)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        print(
            "huggingface_hub is required for direct .nemo download. Install via:\n"
            "    python -m pip install -U huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    try:
        cached_path = hf_hub_download(
            repo_id=args.model,
            filename=source_filename,
            force_download=args.force,
        )
    except Exception as exc:
        print(
            f"Direct download failed for {args.model}/{source_filename}: {exc}",
            file=sys.stderr,
        )
        print(
            "If the repo uses a different .nemo filename, rerun with "
            "--hf-filename <name>.nemo.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    print(f"Saving to {target_path} ...", flush=True)
    shutil.copyfile(cached_path, tmp_path)
    # Atomic-ish replace so a crashed download doesn't leave a partial file
    # that subsequent runs would skip.
    tmp_path.replace(target_path)

    size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"OK: {target_path} ({size_mb:.1f} MB)")

    # Sanity: warn if the slang prompt is missing. We don't generate it here —
    # extract_slang.py handles that — but the docker build expects it next to
    # the model.
    slang_path = out_dir / "slang_prompt.txt"
    if not slang_path.exists():
        print(
            f"WARN: {slang_path} not found. Run extract_slang.py before "
            "`til build asr <tag>` so the slang word list is baked in.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
