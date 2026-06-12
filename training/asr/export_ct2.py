"""Export a LoRA-fine-tuned Whisper checkpoint to CTranslate2 (faster-whisper).

Steps:
  1. Load base distil-whisper + LoRA adapter, ``merge_and_unload``.
  2. Save the merged HF model to a temp dir.
  3. Run ``ct2-transformers-converter`` to produce a CT2 model dir.
  4. Copy the slang prompt next to the model files so the inference container
     can find both with one ``COPY`` in the Dockerfile.

Usage on the GCP Workbench instance::

    python training/asr/export_ct2.py \
        --adapter-dir training/asr/runs/distil-en-v1/best \
        --base-model distil-whisper/distil-large-v3 \
        --slang-file asr/models/slang_prompt.txt \
        --output-dir asr/models \
        --quantization float16
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", type=Path, required=True)
    ap.add_argument(
        "--base-model", type=str, default="distil-whisper/distil-large-v3"
    )
    ap.add_argument(
        "--slang-file",
        type=Path,
        default=Path("asr/models/slang_prompt.txt"),
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--quantization",
        type=str,
        default="float16",
        choices=["float16", "int8_float16", "int8", "float32"],
    )
    args = ap.parse_args()

    print(f"Loading base model: {args.base_model}")
    base = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    processor = WhisperProcessor.from_pretrained(
        args.adapter_dir if (args.adapter_dir / "preprocessor_config.json").exists()
        else args.base_model,
        language="en",
        task="transcribe",
    )

    print(f"Loading LoRA adapter: {args.adapter_dir}")
    model = PeftModel.from_pretrained(base, str(args.adapter_dir))
    print("Merging LoRA into base weights")
    model = model.merge_and_unload()

    # Buffer the slang prompt before the wipe loop below, since on re-export it
    # commonly lives inside output_dir and would otherwise be deleted.
    slang_bytes: bytes | None = None
    if args.slang_file.exists():
        slang_bytes = args.slang_file.read_bytes()
    else:
        print(
            f"WARN: slang file {args.slang_file} not found; "
            "container will run without a slang prompt"
        )

    with tempfile.TemporaryDirectory() as tmp:
        merged_dir = Path(tmp) / "merged"
        model.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))
        print(f"Saved merged HF model to {merged_dir}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        # ct2-transformers-converter requires an empty output dir, so wipe.
        for item in args.output_dir.iterdir():
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)

        cmd = [
            "ct2-transformers-converter",
            "--model",
            str(merged_dir),
            "--output_dir",
            str(args.output_dir),
            "--quantization",
            args.quantization,
            # ct2 errors if output_dir exists at all, even when empty.
            "--force",
            "--copy_files",
            "tokenizer.json",
            "preprocessor_config.json",
            "generation_config.json",
        ]
        print("Running:", " ".join(cmd))
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    if slang_bytes is not None:
        dst = args.output_dir / "slang_prompt.txt"
        dst.write_bytes(slang_bytes)
        print(f"Copied slang prompt to {dst} ({len(slang_bytes)} bytes)")

    print(f"CT2 model ready at {args.output_dir}")


if __name__ == "__main__":
    main()
