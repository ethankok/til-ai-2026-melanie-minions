"""LoRA fine-tune of distil-whisper/distil-large-v3 on the novice ASR set.

Uses HuggingFace Seq2SeqTrainer + PEFT LoRA on the decoder attention
projections. The encoder is kept frozen because distil-large-v3 inherits a very
strong encoder from large-v3 already; training compute should go to the
decoder, which is the part the distillation reduced.

Augmentation:
- SpecAugment on log-mel features.
- Optional online noise mixing if --noise-dir is given.
- Optional speed perturbation.

Usage on the GCP Workbench instance::

    python training/asr/train_distil_whisper.py \
        --data-dir training/asr/data \
        --output-dir training/asr/runs/distil-en-v1 \
        --epochs 3 \
        --per-device-batch-size 16 \
        --noise-dir /home/jupyter/novice/asr_noise  # optional
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
import soundfile as sf
import jiwer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizerFast,
)


WER_TRANSFORMS = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"-": " ", "—": " ", "–": " "}),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def _list_wavs(d: Path) -> list[Path]:
    return [p for p in d.rglob("*") if p.suffix.lower() in {".wav", ".flac", ".ogg"}]


def _load_audio(path: Path, target_sr: int = 16000) -> np.ndarray:
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if sr != target_sr:
        import librosa

        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data.astype(np.float32)


def _mix_noise(
    speech: np.ndarray, noise: np.ndarray, snr_db: float
) -> np.ndarray:
    if len(noise) < len(speech):
        reps = len(speech) // len(noise) + 1
        noise = np.tile(noise, reps)
    noise = noise[: len(speech)]
    speech_p = np.mean(speech**2) + 1e-10
    noise_p = np.mean(noise**2) + 1e-10
    target_noise_p = speech_p / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_p / noise_p)
    return (speech + noise).astype(np.float32)


def _speed_perturb(audio: np.ndarray, factor: float) -> np.ndarray:
    if factor == 1.0:
        return audio
    n = int(round(len(audio) / factor))
    xp = np.linspace(0, 1, len(audio), endpoint=False)
    x = np.linspace(0, 1, n, endpoint=False)
    return np.interp(x, xp, audio).astype(np.float32)


@dataclass
class WhisperCollator:
    processor: WhisperProcessor
    feature_extractor: WhisperFeatureExtractor
    tokenizer: WhisperTokenizerFast
    noise_clips: list[np.ndarray] | None
    p_noise: float
    snr_range: tuple[float, float]
    p_speed: float

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        audios: list[np.ndarray] = []
        texts: list[str] = []
        for ex in batch:
            audio = np.asarray(ex["audio"]["array"], dtype=np.float32)
            if self.p_speed > 0 and random.random() < self.p_speed:
                audio = _speed_perturb(audio, random.choice([0.9, 1.1]))
            if (
                self.noise_clips
                and self.p_noise > 0
                and random.random() < self.p_noise
            ):
                noise = random.choice(self.noise_clips)
                snr = random.uniform(*self.snr_range)
                audio = _mix_noise(audio, noise, snr)
            audios.append(audio)
            texts.append(ex["text"])

        features = self.feature_extractor(
            audios, sampling_rate=16000, return_tensors="pt"
        )
        labels = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=448
        )
        label_ids = labels["input_ids"].masked_fill(
            labels["attention_mask"].ne(1), -100
        )
        return {
            "input_features": features["input_features"],
            "labels": label_ids,
        }


def _build_compute_metrics(processor: WhisperProcessor):
    def compute_metrics(eval_pred):
        pred_ids, label_ids = eval_pred.predictions, eval_pred.label_ids
        label_ids = np.where(label_ids == -100, processor.tokenizer.pad_token_id, label_ids)
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = jiwer.wer(
            label_str,
            pred_str,
            reference_transform=WER_TRANSFORMS,
            hypothesis_transform=WER_TRANSFORMS,
        )
        return {"wer": wer}

    return compute_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("training/asr/data"))
    ap.add_argument(
        "--model-name",
        type=str,
        default="distil-whisper/distil-large-v3",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--per-device-batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.10)
    ap.add_argument("--eval-steps", type=int, default=500)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=50)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--noise-dir", type=Path, default=None)
    ap.add_argument("--p-noise", type=float, default=0.3)
    ap.add_argument("--snr-min", type=float, default=5.0)
    ap.add_argument("--snr-max", type=float, default=20.0)
    ap.add_argument("--p-speed", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading dataset from {args.data_dir}")
    ds: DatasetDict = load_from_disk(str(args.data_dir))

    print(f"Loading model: {args.model_name}")
    processor = WhisperProcessor.from_pretrained(
        args.model_name, language="en", task="transcribe"
    )
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    feature_extractor.mask_time_prob = 0.05
    feature_extractor.mask_feature_prob = 0.05

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    )
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # Freeze encoder; LoRA the decoder attention.
    for p in model.model.encoder.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    noise_clips: list[np.ndarray] | None = None
    if args.noise_dir and args.noise_dir.exists():
        print(f"Loading noise clips from {args.noise_dir}")
        noise_paths = _list_wavs(args.noise_dir)
        noise_clips = []
        for p in noise_paths[:500]:
            try:
                noise_clips.append(_load_audio(p))
            except Exception:
                continue
        print(f"Loaded {len(noise_clips)} noise clips")

    collator = WhisperCollator(
        processor=processor,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        noise_clips=noise_clips,
        p_noise=args.p_noise,
        snr_range=(args.snr_min, args.snr_max),
        p_speed=args.p_speed,
    )

    bf16_ok = torch.cuda.is_bf16_supported()
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        gradient_checkpointing=True,
        bf16=bf16_ok,
        fp16=not bf16_ok,
        predict_with_generate=True,
        generation_max_length=225,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        data_collator=collator,
        compute_metrics=_build_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
    )

    trainer.train()
    trainer.save_model(str(args.output_dir / "best"))
    processor.save_pretrained(str(args.output_dir / "best"))
    print(f"Saved best LoRA adapter + processor to {args.output_dir / 'best'}")


if __name__ == "__main__":
    main()
