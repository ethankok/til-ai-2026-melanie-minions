"""ASR manager backed by faster-whisper distil-large-v3.

Single English-only model, loaded once at startup. Uses an in-world slang prompt
mined from the NLP corpus to bias decoding toward the dataset's vocabulary.
"""

from __future__ import annotations

import io
import os
import re

import ctranslate2
import numpy as np
import soundfile as sf
from asr_postprocess import digits_to_words as _digits_to_words
from faster_whisper import WhisperModel

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


class ASRManager:
    """English ASR backed by faster-whisper (CT2) distil-large-v3."""

    MODELS_DIR = os.environ.get("ASR_MODELS_DIR", "/workspace/models/asr")
    SLANG_PROMPT_PATH = os.environ.get(
        "ASR_SLANG_PROMPT_PATH", "/workspace/models/asr/slang_prompt.txt"
    )
    TARGET_SR = 16000

    def __init__(self):
        # Default to GPU when CUDA is visible to ctranslate2, otherwise CPU.
        # The eval container is built on an NVIDIA image, so GPU is the
        # expected path; the CPU fallback keeps us alive on misconfigured
        # runs (no --gpus, CPU-only debug box) instead of failing at startup.
        has_cuda = False
        try:
            has_cuda = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            has_cuda = False

        default_device = "cuda" if has_cuda else "cpu"
        default_compute = "float16" if has_cuda else "int8"
        self.initial_prompt = self._load_slang_prompt()
        device = os.environ.get("ASR_DEVICE", default_device)
        compute_type = os.environ.get("ASR_COMPUTE_TYPE", default_compute)
        try:
            self._load_model(device=device, compute_type=compute_type)
        except RuntimeError as exc:
            # CTranslate2 can see libcuda while still missing CUDA runtime libs
            # such as libcublas.so.12. Fall back to CPU unless the user
            # explicitly forced ASR_DEVICE.
            if "ASR_DEVICE" in os.environ or device == "cpu":
                raise
            print(
                f"[ASRManager] CUDA startup failed ({exc}); falling back to cpu/int8",
                flush=True,
            )
            self._load_model(device="cpu", compute_type="int8")

    def _load_model(self, device: str, compute_type: str) -> None:
        print(
            f"[ASRManager] device={device} compute_type={compute_type}",
            flush=True,
        )
        self.model = WhisperModel(
            self.MODELS_DIR,
            device=device,
            compute_type=compute_type,
        )
        # Note: BatchedInferencePipeline only batches encoder segments WITHIN one
        # long audio. Since we already loop one clip at a time, it adds no real
        # throughput here, and it forces VAD which was truncating long clips
        # (see ERROR_ANALYSIS.md). Use the plain WhisperModel.transcribe path.
        self._warmup()

    def _load_slang_prompt(self) -> str | None:
        if not os.path.exists(self.SLANG_PROMPT_PATH):
            return None
        with open(self.SLANG_PROMPT_PATH, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text or None

    def _warmup(self) -> None:
        silence = np.zeros(self.TARGET_SR, dtype=np.float32)
        segments, _ = self.model.transcribe(silence, language="en", beam_size=1)
        for _ in segments:
            pass

    def _decode_wav(self, audio_bytes: bytes) -> np.ndarray:
        data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != self.TARGET_SR:
            if _HAS_LIBROSA:
                data = librosa.resample(data, orig_sr=sr, target_sr=self.TARGET_SR)
            else:
                ratio = self.TARGET_SR / sr
                new_len = int(round(len(data) * ratio))
                xp = np.linspace(0, 1, len(data), endpoint=False)
                fp = data
                x = np.linspace(0, 1, new_len, endpoint=False)
                data = np.interp(x, xp, fp).astype(np.float32)
        return np.ascontiguousarray(data, dtype=np.float32)

    def asr_batch(self, audio_bytes_list: list[bytes]) -> list[str]:
        if not audio_bytes_list:
            return []

        audios: list[np.ndarray] = []
        for blob in audio_bytes_list:
            try:
                audios.append(self._decode_wav(blob))
            except Exception:
                audios.append(np.zeros(self.TARGET_SR, dtype=np.float32))

        results: list[str] = [""] * len(audios)
        for i, audio in enumerate(audios):
            try:
                if self._is_probably_silence(audio):
                    results[i] = ""
                    continue
                segments, _info = self.model.transcribe(
                    audio,
                    language="en",
                    task="transcribe",
                    beam_size=1,
                    # vad_filter was eating leading/trailing/middle speech on
                    # long clips (see ERROR_ANALYSIS.md). The audio-level silence
                    # guard above handles the empty-clip hallucination case.
                    vad_filter=False,
                    condition_on_previous_text=False,
                    initial_prompt=self.initial_prompt,
                    without_timestamps=True,
                    # Single greedy decode, no temperature fallback retries.
                    temperature=0.0,
                    # Hallucination guards: skip segments where the decoder is
                    # uncertain or the output is unusually compressed (a Whisper
                    # repetition tell). These thresholds are tighter than the
                    # library defaults.
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    no_speech_threshold=0.6,
                )
                text = " ".join(seg.text for seg in segments)
                text = re.sub(r"\s+", " ", text).strip()
                results[i] = self._postprocess_transcript(text)
            except Exception:
                results[i] = ""
        return results

    def _is_probably_silence(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return True
        # Audio-level silence detector; cheaper than running the model on noise.
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        # Long-clip conservative threshold: only blank essentially digital silence
        # so very quiet real speech is not discarded for its low average energy.
        if rms < 2e-4 and peak < 2e-3:
            return True
        # Short-clip aggressive threshold: sub-1.5 s clips that are still quiet
        # are almost always microphone bumps or breaths, not speech. Whisper
        # hallucinates "Thank you." / "I" on these (see ERROR_ANALYSIS.md
        # sample_924, sample_2373).
        if audio.size < self.TARGET_SR * 1.5 and rms < 1e-2:
            return True
        return False

    def _postprocess_transcript(self, text: str) -> str:
        return _digits_to_words(text)

    def asr(self, audio_bytes: bytes) -> str:
        return self.asr_batch([audio_bytes])[0]
