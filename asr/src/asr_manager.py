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
from faster_whisper import BatchedInferencePipeline, WhisperModel

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False


_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}
_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def _int_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        q, r = divmod(n, 10)
        return _TENS[q] if r == 0 else f"{_TENS[q]} {_ONES[r]}"
    if n < 1000:
        q, r = divmod(n, 100)
        return f"{_ONES[q]} hundred" if r == 0 else f"{_ONES[q]} hundred {_int_to_words(r)}"
    if n < 1_000_000:
        q, r = divmod(n, 1000)
        return f"{_int_to_words(q)} thousand" if r == 0 else f"{_int_to_words(q)} thousand {_int_to_words(r)}"
    return " ".join(_DIGIT_WORDS[d] for d in str(n))


def _digits_to_words(text: str) -> str:
    """Verbalize numerals because the ASR scorer does not normalize digits.

    Official transcripts spell numbers out ("seventy two", "zero six hundred",
    "seven niner"). Faster-Whisper often emits digits, which are counted as
    word errors even when the speech recognition was semantically right.
    """

    def repl_niner(match: re.Match[str]) -> str:
        return f"{_DIGIT_WORDS[match.group(1)]} niner"

    text = re.sub(r"\b([0-9])\s*[- ]\s*9\s*[- ]?er\b", repl_niner, text, flags=re.I)

    def repl_decimal(match: re.Match[str]) -> str:
        whole, frac = match.group(1), match.group(2)
        whole_words = _int_to_words(int(whole)) if whole else "zero"
        frac_words = " ".join(_DIGIT_WORDS[d] for d in frac)
        return f"{whole_words} point {frac_words}"

    text = re.sub(r"\b(\d+)\s*\.\s*(\d+)\b", repl_decimal, text)

    def _time_words(hour: int, minute: int) -> str:
        if minute == 0:
            if hour == 0:
                return "zero zero hundred"
            if hour < 10:
                return f"zero {_ONES[hour]} hundred"
            return f"{_int_to_words(hour)} hundred"
        hour_words = (
            f"zero {_ONES[hour]}" if hour < 10 else " ".join(_DIGIT_WORDS[d] for d in f"{hour:02d}")
        )
        return f"{hour_words} {_int_to_words(minute)}"

    def repl_split_time(match: re.Match[str]) -> str:
        hour = int(match.group(1) + match.group(2))
        minute = int(match.group(3))
        if hour > 23:
            return match.group(0)
        return _time_words(hour, minute)

    text = re.sub(r"\b([0-2])\s*,\s*([0-9])([0-5][0-9])\b", repl_split_time, text)

    def repl_hundreds_time(match: re.Match[str]) -> str:
        return _time_words(int(match.group(1)), 0)

    text = re.sub(r"\b0\s*([1-9])00\b", repl_hundreds_time, text)
    text = re.sub(r"\b([01][0-9]|2[0-3])00\b", repl_hundreds_time, text)

    def repl_four_digit(match: re.Match[str]) -> str:
        value = match.group(0)
        if value.startswith("0"):
            return " ".join(_DIGIT_WORDS[d] for d in value[:2]) + " " + _int_to_words(int(value[2:]))
        return " ".join(_DIGIT_WORDS[d] for d in value)

    text = re.sub(r"\b\d{4}\b", repl_four_digit, text)

    def repl_int(match: re.Match[str]) -> str:
        raw = match.group(0).replace(",", "")
        return _int_to_words(int(raw))

    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", repl_int, text)
    text = re.sub(r"\b\d+\b", repl_int, text)
    return text


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
        self.batched = BatchedInferencePipeline(model=self.model)
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
                segments, _info = self.batched.transcribe(
                    audio,
                    language="en",
                    task="transcribe",
                    beam_size=1,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 500},
                    condition_on_previous_text=False,
                    initial_prompt=self.initial_prompt,
                )
                text = " ".join(seg.text for seg in segments).strip()
                results[i] = self._postprocess_transcript(text)
            except Exception:
                results[i] = ""
        return results

    def _is_probably_silence(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return True
        # Prevent Whisper's common silence hallucinations such as "Thank you."
        # Use both RMS and peak so very quiet real speech is not discarded just
        # because its average energy is low.
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        return rms < 2e-4 and peak < 2e-3

    def _postprocess_transcript(self, text: str) -> str:
        return _digits_to_words(text)

    def asr(self, audio_bytes: bytes) -> str:
        return self.asr_batch([audio_bytes])[0]
