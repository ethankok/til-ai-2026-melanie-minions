"""ASR manager backed by NVIDIA NeMo ASR models.

Drop-in replacement for `asr_manager.ASRManager` for A/B testing against the
legacy faster-whisper distil-large-v3 path. The default checkpoint is the
Parakeet-TDT-v2 English model, staged as the default zero-shot ASR model.

NeMo/Parakeet models emit spelled-out numbers more often than Whisper-family
models (matching the official transcript style: "zero six hundred",
"twenty third", "north-northeast"). The `digits_to_words` post-processor stays
as a safety net.

Container expectations:

* Weights baked into the image at build time. Default lookup path is
  `/workspace/models/asr/parakeet-unified-en-0.6b.nemo`. Override via env
  `ASR_NEMO_MODEL` (filename) or `ASR_MODELS_DIR` (parent directory). To run
  the previous TDT-v2 image shape, set
  `ASR_NEMO_MODEL=parakeet-tdt-0.6b-v2.nemo`.
* Slang prompt at `/workspace/models/asr/slang_prompt.txt` is reused as a
  word-boost list when the loaded NeMo model exposes a context-biasing
  decoding API. If the API is unavailable on this NeMo version, biasing is
  silently skipped (logged at startup) and the manager keeps working.
* GPU expected. CPU fallback works but speed score will collapse.
"""

from __future__ import annotations

import io
import inspect
import os
import time
from typing import Any

import numpy as np
import soundfile as sf
from asr_postprocess import digits_to_words as _digits_to_words

try:
    import librosa
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

try:  # CUDA detection without forcing a hard torch dep at import time.
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# Optional per-batch latency logging for Finals speed tuning. Set ASR_TIMING=1
# to emit one "[ASR-TIMING] ..." line per batch: the decode-vs-infer split plus
# an estimated Finals per-batch time_score. Default-OFF — zero cost and no
# behaviour change when unset. Read at import; the Dockerfile/run sets the env
# before the process starts. NOTE the estimate is manager-only (decode + infer);
# it excludes the server's b64-decode + HTTP, so the harness `time=` is slightly
# lower. Good enough to locate the bottleneck.
_TIMING = bool(os.environ.get("ASR_TIMING"))


def _build_lm_decoding_cfg(
    base_cfg: dict,
    *,
    lm_path: str,
    alpha: float,
    beam_size: int,
    strategy: str,
    pruning_mode: str,
    blank_lm_score_mode: str,
) -> dict:
    """Return a decoding-config dict that enables NGPU-LM n-gram fusion.

    Pure dict transform — no NeMo import — so it is unit-testable on the Mac.
    Sets the top-level ``strategy`` and the NGPU-LM fields under ``beam.*`` per
    the NeMo RNNT/TDT NGPU-LM API. Does not mutate ``base_cfg``; existing keys
    (including unrelated ``beam.*`` sub-keys) are preserved.
    """
    cfg = dict(base_cfg)
    beam = dict(cfg.get("beam") or {})
    beam["beam_size"] = beam_size
    beam["ngram_lm_model"] = lm_path
    beam["ngram_lm_alpha"] = alpha
    beam["pruning_mode"] = pruning_mode
    beam["blank_lm_score_mode"] = blank_lm_score_mode
    cfg["beam"] = beam
    cfg["strategy"] = strategy
    return cfg


class NemoASRManager:
    """English ASR backed by a local NeMo ASR `.nemo` checkpoint."""

    MODELS_DIR = os.environ.get("ASR_MODELS_DIR", "/workspace/models/asr")
    MODEL_FILE = os.environ.get("ASR_NEMO_MODEL", "parakeet-tdt-0.6b-v2.nemo")
    SLANG_PROMPT_PATH = os.environ.get(
        "ASR_SLANG_PROMPT_PATH", "/workspace/models/asr/slang_prompt.txt"
    )
    TARGET_SR = 16000
    # Per-call decode batch. Parakeet's encoder amortizes nicely over a batch;
    # 8 fits well under T4 VRAM (16 GB) at fp16 for 30s clips.
    BATCH_SIZE = int(os.environ.get("ASR_NEMO_BATCH", "8"))

    def __init__(self):
        # Defer the heavy import so syntax-checking the module doesn't require
        # NeMo on the developer machine.
        try:
            import nemo.collections.asr as nemo_asr  # noqa: F401
        except Exception as exc:  # pragma: no cover - import-time only
            raise RuntimeError(
                "nemo_toolkit[asr] is required for NemoASRManager. "
                "Install via requirements-nemo.txt."
            ) from exc

        self._nemo_asr = nemo_asr

        device = os.environ.get("ASR_DEVICE")
        if device is None:
            device = "cuda" if (_HAS_TORCH and torch.cuda.is_available()) else "cpu"
        self.device = device
        self.use_amp = self.device == "cuda" and _HAS_TORCH

        self.slang_terms = self._load_slang_terms()
        self._load_model()
        self._configure_biasing()
        self._configure_lm_fusion()
        self._warmup()

    # ------------------------------------------------------------------ #
    # Setup                                                                #
    # ------------------------------------------------------------------ #
    def _load_model(self) -> None:
        model_path = os.path.join(self.MODELS_DIR, self.MODEL_FILE)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"NeMo model not found at {model_path}. Build the image with "
                "training/asr/download_models_nemo.py to bake weights in."
            )
        print(
            f"[NemoASRManager] device={self.device} model={model_path}",
            flush=True,
        )

        # `restore_from` is the offline loader; it does not phone home and
        # works inside the air-gapped eval container. `EncDecRNNTBPEModel`
        # covers Parakeet-TDT (TDT shares the RNNT decoding interface).
        import nemo
        from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
        from nemo.collections.asr.models import ASRModel

        encoder_params = inspect.signature(ConformerEncoder.__init__).parameters
        print(
            "[NemoASRManager] nemo="
            f"{getattr(nemo, '__version__', 'unknown')} path={getattr(nemo, '__file__', 'unknown')} "
            f"unified_chunking={'att_chunk_context_size' in encoder_params}",
            flush=True,
        )
        if self.MODEL_FILE.startswith("parakeet-unified") and "att_chunk_context_size" not in encoder_params:
            raise RuntimeError(
                "This NeMo runtime cannot load parakeet-unified-en-0.6b: "
                "ConformerEncoder is missing att_chunk_context_size. Rebuild "
                "with the pinned GitHub NeMo commit in requirements-nemo.txt."
            )

        self.model = ASRModel.restore_from(restore_path=model_path)

        if _HAS_TORCH:
            self.model = self.model.to(self.device)
            self.model.eval()
            if self.use_amp:
                # Parakeet ships fp32 weights; cast to fp16 for the speed win.
                # bf16 on Ampere+ would be cleaner but T4 (Turing) has no bf16.
                try:
                    self.model = self.model.half()
                except Exception as exc:
                    print(f"[NemoASRManager] half() failed, staying fp32: {exc}", flush=True)
                    self.use_amp = False

    def _load_slang_terms(self) -> list[str]:
        if not os.path.exists(self.SLANG_PROMPT_PATH):
            return []
        with open(self.SLANG_PROMPT_PATH, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return []
        # Slang prompt is a single line of space-separated terms,
        # highest-frequency first.
        return [t for t in text.split() if t]

    def _configure_biasing(self) -> None:
        """Wire the slang word list into the decoder if the NeMo API supports it.

        NeMo's transducer/TDT decoders support context biasing on newer
        releases (`set_decoding_strategy` + `boosting_words`). Older releases
        don't, in which case we silently skip — the model still benefits from
        being fine-tuned, and Parakeet has decent zero-shot proper-noun recall
        on its own. We never crash on an unsupported NeMo version.
        """
        if not self.slang_terms:
            return

        cfg_set = False
        try:
            from omegaconf import OmegaConf  # type: ignore

            decoding_cfg = self.model.cfg.decoding if hasattr(self.model, "cfg") else None
            if decoding_cfg is not None:
                cfg = OmegaConf.to_container(decoding_cfg, resolve=True)
                if isinstance(cfg, dict):
                    # Newer NeMo: TDT/RNNT decoder accepts a context list.
                    cfg.setdefault("strategy", "greedy_batch")
                    cfg["preserve_alignments"] = False
                    cfg["compute_timestamps"] = False
                    self.model.change_decoding_strategy(OmegaConf.create(cfg))
                    cfg_set = True
        except Exception as exc:
            print(
                f"[NemoASRManager] decoding strategy unchanged ({exc}); using NeMo default",
                flush=True,
            )

        # Best-effort: if the model has a `set_context_biasing` / similar API,
        # use it. We try several known method names and stop on the first one
        # that accepts the slang list. None of these are required to ship; the
        # initial submission can land without biasing.
        for method_name in (
            "set_context_biasing",
            "set_boosting_words",
            "configure_biasing",
        ):
            method = getattr(self.model, method_name, None)
            if method is None:
                continue
            try:
                method(self.slang_terms)
                print(
                    f"[NemoASRManager] context biasing enabled via {method_name} "
                    f"({len(self.slang_terms)} terms)",
                    flush=True,
                )
                return
            except Exception as exc:
                print(
                    f"[NemoASRManager] {method_name} rejected slang list: {exc}",
                    flush=True,
                )

        if cfg_set:
            print(
                "[NemoASRManager] decoding strategy set; no context-biasing API found",
                flush=True,
            )

    def _configure_lm_fusion(self) -> None:
        """Enable NGPU-LM n-gram shallow fusion if an LM path is configured.

        Default-OFF: when ``ASR_NGRAM_LM`` is unset the decoder is left exactly
        as ``_configure_biasing`` left it (greedy), so the shipped image is
        behaviourally identical to nemo-ft-v3 until the env is set.

        Mirrors the ``_configure_biasing`` safety contract: any failure (NGPU-LM
        unavailable on this NeMo build, bad config, missing LM file) is logged
        and the decoder stays on its current strategy — we never crash. The
        ``.ARPA`` emitted alongside the LM keeps the classic ``maes`` strategy
        available as a manual fallback.
        """
        # Runtime-fallback bookkeeping. If the fused (malsd_batch) decode raises
        # at inference time on the eval GPU, we revert to greedy instead of
        # returning empty transcripts (which score exactly 0.000). Initialised
        # here so the attrs always exist, even when fusion is OFF.
        self._lm_fusion_active = False
        self._greedy_decoding_cfg = None

        lm_path = os.environ.get("ASR_NGRAM_LM")
        if not lm_path:
            return
        if not os.path.exists(lm_path):
            print(
                f"[NemoASRManager] ASR_NGRAM_LM={lm_path} not found; fusion OFF",
                flush=True,
            )
            return

        alpha = float(os.environ.get("ASR_NGRAM_LM_ALPHA", "0.3"))
        beam_size = int(os.environ.get("ASR_BEAM_SIZE", "4"))
        strategy = os.environ.get("ASR_LM_STRATEGY", "malsd_batch")
        pruning_mode = os.environ.get("ASR_LM_PRUNING", "late")
        blank_mode = os.environ.get("ASR_LM_BLANK_MODE", "lm_weighted_full")

        try:
            from omegaconf import OmegaConf  # type: ignore

            decoding_cfg = self.model.cfg.decoding if hasattr(self.model, "cfg") else None
            base = OmegaConf.to_container(decoding_cfg, resolve=True) if decoding_cfg is not None else {}
            if not isinstance(base, dict):
                base = {}
            # Snapshot the current greedy strategy BEFORE switching, so a
            # malsd_batch runtime failure can revert to it (see
            # _fallback_to_greedy). _build_lm_decoding_cfg copies base, so this
            # snapshot is unaffected by the switch.
            self._greedy_decoding_cfg = dict(base)
            cfg = _build_lm_decoding_cfg(
                base,
                lm_path=lm_path,
                alpha=alpha,
                beam_size=beam_size,
                strategy=strategy,
                pruning_mode=pruning_mode,
                blank_lm_score_mode=blank_mode,
            )
            self.model.change_decoding_strategy(OmegaConf.create(cfg))
            self._lm_fusion_active = True
            print(
                f"[NemoASRManager] NGPU-LM fusion ON: lm={lm_path} strategy={strategy} "
                f"beam={beam_size} alpha={alpha} pruning={pruning_mode} blank={blank_mode}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[NemoASRManager] LM fusion setup failed ({exc}); staying on greedy",
                flush=True,
            )

    def _fallback_to_greedy(self, reason: str) -> bool:
        """Revert from fused (malsd_batch) decoding to greedy after a runtime
        failure on the serving GPU.

        malsd_batch decodes correctly on the Workbench T4 but the cloud eval
        runs different hardware where the CUDA-graph beam path can raise for
        every batch. The old behaviour swallowed that into empty transcripts
        (cloud score 0.000). Now the first failure permanently reverts the
        decoder to the greedy strategy captured in `_configure_lm_fusion`, so the
        worst case is greedy accuracy, never blank output.

        Returns True if the decoder is now greedy and the caller may retry.
        One-shot: once malsd_batch fails on this hardware it will keep failing,
        so we stop attempting it for the rest of the process.
        """
        if not self._lm_fusion_active or self._greedy_decoding_cfg is None:
            return False
        self._lm_fusion_active = False
        try:
            from omegaconf import OmegaConf  # type: ignore

            self.model.change_decoding_strategy(
                OmegaConf.create(dict(self._greedy_decoding_cfg))
            )
            print(
                f"[NemoASRManager] fused decode failed ({reason}); permanently "
                "reverted to greedy for the rest of this process",
                flush=True,
            )
            return True
        except Exception as exc:
            print(
                f"[NemoASRManager] greedy fallback ALSO failed ({exc})",
                flush=True,
            )
            return False

    def _warmup(self) -> None:
        silence = np.zeros(self.TARGET_SR, dtype=np.float32)
        try:
            self._transcribe_batch([silence])
        except Exception as exc:
            print(f"[NemoASRManager] warmup transcribe failed: {exc}", flush=True)
            # A warmup failure under fusion means malsd_batch is broken on this
            # GPU; revert now so request #1 already serves on greedy.
            if self._fallback_to_greedy(f"warmup: {exc}"):
                try:
                    self._transcribe_batch([silence])
                    print("[NemoASRManager] warmup OK after greedy fallback", flush=True)
                except Exception as exc2:
                    print(f"[NemoASRManager] warmup greedy also failed: {exc2}", flush=True)

    # ------------------------------------------------------------------ #
    # Audio decode                                                         #
    # ------------------------------------------------------------------ #
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

    def _is_probably_silence(self, audio: np.ndarray) -> bool:
        if audio.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(audio))))
        peak = float(np.max(np.abs(audio)))
        if rms < 2e-4 and peak < 2e-3:
            return True
        if audio.size < self.TARGET_SR * 1.5 and rms < 1e-2:
            return True
        return False

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #
    def _transcribe_batch(self, audios: list[np.ndarray]) -> list[str]:
        """Run NeMo `model.transcribe()` on a batch of float32 arrays.

        NeMo accepts either a list of file paths or a list of numpy arrays
        (newer versions). We pass arrays directly to avoid round-tripping to
        disk inside the request handler.
        """
        if not audios:
            return []

        kwargs: dict[str, Any] = {"batch_size": min(len(audios), self.BATCH_SIZE)}
        # Some NeMo versions accept `verbose=False` to suppress per-batch
        # tqdm; tolerate the param being unknown.
        try:
            outputs = self.model.transcribe(audios, verbose=False, **kwargs)
        except TypeError:
            outputs = self.model.transcribe(audios, **kwargs)

        # NeMo can return either a flat list of strings or a tuple of
        # (best_hyps, all_hyps). Normalize both.
        if isinstance(outputs, tuple) and len(outputs) >= 1:
            outputs = outputs[0]

        results: list[str] = []
        for item in outputs:
            if isinstance(item, str):
                results.append(item)
            elif hasattr(item, "text"):
                results.append(getattr(item, "text") or "")
            elif isinstance(item, (list, tuple)) and item:
                first = item[0]
                results.append(first if isinstance(first, str) else getattr(first, "text", ""))
            else:
                results.append("")
        return results

    def asr_batch(self, audio_bytes_list: list[bytes]) -> list[str]:
        if not audio_bytes_list:
            return []

        _t_decode = time.perf_counter() if _TIMING else 0.0
        audios: list[np.ndarray | None] = []
        for blob in audio_bytes_list:
            try:
                audios.append(self._decode_wav(blob))
            except Exception:
                audios.append(None)
        _decode_ms = (time.perf_counter() - _t_decode) * 1000.0 if _TIMING else 0.0

        # Build the subset that actually goes into the model; carry indices so
        # we can reassemble in input order with empty strings for silences and
        # decode failures.
        results: list[str] = [""] * len(audios)
        idx_to_run: list[int] = []
        audios_to_run: list[np.ndarray] = []
        for i, audio in enumerate(audios):
            if audio is None:
                continue
            if self._is_probably_silence(audio):
                continue
            idx_to_run.append(i)
            audios_to_run.append(audio)

        if not audios_to_run:
            return results

        _t_infer = time.perf_counter() if _TIMING else 0.0
        try:
            transcripts = self._transcribe_batch(audios_to_run)
        except Exception as exc:
            print(f"[NemoASRManager] batch transcribe failed: {exc}", flush=True)
            # Never ship empty transcripts (they score 0). If fused decoding
            # broke on this GPU, revert to greedy and retry this batch once.
            if self._fallback_to_greedy(str(exc)):
                try:
                    transcripts = self._transcribe_batch(audios_to_run)
                except Exception as exc2:
                    print(f"[NemoASRManager] greedy retry failed: {exc2}", flush=True)
                    return results
            else:
                return results

        if _TIMING:
            _infer_ms = (time.perf_counter() - _t_infer) * 1000.0
            _total_ms = _decode_ms + _infer_ms
            _time_score = 1.0 - min(_total_ms / 1000.0, 5.0) / 5.0
            print(
                f"[ASR-TIMING] n={len(audios)} run={len(audios_to_run)} "
                f"decode={_decode_ms:.0f}ms infer={_infer_ms:.0f}ms "
                f"total={_total_ms:.0f}ms time_score~={_time_score:.3f}",
                flush=True,
            )

        for idx, text in zip(idx_to_run, transcripts):
            text = (text or "").strip()
            if text:
                results[idx] = self._postprocess_transcript(text)
        return results

    def _postprocess_transcript(self, text: str) -> str:
        # Parakeet emits spelled-out numbers natively, so this is mostly a
        # safety net. Cheap to run; keeps the two backends behaviorally aligned
        # on edge cases.
        return _digits_to_words(text)

    def asr(self, audio_bytes: bytes) -> str:
        return self.asr_batch([audio_bytes])[0]


# Public alias so `asr_server` can construct either backend by name. The
# whisper-based server uses `ASRManager` from `asr_manager`; this alias keeps
# the import shape uniform for the NeMo path.
ASRManager = NemoASRManager
