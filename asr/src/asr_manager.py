"""Baseline ASR manager.

This is intentionally simple: it returns a valid transcript string for every
request without loading a model. Use it as the integration baseline before
replacing it with Whisper/NeMo/etc.
"""


class ASRManager:
    """Valid-but-dumb ASR baseline."""

    def __init__(self):
        # Keep startup fast and deterministic. Heavy ASR models should be loaded
        # here later so inference does not reload weights per request.
        self.default_transcript = ""

    def asr(self, audio_bytes: bytes) -> str:
        """Transcribe one WAV file.

        Args:
            audio_bytes: Raw WAV bytes.

        Returns:
            A transcript string. Empty string is schema-valid but scores badly.
        """
        _ = audio_bytes
        return self.default_transcript
