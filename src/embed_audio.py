"""CLAP audio and text embedding helpers."""

from __future__ import annotations

from pathlib import Path
import wave

import numpy as np

TARGET_SAMPLE_RATE = 48_000
_MODEL = None


def _load_wav_fallback(path: Path) -> np.ndarray:
    """Read basic PCM WAV files when optional librosa is unavailable."""
    with wave.open(str(path), "rb") as stream:
        channels, width, rate, frames = (
            stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.readframes(stream.getnframes())
        )
    if width != 2:
        raise ValueError("WAV fallback supports 16-bit PCM only")
    waveform = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    waveform = waveform.reshape(-1, channels).mean(axis=1)
    if rate != TARGET_SAMPLE_RATE:
        old_times = np.arange(len(waveform), dtype=np.float64) / rate
        new_length = round(len(waveform) * TARGET_SAMPLE_RATE / rate)
        new_times = np.arange(new_length, dtype=np.float64) / TARGET_SAMPLE_RATE
        waveform = np.interp(new_times, old_times, waveform).astype(np.float32)
    return waveform


def load_audio(path: str | Path) -> np.ndarray:
    """Load an audio file as mono float32 samples at 48 kHz."""
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {audio_path}")
    try:
        import librosa
        waveform, _ = librosa.load(str(audio_path), sr=TARGET_SAMPLE_RATE, mono=True)
    except ImportError:
        if audio_path.suffix.lower() != ".wav":
            raise ValueError("librosa is required for non-WAV audio files")
        waveform = _load_wav_fallback(audio_path)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not load audio file: {audio_path}") from exc
    if waveform.size == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    return np.asarray(waveform, dtype=np.float32)


def _get_model():
    global _MODEL
    if _MODEL is None:
        import laion_clap
        _MODEL = laion_clap.CLAP_Module(enable_fusion=False)
        _MODEL.load_ckpt()
    return _MODEL


def _normalize(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def embed_audio(paths: list[str], batch_size: int = 8) -> np.ndarray:
    """Embed audio files in batches into normalized CLAP audio vectors."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not paths:
        return np.empty((0, 512), dtype=np.float32)
    model = _get_model()
    vectors = []
    for start in range(0, len(paths), batch_size):
        waveforms = [load_audio(path) for path in paths[start : start + batch_size]]
        vectors.append(np.asarray(model.get_audio_embedding_from_data(x=waveforms, use_tensor=False), dtype=np.float32))
    return _normalize(np.concatenate(vectors, axis=0)).astype(np.float32)


def embed_text_for_audio(query: str) -> np.ndarray:
    """Embed text with CLAP's text encoder, not CLIP's text encoder."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    model = _get_model()
    vector = np.asarray(model.get_text_embedding([query], use_tensor=False), dtype=np.float32)[0]
    return _normalize(vector[None, :])[0].astype(np.float32)
