import wave

import numpy as np

from src.embed_audio import TARGET_SAMPLE_RATE, load_audio


def _write_pcm_wav(path, sample_rate, seconds=0.05):
    samples = np.zeros(int(sample_rate * seconds), dtype=np.int16).tobytes()
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(samples)


def test_load_audio_resamples_to_clap_rate(tmp_path):
    path = tmp_path / "short-8khz.wav"
    _write_pcm_wav(path, 8_000)
    waveform = load_audio(path)
    assert waveform.dtype == np.float32
    assert len(waveform) == TARGET_SAMPLE_RATE * 0.05


def test_load_audio_keeps_target_rate(tmp_path):
    path = tmp_path / "short-48khz.wav"
    _write_pcm_wav(path, TARGET_SAMPLE_RATE)
    waveform = load_audio(path)
    assert len(waveform) == TARGET_SAMPLE_RATE * 0.05
