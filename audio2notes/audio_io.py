"""Decode any audio format via ffmpeg; read/write wav via soundfile."""
import shutil
import subprocess

import numpy as np
import soundfile as sf


def load_audio(path, sr=44100, channels=2):
    """Decode mp3/m4a/aac/flac/ogg/opus/wav/wma/... to float32 array (channels, samples)."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn",
           "-ac", str(channels), "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).T
    return np.ascontiguousarray(audio)


def read_wav(path):
    """Returns (audio (channels, samples), sr)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.T, sr


def write_wav(path, audio, sr):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.T
    sf.write(str(path), audio, sr, subtype="PCM_16")


def to_mono(audio):
    return audio.mean(axis=0) if audio.ndim == 2 else audio
