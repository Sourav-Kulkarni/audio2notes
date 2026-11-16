"""Synthetic test song: vocal-like melody (Sa=D, +30 cents sharp, vibrato, meend) + drums + pad."""
import sys
import numpy as np
import soundfile as sf

sr = 44100
rng = np.random.default_rng(0)
sa = 62  # D4
# (sargam offset, duration)
melody = [(0,.6),(2,.4),(4,.4),(5,.6),(7,1.0),(None,.3),(7,.4),(9,.4),(7,.4),(5,.4),(4,.8),(None,.3),
          (4,.4),(5,.4),(7,.4),(12,1.0),(11,.3),(9,.3),(7,.8),(None,.3),(5,.4),(4,.4),(2,.4),(0,1.2)]
f0 = []
for off, d in melody:
    n = int(d * sr)
    if off is None:
        f0.append(np.zeros(n)); continue
    f0.append(np.full(n, 440 * 2 ** ((sa + off - 69 + 0.30) / 12)))
f0 = np.concatenate(f0)
t = np.arange(len(f0)) / sr
# glides between notes (meend): smooth log-f0 across voiced regions
lf = np.where(f0 > 0, np.log(np.maximum(f0, 1)), 0)
k = int(0.06 * sr); ker = np.ones(k) / k
v = (f0 > 0).astype(float)
lf_s = np.convolve(lf, ker, "same") / np.maximum(np.convolve(v, ker, "same"), 1e-6)
f = np.where(f0 > 0, np.exp(lf_s) * 2 ** (0.4 / 12 * np.sin(2 * np.pi * 5.5 * t) * np.clip(t % 2, 0, 1)), 0)
phase = 2 * np.pi * np.cumsum(f) / sr
voice = sum((1 / h) * np.sin(h * phase) * np.exp(-((h * f - 700) / 900) ** 2 * 0.5 + 0.0) for h in range(1, 20))
env = np.convolve(v, np.ones(int(0.03 * sr)) / int(0.03 * sr), "same")
voice = voice * env
# accompaniment: drums + pad (D major-ish chord)
N = len(voice)
drums = np.zeros(N)
for b in np.arange(0, N / sr, 0.5):
    i = int(b * sr); L = min(4000, N - i)
    drums[i:i+L] += np.sin(2*np.pi*60*np.arange(L)/sr) * np.exp(-np.arange(L)/800) * 0.8
    j = int((b + 0.25) * sr); L = min(3000, N - j)
    if L > 0: drums[j:j+L] += rng.normal(0, 0.3, L) * np.exp(-np.arange(L)/500)
pad = sum(0.08 * np.sin(2*np.pi*440*2**((m-69)/12)*t) for m in (50, 57, 62, 66))
mix = 0.5 * voice / np.abs(voice).max() + 0.4 * drums + pad
mix = np.stack([mix, mix], 1) / np.abs(mix).max() * 0.9
sf.write(sys.argv[1], mix, sr)
sf.write(sys.argv[1].replace(".wav", "_vocal_only.wav"), voice / np.abs(voice).max() * .9, sr)
print("wrote", sys.argv[1], f"{N/sr:.1f}s")
