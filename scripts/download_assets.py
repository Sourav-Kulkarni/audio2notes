"""Download the instrument sample bank used to render notes as audio.

    .venv\\Scripts\\python scripts\\download_assets.py

GeneralUser GS (S. Christian Collins) - free to use and redistribute, see
https://schristiancollins.com/generaluser.php . Mirrored on Hugging Face in the
projectlosangeles/soundfonts4u dataset. Without it, rendering falls back to a basic synth.
(Demucs / CREPE model weights download automatically on first use.)
"""
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = [
    ("https://huggingface.co/datasets/projectlosangeles/soundfonts4u/resolve/main/GeneralUser_GS_v1_471.sf2",
     ROOT / "third_party" / "soundfonts" / "GeneralUser_GS.sf2", 31_281_186),
]


def download(url, dest, size):
    if dest.exists() and dest.stat().st_size == size:
        print(f"ok      {dest.relative_to(ROOT)} (already present)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"get     {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or size)
        got = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            got += len(chunk)
            print(f"\r        {got / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()
    with open(tmp, "rb") as f:
        if f.read(4) != b"RIFF":
            tmp.unlink()
            sys.exit(f"downloaded file is not a SoundFont: {url}")
    tmp.replace(dest)
    print(f"saved   {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    for url, dest, size in ASSETS:
        download(url, dest, size)
