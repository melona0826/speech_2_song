import pathlib
import shutil
import os

if __name__ == '__main__':
    base_dir = pathlib.Path(__file__).parent.resolve()
    songs_dir = base_dir / "songs"
    vocals_root = base_dir / "vocals"

    for song_dir in songs_dir.iterdir():
        if song_dir.is_dir():
            vocals_path = song_dir / "vocals.wav"
            if vocals_path.exists():
                target_dir = vocals_root / song_dir.name
                os.makedirs(target_dir, exist_ok=True)
                target_path = target_dir / "vocals.wav"
                shutil.copy2(str(vocals_path), str(target_path))
                print(f"Copied {vocals_path} -> {target_path}")
            else:
                print(f"(skip) {vocals_path} not found")
