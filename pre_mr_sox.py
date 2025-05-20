import pathlib
import shutil
import os

if __name__ == '__main__':
    base_dir = pathlib.Path(__file__).parent.resolve()
    songs_dir = base_dir / "songs"
    mr_root = base_dir / "mr"

    for song_dir in songs_dir.iterdir():
        if song_dir.is_dir():
            other_path = song_dir / "other.wav"
            if other_path.exists():
                target_dir = mr_root / song_dir.name
                os.makedirs(target_dir, exist_ok=True)
                target_path = target_dir / "mr.wav"
                shutil.copy2(str(other_path), str(target_path))
                print(f"Copied {other_path} -> {target_path}")
            else:
                print(f"(skip) {other_path} not found")
