"""Download GSM8K train/test splits from OpenAI's grade-school-math repository."""

from pathlib import Path
import requests


BASE_RAW_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data"


def download_file(url: str, output_path: Path) -> bool:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {url} -> {output_path}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        output_path.write_bytes(response.content)
        print(f"  Saved {output_path.stat().st_size} bytes")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        if output_path.exists():
            output_path.unlink()
        return False


def main():
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "gsm8k"

    files = {
        "train": f"{BASE_RAW_URL}/train.jsonl",
        "test": f"{BASE_RAW_URL}/test.jsonl",
    }

    success = 0
    for split, url in files.items():
        out_path = data_dir / f"{split}.jsonl"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"Skipping {split}: already exists at {out_path}")
            success += 1
            continue
        if download_file(url, out_path):
            success += 1

    print(f"\nCompleted: {success}/{len(files)} files available.")


if __name__ == "__main__":
    main()
