"""Download a small public arXiv PDF set into data/pdfs (for demos)."""

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "pdfs"

SAMPLE_PDFS = {
    "1706.03762_attention_is_all_you_need.pdf": "https://arxiv.org/pdf/1706.03762.pdf",
    "1810.04805_bert.pdf": "https://arxiv.org/pdf/1810.04805.pdf",
    "1409.3215_seq2seq.pdf": "https://arxiv.org/pdf/1409.3215.pdf",
    "1508.04025_luong_attention.pdf": "https://arxiv.org/pdf/1508.04025.pdf",
    "1607.06450_layer_normalization.pdf": "https://arxiv.org/pdf/1607.06450.pdf",
}


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    for name, url in SAMPLE_PDFS.items():
        dest = DATA / name
        if dest.exists() and dest.stat().st_size > 10_000:
            print("skip (exists):", dest)
            continue
        print("downloading", url, "->", dest)
        urllib.request.urlretrieve(url, dest)
    print("Done. PDFs in", DATA)


if __name__ == "__main__":
    main()
