"""Fetch and cache the public corpora this project trains and evaluates on.

Three sources, chosen so that every evaluation number can be reproduced from a
clean checkout with no gated feed, no API key, and no account.

TRAIN_CORPUS
    25 DGA families at 13,500 domains each, plus a matched benign set drawn from
    the Alexa top list. This is the only corpus the model ever trains on.

CROSS_DGA
    Pre-generated example domains shipped alongside Johannes Bader's
    reimplementations of DGA algorithms. Different provenance, different author,
    different generation date. Families here that do not appear in TRAIN_CORPUS
    form the cross-source unseen-family test set.

    Only the checked-in example files are read. The generator scripts are never
    executed.

CROSS_BENIGN
    An independently compiled top-domains ranking. Used to check that benign
    accuracy is not an artefact of the specific Alexa snapshot the training
    corpus was built from.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    note: str


TRAIN_CORPUS = Source(
    key="dga_domains_dataset",
    url="https://github.com/chrmor/DGA_domains_dataset.git",
    note="25 DGA families plus Alexa benign, 675k rows",
)

CROSS_DGA = Source(
    key="domain_generation_algorithms",
    url="https://github.com/baderj/domain_generation_algorithms.git",
    note="per-family example_domains.txt, cross-source unseen families",
)

CROSS_BENIGN = Source(
    key="top_domains",
    url="https://github.com/zer0h/top-1000000-domains.git",
    note="independent benign ranking for cross-source benign check",
)

ALL_SOURCES = (TRAIN_CORPUS, CROSS_DGA, CROSS_BENIGN)


def fetch(source: Source, raw_dir: Path | None = None, force: bool = False) -> Path:
    """Shallow clone a source into the raw cache and return its path.

    Idempotent. An existing checkout is reused unless force is set, so repeated
    dataset builds do not re-download roughly 200 MB every time.
    """
    raw_dir = raw_dir or RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / source.key

    if dest.exists():
        if not force:
            return dest
        shutil.rmtree(dest)

    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", source.url, str(dest)],
        check=True,
        timeout=900,
    )
    return dest


def fetch_all(raw_dir: Path | None = None, force: bool = False) -> dict[str, Path]:
    return {s.key: fetch(s, raw_dir=raw_dir, force=force) for s in ALL_SOURCES}
