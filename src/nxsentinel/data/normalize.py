"""Turn a raw domain string into the exact text the model sees.

The single most important decision in this file is that the public suffix is
removed and never shown to the model.

A DGA corpus carries a strong TLD shortcut. In the 25-family training corpus,
corebot samples arrive as ``<random>.ddns.net`` while most benign samples end in
``.com``. A character model handed the full FQDN learns "ends in ddns.net" and
reports a spectacular in-distribution score that collapses the moment a family
switches registrar. Stripping the suffix forces the model to work from the
generated label itself, which is the part that actually characterises a DGA.

TLD reputation is a real production signal and a real defence should use it.
It is excluded here so the reported generalisation numbers describe the string
model alone rather than a registrar lookup in disguise.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

import tldextract

# Bundled public suffix snapshot. suffix_list_urls=() pins it so a build never
# depends on a live fetch, and results stay reproducible across machines.
# include_psl_private_domains=True is what makes dynamic-DNS providers such as
# ddns.net resolve as suffixes rather than as registrable domains.
_EXTRACT = tldextract.TLDExtract(
    suffix_list_urls=(),
    include_psl_private_domains=True,
)

# Characters permitted in a hostname label, plus the dot that joins labels.
_ALLOWED = set(string.ascii_lowercase + string.digits + "-.")

MAX_CORE_LEN = 96


@dataclass(frozen=True)
class Normalized:
    """Result of normalising one raw domain.

    Attributes:
        core: the text fed to the model, lowercase, public suffix removed.
        suffix: the public suffix that was stripped, kept for auditing only.
        ok: whether the record should be used at all.
        reason: why it was rejected, empty when ok is True.
    """

    core: str
    suffix: str
    ok: bool
    reason: str = ""


def normalize(raw: str) -> Normalized:
    """Normalise one domain into model input.

    Returns a Normalized with ok=False rather than raising, because these
    corpora contain a small amount of junk and the caller wants to count
    rejections rather than crash on them.
    """
    if raw is None:
        return Normalized("", "", False, "null")

    d = raw.strip().lower().rstrip(".")
    if not d:
        return Normalized("", "", False, "empty")

    # Punycode domains encode a script the character model has no way to read,
    # and they are rare enough in both corpora that dropping them is cleaner
    # than pretending the model handles them.
    if d.startswith("xn--") or ".xn--" in d:
        return Normalized("", "", False, "punycode")

    if set(d) - _ALLOWED:
        return Normalized("", "", False, "bad_chars")

    ex = _EXTRACT(d)
    if not ex.suffix:
        return Normalized("", "", False, "no_public_suffix")
    if not ex.domain:
        return Normalized("", ex.suffix, False, "suffix_only")

    core = f"{ex.subdomain}.{ex.domain}" if ex.subdomain else ex.domain
    core = core.strip(".")

    if not core:
        return Normalized("", ex.suffix, False, "empty_core")
    if len(core) > MAX_CORE_LEN:
        return Normalized("", ex.suffix, False, "too_long")
    # A bare single character carries no signal and appears in both classes.
    if len(core) < 2:
        return Normalized("", ex.suffix, False, "too_short")

    return Normalized(core, ex.suffix, True)


def normalize_many(raws) -> tuple[list[Normalized], dict[str, int]]:
    """Normalise an iterable of domains and report why records were dropped."""
    out: list[Normalized] = []
    rejects: dict[str, int] = {}
    for r in raws:
        n = normalize(r)
        out.append(n)
        if not n.ok:
            rejects[n.reason] = rejects.get(n.reason, 0) + 1
    return out, rejects


_WS = re.compile(r"\s+")


def read_domain_lines(text: str):
    """Yield candidate domains from a text blob, one per line.

    Handles the two shapes these corpora use, a bare domain per line and a
    ``rank,domain`` pair, without needing a separate parser for each source.
    """
    for line in text.splitlines():
        line = _WS.sub("", line)
        if not line or line.startswith("#"):
            continue
        if "," in line:
            parts = [p for p in line.split(",") if p]
            line = parts[-1] if parts else ""
        if line:
            yield line
