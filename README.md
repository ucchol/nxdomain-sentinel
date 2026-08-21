# nxdomain-sentinel

A domain threat classifier that is scored on malware families it has never seen.

Most published DGA detectors report accuracy above 0.99. That number is close to
meaningless, because it is measured on a random shuffle where every malware
family appears in both training and test. It answers "can you recognise a
family you were trained on", which is a question a signature list already
answers. The question a SOC actually has is whether the detector fires on a
family that did not exist when the model was trained.

This repository is built to measure that second thing honestly, and to serve the
result as a small, cheap, always-on API.

> **Status, August 2026.** Research phase. The data pipeline, model, and
> evaluation harness are complete and tested. One of five folds is trained. The
> serving layer, container, and deployment are not written yet. The
> out-of-distribution gate currently performs at chance, which is documented
> below rather than hidden. Do not read the numbers in this README as final.

---

## The evaluation design

Three levels, in increasing order of how much they tell you.

**Random split.** The conventional shuffle. Included only as a reference point
for how inflated the usual number is.

**Family holdout, five folds.** Whole families are removed from training and
validation and reappear only at test time. Across the five folds every one of
the 25 training families is held out exactly once, so no single lucky partition
drives the result. Each fold also carries a matched in-distribution slice scored
by the *same* model, which is what makes the seen and unseen numbers comparable
without the confound of two separate training runs.

**Cross-source.** Sixteen further families from a different author, a different
generation date, and a different repository, none of which appear in any
training pool. Benign traffic is checked the same way, against an independently
compiled top-domains ranking rather than the list the training corpus was built
from.

### Why the public suffix is stripped

The model never sees a TLD. In the training corpus, corebot arrives as
`<random>.ddns.net` while most benign samples end in `.com`. A character model
handed the full name learns "ends in ddns.net" and reports a spectacular score
that collapses the first time a family changes registrar. Everything to the
right of the public suffix is removed before encoding, using a pinned Public
Suffix List snapshot with private domains enabled so that dynamic DNS providers
resolve as suffixes.

TLD reputation is a real signal and a real defence should use it. It is excluded
here so the reported generalisation describes the string model alone rather than
a registrar lookup in disguise.

---

## Results so far

Fold 0, holding out cryptolocker, dircrypt, padcrypt, ramnit and rovnix.
Character CNN, 96,917 parameters, six epochs, trained on a subsample on a single
CPU core. Undertrained on purpose while the harness was being validated.

| partition | AUC | AUPRC | recall @ 0.1% FPR |
|---|---|---|---|
| seen families | 0.9757 | 0.9751 | 0.5656 |
| unseen families | 0.9942 | 0.9961 | 0.7770 |
| cross-source | 0.9633 | 0.8721 | 0.5417 |

Recall at a fixed 0.1% false positive rate is the headline operating point
rather than accuracy or F1. Domain traffic is overwhelmingly benign, so a
detector tuned at 1% FPR produces an unusable flood of alerts.

**The unseen number is higher than the seen number, and that is a flaw in the
experiment, not a result.** Fold 0 happens to hold out five families that all
generate high-entropy random-looking strings, which are easy whether or not you
trained on them. The seen partition meanwhile retains every hard family,
including the dictionary-based ones that look like real registrations. The
comparison is measuring which families landed in the holdout.

The fix is a paired per-family design. For each family, compare its recall in
the one fold where it was held out against its recall in the four folds where it
was trained. Same family both times, so difficulty cancels out. That requires
all five folds and is the next piece of work.

---

## Open problem: the novelty gate is at chance

A binary score answers "does this look generated". It cannot answer "is this a
family I have seen before", and for a defender those are different questions. A
confident hit on a known family is a signature match. A confident hit that sits
far from every known family is a candidate novel campaign, which is the alert
worth waking someone for.

The gate fits one Gaussian per known family in the classifier's penultimate
space with a shared, shrunk covariance, and scores the Mahalanobis distance to
the nearest centroid. No OOD data is needed to fit it.

It does not currently work. AUC 0.5376 separating unseen families from seen
ones, and 5.83% of unseen families flagged at a 5% false positive rate on seen
families, which is chance.

Two explanations were tested and rejected:

- **LayerNorm flattening the magnitude signal.** Scoring the pre-norm
  representation instead gave 0.4777, worse than chance. Rejected.
- **The embedding lacking family structure.** The family head reaches 0.8021
  accuracy across 20 classes against a 0.05 baseline. The structure is there.
  Rejected.

What survives is that the embedding organises by *generation style* rather than
family identity. Random-hex, random-alphanumeric and dictionary-based look like
the real clusters. Fold 0's held-out families are all random-alphanumeric, a
style already well covered in training, so they land near known centroids and
the gate correctly reports they are not novel.

That is falsifiable. Fold 1 holds out suppobox, a dictionary DGA, while training
on almost no dictionary families. If the account is right, fold 1's gate AUC
should rise sharply and suppobox should drive nearly all of it. If it does not,
the gate needs redesigning rather than reframing.

---

## Data

All three corpora are public, and a clean checkout reproduces the dataset with
no gated feed, no API key and no account.

| role | source | contents |
|---|---|---|
| training | [chrmor/DGA_domains_dataset](https://github.com/chrmor/DGA_domains_dataset) | 25 families at 13,500 each, plus matched benign |
| cross-source DGA | [baderj/domain_generation_algorithms](https://github.com/baderj/domain_generation_algorithms) | per-family example domains, 16 usable unseen families |
| cross-source benign | [zer0h/top-1000000-domains](https://github.com/zer0h/top-1000000-domains) | independent top-domains ranking |

Only the checked-in example files from the generator repository are read. The
generator scripts are never imported or executed.

After normalisation and deduplication: **827,924 domains**, 350,789 malicious
and 477,135 benign.

Two things the build catches that are worth knowing about:

- **42,148 benign domains** appear in both benign corpora. Deduplicated globally
  on the normalised core, training corpus winning.
- **47 label conflicts**, domains present as both benign and malicious. These
  are short-string families such as symmi generating four-character labels that
  collide with real registrations. Dropped rather than arbitrarily assigned.

And one bug worth a regression test: `nan.com`, `inf.com`, `na.com` and
`null.com` are real registrations that pandas silently converts to floats under
default NA handling. Every read path disables NA coercion.

---

## Reproducing

```bash
git clone https://github.com/<you>/nxdomain-sentinel.git
cd nxdomain-sentinel
pip install -e ".[dev]"

python scripts/build_dataset.py          # clones corpora, builds splits, ~2 min
python -m pytest tests/ -q               # 30 tests

python scripts/train.py    --folds 0 1 2 3 4 --epochs 12
python scripts/evaluate.py --folds 0 1 2 3 4
```

Training and evaluation are separate commands with a checkpoint between them, so
an evaluation failure costs seconds rather than the whole training run.

Folds are materialised one at a time. Building all five at once copies 3.4
million rows and was the actual cause of repeated out-of-memory kills during
development.

---

## Layout

```
src/nxsentinel/
  data/normalize.py    public-suffix stripping, the anti-shortcut logic
  data/sources.py      corpus definitions and cached fetching
  data/build.py        assembly, deduplication, conflict handling
  data/splits.py       family holdout, leakage assertions
  model/charcnn.py     character CNN, two heads, shared embedding
  model/ood.py         Mahalanobis novelty gate over that embedding
  train/loop.py        training, gate fitting
  train/evaluate.py    metrics, frozen thresholds, per-family breakdown
scripts/               build_dataset, train, evaluate
tests/                 30 tests, including a deliberately constructed leak
```

The leakage tests matter more than they look. A split bug does not crash, it
quietly raises every score, so the suite asserts that a constructed leak is
*detected* rather than trusting that one never happens.

---

## Not done yet

- Folds 1 through 4, and the paired per-family comparison
- Resolving or redesigning the novelty gate
- FastAPI service, web UI, container
- Terraform for Cloud Run, GitHub Actions deploy via Workload Identity Federation
- Measured p50, p95, p99 under load, cold start time, and the actual monthly bill

## License

MIT. See [LICENSE](LICENSE).
