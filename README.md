# nxdomain-sentinel

A domain threat classifier that is scored on malware families it has never seen,
and that reports how much that costs.

Published DGA detectors routinely report accuracy above 0.99. That number is
close to meaningless, because it is measured on a random shuffle where every
malware family appears in both training and test. It answers "can you recognise
a family you were trained on", which a signature list already answers. The
question a SOC has is whether the detector fires on a family that did not exist
when the model was trained.

This repository measures that, and the answer is not flattering to anybody.

> **Status.** All five folds trained and evaluated. Service, container and
> infrastructure written and tested. Not yet deployed to a live URL. Numbers
> below come from a small model trained on one CPU core for six epochs, so they
> are a floor rather than a ceiling.

---

## Headline result

**Holding a family out costs a median 5.6 points of recall and a mean 14.9
points**, at a fixed 0.1% false positive rate.

That gap is measured by pairing. Every family is held out in exactly one fold
and trained in the other four, so each family is compared against *itself*.
Family difficulty appears on both sides of the subtraction and cancels. This
matters, because the naive comparison is actively misleading: in fold 0 the
unseen partition scored *higher* than the seen one, purely because that fold
happened to hold out five easy random-string families while the seen partition
kept every hard one.

Mean and median diverge sharply because the damage is concentrated:

| family | recall trained | recall held out | cost |
|---|---|---|---|
| vawtrak | 0.7017 | 0.0147 | **+0.687** |
| qadars | 0.9403 | 0.3542 | +0.586 |
| kraken | 0.6033 | 0.2299 | +0.373 |
| pykspa | 0.7592 | 0.4170 | +0.342 |
| simda | 0.3563 | 0.0293 | +0.327 |
| … | | | |
| murofet | 0.9685 | 0.9687 | −0.000 |
| necurs | 0.7576 | 0.7931 | −0.036 |

Most families transfer nearly free. A handful collapse completely. A single
averaged number hides which is which, which is why the per-family table is the
result and the mean is a footnote.

## The finding I did not expect

**The classifier cannot detect dictionary-based DGAs at all, trained or not.**

suppobox, matsnu and gozi build domains by concatenating real words, so their
output looks like ordinary registered names. Recall on suppobox is 0.0000 when
held out, and 0.0028 when it is sitting *in the training set*. matsnu reaches
0.1382 trained. gozi reaches 0.2244 trained.

This is not a generalisation failure. At this capacity the character CNN has no
representation for "real words in an order a human would not choose", and
holding the family out costs almost nothing because there was nothing to lose.
Any honest DGA detector should report dictionary families separately rather than
letting 22 easy random-string families carry the average.

---

## The novelty gate

A binary score answers "does this look generated". It cannot answer "is this a
family I have seen before", and for a defender those are different questions. A
confident hit on a known family is a signature match. A confident hit far from
every known family is a candidate novel campaign.

The gate fits one Gaussian per known family in the classifier's penultimate
space with a pooled, shrunk covariance, and scores Mahalanobis distance to the
nearest centroid. No OOD data is needed to fit it.

**Median per-family AUC 0.7211. Fourteen of 25 families above 0.70, eight below
0.50.**

I initially reported this gate as performing at chance. That was wrong, and the
error is worth recording because it was a measurement bug rather than a model
bug. I was pooling all five held-out families in a fold into one AUC. Four of
fold 1's five families are random-string families whose style is well
represented in training, so they are genuinely *not* novel and the gate
correctly said so. Averaging four correct "not novel" verdicts against one real
novelty produced 0.5237, which reads as broken.

Per family, fold 1 decomposes to suppobox 0.8802, pykspa 0.7354, tinba 0.3705,
ranbyus 0.3213, emotet 0.3111. The pooled number was not a weak signal, it was
the wrong measurement.

**The gate catches what the classifier misses.** suppobox is the clearest case:
recall 0.0000, gate AUC 0.8802, 34% flagged at a 5% false positive rate. The
component that fails and the component that fires are looking at different
things, which is the whole argument for having both.

### Where the style hypothesis stands

I predicted the gate would fire specifically when a held-out family's
*generation style* was absent from training. Dictionary families average 0.8038
against 0.6551 for random-string families, a gap of +0.149 in the predicted
direction.

I would not call that settled. There are only three dictionary families, and the
four highest gate scores in the table are random-string families (qadars and
nymaim at 0.9623, symmi 0.9437, vawtrak 0.9401). The direction is right and the
sample is too small to lean on.

---

## Serving

| batch | p50 | p95 | p99 | per domain |
|---|---|---|---|---|
| 1 | 1.52 ms | 1.71 ms | 1.83 ms | 1.525 ms |
| 10 | 10.55 ms | 11.34 ms | 14.53 ms | 1.055 ms |
| 100 | 102.98 ms | 107.50 ms | 109.71 ms | 1.030 ms |
| 500 | 481.51 ms | 506.97 ms | 512.57 ms | 0.963 ms |

In-process inference on one core, excluding HTTP and network. End-to-end numbers
require the deployed service and are not claimed here.

The serving artifact is **0.57 MB** and carries weights, the novelty gate, the
decision threshold, the novelty threshold, the family list, and the gate's
measured AUC in one file. A threshold separated from the model it was calibrated
on is a production incident waiting to happen.

Because the artifact carries its measured gate AUC, **the API labels its own
novelty field as unproven** whenever that AUC falls below 0.70. A field that
means nothing is worse than no field.

```bash
curl -X POST https://<service>/v1/score \
  -H 'Content-Type: application/json' \
  -d '{"domain":"cvyh1po636avyrsxebwbkn7.ddns.net"}'
```

```json
{
  "domain": "cvyh1po636avyrsxebwbkn7.ddns.net",
  "core": "cvyh1po636avyrsxebwbkn7",
  "verdict": "dga",
  "dga_probability": 0.998,
  "nearest_family": "corebot",
  "novelty_score": 84.2,
  "is_novel": false
}
```

Note `core`. **The model never sees a TLD.** In the training corpus corebot
arrives as `<random>.ddns.net` while most benign samples end in `.com`. A
character model handed the full name learns "ends in ddns.net" and reports a
spectacular score that collapses the first time a family changes registrar.
Everything right of the public suffix is stripped before encoding, using a
pinned Public Suffix List snapshot with private domains enabled. TLD reputation
is a real signal and a real defence should use it. It is excluded so the
reported generalisation describes the string model rather than a registrar
lookup in disguise.

---

## Infrastructure

Cloud Run on the always-free tier, which is 2M requests, 180,000 vCPU-seconds
and 360,000 GiB-seconds a month. At 512 MiB and roughly 1 ms of inference the
binding constraint is request count, not compute.

- `min_instance_count = 0` with `cpu_idle`. Pinning one warm instance costs about
  $65 a month, so a scheduled ping every ten minutes keeps an instance alive
  instead, at roughly 4,300 requests against a 2,000,000 allowance.
- Artifact Registry cleanup keeps three versions. Free storage is 0.5 GB and
  without a policy every CI build accumulates until it silently bills.
- A one dollar budget alert as a tripwire, since any spend at all means
  something is wrong.
- Deploy authenticates through Workload Identity Federation. No long-lived
  service account key exists in this repository.
- CI fails the build above 900 MB, catching the failure mode where wheel
  selection silently pulls 2.5 GB of CUDA into a CPU-only service.

---

## Data

All public, reproducible from a clean checkout with no gated feed and no account.

| role | source | contents |
|---|---|---|
| training | [chrmor/DGA_domains_dataset](https://github.com/chrmor/DGA_domains_dataset) | 25 families at 13,500 each, plus matched benign |
| cross-source DGA | [baderj/domain_generation_algorithms](https://github.com/baderj/domain_generation_algorithms) | per-family example domains, 16 usable unseen families |
| cross-source benign | [zer0h/top-1000000-domains](https://github.com/zer0h/top-1000000-domains) | independent top-domains ranking |

Only checked-in example files are read from the generator repository. The
generator scripts are never imported or executed.

**827,924 domains** after normalisation and deduplication, 350,789 malicious and
477,135 benign. Three things the build catches:

- **42,148 benign domains** appear in both benign corpora. Deduplicated globally
  on the normalised core.
- **47 label conflicts**, domains present as both benign and malicious. These are
  short-string families such as symmi generating four-character labels that
  collide with real registrations. Dropped rather than arbitrarily assigned.
- `nan.com`, `inf.com`, `na.com` and `null.com` are real registrations that
  pandas converts to floats under default NA handling. Every read path disables
  NA coercion, with regression tests.

---

## Reproducing

```bash
make install        # CPU-only torch
make data           # clones corpora, builds splits, ~2 min
make test           # 42 tests
make train          # 5 folds
make eval           # scores every checkpoint
make analyze        # paired per-family comparison
make export bench   # serving artifact and latency table
make serve          # local API on :8080
```

Training and evaluation are separate commands with a checkpoint between them, so
an evaluation failure costs seconds rather than the whole training run. Folds
materialise one at a time; building all five copies 3.4 million rows and was the
actual cause of repeated out-of-memory kills during development.

## Layout

```
src/nxsentinel/
  data/normalize.py    public-suffix stripping, the anti-shortcut logic
  data/build.py        assembly, deduplication, conflict handling
  data/splits.py       family holdout, leakage assertions
  model/charcnn.py     character CNN, 96,917 params, two heads
  model/ood.py         Mahalanobis novelty gate
  train/evaluate.py    metrics, frozen thresholds, per-family breakdowns
  serve/               FastAPI app, predictor, web UI
scripts/               build_dataset, train, evaluate, analyze, export_model
infra/                 Terraform for Cloud Run
bench/                 latency harness
```

Thresholds are calibrated on validation and frozen before any test partition is
touched. The leakage suite asserts that a deliberately constructed leak is
*detected*, because a split bug does not crash, it quietly raises every score.

## Known limitations

- Dictionary DGAs are undetected. This needs word-boundary or lexical features,
  not more epochs.
- Six epochs on one CPU core. Seen-family recall at 0.1% FPR is 0.5656, a floor.
- Only three dictionary families exist in the corpus, so the style analysis is
  underpowered.
- Not deployed yet, so no end-to-end latency and no real cold-start number.

## License

MIT. See [LICENSE](LICENSE).
