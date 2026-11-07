from types import SimpleNamespace

from nxsentinel.ops.lexical import LexicalSignal
from nxsentinel.ops.replay import Event, alert, windows


def test_lexical_signal_flags_new_pairs_in_the_registrable_label(tmp_path):
    model = LexicalSignal.fit(
        ["river", "bank", "orchard"], ["riverbank", "mail.riverbank"]
    )
    assert model.pairs == {"river bank": 2}
    assert not model.flags("mail.riverbank")
    assert not model.flags("river")
    assert not model.flags("river2orchard")
    assert model.flags("mail.riverorchard")

    artifact = tmp_path / "lexicon.json"
    model.save(artifact)
    loaded, digest = LexicalSignal.load(artifact)
    assert len(digest) == 64 and loaded.flags("riverorchard")


def test_lexical_candidate_survives_low_classifier_and_novelty_scores():
    model = LexicalSignal.fit(["river", "bank", "orchard"], ["riverbank"])

    def predict(names):
        assert names == ["riverorchard.com"]
        return [SimpleNamespace(core="riverorchard", verdict="benign",
                                above_novelty_threshold=False, dga_probability=0.0,
                                novelty_score=0.0)]

    row, = windows([Event(10, "device", "riverorchard.com", True)], predict, lexical=model)
    assert row["combined_names"] == 0
    assert row["combined_lexical_names"] == row["lexical_names"] == 1
    assert row["evidence"][0]["rare_word_pair"]
    assert alert(row, 1, 1)
