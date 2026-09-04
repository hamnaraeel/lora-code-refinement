"""Config loading, inheritance and fingerprinting.

A silent inheritance bug means eight sweep arms trained with defaults nobody
chose, so the chain resolution is tested explicitly.
"""

import textwrap

import pytest

from coderefine.config import RunConfig, parse_overrides


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _three_level_chain(tmp_path):
    """base.yaml <- mid.yaml <- arm.yaml, the shape the real sweep configs use."""
    _write(tmp_path, "base.yaml", """
        name: base
        base_model: m
        train: {learning_rate: 2.0e-4, num_epochs: 3, seed: 42}
        lora: {r: 16, alpha: 32}
    """)
    _write(tmp_path, "mid.yaml", """
        extends: base.yaml
        name: mid
        quant: {load_in_4bit: true}
    """)
    return _write(tmp_path, "arm.yaml", """
        extends: mid.yaml
        name: arm
        lora: {r: 8, alpha: 16}
    """)


class TestInheritance:
    def test_grandparent_values_survive(self, tmp_path):
        # The bug that would otherwise be invisible: a sweep arm extending a
        # config that itself extends base.yaml, losing every base default.
        _three_level_chain(tmp_path)
        cfg = RunConfig.from_yaml(tmp_path / "arm.yaml")
        assert cfg.name == "arm"
        assert cfg.lora.r == 8 and cfg.lora.alpha == 16   # from the arm
        assert cfg.quant.load_in_4bit is True             # from the parent
        assert cfg.train.seed == 42                       # from the grandparent
        assert cfg.train.learning_rate == 2.0e-4          # from the grandparent

    def test_nested_dicts_merge_rather_than_replace(self, tmp_path):
        _write(tmp_path, "a.yaml", "name: a\ntrain: {learning_rate: 1.0e-4, num_epochs: 3}\n")
        _write(tmp_path, "b.yaml", "extends: a.yaml\nname: b\ntrain: {num_epochs: 5}\n")
        cfg = RunConfig.from_yaml(tmp_path / "b.yaml")
        assert cfg.train.num_epochs == 5
        assert cfg.train.learning_rate == 1.0e-4  # not clobbered by the partial override

    def test_circular_inheritance_is_an_error(self, tmp_path):
        _write(tmp_path, "x.yaml", "extends: y.yaml\nname: x\n")
        _write(tmp_path, "y.yaml", "extends: x.yaml\nname: y\n")
        with pytest.raises(ValueError, match="Circular"):
            RunConfig.from_yaml(tmp_path / "x.yaml")

    def test_missing_parent_is_an_error(self, tmp_path):
        _write(tmp_path, "x.yaml", "extends: nope.yaml\nname: x\n")
        with pytest.raises(FileNotFoundError):
            RunConfig.from_yaml(tmp_path / "x.yaml")


class TestValidation:
    def test_unknown_top_level_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown config key"):
            RunConfig.from_dict({"nonsense": 1})

    def test_unknown_nested_key_rejected(self):
        with pytest.raises(ValueError, match="Unknown keys in lora"):
            RunConfig.from_dict({"lora": {"rank": 8}})  # it's `r`, not `rank`


class TestFingerprint:
    def test_same_settings_same_fingerprint(self):
        a = RunConfig.from_dict({"name": "one", "lora": {"r": 8}})
        b = RunConfig.from_dict({"name": "two", "lora": {"r": 8}})
        assert a.fingerprint() == b.fingerprint()  # name is not part of identity

    def test_different_settings_differ(self):
        a = RunConfig.from_dict({"lora": {"r": 8}})
        b = RunConfig.from_dict({"lora": {"r": 16}})
        assert a.fingerprint() != b.fingerprint()

    def test_notes_do_not_affect_identity(self):
        a = RunConfig.from_dict({"notes": "first attempt"})
        b = RunConfig.from_dict({"notes": "second attempt"})
        assert a.fingerprint() == b.fingerprint()


class TestOverrides:
    def test_dotted_paths_become_nested(self):
        assert parse_overrides(["train.num_epochs=5"]) == {"train": {"num_epochs": 5}}

    def test_types_are_parsed_not_left_as_strings(self):
        out = parse_overrides(["train.learning_rate=1e-4", "quant.load_in_4bit=true"])
        assert out["train"]["learning_rate"] == 1e-4
        assert out["quant"]["load_in_4bit"] is True

    def test_malformed_override_rejected(self):
        with pytest.raises(ValueError):
            parse_overrides(["no_equals_sign"])

    def test_overrides_apply_on_top_of_a_file(self, tmp_path):
        p = tmp_path / "c.yaml"
        p.write_text("name: c\ntrain: {num_epochs: 3}\n", encoding="utf-8")
        cfg = RunConfig.from_yaml(p, parse_overrides(["train.num_epochs=7"]))
        assert cfg.train.num_epochs == 7
