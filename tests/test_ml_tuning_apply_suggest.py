"""Tests for the safe ``trial.suggest_*`` parser used by :func:`ml_funcs.tuning.ml_tuner`."""

from __future__ import annotations

import pytest

pytest.importorskip("hyperopt", reason="ml_funcs.tuning requires hyperopt")

from dstoolbox.ml_funcs.tuning import _apply_trial_suggest  


class _FakeTrial:
    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []

    def suggest_int(self, name: str, low: int, high: int, **kw):
        self.calls.append(("suggest_int", (name, low, high), kw))
        return (low + high) // 2

    def suggest_float(self, name: str, low: float, high: float, **kw):
        self.calls.append(("suggest_float", (name, low, high), kw))
        return (low + high) / 2

    def suggest_categorical(self, name: str, choices):
        self.calls.append(("suggest_categorical", (name, tuple(choices)), {}))
        return choices[0]


class TestApplyTrialSuggest:
    def test_int_positional(self):
        trial = _FakeTrial()
        assert _apply_trial_suggest("trial.suggest_int('max_depth', 3, 9)", trial) == 6
        assert trial.calls == [("suggest_int", ("max_depth", 3, 9), {})]

    def test_float_with_kwargs(self):
        trial = _FakeTrial()
        _apply_trial_suggest(
            "trial.suggest_float('lr', 0.001, 0.1, log=True)", trial
        )
        assert trial.calls[0][2] == {"log": True}

    def test_categorical_from_list(self):
        trial = _FakeTrial()
        _apply_trial_suggest(
            "trial.suggest_categorical('opt', ['adam', 'sgd'])", trial
        )
        assert trial.calls[0][1] == ("opt", ("adam", "sgd"))

    def test_rejects_non_trial_expression(self):
        with pytest.raises(ValueError):
            _apply_trial_suggest("os.system('rm -rf /')", _FakeTrial())

    def test_rejects_arbitrary_call(self):
        with pytest.raises(ValueError):
            _apply_trial_suggest("trial.report(0.1, step=1)", _FakeTrial())

    def test_rejects_non_literal_args(self):
        # Names in args should be rejected — literal_eval only accepts literals.
        with pytest.raises((ValueError, SyntaxError)):
            _apply_trial_suggest("trial.suggest_int('x', low, 5)", _FakeTrial())

    def test_rejects_syntax_error(self):
        with pytest.raises(ValueError):
            _apply_trial_suggest("trial.suggest_int(", _FakeTrial())
