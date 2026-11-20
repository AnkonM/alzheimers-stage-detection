"""Label-map plugin registry for external-validation analyses.

A label map turns a dataset's raw clinical label (for OASIS-1: the CDR) into a named class and an
ordinal rank. It is used ONLY by the analysis step, after a run is complete. Labels never enter the
engine, and the closed `schema.Stage` enum is untouched.

    @register("my_map", classes=("a", "b"), reference="a", positive="b")
    def my_map(raw): ...

`get(name)` returns a `LabelMap`; unknown names and unknown raw values raise (never a silent default).
"""
import math
from dataclasses import dataclass
from typing import Callable, Dict, Tuple


@dataclass(frozen=True)
class LabelMap:
    name: str
    classes: Tuple[str, ...]        # ordered from least to most severe; the ordinal rank is the index
    reference: str                  # reference class of the primary binary contrast
    positive: str                   # positive class of the primary binary contrast
    fn: Callable[[object], str]

    def __call__(self, raw) -> str:
        label = self.fn(raw)
        if label not in self.classes:
            raise ValueError(f"label map {self.name!r} produced an undeclared class {label!r}")
        return label

    def rank(self, raw) -> int:
        return self.classes.index(self(raw))


_REGISTRY: Dict[str, LabelMap] = {}


def register(name, *, classes, reference, positive):
    if reference not in classes or positive not in classes:
        raise ValueError("reference and positive must be among the declared classes")

    def deco(fn):
        if name in _REGISTRY:
            raise ValueError(f"label map {name!r} is already registered")
        _REGISTRY[name] = LabelMap(name, tuple(classes), reference, positive, fn)
        return fn
    return deco


def unregister(name):
    _REGISTRY.pop(name, None)


def get(name) -> LabelMap:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown label map {name!r}; registered: {sorted(_REGISTRY)}") from None


def names():
    return sorted(_REGISTRY)


CDR_VALUES = (0.0, 0.5, 1.0, 2.0, 3.0)


def parse_cdr(raw) -> float:
    """CDR as a number; accepts 0, '0.5', '2.0', numpy scalars. Blank, NaN and any value outside the
    CDR scale raise ValueError."""
    if raw is None:
        raise ValueError("CDR is missing")
    try:
        v = float(str(raw).strip())
    except ValueError:
        raise ValueError("CDR is not a number") from None
    if math.isnan(v) or v not in CDR_VALUES:
        raise ValueError(f"CDR value outside the scale {CDR_VALUES}")
    return v


@register("cdr_3class", classes=("CN-like", "CDR0.5", "dementia"), reference="CN-like", positive="dementia")
def cdr_3class(raw) -> str:
    """CDR 0 -> 'CN-like'; CDR 0.5 -> 'CDR0.5'; CDR >= 1 -> 'dementia'. CDR 0.5 is OASIS's
    'very mild dementia' (a clinical diagnosis of probable AD; other cohorts may call it MCI)."""
    v = parse_cdr(raw)
    return "CN-like" if v == 0 else "CDR0.5" if v == 0.5 else "dementia"
