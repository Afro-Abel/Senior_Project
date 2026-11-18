"""Frozen style strengths selected before rule-variation experiments."""
from types import MappingProxyType

SELECTED_STYLE_STRENGTHS = MappingProxyType({
    "rear-priority": 0.05,
    "front-priority": 0.01,
    "aggressive": 0.05,
    "defensive": 0.5,
})
