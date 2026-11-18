"""Named rule variants; standard rules remain the default."""
RULE_VARIANTS = ("standard", "single-hit", "relaxed-bar", "no-doubles-bonus", "eight-point-bearoff")


def validate_rule_variant(name):
    if name not in RULE_VARIANTS:
        raise ValueError(f"rule_variant must be one of {RULE_VARIANTS}")
    return name
