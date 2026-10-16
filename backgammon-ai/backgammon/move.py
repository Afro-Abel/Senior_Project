from dataclasses import dataclass, field


BAR = 25
OFF = 0


@dataclass(frozen=True)
class Step:
    start: int
    end: int


@dataclass
class Move:
    steps: list[Step] = field(default_factory=list)