from dataclasses import dataclass


@dataclass
class Frame:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label:2s} ({self.score:4.2f}): [{self.start:4d}, {self.end:4d})"

    def __len__(self):
        return self.end - self.start
