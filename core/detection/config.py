from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional


@dataclass
class DetectorConfig:
    sample_rate: float
    rise_window: float
    decay_window: float
    min_inter_peak: float
    k_prominence: float
    k_height: float
    peak_threshold: float
    use_prefilter: bool = True
    use_detrend: bool = True


@dataclass
class Reason(Enum):
    ACCEPT = "ACCEPT"
    REJECT_LOW_PROMINENCE = "REJECT_LOW_PROMINENCE"
    REJECT_DECAY_FAIL = "REJECT_DECAY_FAIL"
    REJECT_RISE_FAIL = "REJECT_RISE_FAIL"
    MERGED_WITH_NEIGHBOR = "MERGED_WITH_NEIGHBOR"
    EDGE_WINDOW_TRUNCATED = "EDGE_WINDOW_TRUNCATED"
    COMPOUND_OVERLAPPED = "COMPOUND_OVERLAPPED"


@dataclass
class PeakResult:
    index: int
    time_sec: float
    amplitude: float
    baseline: float
    rise_window: Tuple[int, int]
    decay_window: Tuple[int, int]
    flags: List[str]
    reasons: List[Reason]
    is_edge: bool
    is_compound: bool


@dataclass
class SummaryResult:
    peaks: List[PeakResult]
    noise_sigma: float
    params_used: DetectorConfig

