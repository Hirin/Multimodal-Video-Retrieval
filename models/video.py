"""
Data models for video-related entities.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path


@dataclass
class VideoMetadata:
    """Metadata for a video."""
    video_id: str
    dataset: str
    duration: Optional[float] = None
    num_frames: Optional[int] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Keyframe:
    """Represents a single keyframe from a video."""
    video_id: str
    frame_id: int
    timestamp: float
    frame_path: str
    features: Optional[Any] = None  # np.ndarray

    @property
    def time_mmss(self) -> str:
        """Get timestamp in MM:SS format."""
        from utils.helpers import sec_to_mmss
        return sec_to_mmss(self.timestamp)


@dataclass
class SearchResult:
    """Represents a single search result."""
    video_id: str
    frame_id: int
    timestamp: float
    score: float
    frame_path: str
    rank: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_mmss(self) -> str:
        """Get timestamp in MM:SS format."""
        from utils.helpers import sec_to_mmss
        return sec_to_mmss(self.timestamp)


@dataclass
class SearchQuery:
    """Represents a search query."""
    query_text: str
    query_type: str  # 'clip', 'bm25', 'seed', 'video_id', 'clip_rerank'
    datasets: List[str] = field(default_factory=lambda: ['b1', 'b2'])
    top_k: int = 20
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoSegment:
    """Represents a video segment with start and end times."""
    video_id: str
    start_time: float
    end_time: float
    frames: List[Keyframe] = field(default_factory=list)
    transcript: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get segment duration."""
        return self.end_time - self.start_time


@dataclass
class TranscriptEntry:
    """Represents a transcript entry with timing information."""
    video_id: str
    frame_id: int
    timestamp: float
    text: str
    language: str = 'vi'
    confidence: Optional[float] = None


@dataclass
class BM25Result:
    """Represents a BM25 search result."""
    video_id: str
    frame_id: int
    timestamp: float
    score: float
    text: str
    highlighted_text: Optional[str] = None
