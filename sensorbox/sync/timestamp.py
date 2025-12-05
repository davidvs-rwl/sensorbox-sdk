"""Timestamp management for multi-sensor synchronization."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict
import time
import threading


@dataclass
class SyncConfig:
    """Configuration for time synchronization."""
    use_monotonic: bool = True
    latency_compensation: Dict[str, float] = field(default_factory=lambda: {
        "camera": 0.010,
        "lidar": 0.005,
    })
    alignment_tolerance: float = 0.050


class TimestampManager:
    """Centralized timestamp management for synchronized multi-sensor capture."""
    
    def __init__(self, config: Optional[SyncConfig] = None):
        self._config = config or SyncConfig()
        self._lock = threading.Lock()
        self._start_time_mono: Optional[float] = None
        self._start_time_wall: Optional[datetime] = None
        self._running = False
        self._sensor_stats: Dict[str, dict] = {}
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def elapsed_time(self) -> float:
        if self._start_time_mono is None:
            return 0.0
        return time.monotonic() - self._start_time_mono
    
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._start_time_mono = time.monotonic()
            self._start_time_wall = datetime.now()
            self._running = True
            self._sensor_stats.clear()
    
    def stop(self) -> None:
        with self._lock:
            self._running = False
    
    def reset(self) -> None:
        with self._lock:
            self._start_time_mono = time.monotonic()
            self._start_time_wall = datetime.now()
            self._sensor_stats.clear()
    
    def get_timestamp(
        self,
        sensor_type: str = "default",
        compensate_latency: bool = False,
    ) -> tuple:
        if not self._running:
            raise RuntimeError("TimestampManager is not running. Call start() first.")
        
        with self._lock:
            mono = time.monotonic()
            wall = datetime.now()
            relative_ts = mono - self._start_time_mono
            
            if compensate_latency:
                latency = self._config.latency_compensation.get(sensor_type, 0.0)
                relative_ts -= latency
            
            if sensor_type not in self._sensor_stats:
                self._sensor_stats[sensor_type] = {
                    "count": 0,
                    "first_ts": relative_ts,
                    "last_ts": relative_ts,
                }
            
            stats = self._sensor_stats[sensor_type]
            stats["count"] += 1
            stats["last_ts"] = relative_ts
            
            return (relative_ts, wall)
    
    def get_stats(self) -> Dict[str, dict]:
        with self._lock:
            stats = {}
            for sensor_type, s in self._sensor_stats.items():
                duration = s["last_ts"] - s["first_ts"]
                avg_rate = s["count"] / duration if duration > 0 else 0
                stats[sensor_type] = {
                    "count": s["count"],
                    "duration": duration,
                    "avg_rate_hz": avg_rate,
                }
            return stats
    
    def __enter__(self) -> "TimestampManager":
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
