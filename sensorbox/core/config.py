"""Configuration definitions and naming utilities."""

from datetime import datetime
from enum import Enum
from pathlib import Path


class SensorConfig(Enum):
    """Sensor configuration presets."""
    CONF01 = "CSI + LIDAR"
    CONF02 = "CSI + OAK-D"
    CONF03 = "OAK-D Only"
    CONF04 = "CSI Only"
    CONF05 = "All Sensors"


def generate_filename(config: SensorConfig, extension: str = ".h5") -> str:
    """
    Generate filename with format: CONFXX_YYYY_MM_DD_HHMMSS.h5
    
    Args:
        config: Sensor configuration used
        extension: File extension (default: .h5)
    
    Returns:
        Formatted filename string
    
    Example:
        >>> generate_filename(SensorConfig.CONF02)
        'CONF02_2025_12_07_143052.h5'
    """
    now = datetime.now()
    timestamp = now.strftime("%Y_%m_%d_%H%M%S")
    return f"{config.name}_{timestamp}{extension}"


def parse_filename(filename: str) -> dict:
    """
    Parse a recording filename to extract config and timestamp.
    
    Args:
        filename: Filename like CONF02_2025_12_07_143052.h5
    
    Returns:
        Dict with 'config', 'datetime', 'config_description'
    """
    stem = Path(filename).stem
    parts = stem.split("_")
    
    if len(parts) >= 5 and parts[0].startswith("CONF"):
        config_name = parts[0]
        year = int(parts[1])
        month = int(parts[2])
        day = int(parts[3])
        time_str = parts[4]
        
        hour = int(time_str[0:2])
        minute = int(time_str[2:4])
        second = int(time_str[4:6]) if len(time_str) >= 6 else 0
        
        dt = datetime(year, month, day, hour, minute, second)
        
        try:
            config = SensorConfig[config_name]
            description = config.value
        except KeyError:
            config = None
            description = "Unknown"
        
        return {
            "config": config_name,
            "config_description": description,
            "datetime": dt,
            "formatted_date": dt.strftime("%Y-%m-%d %H:%M:%S"),
        }
    
    return {
        "config": None,
        "config_description": "Unknown",
        "datetime": None,
        "formatted_date": None,
    }
