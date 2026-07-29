from typing import List
from pydantic import BaseModel, Field


class Wave(BaseModel):
    sample_id: str = Field(..., description="Sample identifier")
    time: List[float] = Field(..., description="Time points in seconds")
    signal: List[float] = Field(..., description="Acceleration in g")


class CarpetRegion(BaseModel):
    start_hz: float = Field(..., description="Start frequency in Hz")
    end_hz: float = Field(..., description="End frequency in Hz")


class AssetData(BaseModel):
    fit: List[Wave] = Field(..., description="Reference signals (healthy condition)")
    predict: List[Wave] = Field(..., description="Signals to classify")
