from typing import TypeVar, Generic, Literal
from pydantic import BaseModel

# Define a TypeVar with constraints
T = TypeVar('T', int, bool, Literal["red", "yellow", "green"])

class TrafficDirection(Generic[T], BaseModel):
    north: T
    south: T
    east: T
    west: T

class TrafficModel(BaseModel):
    timestamp: str
    vehicles: TrafficDirection[int]
    emergency: TrafficDirection[int]
    light: TrafficDirection[Literal["red", "yellow", "green"]]
