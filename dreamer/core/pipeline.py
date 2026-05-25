"""Pipeline execution framework inspired by DiffSynth-Studio."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import copy


class PipelineContext:
    """Shared context dictionary passed through pipeline units."""

    def __init__(self, config: Any, **kwargs):
        self.config = config
        self.data: Dict[str, Any] = {}
        self.data.update(kwargs)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any):
        self.data[key] = value

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def __setitem__(self, key: str, value: Any):
        self.data[key] = value


class PipelineUnit(ABC):
    """Base class for a single pipeline processing unit."""

    name: str = ""

    @abstractmethod
    def run(self, context: PipelineContext) -> PipelineContext:
        """Execute this unit and return the updated context."""
        pass


class BasePipeline:
    """A chain of PipelineUnits executed in order."""

    def __init__(self, units: Optional[List[PipelineUnit]] = None):
        self.units: List[PipelineUnit] = units or []

    def add_unit(self, unit: PipelineUnit):
        self.units.append(unit)

    def run(self, context: PipelineContext) -> PipelineContext:
        for unit in self.units:
            context = unit.run(context)
        return context

    def __call__(self, context: PipelineContext) -> PipelineContext:
        return self.run(context)
