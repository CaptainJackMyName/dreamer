"""Model adapter registry for extensible multi-model support."""

import os
from typing import Dict, Type, List
from ..models.adapters import BaseModelAdapter

_MODEL_REGISTRY: Dict[str, Type[BaseModelAdapter]] = {}


def register_model(model_id: str):
    """Decorator to register a model adapter.

    Usage:
        @register_model("wan-2.2-t2v-a14b")
        class WanAdapter(BaseModelAdapter):
            ...
    """

    def decorator(cls: Type[BaseModelAdapter]):
        if not issubclass(cls, BaseModelAdapter):
            raise TypeError(f"Adapter {cls.__name__} must inherit from BaseModelAdapter")
        _MODEL_REGISTRY[model_id] = cls
        return cls

    return decorator


def get_adapter(model_id: str) -> Type[BaseModelAdapter]:
    """Retrieve adapter class by model identifier."""
    # if model_id is a local dir path, load model weights from it
    if os.path.isdir(model_id):
        # get dir name from dir path
        dir_name = os.path.basename(model_id)
        if dir_name not in _MODEL_REGISTRY:
            available = ", ".join(list_models())
            raise ValueError(
                f"Model '{dir_name}' is not registered."
                f"Available models: {available}"
            )
        model_id = dir_name
    elif model_id not in _MODEL_REGISTRY:
        available = ", ".join(list_models())
        raise ValueError(
            f"Model '{model_id}' is not registered. "
            f"Available models: {available}"
        )
    return _MODEL_REGISTRY[model_id]


def list_models() -> List[str]:
    """List all registered model identifiers."""
    return list(_MODEL_REGISTRY.keys())


def unregister_model(model_id: str):
    """Remove a model from registry (mainly for testing)."""
    _MODEL_REGISTRY.pop(model_id, None)
