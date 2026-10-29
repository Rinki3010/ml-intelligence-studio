from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass(frozen=True)
class Port:
    name: str
    data_type: str
    required: bool = True

@dataclass(frozen=True)
class ComponentDefinition:
    type: str
    version: str
    name: str
    description: str
    inputs: tuple[Port, ...] = ()
    outputs: tuple[Port, ...] = ()
    config_schema: dict[str, Any] = field(default_factory=dict)
    status: str = 'implemented'
    execute: Callable[..., dict[str, Any]] | None = None
