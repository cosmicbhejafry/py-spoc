from typing import Any
from pydantic import BaseModel, Field

from ._config_schema import MappingsConfig
   

class SchemesConfig(BaseModel):
    schemes: dict[str, dict[str, Any]]


class LongConfig(BaseModel):
    Statistics: dict[str, dict[str, SchemesConfig]] | None = None
    ReducedStatistics: dict[str, dict[str, SchemesConfig]] | None = None
    Reducers: dict[str, dict[str, SchemesConfig]] | None = None
    Mappings: MappingsConfig = Field(default_factory=MappingsConfig)


class ShortConfig(BaseModel):
    Statistics: dict[str, SchemesConfig] | None = None
    ReducedStatistics: dict[str, SchemesConfig] | None = None
    Reducers: dict[str, SchemesConfig] | None = None
    Mappings: MappingsConfig = Field(default_factory=MappingsConfig)
