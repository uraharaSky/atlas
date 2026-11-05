"""Structured schemas for research paper methodology extracted by an LLM"""

from __future__ import annotations
from pydantic import BaseModel, ConfigDict, Field

class PaperMetadata(BaseModel):
    """Basic metadata about the research paper"""

    model_config = ConfigDict(extra = "forbid")

    title: str = Field(min_length = 1)
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1900, le=2100)

class MarketSpec(BaseModel):
    """Market and asset information used by the paper."""

    model_config = ConfigDict(extra="forbid")

    asset: str = Field(min_length=1)
    market: str | None = None
    asset_class: str | None = None


class ModelSpec(BaseModel):
    """Machine-learning or statistical model used by the paper."""

    model_config = ConfigDict(extra="forbid")

    family: str = Field(min_length=1)
    architecture: str | None = None
    parameters: dict[str, object] = Field(default_factory=dict)


class StrategySpec(BaseModel):
    """Trading strategy derived from the paper."""

    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1)
    long_condition: str | None = None
    short_condition: str | None = None
    exit_condition: str | None = None
    position_sizing: str | None = None


class EvaluationSpec(BaseModel):
    """Evaluation criteria reported or used by the paper."""

    model_config = ConfigDict(extra="forbid")

    forecasting_metrics: list[str] = Field(default_factory=list)
    trading_metrics: list[str] = Field(default_factory=list)


class ForecastSpec(BaseModel):
    """Specification of the forecasting methodology described in a paper."""

    target: str = Field(
        min_length=1,
        description="The variable or quantity being forecast."
    )

    horizon: int = Field(
        ge=1,
        description="Number of periods ahead being forecast."
    )

    frequency: str = Field(
        description="Frequency of the forecast, e.g. daily, weekly, monthly."
    )

    model: str = Field(
        min_length=1,
        description="Forecasting model used by the paper."
    )

    features: list[str] = Field(
        default_factory=list,
        description="Input variables/features used for forecasting."
    )

    preprocessing: list[str] = Field(
        default_factory=list,
        description="Preprocessing steps applied before forecasting."
    )

    evaluation_metrics: list[str] = Field(
        default_factory=list,
        description="Metrics used to evaluate forecast performance."
    )

class ResearchSpec(BaseModel):
    """
    Structured representation of a financial research paper's methodology.

    This is the contract between the LLM layer and the deterministic ATLAS
    research/backtesting engine.
    """

    model_config = ConfigDict(extra="forbid")

    paper: PaperMetadata
    market: MarketSpec
    features: list[str] = Field(default_factory=list)
    preprocessing: list[str] = Field(default_factory=list)
    forecasting: ForecastSpec | None = None
    model: ModelSpec
    strategy: StrategySpec | None = None
    evaluation: EvaluationSpec = Field(default_factory=EvaluationSpec)
