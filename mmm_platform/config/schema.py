"""
Pydantic schemas for model configuration.

These schemas define the structure and validation rules for all
configuration options in the MMM platform.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class AdstockType(str, Enum):
    """Adstock decay rate categories."""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class SignConstraint(str, Enum):
    """Sign constraint for control variables."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNCONSTRAINED = "unconstrained"


class PriorConfig(BaseModel):
    """Configuration for a single prior distribution."""
    distribution: str = Field(..., description="Distribution name (e.g., 'Normal', 'LogNormal', 'HalfNormal')")
    mu: Optional[float] = Field(None, description="Location parameter")
    sigma: Optional[float] = Field(None, description="Scale parameter")
    alpha: Optional[float] = Field(None, description="Shape parameter (for Beta)")
    beta: Optional[float] = Field(None, description="Shape parameter (for Beta)")

    class Config:
        extra = "allow"  # Allow additional prior parameters


class ChannelConfig(BaseModel):
    """Configuration for a single media channel."""
    name: str = Field(..., description="Column name in the data")
    display_name: Optional[str] = Field(None, description="Human-readable name for display")
    adstock_type: AdstockType = Field(AdstockType.MEDIUM, description="Adstock decay category")
    roi_prior_low: float = Field(0.1, ge=0, description="Lower bound for ROI prior")
    roi_prior_mid: float = Field(1.0, ge=0, description="Central estimate for ROI prior")
    roi_prior_high: float = Field(5.0, ge=0, description="Upper bound for ROI prior")

    @field_validator("roi_prior_high")
    @classmethod
    def high_greater_than_low(cls, v, info):
        if "roi_prior_low" in info.data and v <= info.data["roi_prior_low"]:
            raise ValueError("roi_prior_high must be greater than roi_prior_low")
        return v

    def get_display_name(self) -> str:
        """Return display name or formatted column name."""
        if self.display_name:
            return self.display_name
        # Convert column name to readable format
        name = self.name.replace("_spend", "").replace("PaidMedia_", "")
        return name.replace("_", " ")


class ControlConfig(BaseModel):
    """Configuration for a control variable."""
    name: str = Field(..., description="Column name in the data")
    display_name: Optional[str] = Field(None, description="Human-readable name")
    sign_constraint: SignConstraint = Field(
        SignConstraint.UNCONSTRAINED,
        description="Expected sign of coefficient"
    )
    is_dummy: bool = Field(False, description="Whether this is a 0/1 dummy variable")
    scale: bool = Field(False, description="Whether to scale this variable")

    def get_display_name(self) -> str:
        """Return display name or formatted column name."""
        return self.display_name or self.name.replace("_", " ").title()


class DummyVariableConfig(BaseModel):
    """Configuration for auto-generated dummy variables."""
    name: str = Field(..., description="Name for the dummy variable")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    sign_constraint: SignConstraint = Field(
        SignConstraint.UNCONSTRAINED,
        description="Expected sign of coefficient"
    )


class MonthDummyConfig(BaseModel):
    """Configuration for month dummy variables."""
    months: list[int] = Field(default_factory=list, description="Month numbers (1-12) to create dummies for")
    sign_constraints: dict[int, SignConstraint] = Field(
        default_factory=dict,
        description="Sign constraints by month number"
    )


class SamplingConfig(BaseModel):
    """Configuration for MCMC sampling."""
    draws: int = Field(1500, ge=100, le=10000, description="Number of posterior draws")
    tune: int = Field(1500, ge=100, le=10000, description="Number of tuning steps")
    chains: int = Field(4, ge=1, le=8, description="Number of chains")
    target_accept: float = Field(0.9, ge=0.5, le=0.99, description="Target acceptance rate")
    random_seed: int = Field(42, description="Random seed for reproducibility")
    cores: Optional[int] = Field(None, description="Number of cores (None = auto)")


class DataConfig(BaseModel):
    """Configuration for data handling."""
    date_column: str = Field("time", description="Name of the date column")
    target_column: str = Field(..., description="Name of the target (KPI) column")
    date_format: Optional[str] = Field(None, description="Date format string (e.g., '%Y-%m-%d')")
    dayfirst: bool = Field(True, description="Whether dates are day-first format")
    revenue_scale: float = Field(1000.0, gt=0, description="Scale factor for revenue")
    spend_scale: float = Field(1000.0, gt=0, description="Scale factor for spend")


class AdstockConfig(BaseModel):
    """Configuration for adstock transformation."""
    l_max: int = Field(8, ge=1, le=52, description="Maximum lag for adstock effect")
    short_decay: float = Field(0.15, ge=0, le=1, description="Decay rate for short adstock")
    medium_decay: float = Field(0.40, ge=0, le=1, description="Decay rate for medium adstock")
    long_decay: float = Field(0.70, ge=0, le=1, description="Decay rate for long adstock")
    prior_concentration: float = Field(20.0, gt=0, description="Beta distribution concentration")


class SaturationConfig(BaseModel):
    """Configuration for saturation transformation."""
    saturation_percentile: int = Field(50, ge=1, le=99, description="Percentile for half-saturation")
    lam_sigma: float = Field(0.3, gt=0, description="Sigma for lambda prior")


class SeasonalityConfig(BaseModel):
    """Configuration for seasonality modeling."""
    yearly_seasonality: int = Field(2, ge=0, le=10, description="Number of Fourier terms for yearly seasonality")


class ModelConfig(BaseModel):
    """Complete model configuration."""
    name: str = Field(..., description="Name for this model configuration")
    description: Optional[str] = Field(None, description="Description of this configuration")

    # Data configuration
    data: DataConfig

    # Channel configurations
    channels: list[ChannelConfig] = Field(..., min_length=1)

    # Control configurations
    controls: list[ControlConfig] = Field(default_factory=list)

    # Auto-generated dummies
    dummy_variables: list[DummyVariableConfig] = Field(default_factory=list)
    month_dummies: Optional[MonthDummyConfig] = Field(None)

    # Transform configurations
    adstock: AdstockConfig = Field(default_factory=AdstockConfig)
    saturation: SaturationConfig = Field(default_factory=SaturationConfig)
    seasonality: SeasonalityConfig = Field(default_factory=SeasonalityConfig)

    # Sampling configuration
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)

    # Control prior configuration
    control_prior: PriorConfig = Field(
        default_factory=lambda: PriorConfig(distribution="HalfNormal", sigma=1.0)
    )

    def get_channel_columns(self) -> list[str]:
        """Get list of channel column names."""
        return [ch.name for ch in self.channels]

    def get_control_columns(self) -> list[str]:
        """Get list of control column names."""
        return [ctrl.name for ctrl in self.controls]

    def get_channel_by_name(self, name: str) -> Optional[ChannelConfig]:
        """Get channel config by column name."""
        for ch in self.channels:
            if ch.name == name:
                return ch
        return None

    def get_roi_dicts(self) -> tuple[dict, dict, dict]:
        """Get ROI prior dictionaries for all channels."""
        roi_low = {}
        roi_mid = {}
        roi_high = {}
        for ch in self.channels:
            roi_low[ch.name] = ch.roi_prior_low
            roi_mid[ch.name] = ch.roi_prior_mid
            roi_high[ch.name] = ch.roi_prior_high
        return roi_low, roi_mid, roi_high

    def get_adstock_type_dict(self) -> dict[str, str]:
        """Get adstock type mapping for all channels."""
        return {ch.name: ch.adstock_type.value for ch in self.channels}

    class Config:
        use_enum_values = True
