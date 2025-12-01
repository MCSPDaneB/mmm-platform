"""
Pydantic schemas for model configuration.

These schemas define the structure and validation rules for all
configuration options in the MMM platform.
"""

from typing import Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


# =============================================================================
# Helper Functions
# =============================================================================

def sharpness_to_percentile(sharpness: int) -> int:
    """
    Convert 0-100 sharpness slider value to saturation_percentile.

    Maps:
    - 0 (Very Gradual) → 20th percentile
    - 50 (Balanced) → 50th percentile
    - 100 (Very Sharp) → 80th percentile

    Parameters
    ----------
    sharpness : int
        Sharpness value from 0 (gradual) to 100 (sharp)

    Returns
    -------
    int
        Saturation percentile value (20-80)
    """
    # Linear mapping: 0→20, 50→50, 100→80
    return 20 + int(sharpness * 0.6)


def sharpness_label_to_value(label: str) -> int:
    """
    Convert sharpness label to numeric value.

    Parameters
    ----------
    label : str
        One of 'gradual', 'balanced', 'sharp', or 'default'

    Returns
    -------
    int
        Corresponding sharpness value (0-100)
    """
    mapping = {
        "gradual": 25,
        "balanced": 50,
        "sharp": 75,
        "default": None,  # Use global setting
    }
    return mapping.get(label.lower() if label else "default")


# =============================================================================
# Enums
# =============================================================================

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


class CategoryColumnConfig(BaseModel):
    """Definition of a custom category column for grouping variables."""
    name: str = Field(..., description="Column name (e.g., 'Channel Type', 'Funnel Stage')")
    options: list[str] = Field(default_factory=list, description="Available options for this column")


class ChannelConfig(BaseModel):
    """Configuration for a single media channel."""
    name: str = Field(..., description="Column name in the data")
    display_name: Optional[str] = Field(None, description="Human-readable name for display")
    categories: dict[str, str] = Field(default_factory=dict, description="Category values keyed by column name")
    adstock_type: AdstockType = Field(AdstockType.MEDIUM, description="Adstock decay category")
    roi_prior_low: float = Field(0.1, ge=0, description="Lower bound for ROI prior")
    roi_prior_mid: float = Field(1.0, ge=0, description="Central estimate for ROI prior")
    roi_prior_high: float = Field(5.0, ge=0, description="Upper bound for ROI prior")
    curve_sharpness_override: Optional[str] = Field(None,
        description="Per-channel curve sharpness override: 'gradual', 'balanced', 'sharp', or None for global default")

    @model_validator(mode="before")
    @classmethod
    def migrate_old_category_field(cls, data: Any) -> Any:
        """Migrate old 'category' field name to 'categories' dict."""
        if isinstance(data, dict):
            # If old 'category' field exists, migrate it to 'categories'
            if "category" in data and "categories" not in data:
                old_category = data.pop("category")
                if old_category:
                    data["categories"] = {"Category": old_category}
                else:
                    data["categories"] = {}
        return data

    @field_validator("categories", mode="before")
    @classmethod
    def migrate_old_category(cls, v):
        """Migrate old single category field to categories dict."""
        if v is None:
            return {}
        if isinstance(v, str):
            # Old format: single category string
            return {"Category": v}
        return v

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

    def get_category(self, column_name: str = "Category") -> str:
        """Return category value for a specific column, or infer from column name."""
        if column_name in self.categories and self.categories[column_name]:
            return self.categories[column_name]
        # Auto-detect from column name (extract meaningful part from PaidMedia_X_spend pattern)
        name = self.name
        if name.startswith("PaidMedia_"):
            parts = name.replace("PaidMedia_", "").replace("_spend", "").split("_")
            if parts:
                return parts[0].title()
        elif "_" in name:
            return name.split("_")[0].title()
        return "Paid Media"


class OwnedMediaConfig(BaseModel):
    """Configuration for owned media channels (email, organic social, etc.).

    Owned media always has adstock and saturation applied (like paid media channels).
    ROI tracking is optional - only set ROI priors if you have cost/spend data.
    """
    name: str = Field(..., description="Column name in the data")
    display_name: Optional[str] = Field(None, description="Human-readable name for display")
    categories: dict[str, str] = Field(default_factory=dict, description="Category values keyed by column name")

    # Adstock settings (always applied)
    adstock_type: AdstockType = Field(AdstockType.MEDIUM, description="Adstock decay category")

    # Saturation settings (always applied)
    curve_sharpness_override: Optional[str] = Field(None,
        description="Per-variable curve sharpness override: 'gradual', 'balanced', 'sharp', or None for global default")

    # ROI priors - OPTIONAL. Only set if you have cost/spend data for this variable.
    # If roi_prior_mid is None, variable won't appear in ROI calculations.
    roi_prior_low: Optional[float] = Field(None, ge=0, description="Lower bound for ROI prior (optional)")
    roi_prior_mid: Optional[float] = Field(None, ge=0, description="Central estimate for ROI prior (optional)")
    roi_prior_high: Optional[float] = Field(None, ge=0, description="Upper bound for ROI prior (optional)")

    # Backward compatibility - these fields are ignored but accepted for old configs
    apply_adstock: Optional[bool] = Field(None, description="DEPRECATED: adstock is always applied")
    apply_saturation: Optional[bool] = Field(None, description="DEPRECATED: saturation is always applied")
    include_roi: Optional[bool] = Field(None, description="DEPRECATED: determined by roi_prior_mid being set")

    @property
    def has_roi_priors(self) -> bool:
        """Check if ROI priors are configured (for inclusion in ROI calculations)."""
        return self.roi_prior_mid is not None

    @model_validator(mode="before")
    @classmethod
    def migrate_old_category_field(cls, data: Any) -> Any:
        """Migrate old 'category' field name to 'categories' dict."""
        if isinstance(data, dict):
            if "category" in data and "categories" not in data:
                old_category = data.pop("category")
                if old_category:
                    data["categories"] = {"Category": old_category}
                else:
                    data["categories"] = {}
        return data

    @field_validator("categories", mode="before")
    @classmethod
    def migrate_old_category(cls, v):
        """Migrate old single category field to categories dict."""
        if v is None:
            return {}
        if isinstance(v, str):
            return {"Category": v}
        return v

    def get_display_name(self) -> str:
        """Return display name or formatted column name."""
        if self.display_name:
            return self.display_name
        return self.name.replace("_", " ").title()

    def get_category(self, column_name: str = "Category") -> str:
        """Return category value for a specific column."""
        if column_name in self.categories and self.categories[column_name]:
            return self.categories[column_name]
        return "Owned Media"


class CompetitorConfig(BaseModel):
    """Configuration for competitor activity variables.

    Competitor variables have adstock only (no saturation).
    Coefficient is always constrained negative (competitor activity hurts your sales).
    No ROI calculation (can't optimize competitor spend).
    """
    name: str = Field(..., description="Column name in the data")
    display_name: Optional[str] = Field(None, description="Human-readable name for display")
    categories: dict[str, str] = Field(default_factory=dict, description="Category values keyed by column name")

    # Adstock with shorter default decay
    adstock_type: AdstockType = Field(AdstockType.SHORT, description="Adstock decay category (default: short)")

    @model_validator(mode="before")
    @classmethod
    def migrate_old_category_field(cls, data: Any) -> Any:
        """Migrate old 'category' field name to 'categories' dict."""
        if isinstance(data, dict):
            if "category" in data and "categories" not in data:
                old_category = data.pop("category")
                if old_category:
                    data["categories"] = {"Category": old_category}
                else:
                    data["categories"] = {}
        return data

    @field_validator("categories", mode="before")
    @classmethod
    def migrate_old_category(cls, v):
        """Migrate old single category field to categories dict."""
        if v is None:
            return {}
        if isinstance(v, str):
            return {"Category": v}
        return v

    def get_display_name(self) -> str:
        """Return display name or formatted column name."""
        if self.display_name:
            return self.display_name
        return self.name.replace("_", " ").title()

    def get_category(self, column_name: str = "Category") -> str:
        """Return category value for a specific column."""
        if column_name in self.categories and self.categories[column_name]:
            return self.categories[column_name]
        return "Competitor"


class ControlConfig(BaseModel):
    """Configuration for a control variable.

    Controls have optional lag/carryover effect via adstock toggle.
    """
    name: str = Field(..., description="Column name in the data")
    display_name: Optional[str] = Field(None, description="Human-readable name")
    categories: dict[str, str] = Field(default_factory=dict, description="Category values keyed by column name")
    sign_constraint: SignConstraint = Field(
        SignConstraint.UNCONSTRAINED,
        description="Expected sign of coefficient"
    )
    is_dummy: bool = Field(False, description="Whether this is a 0/1 dummy variable")
    scale: bool = Field(False, description="Whether to scale this variable")

    # Optional lag/carryover effect
    apply_adstock: bool = Field(False, description="Whether to apply adstock transformation for lag/carryover")
    adstock_type: AdstockType = Field(AdstockType.SHORT, description="Adstock decay category (only used if apply_adstock=True)")

    @model_validator(mode="before")
    @classmethod
    def migrate_old_category_field(cls, data: Any) -> Any:
        """Migrate old 'category' field name to 'categories' dict."""
        if isinstance(data, dict):
            # If old 'category' field exists, migrate it to 'categories'
            if "category" in data and "categories" not in data:
                old_category = data.pop("category")
                if old_category:
                    data["categories"] = {"Category": old_category}
                else:
                    data["categories"] = {}
        return data

    @field_validator("categories", mode="before")
    @classmethod
    def migrate_old_category(cls, v):
        """Migrate old single category field to categories dict."""
        if v is None:
            return {}
        if isinstance(v, str):
            # Old format: single category string
            return {"Category": v}
        return v

    def get_display_name(self) -> str:
        """Return display name or formatted column name."""
        return self.display_name or self.name.replace("_", " ").title()

    def get_category(self, column_name: str = "Category") -> str:
        """Return category value for a specific column, or infer from column name."""
        if column_name in self.categories and self.categories[column_name]:
            return self.categories[column_name]
        # Auto-detect from column name patterns
        name_lower = self.name.lower()
        if "promo" in name_lower:
            return "Promotions"
        if "season" in name_lower or "fourier" in name_lower:
            return "Seasonality"
        if "month_" in name_lower:
            return "Month Effects"
        if "dummy_" in name_lower or "shock" in name_lower:
            return "Events/Dummies"
        if "trend" in name_lower or self.name == "t":
            return "Trend"
        if "_" in self.name:
            return self.name.split("_")[0].title()
        return "Other"


class DummyVariableConfig(BaseModel):
    """Configuration for auto-generated dummy variables."""
    name: str = Field(..., description="Name for the dummy variable")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    categories: dict[str, str] = Field(default_factory=dict, description="Category values keyed by column name")
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
    brand: Optional[str] = Field(None, description="Brand name for exports")
    model_start_date: Optional[str] = Field(None, description="Start date for modeling (YYYY-MM-DD)")
    model_end_date: Optional[str] = Field(None, description="End date for modeling (YYYY-MM-DD)")
    include_trend: bool = Field(True, description="Include linear time trend as control variable")


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
    curve_sharpness: int = Field(50, ge=0, le=100,
        description="Curve sharpness: 0=very gradual, 50=balanced, 100=very sharp")


class SeasonalityConfig(BaseModel):
    """Configuration for seasonality modeling."""
    yearly_seasonality: int = Field(2, ge=0, le=10, description="Number of Fourier terms for yearly seasonality")


class ModelConfig(BaseModel):
    """Complete model configuration."""
    name: str = Field(..., description="Name for this model configuration")
    description: Optional[str] = Field(None, description="Description of this configuration")
    client: Optional[str] = Field(None, description="Client name for organizing saved configs/models")

    # Data configuration
    data: DataConfig

    # Channel configurations (paid media - required)
    channels: list[ChannelConfig] = Field(..., min_length=1)

    # Owned media configurations (optional)
    owned_media: list[OwnedMediaConfig] = Field(default_factory=list)

    # Competitor configurations (optional)
    competitors: list[CompetitorConfig] = Field(default_factory=list)

    # Control configurations (optional)
    controls: list[ControlConfig] = Field(default_factory=list)

    # Custom category columns for grouping (max 5)
    category_columns: list[CategoryColumnConfig] = Field(
        default_factory=list,
        description="Custom category columns for grouping variables in results"
    )

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

    def get_control_by_name(self, name: str) -> Optional[ControlConfig]:
        """Get control config by column name."""
        for ctrl in self.controls:
            if ctrl.name == name:
                return ctrl
        return None

    def get_owned_media_columns(self) -> list[str]:
        """Get list of owned media column names."""
        return [om.name for om in self.owned_media]

    def get_competitor_columns(self) -> list[str]:
        """Get list of competitor column names."""
        return [comp.name for comp in self.competitors]

    def get_owned_media_by_name(self, name: str) -> Optional[OwnedMediaConfig]:
        """Get owned media config by column name."""
        for om in self.owned_media:
            if om.name == name:
                return om
        return None

    def get_competitor_by_name(self, name: str) -> Optional[CompetitorConfig]:
        """Get competitor config by column name."""
        for comp in self.competitors:
            if comp.name == name:
                return comp
        return None

    def get_roi_dicts(self) -> tuple[dict, dict, dict]:
        """Get ROI prior dictionaries for all channels and owned media with ROI."""
        roi_low = {}
        roi_mid = {}
        roi_high = {}
        # Channels (paid media) always have ROI
        for ch in self.channels:
            roi_low[ch.name] = ch.roi_prior_low
            roi_mid[ch.name] = ch.roi_prior_mid
            roi_high[ch.name] = ch.roi_prior_high
        # Owned media only if include_roi is True
        for om in self.owned_media:
            if om.include_roi:
                roi_low[om.name] = om.roi_prior_low
                roi_mid[om.name] = om.roi_prior_mid
                roi_high[om.name] = om.roi_prior_high
        return roi_low, roi_mid, roi_high

    def get_adstock_type_dict(self) -> dict[str, str]:
        """Get adstock type mapping for channels, owned media (if enabled), competitors, and controls (if enabled)."""
        result = {}
        # Channels always have adstock
        for ch in self.channels:
            result[ch.name] = ch.adstock_type.value
        # Owned media if apply_adstock is True
        for om in self.owned_media:
            if om.apply_adstock:
                result[om.name] = om.adstock_type.value
        # Competitors always have adstock
        for comp in self.competitors:
            result[comp.name] = comp.adstock_type.value
        # Controls if apply_adstock is True
        for ctrl in self.controls:
            if ctrl.apply_adstock:
                result[ctrl.name] = ctrl.adstock_type.value
        return result

    def get_owned_media_category_map(self, column_name: str = "Category") -> dict[str, str]:
        """Get mapping of owned media names to category values for a specific column."""
        return {om.name: om.get_category(column_name) for om in self.owned_media}

    def get_competitor_category_map(self, column_name: str = "Category") -> dict[str, str]:
        """Get mapping of competitor names to category values for a specific column."""
        return {comp.name: comp.get_category(column_name) for comp in self.competitors}

    def get_category_column_names(self) -> list[str]:
        """Get list of category column names."""
        return [col.name for col in self.category_columns]

    def get_channel_category_map(self, column_name: str = "Category") -> dict[str, str]:
        """Get mapping of channel names to category values for a specific column."""
        return {ch.name: ch.get_category(column_name) for ch in self.channels}

    def get_control_category_map(self, column_name: str = "Category") -> dict[str, str]:
        """Get mapping of control names to category values for a specific column."""
        return {ctrl.name: ctrl.get_category(column_name) for ctrl in self.controls}

    def get_all_channel_category_maps(self) -> dict[str, dict[str, str]]:
        """Get all category mappings for all columns for channels."""
        return {col.name: self.get_channel_category_map(col.name) for col in self.category_columns}

    def get_all_control_category_maps(self) -> dict[str, dict[str, str]]:
        """Get all category mappings for all columns for controls."""
        return {col.name: self.get_control_category_map(col.name) for col in self.category_columns}

    # Legacy methods for backward compatibility
    def get_channel_categories(self, column_name: str = "Category") -> list[str]:
        """Get unique channel categories for a specific column."""
        return list(set(ch.get_category(column_name) for ch in self.channels))

    def get_control_categories(self, column_name: str = "Category") -> list[str]:
        """Get unique control categories for a specific column."""
        return list(set(ctrl.get_category(column_name) for ctrl in self.controls))

    def get_channels_by_category(self, category: str, column_name: str = "Category") -> list[ChannelConfig]:
        """Get channels in a specific category for a specific column."""
        return [ch for ch in self.channels if ch.get_category(column_name) == category]

    def get_controls_by_category(self, category: str, column_name: str = "Category") -> list[ControlConfig]:
        """Get controls in a specific category for a specific column."""
        return [ctrl for ctrl in self.controls if ctrl.get_category(column_name) == category]

    class Config:
        use_enum_values = True
