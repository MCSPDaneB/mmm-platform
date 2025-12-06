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

class KPIType(str, Enum):
    """Type of KPI being modeled - determines display terminology."""
    REVENUE = "revenue"  # Currency-based KPIs (revenue, sales, value) -> "ROI" terminology
    COUNT = "count"      # Count-based KPIs (installs, leads, volume) -> "Cost Per X" terminology


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


class DisaggregationMappingConfig(BaseModel):
    """Configuration for disaggregating model results to granular level.

    Stores the mapping configuration (not the file data) so users can
    quickly re-apply disaggregation when re-uploading a granular file.
    """
    id: str = Field(..., description="Unique identifier for this config")
    name: str = Field(..., description="User-friendly name (e.g., 'Placements by Spend')")
    created_at: str = Field(..., description="ISO timestamp when config was created")
    is_active: bool = Field(False, description="Whether this is the active disaggregation config")
    granular_name_cols: list[str] = Field(..., description="Column(s) forming the entity identifier")
    date_column: str = Field(..., description="Date column in granular file")
    weight_column: str = Field(..., description="Weight column for proportional allocation")
    include_columns: list[str] = Field(default_factory=list, description="Additional columns to include from granular file (e.g., impressions, clicks)")
    entity_to_channel_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Maps entity identifiers to model channel names"
    )
    notes: Optional[str] = Field(None, description="Optional notes about this config")


# =============================================================================
# Export Column Schema Models
# =============================================================================

class ColumnSchemaEntry(BaseModel):
    """Configuration for a single column in an export schema."""
    original_name: str = Field(..., description="Original column name from the data")
    display_name: Optional[str] = Field(None, description="Renamed column name for export (None = use original)")
    visible: bool = Field(True, description="Whether to include this column in export")
    order: int = Field(0, description="Position in output (lower = earlier)")


class DatasetColumnSchema(BaseModel):
    """Column schema for a single dataset (decomps, media, or actual_vs_fitted)."""
    columns: list[ColumnSchemaEntry] = Field(default_factory=list)

    def get_visible_columns_ordered(self) -> list[ColumnSchemaEntry]:
        """Return visible columns sorted by order."""
        return sorted(
            [c for c in self.columns if c.visible],
            key=lambda x: x.order
        )

    def get_column_mapping(self) -> dict[str, str]:
        """Return mapping of original_name -> display_name for renaming."""
        return {
            c.original_name: (c.display_name or c.original_name)
            for c in self.columns if c.visible
        }


class ExportColumnSchema(BaseModel):
    """Complete export column schema covering all three datasets.

    Schemas can be saved at client level (shared across models) or
    at model level (as an override for a specific model).

    Disaggregated schemas (decomps_stacked_disagg, media_results_disagg) are optional.
    If not set, they inherit from the base schema and auto-extend with extra columns.
    If explicitly set, they override the base schema entirely.
    """
    id: str = Field(..., description="Unique identifier for this schema")
    name: str = Field(..., description="User-friendly name (e.g., 'Standard BI Export')")
    description: Optional[str] = Field(None, description="Optional description")
    created_at: str = Field(..., description="ISO timestamp when schema was created")
    updated_at: Optional[str] = Field(None, description="ISO timestamp when schema was last updated")

    # Per-dataset schemas (base files)
    decomps_stacked: DatasetColumnSchema = Field(default_factory=DatasetColumnSchema)
    media_results: DatasetColumnSchema = Field(default_factory=DatasetColumnSchema)
    actual_vs_fitted: DatasetColumnSchema = Field(default_factory=DatasetColumnSchema)

    # Disaggregated schemas (optional - inherit from base if None)
    decomps_stacked_disagg: Optional[DatasetColumnSchema] = Field(
        None,
        description="Schema for disaggregated decomps. If None, inherits from decomps_stacked."
    )
    media_results_disagg: Optional[DatasetColumnSchema] = Field(
        None,
        description="Schema for disaggregated media results. If None, inherits from media_results."
    )

    # Model-level override tracking
    is_model_override: bool = Field(False, description="True if this is a model-level override of a client schema")
    parent_schema_id: Optional[str] = Field(None, description="ID of parent client schema if this is an override")


class SchemaValidationResult(BaseModel):
    """Result of validating a schema against actual data columns."""
    dataset_name: str = Field(..., description="Name of the dataset being validated")
    is_valid: bool = Field(True, description="Whether the schema is fully compatible")
    matched_columns: list[str] = Field(default_factory=list, description="Columns that match between schema and data")
    new_columns: list[str] = Field(default_factory=list, description="Columns in data but not in schema")
    removed_columns: list[str] = Field(default_factory=list, description="Columns in schema but not in data")
    drift_severity: Literal["none", "minor", "major"] = Field("none", description="Severity of schema drift")


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

    # ROI configuration - checkbox controls whether ROI priors are used
    # Only enable if you have cost/spend data for this variable
    include_roi: bool = Field(False, description="Whether to include ROI priors for this variable")
    roi_prior_low: Optional[float] = Field(None, ge=0, description="Lower bound for ROI prior (required if include_roi=True)")
    roi_prior_mid: Optional[float] = Field(None, ge=0, description="Central estimate for ROI prior (required if include_roi=True)")
    roi_prior_high: Optional[float] = Field(None, ge=0, description="Upper bound for ROI prior (required if include_roi=True)")

    # Backward compatibility - these fields are ignored but accepted for old configs
    apply_adstock: Optional[bool] = Field(None, description="DEPRECATED: adstock is always applied")
    apply_saturation: Optional[bool] = Field(None, description="DEPRECATED: saturation is always applied")

    @property
    def has_roi_priors(self) -> bool:
        """Check if ROI priors are configured (for inclusion in ROI calculations)."""
        return self.include_roi and self.roi_prior_mid is not None

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

    @model_validator(mode="after")
    def validate_roi_priors(self) -> "OwnedMediaConfig":
        """Validate that ROI priors are set when include_roi is True."""
        if self.include_roi:
            if self.roi_prior_mid is None:
                raise ValueError("ROI prior mid value is required when Include ROI Priors is enabled")
            if self.roi_prior_low is None:
                raise ValueError("ROI prior low value is required when Include ROI Priors is enabled")
            if self.roi_prior_high is None:
                raise ValueError("ROI prior high value is required when Include ROI Priors is enabled")
            if self.roi_prior_high <= self.roi_prior_low:
                raise ValueError("ROI prior high must be greater than ROI prior low")
        return self

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
    target_scale: float = Field(1.0, gt=0, description="Scale factor for target KPI (deprecated - PyMC-Marketing handles scaling)")
    spend_scale: float = Field(1.0, gt=0, description="Scale factor for spend (deprecated - PyMC-Marketing handles scaling)")
    brand: Optional[str] = Field(None, description="Brand name for exports")
    model_start_date: Optional[str] = Field(None, description="Start date for modeling (YYYY-MM-DD)")
    model_end_date: Optional[str] = Field(None, description="End date for modeling (YYYY-MM-DD)")
    include_trend: bool = Field(True, description="Include linear time trend as control variable")

    # KPI type configuration for dynamic labeling
    kpi_type: KPIType = Field(KPIType.REVENUE, description="Type of KPI (revenue/count) for display terminology")
    kpi_display_name: Optional[str] = Field(None, description="Custom display name for KPI (e.g., 'Install' for count KPIs)")

    @model_validator(mode="before")
    @classmethod
    def migrate_revenue_scale(cls, data: Any) -> Any:
        """Migrate old 'revenue_scale' field to 'target_scale'."""
        if isinstance(data, dict):
            if "revenue_scale" in data and "target_scale" not in data:
                data["target_scale"] = data.pop("revenue_scale")
        return data

    @property
    def revenue_scale(self) -> float:
        """Backward compatibility alias for target_scale."""
        return self.target_scale


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
    beta_sigma_multiplier: float = Field(
        1.0, ge=0.1, le=3.0,
        description="Multiplier for beta prior sigma. <1 = tighter ROI priors, >1 = looser"
    )


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

    # Base component categories (intercept, trend, seasonality)
    base_component_categories: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description="Categories for base components keyed by variable name (e.g., 'intercept', 'trend', 'sin_order_1')"
    )

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

    def get_efficiency_dicts(self) -> tuple[dict, dict, dict]:
        """Get efficiency prior dictionaries for all channels and owned media with efficiency priors.

        Returns efficiency priors (ROI for revenue KPIs, or inverted cost-per for count KPIs).
        """
        eff_low = {}
        eff_mid = {}
        eff_high = {}
        # Channels (paid media) always have efficiency priors
        for ch in self.channels:
            eff_low[ch.name] = ch.roi_prior_low
            eff_mid[ch.name] = ch.roi_prior_mid
            eff_high[ch.name] = ch.roi_prior_high
        # Owned media only if include_roi is True
        for om in self.owned_media:
            if om.include_roi:
                eff_low[om.name] = om.roi_prior_low
                eff_mid[om.name] = om.roi_prior_mid
                eff_high[om.name] = om.roi_prior_high
        return eff_low, eff_mid, eff_high

    def get_roi_dicts(self) -> tuple[dict, dict, dict]:
        """Backward compatibility alias for get_efficiency_dicts()."""
        return self.get_efficiency_dicts()

    def get_efficiency_label(self) -> str:
        """Get display label for efficiency metric based on KPI type.

        Returns 'ROI' for revenue KPIs, 'Cost Per {X}' for count KPIs.
        """
        if self.data.kpi_type == KPIType.REVENUE:
            return "ROI"
        else:
            kpi_name = self.data.kpi_display_name or self.data.target_column
            return f"Cost Per {kpi_name.title()}"

    def is_revenue_type(self) -> bool:
        """Check if KPI is revenue-type (uses ROI terminology)."""
        return self.data.kpi_type == KPIType.REVENUE

    def format_efficiency_value(self, value: float) -> str:
        """Format efficiency value for display based on KPI type.

        For revenue KPIs: displays as ROI (e.g., '$3.50')
        For count KPIs: inverts and displays as cost-per (e.g., '$5.00')
        """
        if self.data.kpi_type == KPIType.REVENUE:
            return f"${value:.2f}"
        else:
            # Cost per = 1 / efficiency
            cost_per = 1 / value if value > 0 else float('inf')
            return f"${cost_per:.2f}"

    def get_adstock_type_dict(self) -> dict[str, str]:
        """Get adstock type mapping for channels, owned media, competitors, and controls (if enabled)."""
        result = {}
        # Channels always have adstock
        for ch in self.channels:
            result[ch.name] = ch.adstock_type.value
        # Owned media always has adstock
        for om in self.owned_media:
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
        """Get unique channel categories for a specific column (preserves insertion order)."""
        seen = dict.fromkeys(ch.get_category(column_name) for ch in self.channels)
        return list(seen.keys())

    def get_control_categories(self, column_name: str = "Category") -> list[str]:
        """Get unique control categories for a specific column (preserves insertion order)."""
        seen = dict.fromkeys(ctrl.get_category(column_name) for ctrl in self.controls)
        return list(seen.keys())

    def get_channels_by_category(self, category: str, column_name: str = "Category") -> list[ChannelConfig]:
        """Get channels in a specific category for a specific column."""
        return [ch for ch in self.channels if ch.get_category(column_name) == category]

    def get_controls_by_category(self, category: str, column_name: str = "Category") -> list[ControlConfig]:
        """Get controls in a specific category for a specific column."""
        return [ctrl for ctrl in self.controls if ctrl.get_category(column_name) == category]

    class Config:
        use_enum_values = True
