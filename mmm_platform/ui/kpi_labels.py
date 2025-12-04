"""
KPI-aware labeling system for UI components.

Provides dynamic labels and formatting based on KPI type (revenue vs count).
- Revenue KPIs use "ROI" terminology (e.g., "ROI: $3.50")
- Count KPIs use "Cost Per X" terminology (e.g., "Cost Per Install: $5.00")
"""

from typing import Optional
from mmm_platform.config.schema import ModelConfig, KPIType


class KPILabels:
    """Provides dynamic labels and formatting based on KPI type.

    Usage:
        labels = KPILabels(config)
        st.header(f"Channel {labels.efficiency_label}")
        formatted = labels.format_efficiency(3.5)  # "$3.50" or "$0.29"
    """

    def __init__(self, config: ModelConfig):
        """Initialize with model config.

        Args:
            config: ModelConfig containing KPI type and display settings
        """
        self.config = config
        self.kpi_type = config.data.kpi_type
        self.target_name = config.data.kpi_display_name or config.data.target_column

    @property
    def is_revenue_type(self) -> bool:
        """Check if KPI is revenue-type (uses ROI terminology)."""
        return self.kpi_type == KPIType.REVENUE

    @property
    def efficiency_label(self) -> str:
        """Main efficiency metric label.

        Returns:
            'ROI' for revenue KPIs, 'Cost Per {X}' for count KPIs
        """
        if self.is_revenue_type:
            return "ROI"
        return f"Cost Per {self._format_kpi_name()}"

    @property
    def efficiency_column_label(self) -> str:
        """Abbreviated column header for efficiency in tables.

        Returns:
            'ROI' for revenue KPIs, 'Cost/{X}' for count KPIs (abbreviated)
        """
        if self.is_revenue_type:
            return "ROI"
        # Abbreviate for table columns
        kpi_abbrev = self._format_kpi_name()[:6]
        return f"Cost/{kpi_abbrev}"

    @property
    def marginal_efficiency_label(self) -> str:
        """Label for marginal efficiency metric.

        Returns:
            'Marginal ROI' or 'Marginal Cost Per {X}'
        """
        return f"Marginal {self.efficiency_label}"

    @property
    def prior_input_label(self) -> str:
        """Label for prior input fields in configuration UI.

        Returns:
            'Expected ROI' or 'Expected Cost Per {X}'
        """
        return f"Expected {self.efficiency_label}"

    @property
    def prior_help_text(self) -> str:
        """Help text explaining prior inputs.

        Returns:
            Description of what the prior represents
        """
        if self.is_revenue_type:
            return "Expected return on investment (target value / spend)"
        return f"Expected cost per {self._format_kpi_name().lower()} (spend / target)"

    @property
    def target_label(self) -> str:
        """Label for the target KPI column.

        Returns:
            Custom display name or formatted target column name
        """
        return self._format_kpi_name()

    def _format_kpi_name(self) -> str:
        """Format KPI name for display (title case, cleaned up)."""
        name = self.target_name
        # Clean up common patterns
        name = name.replace("_", " ")
        return name.title()

    def format_efficiency(self, value: float, include_symbol: bool = True) -> str:
        """Format efficiency value for display based on KPI type.

        For revenue KPIs: displays as ROI (e.g., '$3.50')
        For count KPIs: inverts and displays as cost-per (e.g., '$5.00')

        Args:
            value: Efficiency value (target/spend ratio)
            include_symbol: Whether to include $ symbol

        Returns:
            Formatted string
        """
        if value <= 0:
            return "N/A"

        if self.is_revenue_type:
            formatted = f"{value:.2f}"
        else:
            # Cost per = 1 / efficiency
            cost_per = 1 / value
            formatted = f"{cost_per:.2f}"

        if include_symbol:
            return f"${formatted}"
        return formatted

    def format_efficiency_range(self, low: float, mid: float, high: float) -> str:
        """Format efficiency range for display.

        Args:
            low: Low bound efficiency
            mid: Mid point efficiency
            high: High bound efficiency

        Returns:
            Formatted range string like '$1.50 - $3.50 (mid: $2.50)'
        """
        if self.is_revenue_type:
            return f"${low:.2f} - ${high:.2f} (mid: ${mid:.2f})"
        else:
            # Invert for cost-per display (high efficiency = low cost)
            cost_low = 1 / high if high > 0 else float('inf')
            cost_mid = 1 / mid if mid > 0 else float('inf')
            cost_high = 1 / low if low > 0 else float('inf')
            return f"${cost_low:.2f} - ${cost_high:.2f} (mid: ${cost_mid:.2f})"

    def convert_input_to_internal(self, input_value: float) -> float:
        """Convert user input to internal efficiency format.

        For revenue KPIs: input is already efficiency (ROI)
        For count KPIs: input is cost-per, convert to efficiency (1/cost)

        Args:
            input_value: Value entered by user

        Returns:
            Internal efficiency value (target/spend ratio)
        """
        if self.is_revenue_type:
            return input_value
        else:
            # User enters cost per X, convert to efficiency
            return 1 / input_value if input_value > 0 else 0.01

    def convert_internal_to_display(self, internal_value: float) -> float:
        """Convert internal efficiency to display value.

        For revenue KPIs: return as-is (ROI)
        For count KPIs: invert to cost-per

        Args:
            internal_value: Internal efficiency value

        Returns:
            Display value (ROI or cost-per depending on KPI type)
        """
        if self.is_revenue_type:
            return internal_value
        else:
            return 1 / internal_value if internal_value > 0 else float('inf')

    def get_prior_column_names(self) -> tuple[str, str, str]:
        """Get column names for prior input table.

        Returns:
            Tuple of (low_col, mid_col, high_col) column names
        """
        if self.is_revenue_type:
            return ("ROI Low", "ROI Mid", "ROI High")
        else:
            # For cost-per, low cost = high efficiency, so labels are inverted
            return ("Cost High", "Cost Mid", "Cost Low")

    def get_tab_names(self) -> dict[str, str]:
        """Get dynamic tab names for results page.

        Returns:
            Dict mapping tab keys to display names
        """
        eff = self.efficiency_label
        return {
            "channel_efficiency": f"Channel {eff}",
            "marginal_efficiency": f"Marginal {eff} & Priority",
            "efficiency_validation": f"{eff} Prior Validation",
        }


def create_kpi_labels(config: Optional[ModelConfig]) -> Optional[KPILabels]:
    """Factory function to create KPILabels from config.

    Args:
        config: ModelConfig or None

    Returns:
        KPILabels instance or None if config is None
    """
    if config is None:
        return None
    return KPILabels(config)
