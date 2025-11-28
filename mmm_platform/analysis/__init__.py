"""Analysis and diagnostics for MMM Platform."""

from .diagnostics import ModelDiagnostics
from .contributions import ContributionAnalyzer
from .reporting import ReportGenerator
from .bayesian_significance import (
    BayesianSignificanceAnalyzer,
    BayesianSignificanceReport,
    CredibleIntervalResult,
    ProbabilityOfDirectionResult,
    ROPEResult,
    ROICredibleIntervalResult,
    PriorSensitivityResult,
    get_interpretation_guide,
)
from .visualizations import (
    create_forest_plot,
    create_roi_posterior_plot,
    create_probability_of_direction_plot,
    create_prior_vs_posterior_plot,
    create_significance_dashboard,
    create_model_decomposition_plot,
    create_residual_analysis_plots,
    create_channel_roi_bar_chart,
    create_contribution_waterfall_chart,
)

__all__ = [
    "ModelDiagnostics",
    "ContributionAnalyzer",
    "ReportGenerator",
    "BayesianSignificanceAnalyzer",
    "BayesianSignificanceReport",
    "CredibleIntervalResult",
    "ProbabilityOfDirectionResult",
    "ROPEResult",
    "ROICredibleIntervalResult",
    "PriorSensitivityResult",
    "get_interpretation_guide",
    "create_forest_plot",
    "create_roi_posterior_plot",
    "create_probability_of_direction_plot",
    "create_prior_vs_posterior_plot",
    "create_significance_dashboard",
    "create_model_decomposition_plot",
    "create_residual_analysis_plots",
    "create_channel_roi_bar_chart",
    "create_contribution_waterfall_chart",
]
