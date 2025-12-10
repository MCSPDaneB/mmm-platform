"""Analysis and diagnostics for MMM Platform."""

from .diagnostics import ModelDiagnostics
from .contributions import ContributionAnalyzer
from .reporting import ReportGenerator
from .backtest import BacktestValidator
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
from .marginal_roi import (
    MarginalROIAnalyzer,
    ChannelMarginalROI,
    InvestmentPriorityResult,
    logistic_saturation_derivative,
    calculate_marginal_roi,
    find_breakeven_spend,
)
from .executive_summary import (
    ExecutiveSummaryGenerator,
    ExecutiveSummaryResult,
    ReallocationRecommendation,
)
from .combined_models import (
    CombinedModelAnalyzer,
    CombinedChannelAnalysis,
    CombinedModelResult,
    ViewRecommendation,
)
from .demo import (
    create_demo_scenario,
    DemoScenario,
    run_full_demo,
    test_marginal_roi,
    test_executive_summary,
    test_combined_models,
    test_visualizations,
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
    # New executive visualizations
    create_baseline_channels_donut,
    create_contribution_rank_over_time,
    create_roi_effectiveness_bubble,
    create_response_curves,
    create_current_vs_marginal_roi,
    create_spend_vs_breakeven,
    create_stacked_contributions_area,
)

__all__ = [
    # Analyzers
    "ModelDiagnostics",
    "ContributionAnalyzer",
    "ReportGenerator",
    "BacktestValidator",
    "BayesianSignificanceAnalyzer",
    "MarginalROIAnalyzer",
    "ExecutiveSummaryGenerator",
    # Bayesian significance results
    "BayesianSignificanceReport",
    "CredibleIntervalResult",
    "ProbabilityOfDirectionResult",
    "ROPEResult",
    "ROICredibleIntervalResult",
    "PriorSensitivityResult",
    "get_interpretation_guide",
    # Marginal ROI results
    "ChannelMarginalROI",
    "InvestmentPriorityResult",
    "logistic_saturation_derivative",
    "calculate_marginal_roi",
    "find_breakeven_spend",
    # Executive summary results
    "ExecutiveSummaryResult",
    "ReallocationRecommendation",
    # Combined model analysis
    "CombinedModelAnalyzer",
    "CombinedChannelAnalysis",
    "CombinedModelResult",
    "ViewRecommendation",
    # Demo utilities
    "create_demo_scenario",
    "DemoScenario",
    "run_full_demo",
    "test_marginal_roi",
    "test_executive_summary",
    "test_combined_models",
    "test_visualizations",
    # Visualizations
    "create_forest_plot",
    "create_roi_posterior_plot",
    "create_probability_of_direction_plot",
    "create_prior_vs_posterior_plot",
    "create_significance_dashboard",
    "create_model_decomposition_plot",
    "create_residual_analysis_plots",
    "create_channel_roi_bar_chart",
    "create_contribution_waterfall_chart",
    # New executive visualizations
    "create_baseline_channels_donut",
    "create_contribution_rank_over_time",
    "create_roi_effectiveness_bubble",
    "create_response_curves",
    "create_current_vs_marginal_roi",
    "create_spend_vs_breakeven",
    "create_stacked_contributions_area",
]
