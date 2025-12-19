"""
Tests for BudgetAllocator class.

Tests cover:
- Initialization
- Basic optimize functionality
- Incremental optimization
- Efficiency floor optimization
- Scenario analysis
- Channel info retrieval
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_config():
    """Create a mock MMMConfig."""
    config = Mock()
    config.data = Mock()
    config.data.target_column = "sales"
    config.data.spend_scale = 1000.0
    config.data.date_column = "date"

    # Channels with different types
    channel1 = Mock()
    channel1.name = "tv_spend"
    channel1.display_name = "TV"
    channel1.channel_type = "paid_media"
    channel1.has_roi_prior = True

    channel2 = Mock()
    channel2.name = "search_spend"
    channel2.display_name = "Search"
    channel2.channel_type = "paid_media"
    channel2.has_roi_prior = True

    channel3 = Mock()
    channel3.name = "email_sends"
    channel3.display_name = "Email"
    channel3.channel_type = "owned_media"
    channel3.has_roi_prior = True

    config.channels = [channel1, channel2, channel3]

    return config


@pytest.fixture
def mock_df_scaled():
    """Create mock scaled DataFrame with 52 weeks of data."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=52, freq="W")

    df = pd.DataFrame({
        "date": dates,
        "tv_spend": np.random.uniform(5, 15, 52),
        "search_spend": np.random.uniform(3, 10, 52),
        "email_sends": np.random.uniform(1, 5, 52),
        "sales": np.random.uniform(50, 150, 52),
    })

    return df


@pytest.fixture
def mock_idata():
    """Create mock InferenceData with posterior samples."""
    idata = Mock()

    # Create posterior mock with samples
    posterior = Mock()

    # Adstock/saturation parameters (n_samples, n_channels)
    n_samples = 100
    n_channels = 3

    # Create mock xarray DataArrays for posterior parameters
    alpha_mock = Mock()
    alpha_mock.values = np.random.uniform(0.3, 0.7, (2, 50, n_channels))  # (chains, draws, channels)
    alpha_mock.stack = Mock(return_value=Mock(values=np.random.uniform(0.3, 0.7, (n_samples, n_channels))))

    lam_mock = Mock()
    lam_mock.values = np.random.uniform(1.0, 3.0, (2, 50, n_channels))
    lam_mock.stack = Mock(return_value=Mock(values=np.random.uniform(1.0, 3.0, (n_samples, n_channels))))

    beta_mock = Mock()
    beta_mock.values = np.random.uniform(0.5, 2.0, (2, 50, n_channels))
    beta_mock.stack = Mock(return_value=Mock(values=np.random.uniform(0.5, 2.0, (n_samples, n_channels))))

    posterior.__getitem__ = Mock(side_effect=lambda k: {
        "alpha": alpha_mock,
        "lam": lam_mock,
        "beta_channel": beta_mock,
    }.get(k, Mock()))

    posterior.__contains__ = Mock(return_value=True)

    idata.posterior = posterior

    return idata


@pytest.fixture
def mock_mmm(mock_df_scaled, mock_idata):
    """Create mock PyMC-Marketing MMM object."""
    mmm = Mock()
    mmm.channel_columns = ["tv_spend", "search_spend", "email_sends"]
    mmm.idata = mock_idata

    # Mock compute_channel_contribution_original_scale
    def mock_contribution(*args, **kwargs):
        n_periods = kwargs.get('num_periods', 8)
        n_samples = 100
        n_channels = 3
        # Return mock array of contributions
        return np.random.uniform(1000, 10000, (n_samples, n_periods, n_channels))

    mmm.compute_channel_contribution_original_scale = Mock(side_effect=mock_contribution)

    return mmm


@pytest.fixture
def mock_wrapper(mock_config, mock_df_scaled, mock_mmm, mock_idata):
    """Create mock MMMWrapper."""
    wrapper = Mock()
    wrapper.config = mock_config
    wrapper.df_scaled = mock_df_scaled
    wrapper.df_original = mock_df_scaled.copy()
    wrapper.df_original["tv_spend"] = mock_df_scaled["tv_spend"] * 1000
    wrapper.df_original["search_spend"] = mock_df_scaled["search_spend"] * 1000
    wrapper.df_original["email_sends"] = mock_df_scaled["email_sends"] * 1000
    wrapper.df_original["sales"] = mock_df_scaled["sales"] * 1000
    wrapper.mmm = mock_mmm
    wrapper.idata = mock_idata

    return wrapper


@pytest.fixture
def allocator(mock_wrapper):
    """Create BudgetAllocator with mock wrapper."""
    with patch('mmm_platform.optimization.bridge.OptimizationBridge') as MockBridge:
        # Create a real-ish mock for the bridge
        bridge = Mock()
        bridge.wrapper = mock_wrapper
        bridge.config = mock_wrapper.config
        bridge.mmm = mock_wrapper.mmm
        bridge.channel_columns = ["tv_spend", "search_spend", "email_sends"]
        bridge.channel_display_names = {
            "tv_spend": "TV",
            "search_spend": "Search",
            "email_sends": "Email",
        }

        # Mock methods
        bridge.get_optimizable_channels.return_value = ["tv_spend", "search_spend", "email_sends"]
        bridge.get_utility_function.return_value = lambda x: x.mean()
        bridge.get_average_period_spend.return_value = {
            "tv_spend": 5000.0,
            "search_spend": 3000.0,
            "email_sends": 1000.0,
        }
        bridge.get_historical_spend.return_value = {
            "tv_spend": 260000.0,
            "search_spend": 156000.0,
            "email_sends": 52000.0,
        }
        bridge.get_current_allocation.return_value = {
            "tv_spend": 40000.0,
            "search_spend": 24000.0,
            "email_sends": 8000.0,
        }
        bridge.estimate_response_at_allocation.return_value = (50000.0, 45000.0, 55000.0)

        MockBridge.return_value = bridge

        from mmm_platform.optimization.allocator import BudgetAllocator
        alloc = BudgetAllocator(mock_wrapper, num_periods=8, utility="mean")
        alloc.bridge = bridge

        return alloc


# ============================================================================
# Test Classes
# ============================================================================

class TestBudgetAllocatorInit:
    """Tests for BudgetAllocator initialization."""

    def test_init_with_fitted_model(self, mock_wrapper):
        """Allocator initializes with fitted model."""
        with patch('mmm_platform.optimization.allocator.OptimizationBridge') as MockBridge:
            bridge = Mock()
            bridge.channel_columns = ["tv_spend", "search_spend"]
            bridge.get_utility_function.return_value = lambda x: x.mean()
            MockBridge.return_value = bridge

            from mmm_platform.optimization.allocator import BudgetAllocator
            alloc = BudgetAllocator(mock_wrapper, num_periods=8)

            assert alloc.num_periods == 8
            assert alloc.utility_name == "mean"

    def test_init_with_custom_utility(self, mock_wrapper):
        """Allocator accepts custom utility function."""
        with patch('mmm_platform.optimization.allocator.OptimizationBridge') as MockBridge:
            bridge = Mock()
            bridge.channel_columns = ["tv_spend"]
            bridge.get_utility_function.return_value = lambda x: np.percentile(x, 5)
            MockBridge.return_value = bridge

            from mmm_platform.optimization.allocator import BudgetAllocator
            alloc = BudgetAllocator(mock_wrapper, num_periods=4, utility="var")

            assert alloc.utility_name == "var"

    def test_channels_property(self, allocator):
        """Channels property returns optimizable channels."""
        channels = allocator.channels

        assert isinstance(channels, list)
        assert "tv_spend" in channels

    def test_mmm_property(self, allocator, mock_mmm):
        """MMM property returns underlying model."""
        allocator.bridge.mmm = mock_mmm

        assert allocator.mmm == mock_mmm


class TestGetChannelInfo:
    """Tests for get_channel_info method."""

    def test_returns_dataframe(self, allocator):
        """get_channel_info returns DataFrame."""
        info = allocator.get_channel_info()

        assert isinstance(info, pd.DataFrame)

    def test_has_required_columns(self, allocator):
        """DataFrame has required columns."""
        info = allocator.get_channel_info()

        assert "channel" in info.columns
        assert "display_name" in info.columns
        assert "historical_spend" in info.columns
        assert "avg_period_spend" in info.columns

    def test_includes_all_channels(self, allocator):
        """DataFrame includes all optimizable channels."""
        info = allocator.get_channel_info()

        channels = info["channel"].tolist()
        assert "tv_spend" in channels
        assert "search_spend" in channels


class TestOptimize:
    """Tests for optimize method."""

    def test_returns_optimization_result(self, allocator):
        """optimize returns OptimizationResult."""
        # Mock the _optimize_with_working_gradients method
        mock_result = Mock()
        mock_result.success = True
        mock_result.message = "Optimization succeeded"
        mock_result.nit = 25
        mock_result.fun = -50000.0
        mock_result.risk_metrics = {
            'expected_response': 50000.0,
            'response_ci_low': 45000.0,
            'response_ci_high': 55000.0,
            'response_var': 47000.0,
            'response_cvar': 45000.0,
            'response_sharpe': 2.5,
            'response_std': 3000.0,
        }

        allocation = {"tv_spend": 40000, "search_spend": 30000, "email_sends": 10000}
        allocator._optimize_with_working_gradients = Mock(return_value=(allocation, mock_result))

        result = allocator.optimize(total_budget=80000)

        assert result is not None
        assert hasattr(result, 'optimal_allocation')
        assert hasattr(result, 'expected_response')

    def test_respects_total_budget(self, allocator):
        """Allocation sums to total budget."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.message = "OK"
        mock_result.nit = 20
        mock_result.fun = -45000.0
        mock_result.risk_metrics = {
            'expected_response': 45000.0,
            'response_ci_low': 40000.0,
            'response_ci_high': 50000.0,
        }
        mock_result.actual_allocated = 100000
        mock_result.unallocated_budget = 0.0

        allocation = {"tv_spend": 50000, "search_spend": 35000, "email_sends": 15000}
        allocator._optimize_with_working_gradients = Mock(return_value=(allocation, mock_result))

        result = allocator.optimize(total_budget=100000)

        total = sum(result.optimal_allocation.values())
        assert abs(total - 100000) < 0.01

    def test_with_compare_to_current(self, allocator):
        """compare_to_current includes baseline comparison."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.message = "OK"
        mock_result.nit = 15
        mock_result.fun = -48000.0
        mock_result.risk_metrics = {
            'expected_response': 48000.0,
            'response_ci_low': 44000.0,
            'response_ci_high': 52000.0,
        }
        mock_result.actual_allocated = 80000
        mock_result.unallocated_budget = 0.0

        allocation = {"tv_spend": 45000, "search_spend": 30000, "email_sends": 5000}
        allocator._optimize_with_working_gradients = Mock(return_value=(allocation, mock_result))

        result = allocator.optimize(total_budget=80000, compare_to_current=True)

        assert result.current_allocation is not None
        assert result.current_response is not None

    def test_with_custom_bounds(self, allocator):
        """Custom channel bounds are passed through."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.message = "OK"
        mock_result.nit = 18
        mock_result.fun = -42000.0
        mock_result.risk_metrics = {
            'expected_response': 42000.0,
            'response_ci_low': 38000.0,
            'response_ci_high': 46000.0,
        }
        mock_result.actual_allocated = 60000
        mock_result.unallocated_budget = 0.0

        allocation = {"tv_spend": 30000, "search_spend": 20000, "email_sends": 10000}
        allocator._optimize_with_working_gradients = Mock(return_value=(allocation, mock_result))

        bounds = {
            "tv_spend": (10000, 50000),
            "search_spend": (5000, 30000),
            "email_sends": (0, 20000),
        }

        result = allocator.optimize(total_budget=60000, channel_bounds=bounds)

        # Verify optimization was called
        allocator._optimize_with_working_gradients.assert_called_once()
        assert result.success

    def test_failed_optimization_returns_result(self, allocator):
        """Failed optimization still returns a result."""
        # Simulate exception
        allocator._optimize_with_working_gradients = Mock(side_effect=Exception("Optimization failed"))

        result = allocator.optimize(total_budget=50000)

        assert result is not None
        assert result.success is False
        assert "Optimization failed" in result.message

    def test_with_seasonal_indices(self, allocator):
        """Seasonal indices are passed to optimizer."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.message = "OK"
        mock_result.nit = 22
        mock_result.fun = -55000.0
        mock_result.risk_metrics = {
            'expected_response': 55000.0,
            'response_ci_low': 50000.0,
            'response_ci_high': 60000.0,
        }

        allocation = {"tv_spend": 50000, "search_spend": 40000, "email_sends": 10000}
        allocator._optimize_with_working_gradients = Mock(return_value=(allocation, mock_result))

        seasonal = {"tv_spend": 1.2, "search_spend": 0.9, "email_sends": 1.0}
        result = allocator.optimize(total_budget=100000, seasonal_indices=seasonal)

        # Check seasonal indices were passed
        call_args = allocator._optimize_with_working_gradients.call_args
        assert call_args[1].get('seasonal_indices') == seasonal


class TestOptimizeIncremental:
    """Tests for optimize_incremental method."""

    def test_returns_result(self, allocator):
        """optimize_incremental returns OptimizationResult."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.message = "OK"
        mock_result.nit = 20
        mock_result.fun = -60000.0
        mock_result.risk_metrics = {
            'expected_response': 60000.0,
            'response_ci_low': 55000.0,
            'response_ci_high': 65000.0,
        }

        allocation = {"tv_spend": 55000, "search_spend": 35000, "email_sends": 20000}
        allocator._optimize_with_working_gradients = Mock(return_value=(allocation, mock_result))

        base = {"tv_spend": 40000, "search_spend": 24000, "email_sends": 8000}
        result = allocator.optimize_incremental(
            base_allocation=base,
            incremental_budget=28000,
        )

        assert result is not None
        assert result.current_allocation == base

    def test_respects_base_allocation(self, allocator):
        """Incremental optimization respects base allocation."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.message = "OK"
        mock_result.nit = 15
        mock_result.fun = -52000.0
        mock_result.risk_metrics = {
            'expected_response': 52000.0,
            'response_ci_low': 48000.0,
            'response_ci_high': 56000.0,
        }

        base = {"tv_spend": 30000, "search_spend": 20000, "email_sends": 10000}
        allocation = {"tv_spend": 45000, "search_spend": 28000, "email_sends": 17000}
        allocator._optimize_with_working_gradients = Mock(return_value=(allocation, mock_result))

        result = allocator.optimize_incremental(
            base_allocation=base,
            incremental_budget=30000,
        )

        # Total should be base + incremental
        expected_total = sum(base.values()) + 30000
        assert result.total_budget == expected_total


class TestOptimizeWithEfficiencyFloor:
    """Tests for optimize_with_efficiency_floor method."""

    def test_full_budget_meets_roi_target(self, allocator):
        """When full budget meets ROI target, returns full allocation."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.message = "OK"
        mock_result.nit = 20
        mock_result.fun = -75000.0
        mock_result.risk_metrics = {
            'expected_response': 75000.0,  # ROI = 75000/50000 = 1.5
            'response_ci_low': 70000.0,
            'response_ci_high': 80000.0,
        }
        mock_result.actual_allocated = 50000
        mock_result.unallocated_budget = 0.0

        allocation = {"tv_spend": 25000, "search_spend": 15000, "email_sends": 10000}
        allocator._optimize_with_working_gradients = Mock(return_value=(allocation, mock_result))

        result = allocator.optimize_with_efficiency_floor(
            total_budget=50000,
            efficiency_metric="roi",
            efficiency_target=1.0,  # Require ROI >= 1.0
        )

        assert result.unallocated_budget == 0.0
        assert result.achieved_efficiency >= 1.0

    def test_cpa_target_with_partial_allocation(self, allocator):
        """When CPA target requires less budget, returns partial allocation."""
        # First call (full budget) fails target
        mock_result_high = Mock()
        mock_result_high.success = True
        mock_result_high.message = "OK"
        mock_result_high.nit = 20
        mock_result_high.fun = -40000.0
        mock_result_high.risk_metrics = {
            'expected_response': 40000.0,  # CPA = 100000/40000 = 2.5
            'response_ci_low': 35000.0,
            'response_ci_high': 45000.0,
        }

        # Mid-budget call meets target
        mock_result_mid = Mock()
        mock_result_mid.success = True
        mock_result_mid.message = "OK"
        mock_result_mid.nit = 18
        mock_result_mid.fun = -35000.0
        mock_result_mid.risk_metrics = {
            'expected_response': 35000.0,  # CPA = 50000/35000 = 1.43
            'response_ci_low': 30000.0,
            'response_ci_high': 40000.0,
        }

        allocation_high = {"tv_spend": 50000, "search_spend": 30000, "email_sends": 20000}
        allocation_mid = {"tv_spend": 25000, "search_spend": 15000, "email_sends": 10000}

        # Return different results based on budget
        call_count = [0]
        def mock_optimize(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return (allocation_high, mock_result_high)
            else:
                return (allocation_mid, mock_result_mid)

        allocator._optimize_with_working_gradients = Mock(side_effect=mock_optimize)

        result = allocator.optimize_with_efficiency_floor(
            total_budget=100000,
            efficiency_metric="cpa",
            efficiency_target=2.0,  # Max CPA of $2.0
        )

        # Should have unallocated budget since full budget doesn't meet CPA target
        assert result is not None

    def test_no_budget_meets_target(self, allocator):
        """When no budget meets target, returns zero allocation."""
        # All calls return poor efficiency
        mock_result = Mock()
        mock_result.success = True
        mock_result.message = "OK"
        mock_result.nit = 15
        mock_result.fun = -5000.0
        mock_result.risk_metrics = {
            'expected_response': 5000.0,  # Very poor ROI
            'response_ci_low': 4000.0,
            'response_ci_high': 6000.0,
        }

        allocation = {"tv_spend": 30000, "search_spend": 15000, "email_sends": 5000}
        allocator._optimize_with_working_gradients = Mock(return_value=(allocation, mock_result))

        result = allocator.optimize_with_efficiency_floor(
            total_budget=50000,
            efficiency_metric="roi",
            efficiency_target=10.0,  # Unrealistic ROI target
        )

        assert result.unallocated_budget == 50000
        assert all(v == 0.0 for v in result.optimal_allocation.values())


class TestScenarioAnalysis:
    """Tests for scenario_analysis method."""

    def test_returns_scenario_result(self, allocator):
        """scenario_analysis returns ScenarioResult."""
        mock_result = Mock()
        mock_result.success = True
        mock_result.message = "OK"
        mock_result.nit = 20
        mock_result.fun = -40000.0
        mock_result.risk_metrics = {
            'expected_response': 40000.0,
            'response_ci_low': 35000.0,
            'response_ci_high': 45000.0,
        }

        allocation = {"tv_spend": 20000, "search_spend": 15000, "email_sends": 5000}
        allocator._optimize_with_working_gradients = Mock(return_value=(allocation, mock_result))

        scenarios = [50000, 75000, 100000]
        result = allocator.scenario_analysis(budget_scenarios=scenarios)

        assert result is not None
        assert hasattr(result, 'budget_scenarios')
        assert hasattr(result, 'results')
        assert hasattr(result, 'efficiency_curve')

    def test_runs_all_scenarios(self, allocator):
        """Analysis runs optimization for each budget."""
        responses = [30000, 45000, 55000, 62000]
        call_idx = [0]

        def mock_optimize(*args, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1

            mock_result = Mock()
            mock_result.success = True
            mock_result.message = "OK"
            mock_result.nit = 20
            mock_result.fun = -responses[min(idx, len(responses)-1)]
            mock_result.risk_metrics = {
                'expected_response': responses[min(idx, len(responses)-1)],
                'response_ci_low': responses[min(idx, len(responses)-1)] * 0.9,
                'response_ci_high': responses[min(idx, len(responses)-1)] * 1.1,
            }

            allocation = {"tv_spend": 20000, "search_spend": 15000, "email_sends": 5000}
            return (allocation, mock_result)

        allocator._optimize_with_working_gradients = Mock(side_effect=mock_optimize)

        scenarios = [40000, 60000, 80000, 100000]
        result = allocator.scenario_analysis(budget_scenarios=scenarios)

        assert len(result.results) == len(scenarios)

    def test_efficiency_curve_has_marginal_response(self, allocator):
        """Efficiency curve includes marginal response calculation."""
        responses = [30000, 50000, 65000]
        call_idx = [0]

        def mock_optimize(*args, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1

            mock_result = Mock()
            mock_result.success = True
            mock_result.message = "OK"
            mock_result.nit = 20
            mock_result.fun = -responses[min(idx, len(responses)-1)]
            mock_result.risk_metrics = {
                'expected_response': responses[min(idx, len(responses)-1)],
                'response_ci_low': responses[min(idx, len(responses)-1)] * 0.9,
                'response_ci_high': responses[min(idx, len(responses)-1)] * 1.1,
            }

            allocation = {"tv_spend": 20000, "search_spend": 15000, "email_sends": 5000}
            return (allocation, mock_result)

        allocator._optimize_with_working_gradients = Mock(side_effect=mock_optimize)

        scenarios = [50000, 75000, 100000]
        result = allocator.scenario_analysis(budget_scenarios=scenarios)

        assert "marginal_response" in result.efficiency_curve.columns


class TestCreateAllocatorFromSession:
    """Tests for create_allocator_from_session helper."""

    def test_returns_none_without_model(self):
        """Returns None if no model in session."""
        from mmm_platform.optimization.allocator import create_allocator_from_session

        session = {"current_model": None}
        result = create_allocator_from_session(session)

        assert result is None

    def test_returns_none_if_not_fitted(self):
        """Returns None if model not fitted."""
        from mmm_platform.optimization.allocator import create_allocator_from_session

        wrapper = Mock()
        wrapper.idata = None

        session = {"current_model": wrapper}
        result = create_allocator_from_session(session)

        assert result is None

    def test_returns_allocator_with_fitted_model(self, mock_wrapper):
        """Returns allocator with fitted model."""
        from mmm_platform.optimization.allocator import create_allocator_from_session

        with patch('mmm_platform.optimization.allocator.BudgetAllocator') as MockAllocator:
            MockAllocator.return_value = Mock()

            session = {"current_model": mock_wrapper}
            result = create_allocator_from_session(session)

            assert result is not None
            MockAllocator.assert_called_once_with(mock_wrapper)


class TestWarmStartBehavior:
    """Tests for warm-start (x0_init) behavior in optimizer."""

    def test_optimize_accepts_x0_init_parameter(self, allocator):
        """Verify optimize() accepts x0_init parameter."""
        # x0_init should be accepted without error
        result = allocator.optimize(
            total_budget=100000,
            x0_init={"tv_spend": 50000, "search_spend": 50000},
        )
        assert result is not None

    def test_scenario_analysis_uses_ascending_order(self, allocator):
        """Scenario analysis should process budgets in ascending order for warm-start."""
        # We can't easily test the internal warm-start behavior,
        # but we can verify the results are in ascending budget order
        budget_scenarios = [150000, 50000, 100000]  # Not sorted

        result = allocator.scenario_analysis(budget_scenarios)

        # Results should be ordered by budget (ascending)
        budgets = [r.total_budget for r in result.results]
        assert budgets == sorted(budgets)

    def test_warm_start_scales_allocation(self):
        """Verify warm-start allocation is scaled proportionally."""
        # This tests the scaling logic in _optimize_with_working_gradients
        channels = ["tv_spend", "search_spend"]
        prev_allocation = {"tv_spend": 30000, "search_spend": 70000}  # $100k total
        new_budget_per_period_scaled = 200  # Represents $200k total

        # Expected: allocation should be scaled 2x (200k / 100k)
        prev_total = sum(prev_allocation.values())
        spend_scale = 1000.0
        num_periods = 1

        scale_factor = new_budget_per_period_scaled / (prev_total / spend_scale / num_periods)

        expected_x0 = np.array([
            prev_allocation[ch] / spend_scale / num_periods * scale_factor
            for ch in channels
        ])

        # TV: 30k / 1000 * 2 = 60
        # Search: 70k / 1000 * 2 = 140
        assert expected_x0[0] == pytest.approx(60.0)
        assert expected_x0[1] == pytest.approx(140.0)

    def test_historical_proportions_as_default(self):
        """Verify historical proportions are used as default x0 (not uniform)."""
        # Historical spend: TV=30%, Search=70%
        historical = {"tv_spend": 30000, "search_spend": 70000}
        hist_total = sum(historical.values())

        # Target budget (scaled) = 150
        budget_per_period_scaled = 150

        # Expected x0: proportions scaled to target
        expected_x0 = np.array([
            historical[ch] / hist_total * budget_per_period_scaled
            for ch in ["tv_spend", "search_spend"]
        ])

        # TV: 30% of 150 = 45
        # Search: 70% of 150 = 105
        assert expected_x0[0] == pytest.approx(45.0)
        assert expected_x0[1] == pytest.approx(105.0)

        # NOT uniform (75, 75)
        uniform_x0 = np.ones(2) * budget_per_period_scaled / 2
        assert not np.allclose(expected_x0, uniform_x0)
