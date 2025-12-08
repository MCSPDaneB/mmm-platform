"""
Constraint builder helpers for budget optimization.

This module provides easy-to-use factory methods for creating
common optimization constraints.
"""

from typing import Any, Callable
import pytensor.tensor as pt
import logging

logger = logging.getLogger(__name__)


class ConstraintBuilder:
    """
    Factory for common optimization constraints.

    Constraints are used to enforce business rules during optimization,
    such as minimum spend per channel, maximum ratios, etc.

    Examples
    --------
    >>> constraints = [
    ...     ConstraintBuilder.min_spend("search_spend", 10000),
    ...     ConstraintBuilder.max_ratio("tv_spend", 0.4),
    ... ]
    >>> result = allocator.optimize(100000, constraints=constraints)
    """

    @staticmethod
    def min_spend(channel: str, amount: float) -> dict:
        """
        Ensure a channel receives at least a minimum amount.

        Parameters
        ----------
        channel : str
            Channel column name.
        amount : float
            Minimum spend required.

        Returns
        -------
        dict
            Constraint specification for PyMC-Marketing.
        """
        def constraint_fn(
            budgets_sym: pt.TensorVariable,
            total_budget_sym: pt.TensorVariable,
            optimizer: Any,
        ) -> pt.TensorVariable:
            channel_idx = optimizer.mmm_model.channel_columns.index(channel)
            channel_budget = budgets_sym[channel_idx]
            return channel_budget - amount  # >= 0 for inequality

        return {
            "name": f"min_spend_{channel}",
            "type": "ineq",
            "fun": constraint_fn,
        }

    @staticmethod
    def max_spend(channel: str, amount: float) -> dict:
        """
        Ensure a channel does not exceed a maximum amount.

        Parameters
        ----------
        channel : str
            Channel column name.
        amount : float
            Maximum spend allowed.

        Returns
        -------
        dict
            Constraint specification for PyMC-Marketing.
        """
        def constraint_fn(
            budgets_sym: pt.TensorVariable,
            total_budget_sym: pt.TensorVariable,
            optimizer: Any,
        ) -> pt.TensorVariable:
            channel_idx = optimizer.mmm_model.channel_columns.index(channel)
            channel_budget = budgets_sym[channel_idx]
            return amount - channel_budget  # >= 0 for inequality

        return {
            "name": f"max_spend_{channel}",
            "type": "ineq",
            "fun": constraint_fn,
        }

    @staticmethod
    def min_ratio(channel: str, ratio: float) -> dict:
        """
        Ensure a channel receives at least a percentage of total budget.

        Parameters
        ----------
        channel : str
            Channel column name.
        ratio : float
            Minimum ratio (0.1 = 10% of total).

        Returns
        -------
        dict
            Constraint specification for PyMC-Marketing.
        """
        def constraint_fn(
            budgets_sym: pt.TensorVariable,
            total_budget_sym: pt.TensorVariable,
            optimizer: Any,
        ) -> pt.TensorVariable:
            channel_idx = optimizer.mmm_model.channel_columns.index(channel)
            channel_budget = budgets_sym[channel_idx]
            min_required = total_budget_sym * ratio
            return channel_budget - min_required  # >= 0 for inequality

        return {
            "name": f"min_ratio_{channel}",
            "type": "ineq",
            "fun": constraint_fn,
        }

    @staticmethod
    def max_ratio(channel: str, ratio: float) -> dict:
        """
        Ensure a channel does not exceed a percentage of total budget.

        Parameters
        ----------
        channel : str
            Channel column name.
        ratio : float
            Maximum ratio (0.4 = 40% of total).

        Returns
        -------
        dict
            Constraint specification for PyMC-Marketing.
        """
        def constraint_fn(
            budgets_sym: pt.TensorVariable,
            total_budget_sym: pt.TensorVariable,
            optimizer: Any,
        ) -> pt.TensorVariable:
            channel_idx = optimizer.mmm_model.channel_columns.index(channel)
            channel_budget = budgets_sym[channel_idx]
            max_allowed = total_budget_sym * ratio
            return max_allowed - channel_budget  # >= 0 for inequality

        return {
            "name": f"max_ratio_{channel}",
            "type": "ineq",
            "fun": constraint_fn,
        }

    @staticmethod
    def channel_ratio(
        channel1: str,
        channel2: str,
        max_ratio: float,
    ) -> dict:
        """
        Ensure channel1 spend <= max_ratio * channel2 spend.

        Useful for maintaining relative spend between channels,
        e.g., "TV should not exceed 2x Search spend".

        Parameters
        ----------
        channel1 : str
            First channel (constrained).
        channel2 : str
            Second channel (reference).
        max_ratio : float
            Maximum ratio of channel1 to channel2.

        Returns
        -------
        dict
            Constraint specification for PyMC-Marketing.
        """
        def constraint_fn(
            budgets_sym: pt.TensorVariable,
            total_budget_sym: pt.TensorVariable,
            optimizer: Any,
        ) -> pt.TensorVariable:
            ch1_idx = optimizer.mmm_model.channel_columns.index(channel1)
            ch2_idx = optimizer.mmm_model.channel_columns.index(channel2)
            budget1 = budgets_sym[ch1_idx]
            budget2 = budgets_sym[ch2_idx]
            # budget1 <= max_ratio * budget2
            # Rearranged: max_ratio * budget2 - budget1 >= 0
            return max_ratio * budget2 - budget1

        return {
            "name": f"ratio_{channel1}_to_{channel2}",
            "type": "ineq",
            "fun": constraint_fn,
        }

    @staticmethod
    def min_channel_ratio(
        channel1: str,
        channel2: str,
        min_ratio: float,
    ) -> dict:
        """
        Ensure channel1 spend >= min_ratio * channel2 spend.

        Parameters
        ----------
        channel1 : str
            First channel (constrained).
        channel2 : str
            Second channel (reference).
        min_ratio : float
            Minimum ratio of channel1 to channel2.

        Returns
        -------
        dict
            Constraint specification for PyMC-Marketing.
        """
        def constraint_fn(
            budgets_sym: pt.TensorVariable,
            total_budget_sym: pt.TensorVariable,
            optimizer: Any,
        ) -> pt.TensorVariable:
            ch1_idx = optimizer.mmm_model.channel_columns.index(channel1)
            ch2_idx = optimizer.mmm_model.channel_columns.index(channel2)
            budget1 = budgets_sym[ch1_idx]
            budget2 = budgets_sym[ch2_idx]
            # budget1 >= min_ratio * budget2
            # Rearranged: budget1 - min_ratio * budget2 >= 0
            return budget1 - min_ratio * budget2

        return {
            "name": f"min_ratio_{channel1}_to_{channel2}",
            "type": "ineq",
            "fun": constraint_fn,
        }

    @staticmethod
    def group_min_spend(channels: list[str], amount: float) -> dict:
        """
        Ensure a group of channels together receives at least a minimum.

        Parameters
        ----------
        channels : list[str]
            List of channel column names in the group.
        amount : float
            Minimum combined spend for the group.

        Returns
        -------
        dict
            Constraint specification for PyMC-Marketing.
        """
        def constraint_fn(
            budgets_sym: pt.TensorVariable,
            total_budget_sym: pt.TensorVariable,
            optimizer: Any,
        ) -> pt.TensorVariable:
            group_total = 0
            for ch in channels:
                ch_idx = optimizer.mmm_model.channel_columns.index(ch)
                group_total = group_total + budgets_sym[ch_idx]
            return group_total - amount  # >= 0 for inequality

        group_name = "_".join(channels[:3])  # Truncate for readability
        return {
            "name": f"group_min_{group_name}",
            "type": "ineq",
            "fun": constraint_fn,
        }

    @staticmethod
    def group_max_ratio(channels: list[str], ratio: float) -> dict:
        """
        Ensure a group of channels together does not exceed a ratio.

        Parameters
        ----------
        channels : list[str]
            List of channel column names in the group.
        ratio : float
            Maximum combined ratio of total budget.

        Returns
        -------
        dict
            Constraint specification for PyMC-Marketing.
        """
        def constraint_fn(
            budgets_sym: pt.TensorVariable,
            total_budget_sym: pt.TensorVariable,
            optimizer: Any,
        ) -> pt.TensorVariable:
            group_total = 0
            for ch in channels:
                ch_idx = optimizer.mmm_model.channel_columns.index(ch)
                group_total = group_total + budgets_sym[ch_idx]
            max_allowed = total_budget_sym * ratio
            return max_allowed - group_total  # >= 0 for inequality

        group_name = "_".join(channels[:3])
        return {
            "name": f"group_max_ratio_{group_name}",
            "type": "ineq",
            "fun": constraint_fn,
        }


def build_bounds_from_constraints(
    channels: list[str],
    total_budget: float,
    min_spend: dict[str, float] | None = None,
    max_spend: dict[str, float] | None = None,
    min_ratio: dict[str, float] | None = None,
    max_ratio: dict[str, float] | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Build channel bounds dictionary from simple constraints.

    This is a convenience function that converts common constraint
    specifications into the bounds format expected by the optimizer.

    Parameters
    ----------
    channels : list[str]
        List of channel column names.
    total_budget : float
        Total budget being allocated.
    min_spend : dict, optional
        {channel: min_amount} minimum spend per channel.
    max_spend : dict, optional
        {channel: max_amount} maximum spend per channel.
    min_ratio : dict, optional
        {channel: min_ratio} minimum ratio of total budget.
    max_ratio : dict, optional
        {channel: max_ratio} maximum ratio of total budget.

    Returns
    -------
    dict[str, tuple[float, float]]
        {channel: (lower_bound, upper_bound)}
    """
    min_spend = min_spend or {}
    max_spend = max_spend or {}
    min_ratio = min_ratio or {}
    max_ratio = max_ratio or {}

    bounds = {}
    for ch in channels:
        # Start with default bounds
        lower = 0.0
        upper = total_budget

        # Apply min_spend
        if ch in min_spend:
            lower = max(lower, min_spend[ch])

        # Apply min_ratio
        if ch in min_ratio:
            lower = max(lower, total_budget * min_ratio[ch])

        # Apply max_spend
        if ch in max_spend:
            upper = min(upper, max_spend[ch])

        # Apply max_ratio
        if ch in max_ratio:
            upper = min(upper, total_budget * max_ratio[ch])

        # Ensure valid bounds
        if upper < lower:
            logger.warning(
                f"Invalid bounds for {ch}: lower={lower}, upper={upper}. "
                f"Setting upper=lower."
            )
            upper = lower

        bounds[ch] = (lower, upper)

    return bounds
