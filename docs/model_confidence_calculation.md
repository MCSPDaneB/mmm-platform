# Model Confidence Calculation

This document describes the methodology for calculating model confidence from Bayesian posterior distributions, specifically for use in discounting optimizer predictions.

## Overview

Model confidence measures how certain the model is about its channel effectiveness estimates. Channels with wide posterior distributions (high uncertainty) should contribute less to our confidence in optimization results, especially if those uncertain channels receive large budget allocations.

## Formula

```
Model Confidence = Σ (channel_confidence × spend_weight)
```

Where:
- `channel_confidence` = confidence for each channel based on posterior uncertainty
- `spend_weight` = proportion of total optimized spend allocated to that channel

## Channel Confidence Calculation

For each channel, confidence is derived from the **coefficient of variation (CV)** of the posterior samples for the channel's beta coefficient:

```python
CV = std(beta_samples) / |mean(beta_samples)|

channel_confidence = clip(1 - CV, 0.3, 1.0)
```

### Interpretation

| CV Value | Channel Confidence | Meaning |
|----------|-------------------|---------|
| 0.0 | 1.00 (100%) | Perfect certainty - all posterior samples agree |
| 0.3 | 0.70 (70%) | Moderate uncertainty |
| 0.5 | 0.50 (50%) | High uncertainty |
| 0.7+ | 0.30 (30%) | Very high uncertainty (capped at floor) |

### Why Coefficient of Variation?

CV normalizes the standard deviation by the mean, making it scale-invariant. A channel with beta mean of 1000 and std of 100 has the same CV (0.1) as a channel with mean 10 and std 1. This allows fair comparison across channels with different effect sizes.

## Spend-Weighted Aggregation

The overall model confidence weights each channel's confidence by its share of the optimized budget:

```python
total_spend = sum(optimized_allocation.values())
weighted_confidence = 0.0

for channel, spend in optimized_allocation.items():
    spend_weight = spend / total_spend
    weighted_confidence += channel_confidence[channel] * spend_weight
```

### Why Spend-Weighting?

If the optimizer recommends putting 80% of budget into a channel with high uncertainty, that should significantly reduce our overall confidence. Conversely, uncertainty in a channel receiving only 5% of budget matters less.

## Implementation Example

```python
import numpy as np

def calculate_model_confidence(
    posterior_samples,  # shape: (n_samples, n_channels)
    optimized_allocation: dict[str, float],
    channel_names: list[str],
) -> float:
    """
    Calculate spend-weighted model confidence from posterior uncertainty.

    Parameters
    ----------
    posterior_samples : np.ndarray
        Beta coefficient samples, shape (n_samples, n_channels)
    optimized_allocation : dict
        {channel_name: spend_amount} from optimization
    channel_names : list
        Channel names in order matching posterior_samples columns

    Returns
    -------
    float
        Overall model confidence between 0.3 and 1.0
    """
    total_spend = sum(optimized_allocation.values())
    if total_spend <= 0:
        return 0.8  # Default if no spend

    weighted_confidence = 0.0

    for ch_idx, channel in enumerate(channel_names):
        spend = optimized_allocation.get(channel, 0)
        if spend <= 0:
            continue

        # Get beta samples for this channel
        beta_samples = posterior_samples[:, ch_idx]

        # Calculate coefficient of variation
        mean = np.mean(beta_samples)
        std = np.std(beta_samples)

        if abs(mean) < 1e-10:
            # Mean near zero indicates unreliable estimate
            channel_conf = 0.5
        else:
            cv = std / abs(mean)
            # Confidence = 1 - CV, clipped to [0.3, 1.0]
            channel_conf = np.clip(1 - cv, 0.3, 1.0)

        # Weight by spend allocation
        spend_weight = spend / total_spend
        weighted_confidence += channel_conf * spend_weight

    return float(np.clip(weighted_confidence, 0.3, 1.0))
```

## Accessing Posterior Samples in PyMC-Marketing

In our MMM platform, posterior samples are extracted from the fitted model's InferenceData:

```python
from mmm_platform.optimization.risk_objectives import PosteriorSamples

# Extract samples from fitted model
posterior_samples = PosteriorSamples.from_idata(mmm.idata, n_samples=500)

# Access beta samples: shape (n_samples, n_channels)
beta_samples = posterior_samples.beta_samples

# Channel order matches the model's channel configuration
channel_names = wrapper.config.channels  # or similar
```

The `saturation_beta` parameter in PyMC-Marketing represents the channel effectiveness coefficient after saturation transformation.

## Use Cases

1. **Optimizer Result Discounting**: Reduce expected uplift when the model is uncertain about key channels
2. **Reporting Confidence Intervals**: Communicate uncertainty to stakeholders
3. **Channel Prioritization**: Flag channels where more data collection would improve model certainty
4. **Model Comparison**: Compare confidence across different model specifications

## Limitations

- Assumes posterior samples are available (requires MCMC, not just point estimates)
- CV can be misleading when mean is close to zero
- Doesn't account for correlation between channel estimates
- Floor of 0.3 is arbitrary but prevents complete discounting

## References

- Coefficient of Variation: https://en.wikipedia.org/wiki/Coefficient_of_variation
- Bayesian Credible Intervals: https://en.wikipedia.org/wiki/Credible_interval
- PyMC-Marketing MMM: https://www.pymc-marketing.io/
