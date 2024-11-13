import numpy as np
import pandas as pd
from scipy import stats


def plot_beta_dist(
    alpha: int,
    beta: int,
    size: int = 100,
    from_cdf_prob: float = 0.001,
    to_cdf_prob: float = 0.999,
) -> pd.DataFrame:
    x = np.linspace(
        stats.beta.ppf(
            from_cdf_prob, alpha, beta
        ),  # get the x value for which P(X <= x) = 0.01
        stats.beta.ppf(
            to_cdf_prob, alpha, beta
        ),  # get the x value for which P(X <= x) = 0.99
        size,
    )

    return pd.DataFrame(dict(x=x)).assign(
        pdf=lambda x: stats.beta.pdf(x.x, alpha, beta),
    )
