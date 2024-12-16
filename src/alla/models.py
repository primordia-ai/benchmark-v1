"""
ALLA Models
"""
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy import stats
from tqdm import tqdm

from alla.utils import stats as alla_stats, synthetic as synth


class ALLA:
    """
    Active Lifelong Learning Architecture

    For our benchmarking experiment purposes.
    Stores a belief model of stock classes and their indicator likelihoods, etc.
    Stocks are one of High Growth, Stable, or Time Bomb.
    """

    def __init__(self, stocks_dataset: xr.Dataset, asset_memory_depth: int = 20):
        """
        Copy the stocks Dataset, and add new dimensions for ALLA's class prediction, and quantile predictions

        :param stocks_dataset:      xr.Dataset from synth.generate_synthetic_stocks
        :param asset_memory_depth:  How many Bayesian updates to remember for each asset
        """
        # Copy the stocks Dataset and add new dimension for ALLA's class prediction, and quantile predictions
        self.stocks: xr.Dataset = stocks_dataset.copy()
        self.asset_memory_depth = asset_memory_depth

        self.n_assets = len(self.stocks.asset)
        self.classes = {
            "high_growth": "High Growth",
            "stable": "Stable",
            "time_bomb": "Time Bomb"
        }
        self.class_idx = {
            "high_growth": 0,
            "stable": 1,
            "time_bomb": 2
        }
        self.n_classes = len(self.classes)

        # Known growth params
        self.mu_g = synth.GROWTH_MU
        self.sigma_g = synth.GROWTH_SCALE

        # Ordered list of soft indicator names
        self.soft_indicators = stocks_dataset.coords['indicator'].values[
            stocks_dataset.coords['indicator_type'].values == 'soft'
        ]
        self.n_soft = len(self.soft_indicators)

        # Get the soft likelihood dataset. Dimensions are indicator, asset-class, and color
        self.soft_likelihoods: xr.Dataset = synth.get_soft_indicator_xarray()

        # Init our results data, essentially
        self.indicator_utility_avg = None
        self.reset()

    def reset(self):
        """
        Reset the model to its initial state
        """
        # Quantile predictions -- All values initialized to 0
        # Updates will push oldest values out, where idx 0 is oldest
        # TODO -- update these to actual quantiles
        self.stocks['quantile_pred'] = xr.DataArray(
            np.zeros((self.stocks.sizes['asset'], 3, self.asset_memory_depth)),
            dims=['asset', 'quantile', 'time'],
            coords={
                # 'quantile': [0.025, 0.16, 0.5, 0.84, 0.975],
                'quantile': [0.33, 0.5, 0.66],
                'time': list(range(self.asset_memory_depth))
            }
        )
        # Asset priors -- All values initialized to 1/3 (across time as well)
        # Updates will push oldest values out, where idx 0 is oldest
        self.stocks['asset_prior'] = xr.DataArray(
            np.full((self.n_assets, self.n_classes, self.asset_memory_depth), 1 / self.n_classes),
            dims=['asset', 'asset_class', 'time'],
            coords={
                'asset_class': list(self.classes.keys()),
                'time': list(range(self.asset_memory_depth))
            },
        )

        # Indicator utility -- how much information in each indicator (for ranking)
        # We use a running average where [i, 0] and [i, 1] are the number of observations and current mean
        # for indicator i, respectively
        self.indicator_utility_avg = np.zeros((self.n_soft, 2))
        self.indicator_utility_avg[:, 1] = -np.inf

    def _update_quantile_pred(self, asset: xr.Dataset, quantile_pred):
        """
        Update the quantile history

        :param asset: xr.Dataset for the selected asset
        """
        # Update the quantile predictions -- place the new value at the end
        self.stocks['quantile_pred'].loc[asset.asset] = np.roll(
            self.stocks['quantile_pred'].loc[asset.asset], -1, axis=1
        )
        self.stocks['quantile_pred'].loc[asset.asset][:, -1] = quantile_pred

    def _update_asset_prior(self, asset: xr.Dataset, asset_prior):
        """
        Update the asset prior history

        :param asset: xr.Dataset for the selected asset
        """
        # Update the asset priors -- place the new value at the end
        self.stocks['asset_prior'].loc[asset.asset] = np.roll(
            self.stocks['asset_prior'].loc[asset.asset], -1, axis=1
        )
        self.stocks['asset_prior'].loc[asset.asset][:, -1] = asset_prior

    def select_next_indicator(self, prior, candidates):
        """
        Select next indicator using expected log Bayes factor for growth prediction

        Also updates observed indicator utility average
        """
        max_utility = -float('inf')
        best_j = None

        # Current asset-class probabilities
        p_k = prior / np.sum(prior)

        # Current growth mixture params
        # prior_mean, prior_std = alla_stats.mixture_stats(p_k, self.mu_g, self.sigma_g)

        for j in candidates:
            utility = 0

            indicator_name = self.soft_indicators[j]
            class_color_matrix = self.soft_likelihoods[indicator_name].values  # [classes, colors]
            # Sum over the colors for each asset-class
            # class_likelihood = i_likes.sum(axis=0)
            # class_likelihood /= class_likelihood.sum()
            color_likelihoods = np.dot(p_k, class_color_matrix)  # [p(red|p_k), p(yellow|p_k), p(green|p_k)]

            # TODO -- optimize
            for ic, c in enumerate(["red", "yellow", "green"]):
                # P(c | p_k)
                p_col = color_likelihoods[ic]

                if p_col > 0:
                    # Posterior class probabilities if we observe color
                    cls_likelihood = class_color_matrix[:, ic]
                    gamma = p_k * cls_likelihood
                    if np.sum(gamma) > 1e-6:
                        gamma = gamma / np.sum(gamma)

                        # Expected log Bayes factor is expected log ratio of the densities
                        # Note: we need to integrate over possible g values
                        g_points = np.linspace(
                            min(self.mu_g - 3 * max(self.sigma_g)),
                            max(self.mu_g + 3 * max(self.sigma_g)),
                            100
                        )

                        for g in g_points:
                            # P(g|posterior)
                            post_prob = sum(
                                gamma[k] * stats.norm.pdf(g, self.mu_g[k], self.sigma_g[k])
                                    for k in range(self.n_classes)
                            )
                            # P(g|prior)
                            prior_prob = sum(
                                p_k[k] * stats.norm.pdf(g, self.mu_g[k], self.sigma_g[k])
                                    for k in range(self.n_classes)
                            )

                            if post_prob > 0 and prior_prob > 0:
                                # Weight by posterior probability of g
                                utility += p_col * post_prob * (np.log(post_prob) - np.log(prior_prob)) * \
                                           (g_points[1] - g_points[0])

            # Update the utility of this indicator
            if self.indicator_utility_avg[j, 0] == 0:
                # First observation
                self.indicator_utility_avg[j, 0] = 1
                self.indicator_utility_avg[j, 1] = utility
                avg_utility = utility
            else:
                # Running average
                self.indicator_utility_avg[j, 0] += 1
                self.indicator_utility_avg[j, 1] += ((utility - self.indicator_utility_avg[j, 1]) /
                                                     self.indicator_utility_avg[j, 0])
                avg_utility = self.indicator_utility_avg[j, 1]

            if avg_utility > max_utility:
                max_utility = avg_utility
                best_j = j

        return best_j, max_utility

    def process_asset(self, asset: Union[xr.Dataset, str], soft_only=True) -> dict:
        """
        Process an asset to infer its class and growth forecast
        """
        if not soft_only:
            raise NotImplementedError("Only soft indicators are supported for now.")

        if isinstance(asset, str):
            # Get the asset by its name
            asset = self.stocks.sel(asset=asset)

        # Get the soft indicators, by masking with indicator_type
        observed_colors = asset.ryg.values[asset.indicator_type.values == 'soft']
        # Get most recent prior
        prior = self.stocks.asset_prior.sel(asset=asset.asset).values[:, -1]
        candidates = list(range(self.n_soft))

        selected_js = []
        utilities = []
        prior_history = []
        growth_means = []
        growth_stds = []

        for count in tqdm(range(self.n_soft), desc="Soft Ind."):
            p_k = prior.copy()
            prior_history.append(p_k)

            # Track growth predictions
            mean_g, std_g = alla_stats.mixture_stats(p_k, self.mu_g, self.sigma_g)
            growth_means.append(mean_g)
            growth_stds.append(std_g)

            j, utility = self.select_next_indicator(prior, candidates)
            utilities.append(utility)
            selected_js.append(j)
            candidates.remove(j)
            indicator_name = self.soft_indicators[j]

            # Observed
            observed_color = observed_colors[j]

            # Inference
            # Likelihood of asset-class given color
            likelihood = self.soft_likelihoods[indicator_name].sel(color=observed_color).values  # [3,]
            posterior = p_k * likelihood
            posterior /= np.sum(posterior)
            prior = posterior * np.sum(prior)
            prior /= np.sum(prior)

            # Update the stocks dataset with class probabilities and growth quantiles
            self._update_asset_prior(asset, prior)
            # TODO -- Calculate the quantiles properly
            quantiles = [
                mean_g - std_g,
                mean_g,
                mean_g + std_g
            ]
            self._update_quantile_pred(asset, quantiles)

        # return {
        #     "selected_js": selected_js,
        #     "utilities": utilities,
        #     "class_probs": class_probs,
        #     "prior_history": prior_history,
        #     "growth_means": growth_means,
        #     "growth_stds": growth_stds,
        # }

    def plot_prior_history(self, asset: str, save_path: str | None = None):
        """
        Plot the history of an asset's prior class probabilities and growth forecast
        TODO -- Should be separated out of ALLA to a plot.py file

        :param asset:       str identifying the asset
        :param save_path:   path to save the plot
        :return:
        """
        asset = self.stocks.sel(asset=asset)
        asset_class = str(asset.category.values)
        time = self.stocks.time.values

        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

        # Plot class probabilities
        class_probs = self.stocks.asset_prior.sel(asset=asset.asset).values
        for name, idx in self.class_idx.items():
            lbl = self.classes[name]
            if name == asset_class:
                # True class
                color = 'r'
                ls = '-'
                lbl += " (true)"
                true_mu_g = self.mu_g[idx]
            else:
                color = 'gray'
                ls = '--'

            ax2.plot(time, class_probs[idx, :], color=color, linestyle=ls, label=lbl)

            # ax2.fill_between(steps,
            #                  np.clip(class_probs[:, k] - 2 * std, 0, 1),
            #                  np.clip(class_probs[:, k] + 2 * std, 0, 1),
            #                  color=color, alpha=0.2)

        ax2.set_xlabel('Step')
        ax2.set_ylabel('Class Probability')
        ax2.legend()
        ax2.grid(True)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        # # Plot growth forecast
        # growth_means = np.array(a0_results['growth_means'])
        # growth_stds = np.array(a0_results['growth_stds'])
        #
        # # Plot utilities
        # ax1.plot(steps, a0_results['utilities'], 'b-', label='Expected Log Bayes Factor for Growth')
        # ax1.set_xlabel('Step')
        # ax1.set_ylabel('Expected Utility')
        # ax1.legend()
        # ax1.grid(True)
        # ax1.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        if save_path:
            plt.savefig(save_path)
            print(f"Saving plot to {save_path}")

        plt.show()
