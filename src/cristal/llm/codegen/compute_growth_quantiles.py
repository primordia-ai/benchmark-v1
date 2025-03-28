from typing import Union
import xarray as xr
import cristal.synthetic as synth

def compute_growth_quantiles_codegen(
        self,
        quantiles: tuple[float] = (0.025, 0.16, 0.5, 0.84, 0.975),

        asset: Union[xr.Dataset, str] | None = None,
    ) -> None:
        """
        Compute the growth quantiles for either a specific asset, or if None, all assets
        Returns None because the values will be updated onto the stocks dataset

        :param quantiles:   List of quantiles to compute
        :param asset:       Optional asset to compute for
        :return:            None
        """
        if asset is None:
            for asset in self.stocks.asset.values:
                self.compute_growth_quantiles_codegen(quantiles, self.stocks.sel(asset=asset))
        else:
            # Compute the quantiles
            # We use the known Mixture means and stds, and then the asset mixture weights
            prior = None
            quantile_values = synth.compute_growth_quantiles_codegen(prior, list(quantiles))
            self._update_quantile_pred(asset, quantile_values)