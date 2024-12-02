import xsarsea.gradients
import cv2
import xarray as xr
import xarray as xr
from scipy.ndimage import binary_dilation
import numpy as np
import logging

import logging
logger = logging.getLogger('grdwindinversion.gradientFeatures')
logger.addHandler(logging.NullHandler())


class GradientFeatures:
    def __init__(self, xr_dataset, xr_dataset_100, windows_sizes, downscales_factors, window_step=1):
        """
        Initialize variables and xsarsea.gradients.Gradients.

        Parameters
        ----------
        xr_dataset : xarray.Dataset
            xarray.Dataset containing the SAR data.
        xr_dataset_100 : xarray.Dataset
            xarray.Dataset containing the 100m resolution SAR data.
        windows_sizes : list
            List of window sizes for gradient computation.
        downscales_factors : list
            List of downscale factors for gradient computation.
        window_step : int
            Step size for the window (default is 1).

        Returns
        -------
        None
        """
        self.xr_dataset = xr_dataset
        self.xr_dataset_100 = xr_dataset_100
        self.windows_sizes = windows_sizes
        self.downscales_factors = downscales_factors
        self.window_step = window_step
        self.gradients = None
        self.hist = None
        self._compute_gradients()

    def _compute_gradients(self):
        """
        Instantiate the gradients object and compute the histogram.

        Parameters
        ----------
        None

        Returns
        -------
        None

        """
        self.gradients = xsarsea.gradients.Gradients(
            self.xr_dataset_100['sigma0_detrend'],
            windows_sizes=self.windows_sizes,
            downscales_factors=self.downscales_factors,
            window_step=self.window_step
        )
        self.hist = self.gradients.histogram
        # Get orthogonal gradients
        self.hist['angles'] = self.hist['angles'] + np.pi / 2

    def get_heterogeneity_mask(self, config):
        """
        Compute the heterogeneity mask.

        Parameters
        ----------
        config : dict
            Configuration parameters.

        Returns
        -------
        dict
            Dictionary containing the dataArrays related toheterogeneity mask.

        """
        dual_pol = config["l2_params"]["dual_pol"]

        new_dataArrays = {}

        try:

            sigma0_400_co = [da.sigma0 for da in self.gradients.gradients_list if (
                da.sigma0["pol"] == config["l2_params"]["copol"] and da.sigma0.downscale_factor == 4)][0]
            sigs = [sigma0_400_co]

            if dual_pol:
                sigma0_800_cross = [da.sigma0 for da in self.gradients.gradients_list if (
                    da.sigma0["pol"] == config["l2_params"]["crosspol"] and da.sigma0.downscale_factor == 8)][0]
                sigs.append(sigma0_800_cross)

            filters = {}
            for sig in sigs:

                pol = sig["pol"].values
                res = 100 * sig.downscale_factor.values

                # delete useless coords : could be problematic to have it later
                if 'downscale_factor' in sig.coords:
                    sig = sig.reset_coords("downscale_factor", drop=True)

                if 'window_size' in sig.coords:
                    sig = sig.reset_coords("window_size", drop=True)
                # mask
                sig = xr.where(sig <= 0, 1e-15, sig)

                # map incidence for detrend
                incidence = xr.DataArray(data=cv2.resize(
                    self.xr_dataset_100.incidence.values, sig.shape[::-1], cv2.INTER_NEAREST), dims=sig.dims, coords=sig.coords)

                sigma0_detrend = xsarsea.sigma0_detrend(sig, incidence)

                filter_name = str(res)+"_"+str(pol)
                I = sigma0_detrend
                f1, f2, f3, f4, f = xsarsea.gradients.filtering_parameters(I)
                filters[filter_name] = f

            thresholds = [0.78]  # < is unusable
            if dual_pol:
                # Seuil pour crosspol si dual_pol est activé
                thresholds.append(0.71)

            for idx_filter, filter in enumerate(filters):
                # interp to user resolution and map on dataset grid
                new_dataArrays[filter] = filters[filter].interp(
                    line=self.xr_dataset.line, sample=self.xr_dataset.sample, method="nearest")
                new_dataArrays[filter+"_mask"] = xr.where(
                    new_dataArrays[filter] > thresholds[idx_filter], True, False)

            varname_400_copol_mask = f'400_{config["l2_params"]["copol"]}_mask'
            varname_800_crosspol_mask = f'800_{config["l2_params"]["crosspol"]}_mask'

            # Cas 0 : no heterogeneity
            new_dataArrays["heterogeneity_mask"] = xr.full_like(
                new_dataArrays[varname_400_copol_mask], 0)

            if dual_pol:
                # Cas 3 : Dual-polarization
                new_dataArrays["heterogeneity_mask"] = xr.where(
                    new_dataArrays[varname_400_copol_mask] & new_dataArrays[varname_800_crosspol_mask], 3, new_dataArrays["heterogeneity_mask"])

                # Cas 1 : Co-polarization only
                new_dataArrays["heterogeneity_mask"] = xr.where(
                    new_dataArrays[varname_400_copol_mask] & ~new_dataArrays[varname_800_crosspol_mask], 1, new_dataArrays["heterogeneity_mask"])

                # Cas 2 : Cross-polarization only
                new_dataArrays["heterogeneity_mask"] = xr.where(
                    ~new_dataArrays[varname_400_copol_mask] & new_dataArrays[varname_800_crosspol_mask], 2, new_dataArrays["heterogeneity_mask"])

                # Attributes
                new_dataArrays["heterogeneity_mask"].attrs["valid_range"] = np.array([
                    0, 3])
                new_dataArrays["heterogeneity_mask"].attrs["flag_values"] = np.array([
                    0, 1, 2, 3])
                new_dataArrays["heterogeneity_mask"].attrs["flag_meanings"] = (
                    "homogeneous_NRCS, heterogeneous_from_co-polarization_NRCS, "
                    "heterogeneous_from_cross-polarization_NRCS, heterogeneous_from_dual-polarization_NRCS"
                )
            else:
                # no crosspol
                new_dataArrays["heterogeneity_mask"] = xr.where(
                    new_dataArrays[varname_400_copol_mask], 1, new_dataArrays["heterogeneity_mask"])

                # Attributs pour le cas single-pol
                new_dataArrays["heterogeneity_mask"].attrs["valid_range"] = np.array([
                    0, 1])
                new_dataArrays["heterogeneity_mask"].attrs["flag_values"] = np.array([
                    0, 1])
                new_dataArrays["heterogeneity_mask"].attrs["flag_meanings"] = (
                    "homogeneous_NRCS, heterogeneous_from_co-polarization_NRCS"
                )

            # Attributs généraux
            new_dataArrays["heterogeneity_mask"].attrs["long_name"] = "Quality flag taking into account the local heterogeneity"
            return new_dataArrays

        except Exception as e:
            logging.error("Error in get_heterogeneity_mask: %s", e)

            new_dataArrays["heterogeneity_mask"] = xr.DataArray(data=np.nan * np.ones([len(self.xr_dataset.coords[dim]) for dim in ['line', 'sample']]),
                                                                dims=[
                'line', 'sample'],
                coords=[self.xr_dataset.coords[dim]
                        for dim in ['line', 'sample']],
                attrs={"comment": "no heterogeneity mask found"})

            return new_dataArrays

    def _remove_ambiguity(self, streaks):
        """
        Remove direction ambiguity using ancillary wind data.

        Parameters
        ----------
        streaks : xarray.Dataset
            Dataset containing the streaks.

        Returns
        -------
        xarray.Dataset
            Dataset containing the streaks with ambiguity removed.
        """

        # Load ancillary wind in antenna convention
        ancillary_wind = self.xr_dataset['ancillary_wind'].interp(
            line=streaks.line,
            sample=streaks.sample,
            method='nearest'
        ).compute()

        # Convert angles to complex numbers
        streaks_c = streaks['weight'] * np.exp(1j * streaks['angle'])
        # Calculate the difference in angle
        diff_angle = xr.apply_ufunc(np.angle, ancillary_wind / streaks_c)

        # Remove ambiguity
        streaks_c = xr.where(np.abs(diff_angle) > np.pi /
                             2, -streaks_c, streaks_c)

        # Update streaks with corrected values
        streaks['weight'] = np.abs(streaks_c)
        streaks['angle'] = xr.apply_ufunc(np.angle, streaks_c)
        return streaks

    def convert_to_meteo_convention(self, streaks):
        """
        Convert wind direction to meteorological convention by creating a new 'angle' DataArray.

        Parameters
        ----------
        streaks : xarray.Dataset
            Dataset containing the streaks.

        Returns
        -------
        xarray.Dataset
            Dataset containing the streaks with wind direction in meteorological convention.

        """
        streaks_meteo = self.xr_dataset[['longitude', 'latitude', 'ground_heading', 'ancillary_wind']].interp(
            line=streaks.line,
            sample=streaks.sample,
            method='nearest')

        streaks_meteo['angle'] = xsarsea.dir_sample_to_meteo(
            np.rad2deg(streaks['angle']), streaks_meteo['ground_heading'])
        streaks_meteo['angle'].attrs[
            'winddir_convention'] = "Wind direction in meteorological convention (clockwise, from), ex: 0°=from north, 90°=from east"

        return streaks_meteo

    def streaks_smooth_mean(self):
        """
        Compute streaks by smoothing the histograms first and then computing the mean.

        Parameters
        ----------
        None

        Returns
        -------
        xarray.DataArray
            DataArray containing the streaks.           
        """

        try:
            hist_smooth = self.hist.copy()
            hist_smooth['weight'] = xsarsea.gradients.circ_smooth(
                hist_smooth['weight'])

            # Compute the mean across 'downscale_factor', 'window_size', and 'pol'
            hist_smooth_mean = hist_smooth.mean(
                ['downscale_factor', 'window_size', 'pol'])

            # Select histogram peak
            iangle_smooth_mean = hist_smooth_mean['weight'].fillna(
                0).argmax(dim='angles')
            streaks_dir_smooth_mean = hist_smooth_mean['angles'].isel(
                angles=iangle_smooth_mean)
            streaks_weight_smooth_mean = hist_smooth_mean['weight'].isel(
                angles=iangle_smooth_mean)

            # Combine angles and weights into a dataset
            streaks_smooth_mean = xr.Dataset({
                'angle': streaks_dir_smooth_mean,
                'weight': streaks_weight_smooth_mean
            })

            # Remove 'angles' coordinate
            streaks_smooth_mean = streaks_smooth_mean.reset_coords(
                'angles', drop=True)

            # Remove ambiguity with ancillary wind
            streaks_smooth_mean = self._remove_ambiguity(
                streaks_smooth_mean)

            # Convert to meteo convention
            streaks_smooth_mean = self.convert_to_meteo_convention(
                streaks_smooth_mean)

            # Set attributes
            streaks_smooth_mean['angle'].attrs['description'] = 'Wind direction estimated from local gradient; histograms smoothed first, then mean computed'

            return streaks_smooth_mean

        except Exception as e:
            logging.error("Error in streaks_smooth_mean: %s", e)

            streaks_dir_smooth_mean_interp = xr.DataArray(data=np.nan * np.ones([len(self.xr_dataset.coords[dim]) for dim in ['line', 'sample']]),
                                                          dims=[
                'line', 'sample'],
                coords=[self.xr_dataset.coords[dim]
                        for dim in ['line', 'sample']],
                attrs={"comment": "no streaks_smooth_mean found"})

            return streaks_dir_smooth_mean_interp

    def streaks_mean_smooth(self):
        """
        Compute streaks by meaning the histograms first and then smoothing.

        Parameters
        ----------
        None

        Returns
        -------
        xarray.DataArray
            DataArray containing the streaks.           
        """
        try:
            # Compute the mean of the histograms
            hist_mean = self.hist.copy().mean(
                ['downscale_factor', 'window_size', 'pol'])

            # Smooth the mean histogram
            hist_mean_smooth = hist_mean.copy()
            hist_mean_smooth['weight'] = xsarsea.gradients.circ_smooth(
                hist_mean['weight'])

            # Select histogram peak
            iangle_mean_smooth = hist_mean_smooth['weight'].fillna(
                0).argmax(dim='angles')
            streaks_dir_mean_smooth = hist_mean_smooth['angles'].isel(
                angles=iangle_mean_smooth)
            streaks_weight_mean_smooth = hist_mean_smooth['weight'].isel(
                angles=iangle_mean_smooth)

            # Combine angles and weights into a dataset
            streaks_mean_smooth = xr.Dataset({
                'angle': streaks_dir_mean_smooth,
                'weight': streaks_weight_mean_smooth
            })

            # Remove 'angles' coordinate
            streaks_mean_smooth = streaks_mean_smooth.reset_coords(
                'angles', drop=True)

            # Remove ambiguity with ancillary wind
            streaks_mean_smooth = self._remove_ambiguity(
                streaks_mean_smooth)

            # Convert to meteo convention
            streaks_mean_smooth = self.convert_to_meteo_convention(
                streaks_mean_smooth)

            # Set attributes
            streaks_mean_smooth['angle'].attrs['description'] = 'Wind direction estimated from local gradient; histograms mean first, then smooth computed'

            return streaks_mean_smooth

        except Exception as e:
            logging.error("Error in streaks_mean_smooth: %s", e)

            streaks_mean_smooth = xr.DataArray(data=np.nan * np.ones([len(self.xr_dataset.coords[dim]) for dim in ['line', 'sample']]),
                                               dims=[
                'line', 'sample'],
                coords=[self.xr_dataset.coords[dim]
                        for dim in ['line', 'sample']],
                attrs={"comment": "no streaks_mean_smooth found"})

            return streaks_mean_smooth

    def streaks_individual(self):
        """
        Compute streaks by smoothing the histogram.

        Parameters
        ----------
        None

        Returns
        -------
        xarray.DataArray
            DataArray containing the individual streaks for each window_size, downscale_factor, polarisation (no combination).           
        """
        try:
            # Compute the mean of the histograms
            hist_smooth = self.hist.copy()
            hist_smooth['weight'] = xsarsea.gradients.circ_smooth(
                hist_smooth['weight'])

            # Select histogram peak for each individual solution
            iangle_individual = hist_smooth['weight'].fillna(
                0).argmax(dim='angles')
            streaks_dir_individual = hist_smooth['angles'].isel(
                angles=iangle_individual)
            streaks_weight_individual = hist_smooth['weight'].isel(
                angles=iangle_individual)
            # Combine angles and weights into a dataset
            streaks_individual = xr.Dataset({
                'angle': streaks_dir_individual,
                'weight': streaks_weight_individual
            })
            # Remove 'angles' coordinate
            streaks_individual = streaks_individual.reset_coords(
                'angles', drop=True)

            # Remove ambiguity with ancillary wind for each individual solution
            streaks_individual = self._remove_ambiguity(
                streaks_individual)

            # Convert to meteo convention
            streaks_individual = self.convert_to_meteo_convention(
                streaks_individual)

            # Set attributes
            streaks_individual['angle'].attrs['description'] = 'Wind direction estimated from local gradient for each individual solution; histograms smoothed individually'

            return streaks_individual

        except Exception as e:
            logging.error("Error in streaks_individual: %s", e)

            streaks_individual = xr.DataArray(data=np.nan * np.ones([len(self.xr_dataset.coords[dim]) for dim in ['line', 'sample']]),
                                              dims=[
                'line', 'sample'],
                coords=[self.xr_dataset.coords[dim]
                        for dim in ['line', 'sample']],
                attrs={"comment": "no streaks_individual found"})

            return streaks_individual
