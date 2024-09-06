import xarray as xr
import xsarsea.gradients
import xarray as xr
from scipy.ndimage import binary_dilation
import numpy as np


def get_streaks(xr_dataset, xr_dataset_100):
    """
    Get the streaks from the wind field.

    Parameters
    ----------
    xr_dataset : xarray.Dataset
        dataset at user resolution.
    xr_dataset_100 : xarray.Dataset
        dataset at 100m resolution.

    Returns
    -------
    xarray.Dataset
        Extract wind direction from Koch Method using xsarsea tools.
    """

    # return empy dataArray, waiting for solution
    return xr.DataArray(data=np.nan * np.ones([len(xr_dataset.coords[dim]) for dim in ['line','sample']]),
                        dims=['line','sample'],
                        coords=[xr_dataset.coords[dim] for dim in ['line','sample']])
    #

    """
    gradients = xsarsea.gradients.Gradients(xr_dataset_100['sigma0_detrend'], windows_sizes=[
                                            1600, 3200], downscales_factors=[1, 2], window_step=1)

    # get gradients histograms as an xarray dataset
    hist = gradients.histogram

    # get orthogonals gradients
    hist['angles'] = hist['angles'] + np.pi/2

    # mean
    hist_mean = hist.mean(['downscale_factor', 'window_size', 'pol'])

    # smooth
    hist_mean_smooth = hist_mean.copy()
    hist_mean_smooth['weight'] = xsarsea.gradients.circ_smooth(
        hist_mean['weight'])

    # smooth only
    # hist_smooth = hist.copy()
    # hist_smooth['weight'] = xsarsea.gradients.circ_smooth(hist_smooth['weight'])

    # select histogram peak
    iangle = hist_mean_smooth['weight'].fillna(0).argmax(dim='angles')
    streaks_dir = hist_mean_smooth.angles.isel(angles=iangle)
    streaks_weight = hist_mean_smooth['weight'].isel(angles=iangle)
    streaks = xr.merge(
        [dict(angle=streaks_dir, weight=streaks_weight)]).drop('angles')

    # streaks are [0, pi]. Remove ambiguity with anciallary wind
    ancillary_wind = xr_dataset_100['ancillary_wind'].sel(line=streaks.line,
                                                          sample=streaks.sample,
                                                          method='nearest').compute()
    streaks_c = streaks['weight'] * np.exp(1j * streaks['angle'])
    diff_angle = xr.apply_ufunc(np.angle, ancillary_wind / streaks_c)
    streaks_c = xr.where(np.abs(diff_angle) > np.pi/2, -streaks_c, streaks_c)
    streaks['weight'] = np.abs(streaks_c)
    streaks['angle'] = xr.apply_ufunc(np.angle, streaks_c)

    streaks_dir = xr.apply_ufunc(
        np.angle, streaks_c.interp(line=xr_dataset.line, sample=xr_dataset.sample))
    streaks_dir = xr.where(
        xr_dataset['land_mask'], np.nan, streaks_dir)
    streaks_dir.attrs['comment'] = 'angle in radians, anticlockwise, 0=line'
    streaks_dir.attrs['description'] = 'wind direction estimated from local gradient, and direction ambiguity removed with ancillary wind'
    
    return streaks_dir
    
    """
