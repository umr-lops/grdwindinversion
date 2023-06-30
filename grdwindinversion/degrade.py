import os,sys
import numpy as np
from scipy.ndimage import generic_filter
from shutil import copyfile
import xarray as xr

import logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import matplotlib.pyplot as plt

def mean(subArray):
    return np.nanmean(subArray)

def createLowerResL2(inputfile, resolution):
    """

    :param inputfile: str
    :param resolution: int e.g. 100 in meters
    :return:
    """
    outputfile = inputfile.replace(".nc","__"+str(resolution)+".nc")
    if (os.path.exists(outputfile)):
        print("ok", outputfile, "already existing")
        return
    if (os.path.exists(inputfile) == False):
        print("not ok", inputfile, "not existing")
        return


    try:
        ds_base = xr.open_dataset(inputfile)
    except Exception as e:
        logger.error('cannot open file %s : %s. Exiting' % (inputfile,  str(e)))
        sys.exit(1)

    ds_res = xr.Dataset()
    ds_res = ds_res.assign_coords(ds_base.coords)
    ds_res = ds_res.assign_attrs(ds_base.attrs)

    logger.debug("Generating footprint...")
    # The footprint is circle-like shape. Its radius depends on the wanted resolution
    # Each data point will be assigned as value the mean of the other data within the footprint
    footprint = np.ones((resolution, resolution))
    a, b = resolution / 2, resolution / 2
    r = resolution / 2
    if resolution % 2 == 0:
        a = a - 0.5
        b = b - 0.5
        r = r - 0.5
    epsilon = resolution * (30.0/100.0)
    for x in range(0, len(footprint)):
        for y in range(0, len(footprint[x])):
            if ((x-a)**2 + (y-b)**2) >= r**2 + epsilon:
                footprint[x][y] = 0
    logger.debug("Finished generating footprint.")

    ## mask
    mask_land = np.ma.getmaskarray(ds_base.owiLandFlag)
    for varName in ds_base.variables:
        if (varName in ["owiAzSize","owiRaSize","spatial_ref","owiLandFlag","owiMask","pol"]):
            ds_res[varName] = ds_base[varName]
            continue
        dims = ds_base[varName].dims
        attrs =  ds_base[varName].attrs

        mask_var_final = (np.isnan(ds_base[varName]) | mask_land).values
        ds_base[varName].values[mask_var_final] = np.nan

        arrayOut = np.empty_like(ds_base[varName])

        generic_filter(ds_base[varName].values, mean, footprint=footprint, mode='constant', output=arrayOut, cval=np.nan)#Computing lower resolution

        ds_res[varName] = xr.DataArray(
            data=arrayOut,
            dims=ds_base[varName].dims,
            coords=ds_base[varName].coords,
            attrs=ds_base[varName].attrs,
        )
        ds_res[varName].attrs["comment_resolution"] = "variable at " + str(resolution) + "km resolution"

    #plt.figure()
    #plt.pcolormesh(ds_res[varName],cmap='jet') ; plt.colorbar()
    #plt.title(varName)

    logger.debug("Finished to compute multi-res data. Closing files...")
    ds_res.to_netcdf(outputfile)
    df = ds_res.to_dataframe()
    df = df.assign(**ds_res.attrs)
    df.reset_index(drop=False,inplace=True)
    df.to_csv(outputfile.replace(".nc",".csv"))

    logging.info("ok outputfile=%s", outputfile)

    ds_res.close()
    ds_base.close()
    logger.debug("Files closed. Exiting.")

