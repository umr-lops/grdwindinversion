# To place here in the code to not have errors with cv2.
#  if placed in main => error ..
import tempfile
import traceback
import xsar
import xsarsea
from xsarsea import windspeed
import grdwindinversion
import xarray as xr
import numpy as np
import sys
import datetime
import yaml
from scipy.ndimage import binary_dilation
import re
import os
import logging
import string

from grdwindinversion.utils import check_incidence_range, get_pol_ratio_name, timing, convert_polarization_name
from grdwindinversion.load_config import getConf
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
try:
    import cv2
except:
    import cv2
cv2.setNumThreads(1)


root_logger = logging.getLogger("grdwindinversion.inversion")

# Sensor metadata registry
SENSOR_METADATA = {
    "S1A": ("S1A", "SENTINEL-1 A", xsar.Sentinel1Meta, xsar.Sentinel1Dataset),
    "S1B": ("S1B", "SENTINEL-1 B", xsar.Sentinel1Meta, xsar.Sentinel1Dataset),
    "S1C": ("S1C", "SENTINEL-1 C", xsar.Sentinel1Meta, xsar.Sentinel1Dataset),
    "S1D": ("S1D", "SENTINEL-1 D", xsar.Sentinel1Meta, xsar.Sentinel1Dataset),
    "RS2": ("RS2", "RADARSAT-2", xsar.RadarSat2Meta, xsar.RadarSat2Dataset),
    "RCM1": ("RCM", "RADARSAT Constellation 1", xsar.RcmMeta, xsar.RcmDataset),
    "RCM2": ("RCM", "RADARSAT Constellation 2", xsar.RcmMeta, xsar.RcmDataset),
    "RCM3": ("RCM", "RADARSAT Constellation 3", xsar.RcmMeta, xsar.RcmDataset),
}

# Mask naming convention used by xsar
XSAR_MASK_SUFFIX = "_mask"


def getSensorMetaDataset(filename):
    """
    Find the sensor name and the corresponding meta and dataset functions

    Parameters
    ----------
    filename : str
        input filename

    Returns
    -------
    tuple
        sensor name, sensor long name, meta function, dataset function
    """
    for sensor_key, sensor_info in SENSOR_METADATA.items():
        if sensor_key in filename:
            return sensor_info

    supported_sensors = "|".join(SENSOR_METADATA.keys())
    raise ValueError(
        f"must be {supported_sensors}, got filename {filename}"
    )


def getOutputName(
    input_file, outdir, sensor, meta_start_date, meta_stop_date, subdir=True
):
    """
    Create output filename for L2-GRD product

    Parameters
    ----------
    input_file : str
        input filename
    outdir : str
        output folder
    sensor : str
        sensor name
    start_date : str
        start date
    stop_date : str
        stop date

    Returns
    -------
    outfile : str
        output filename
    """
    basename = os.path.basename(input_file)

    if sensor in ["S1A", "S1B", "S1C", "S1D"]:
        # Example: S1A_IW_GRDH_1SDV_20210909T130650_20210909T130715_039605_04AE83_C34F.SAFE
        regex = re.compile(
            r"(...)_(..)_(...)(.)_(.)(.)(..)_(........T......)_(........T......)_(......)_(......)_(....).SAFE"
        )
        match = regex.match(basename)
        if not match:
            raise AttributeError(
                f"S1 file {basename} does not match the expected pattern"
            )

        (
            MISSIONID,
            SWATH,
            PRODUCT,
            RESOLUTION,
            LEVEL,
            CLASS,
            POLARIZATION,
            STARTDATE,
            STOPDATE,
            ORBIT,
            TAKEID,
            PRODID,
        ) = match.groups()
        new_format = f"{MISSIONID.lower()}-{SWATH.lower()}-owi-{POLARIZATION.lower()}-{STARTDATE.lower()}-{STOPDATE.lower()}-{ORBIT}-{TAKEID}.nc"

    elif sensor == "RS2":
        # Example: RS2_OK141302_PK1242223_DK1208537_SCWA_20220904_093402_VV_VH_SGF
        regex = re.compile(
            r"(RS2)_OK([0-9]+)_PK([0-9]+)_DK([0-9]+)_(....)_(........)_(......)_(.._?.?.?)_(S.F)"
        )
        match = regex.match(basename)
        if not match:
            raise AttributeError(
                f"RS2 file {basename} does not match the expected pattern"
            )

        MISSIONID, DATA1, DATA2, DATA3, SWATH, DATE, TIME, POLARIZATION, LAST = (
            match.groups()
        )
        new_format = f"{MISSIONID.lower()}-{SWATH.lower()}-owi-{convert_polarization_name(POLARIZATION)}-{meta_start_date.lower()}-{meta_stop_date.lower()}-xxxxx-xxxxx.nc"

    elif sensor == "RCM":
        # Example: RCM1_OK2767220_PK2769320_1_SCLND_20230930_214014_VV_VH_GRD
        regex = re.compile(
            r"(RCM[0-9])_OK([0-9]+)_PK([0-9]+)_([0-9]+)_([A-Z0-9]+)_(\d{8})_(\d{6})_([A-Z]{2}(?:_[A-Z]{2})?)_([A-Z]+)$"
        )
        match = regex.match(basename)
        if not match:
            raise AttributeError(
                f"RCM file {basename} does not match the expected pattern"
            )

        MISSIONID, DATA1, DATA2, DATA3, SWATH, DATE, TIME, POLARIZATION, PRODUCT = (
            match.groups()
        )
        new_format = f"{MISSIONID.lower()}-{SWATH.lower()}-owi-{convert_polarization_name(POLARIZATION)}-{meta_start_date.lower()}-{meta_stop_date.lower()}-xxxxx-xxxxx.nc"

    else:
        raise ValueError(
            f"sensor must be S1A|S1B|S1C|S1D|RS2|RCM, got sensor {sensor}"
        )

    if subdir:
        out_file = os.path.join(outdir, basename, new_format)
    else:
        out_file = os.path.join(outdir, new_format)
    return out_file


def addMasks_toMeta(meta: xsar.BaseMeta) -> dict:
    """
    Add high-resolution masks (land, ice, lakes, etc.) from shapefiles to meta object.

    Configuration format:
      masks:
        land:
          - name: 'gshhsH'
            path: '/path/to/mask.shp'
          - name: 'custom_land'
            path: '/path/to/custom.shp'
        ice:
          - name: 'ice_mask'
            path: '/path/to/ice.shp'

    Note: xsar will automatically add '_mask' suffix to the variable names in the dataset.
    For example, 'gshhsH' becomes 'gshhsH_mask' in the xarray dataset.

    Parameters
    ----------
    meta : xsar.BaseMeta
        Metadata object to add mask features to. Must have a set_mask_feature method.

    Returns
    -------
    dict
        Dictionary with mask categories as keys and lists of mask names as values.
        Names are returned WITHOUT the '_mask' suffix that xsar adds internally.
        Example: {'land': ['gshhsH', 'custom_land'], 'ice': ['ice_mask']}

    Raises
    ------
    AttributeError
        If meta object doesn't have set_mask_feature method
    """
    # Validate meta object has required method
    if not hasattr(meta, 'set_mask_feature'):
        raise AttributeError(
            f"Meta object of type {type(meta).__name__} must have a 'set_mask_feature' method")

    conf = getConf()
    masks_by_category = {}

    # Check for 'masks' key
    if "masks" in conf and isinstance(conf["masks"], dict):
        logging.debug("Found 'masks' configuration")

        for category, mask_list in conf["masks"].items():
            if isinstance(mask_list, list):
                masks_by_category[category] = []
                for mask_item in mask_list:
                    if isinstance(mask_item, dict) and "path" in mask_item and "name" in mask_item:
                        mask_name = mask_item["name"]
                        mask_path = mask_item["path"]
                        try:
                            logging.debug("%s path: %s", mask_name, mask_path)
                            meta.set_mask_feature(mask_name, mask_path)
                            logging.info(
                                "Mask feature '%s' set from %s", mask_name, mask_path)
                            masks_by_category[category].append(mask_name)
                        except (IOError, OSError, FileNotFoundError) as e:
                            logging.error(
                                "Failed to load mask file '%s' from path '%s': %s",
                                mask_name, mask_path, str(e))
                            logging.debug("%s", traceback.format_exc())
                        except (ValueError, RuntimeError) as e:
                            logging.error(
                                "Failed to process mask '%s': %s", mask_name, str(e))
                            logging.debug("%s", traceback.format_exc())
                    else:
                        logging.warning(
                            "Invalid mask configuration in category '%s': missing 'name' or 'path' field",
                            category)
            else:
                logging.warning(
                    "Mask category '%s' should contain a list, got %s",
                    category, type(mask_list).__name__
                )

    return masks_by_category


def mergeLandMasks(xr_dataset: xr.Dataset, land_mask_names: list) -> xr.Dataset:
    """
    Merge multiple land masks into the main land_mask variable.

    This function takes all individual land masks added via addMasks_toMeta() and combines
    them using a logical OR operation to create a unified land mask that covers
    all land areas from all sources.

    Parameters
    ----------
    xr_dataset : xr.Dataset
        Dataset containing individual land mask variables. Must contain a 'land_mask' variable.
    land_mask_names : list of str
        Names of the land mask variables to merge (WITHOUT the '_mask' suffix).
        For example: ['gshhsH', 'custom_land'].
        These names will have XSAR_MASK_SUFFIX automatically appended to match
        the variable names in the dataset.

    Returns
    -------
    xr.Dataset
        The input dataset with its land_mask variable updated by merging all specified masks.
        Note: The dataset is modified in place AND returned for convenience.

    Raises
    ------
    ValueError
        If 'land_mask' variable is not present in the dataset
    """
    # Validate that land_mask exists in the dataset
    if "land_mask" not in xr_dataset:
        raise ValueError(
            "Dataset must contain a 'land_mask' variable. "
            f"Available variables: {list(xr_dataset.data_vars.keys())}")

    if not land_mask_names:
        logging.debug("No additional land masks to merge")
        return xr_dataset

    logging.info("Merging %d land masks: %s", len(
        land_mask_names), land_mask_names)

    # Start with the default land_mask from xsar
    merged_mask = xr_dataset["land_mask"].values.astype("uint8")

    # Merge all configured land masks
    for mask_name in land_mask_names:
        # xsar adds XSAR_MASK_SUFFIX to mask names in the dataset
        dataset_mask_name = f"{mask_name}{XSAR_MASK_SUFFIX}"

        if dataset_mask_name in xr_dataset:
            logging.info("Merging mask '%s' into land_mask", dataset_mask_name)
            mask_values = xr_dataset[dataset_mask_name].values.astype("uint8")
            # Logical OR: any pixel marked as land (1) in any mask becomes land
            merged_mask = np.maximum(merged_mask, mask_values)
        else:
            logging.warning(
                "Mask '%s' not found in dataset, skipping", dataset_mask_name)

    # Update the main land_mask
    xr_dataset.land_mask.values = merged_mask
    logging.info("Land masks merged")

    return xr_dataset


def processLandMask(xr_dataset, dilation_iterations=3, merged_masks=None):
    """
    Process land mask to create a 3-level mask system with coastal zone detection.

    This function:
    1. Takes the original land_mask (merged from all configured sources)
    2. Applies binary dilation to detect coastal zones
    3. Creates a 3-level land_mask:
       - 0 = ocean (water far from coast)
       - 1 = coastal (zone between original mask and dilated mask)
       - 2 = land (original land mask)

    Parameters
    ----------
    xr_dataset : xarray.Dataset
        Dataset containing the land_mask variable
    dilation_iterations : int, optional
        Number of dilation iterations to define coastal zone width (default: 3)
    merged_masks : list of str, optional
        Names of masks that were merged into land_mask (for history tracking)

    Returns
    -------
    None
        Modifies xr_dataset.land_mask in place
    """
    logging.info("Processing land mask and adding a coastal zone")

    # Store original land mask (2 = land)
    original_land_mask = xr_dataset["land_mask"].values.astype("uint8")

    # Apply dilation to create coastal zone
    dilated_mask = binary_dilation(
        original_land_mask,
        structure=np.ones((3, 3), np.uint8),
        iterations=dilation_iterations,
    )

    # Create 3-level mask
    # Start with all zeros (ocean)
    three_level_mask = np.zeros_like(original_land_mask, dtype="uint8")

    # Mark land areas (2)
    three_level_mask[original_land_mask == 1] = 2

    # Mark coastal areas (1) - dilated area minus original land
    coastal_zone = (dilated_mask == 1) & (original_land_mask == 0)
    three_level_mask[coastal_zone] = 1

    # Update the land_mask with 3-level system
    xr_dataset.land_mask.values = three_level_mask

    # Update attributes
    xr_dataset.land_mask.attrs["long_name"] = "Land mask with coastal zone"
    xr_dataset.land_mask.attrs["valid_range"] = np.array([0, 2])
    xr_dataset.land_mask.attrs["flag_values"] = np.array([0, 1, 2])
    xr_dataset.land_mask.attrs["flag_meanings"] = "ocean coastal land"
    xr_dataset.land_mask.attrs["meaning"] = "0: ocean, 1: coastal, 2: land"

    # Append to history instead of replacing
    existing_history = xr_dataset.land_mask.attrs.get("history", "")

    # Build history message
    if merged_masks:
        merge_info = f"merged with {', '.join(merged_masks)}"
    else:
        merge_info = ""

    new_history = f"{merge_info}3-level land mask with coastal zone detection via binary dilation"

    if existing_history:
        xr_dataset.land_mask.attrs["history"] = existing_history + \
            "; " + new_history
    else:
        xr_dataset.land_mask.attrs["history"] = new_history


def getAncillary(meta, ancillary_name="ecmwf"):
    """
    Map ancillary wind from "ecmwf" or "era5" or other sources.
    This function is used to check if the model files are available and to map the model to the SAR data.
    This function will use with priority the first model of the config file.

    Parameters
    ----------
    meta: obj `xsar.BaseMeta` (one of the supported SAR mission)
    ancillary_name: str
        Name of the ancillary source (ecmwf or era5)

    Returns
    -------
    tuple
        (map_model, metadata) where:
        - map_model (dict): mapping of model variables to SAR data
        - metadata (dict): ancillary metadata with 'source' and 'source_path' keys
    """
    logging.debug("conf: %s", getConf())
    conf = getConf()
    if 'ancillary_sources' not in conf:
        raise ValueError("Configuration must contain 'ancillary_sources'")

    if ancillary_name not in conf['ancillary_sources']:
        raise ValueError(
            f"Configuration 'ancillary_sources' must contain '{ancillary_name}'")

    if ancillary_name not in ["ecmwf", "era5"]:
        logging.warning("We advice to use either ecmwf or era5.")

    ancillary_sources = conf['ancillary_sources'][ancillary_name]
    if not ancillary_sources:
        raise ValueError(
            f"At least one ancillary model {ancillary_name} must be configured in ancillary_sources")

    map_model = None
    selected_name = None
    selected_path = None
    tried_names = []

    # Loop through models in config order to find the first one that exists
    for source in ancillary_sources:
        model_name = source['name']
        model_path = source['path']
        logging.debug("%s : %s", model_name, model_path)

        # Set raster to check if file exists
        meta.set_raster(model_name, model_path)
        tried_names.append(model_name)

        model_info = meta.rasters.loc[model_name]

        model_file = model_info["get_function"](
            model_info["resource"],
            date=datetime.datetime.strptime(
                meta.start_date, "%Y-%m-%d %H:%M:%S.%f"
            ),
        )[1]

        if os.path.isfile(model_file):
            # File exists! This is our selection
            selected_name = model_name
            selected_path = model_file
            map_model = {
                "%s_%s" % (selected_name, uv): "model_%s" % uv for uv in ["U10", "V10"]
            }
            # Log selection
            if len(ancillary_sources) > 1:
                logging.info(
                    f"Multiple {ancillary_name} models configured. Using {selected_name} (priority order)")
            else:
                logging.info(
                    f"Only one {ancillary_name} model configured: using {selected_name}")
            break

    # Clean up: remove all tried models EXCEPT the selected one
    if selected_name is not None:
        for name in tried_names:
            if name != selected_name:
                meta.rasters = meta.rasters.drop([name])

    # Prepare metadata for traceability
    ancillary_metadata = None
    if selected_name is not None:
        ancillary_metadata = {
            'ancillary_source_model': selected_name,
            'ancillary_source_path': selected_path
        }

    return map_model, ancillary_metadata


@timing(logger=root_logger.debug)
def inverse_dsig_wspd(
    dual_pol,
    inc,
    sigma0,
    sigma0_dual,
    ancillary_wind,
    nesz_cr,
    dsig_cr_name,
    model_co,
    model_cross,
    **kwargs,
):
    """
    Invert sigma0 to retrieve wind using model (lut or gmf).

    Parameters
    ----------
    dual_pol: bool
        True if dualpol, False if singlepol
    inc: xarray.DataArray
        incidence angle
    sigma0: xarray.DataArray
        sigma0 to be inverted
    sigma0_dual: xarray.DataArray
        sigma0 to be inverted for dualpol
    ancillary_wind=: xarray.DataArray (numpy.complex28)
        ancillary wind
            | (for example ecmwf winds), in **ANTENNA convention**,
    nesz_cr: xarray.DataArray
        noise equivalent sigma0 | flattened or not
    dsig_cr_name:  str
        dsig_cr name
    model_co: str
        model to use for VV or HH polarization.
    model_cross: str
        model to use for VH or HV polarization.

    Returns
    -------
    xarray.DataArray
        inverted wind in copol in ** antenna convention** .
    xarray.DataArray
        inverted wind in dualpol in ** antenna convention** .
    xarray.DataArray
        inverted wind in crosspol in ** antenna convention** .
    xarray.DataArray | array
        alpha (ponderation between co and crosspol)

    See Also
    --------
    xsarsea documentation
    https://cerweb.ifremer.fr/datarmor/doc_sphinx/xsarsea/
    """

    # dsig_cr_step == "wspd":

    wind_co = xsarsea.windspeed.invert_from_model(
        inc,
        sigma0,
        ancillary_wind=ancillary_wind,
        model=model_co,
        **kwargs
    )

    if dual_pol:

        wind_cross = windspeed.invert_from_model(
            inc.values,
            sigma0_dual.values,
            model=model_cross,
            **kwargs,
        )

        wspd_co = np.abs(wind_co)
        wspd_cross = np.abs(wind_cross)
        SNR_cross = sigma0_dual.values/nesz_cr.values
        alpha = windspeed.get_dsig_wspd(dsig_cr_name, wind_cross, SNR_cross)

        wpsd_dual = alpha * wspd_co + (1 - alpha) * wspd_cross
        wind_dual = wpsd_dual * np.exp(1j * np.angle(wind_co))

        return wind_co, wind_dual, wind_cross, alpha

    return wind_co, None, None, None


@timing(logger=root_logger.debug)
def inverse(
    dual_pol,
    inc,
    sigma0,
    sigma0_dual,
    ancillary_wind,
    dsig_cr,
    model_co,
    model_cross,
    **kwargs,
):
    """
    Invert sigma0 to retrieve wind using model (lut or gmf).

    Parameters
    ----------
    dual_pol: bool
        True if dualpol, False if singlepol
    inc: xarray.DataArray
        incidence angle
    sigma0: xarray.DataArray
        sigma0 to be inverted
    sigma0_dual: xarray.DataArray
        sigma0 to be inverted for dualpol
    ancillary_wind=: xarray.DataArray (numpy.complex28)
        ancillary wind
            | (for example ecmwf winds), in **ANTENNA convention**,
    dsig_cr=: float or xarray.DataArray
        parameters used for

            | `Jsig_cr=((sigma0_gmf - sigma0) / dsig_cr) ** 2`
    model_co=: str
        model to use for VV or HH polarization.
    model_cross=: str
        model to use for VH or HV polarization.

    Returns
    -------
    xarray.DataArray
        inverted wind in copol in ** antenna convention** .
    xarray.DataArray
        inverted wind in dualpol in ** antenna convention** .
    xarray.DataArray
        inverted wind in crosspol in ** antenna convention** .

    See Also
    --------
    xsarsea documentation
    https://cerweb.ifremer.fr/datarmor/doc_sphinx/xsarsea/
    """
    logging.debug("inversion")

    list_mods = (
        windspeed.available_models().index.tolist()
        + windspeed.available_models().alias.tolist()
        + [None]
    )
    if model_co not in list_mods:
        raise ValueError(
            f"model_co {model_co} not in windspeed.available_models() : not going further"
        )
    if model_cross not in list_mods:
        raise ValueError(
            f"model_cross {model_cross} not in windspeed.available_models() : not going further"
        )

    winds = windspeed.invert_from_model(
        inc,
        sigma0,
        sigma0_dual,
        ancillary_wind=ancillary_wind,
        dsig_cr=dsig_cr,
        model=(model_co, model_cross),
        **kwargs,
    )

    if dual_pol:
        wind_co, wind_dual = winds

        wind_cross = windspeed.invert_from_model(
            inc.values,
            sigma0_dual.values,
            dsig_cr=dsig_cr.values,
            model=model_cross,
            **kwargs,
        )

        return wind_co, wind_dual, wind_cross
    else:
        wind_co = winds

    return wind_co, None, None


@timing(logger=root_logger.debug)
def makeL2asOwi(xr_dataset, config):
    """
    Rename xr_dataset variables and attributes to match naming convention.

    Parameters
    ----------
    xr_dataset: xarray.Dataset
        dataset to rename
    config: dict
        configuration dict

    Returns
    -------
    xarray.Dataset
        final dataset
    dict
        encoding dict

    See Also
    --------
    """

    xr_dataset = xr_dataset.rename(
        {
            "longitude": "owiLon",
            "latitude": "owiLat",
            "incidence": "owiIncidenceAngle",
            "elevation": "owiElevationAngle",
            "ground_heading": "owiHeading",
            "land_mask": "owiLandFlag",
            "mask": "owiMask",
            "windspeed_co": "owiWindSpeed_co",
            "winddir_co": "owiWindDirection_co",
            "ancillary_wind_speed": "owiAncillaryWindSpeed",
            "ancillary_wind_direction": "owiAncillaryWindDirection",
            "sigma0_detrend": "owiNrcs_detrend",
        }
    )

    if "offboresight" in xr_dataset:
        xr_dataset = xr_dataset.rename(
            {"offboresight": "owiOffboresightAngle"})

    if config["add_nrcs_model"]:
        xr_dataset = xr_dataset.rename({"ancillary_nrcs": "owiAncillaryNrcs"})
        xr_dataset.owiAncillaryNrcs.attrs["units"] = "m^2 / m^2"
        xr_dataset.owiAncillaryNrcs.attrs["long_name"] = (
            f"Ancillary Normalized Radar Cross Section - simulated from {config['l2_params']['copol_gmf']} & ancillary wind"
        )

        if config["l2_params"]["dual_pol"]:
            xr_dataset = xr_dataset.rename(
                {"ancillary_nrcs_cross": "owiAncillaryNrcs_cross"}
            )
            xr_dataset.owiAncillaryNrcs_cross.attrs["units"] = "m^2 / m^2"
            xr_dataset.owiAncillaryNrcs_cross.attrs["long_name"] = (
                f"Ancillary Normalized Radar Cross Section - simulated from {config['l2_params']['crosspol_gmf']} & ancillary wind"
            )

    xr_dataset.owiLon.attrs["units"] = "degrees_east"
    xr_dataset.owiLon.attrs["long_name"] = "Longitude at wind cell center"
    xr_dataset.owiLon.attrs["standard_name"] = "longitude"

    xr_dataset.owiLat.attrs["units"] = "degrees_north"
    xr_dataset.owiLat.attrs["long_name"] = "Latitude at wind cell center"
    xr_dataset.owiLat.attrs["standard_name"] = "latitude"

    xr_dataset.owiIncidenceAngle.attrs["units"] = "degrees"
    xr_dataset.owiIncidenceAngle.attrs["long_name"] = (
        "Incidence angle at wind cell center"
    )
    xr_dataset.owiIncidenceAngle.attrs["standard_name"] = "incidence"

    xr_dataset.owiElevationAngle.attrs["units"] = "degrees"
    xr_dataset.owiElevationAngle.attrs["long_name"] = (
        "Elevation angle at wind cell center"
    )
    xr_dataset.owiElevationAngle.attrs["standard_name"] = "elevation"

    xr_dataset["owiNrcs"] = xr_dataset["sigma0_ocean"].sel(
        pol=config["l2_params"]["copol"]
    )
    xr_dataset.owiNrcs.attrs = xr_dataset.sigma0_ocean.attrs
    xr_dataset.owiNrcs.attrs["units"] = "m^2 / m^2"
    xr_dataset.owiNrcs.attrs["long_name"] = "Normalized Radar Cross Section"
    xr_dataset.owiNrcs.attrs["definition"] = "owiNrcs_no_noise_correction - owiNesz"

    xr_dataset["owiMask_Nrcs"] = xr_dataset["sigma0_mask"].sel(
        pol=config["l2_params"]["copol"]
    )
    xr_dataset.owiMask_Nrcs.attrs = xr_dataset.sigma0_mask.attrs

    # NESZ & DSIG
    xr_dataset = xr_dataset.assign(
        owiNesz=(
            ["line", "sample"],
            xr_dataset.nesz.sel(pol=config["l2_params"]["copol"]).values,
        )
    )
    xr_dataset.owiNesz.attrs["units"] = "m^2 / m^2"
    xr_dataset.owiNesz.attrs["long_name"] = "Noise Equivalent SigmaNaught"

    xr_dataset["owiNrcs_no_noise_correction"] = xr_dataset["sigma0_ocean_raw"].sel(
        pol=config["l2_params"]["copol"]
    )
    xr_dataset.owiNrcs_no_noise_correction.attrs = xr_dataset.sigma0_ocean_raw.attrs
    xr_dataset.owiNrcs_no_noise_correction.attrs["units"] = "m^2 / m^2"
    xr_dataset.owiNrcs_no_noise_correction.attrs["long_name"] = (
        "Normalized Radar Cross Section ; no noise correction applied"
    )
    xr_dataset.owiNrcs_no_noise_correction.attrs["comment"] = (
        "owiNrcs_no_noise_correction ; no recalibration"
    )

    if "swath_number" in xr_dataset:
        xr_dataset = xr_dataset.rename(
            {
                "swath_number": "owiSwathNumber",
                "swath_number_flag": "owiSwathNumberFlag",
            }
        )

        xr_dataset["owiSwathNumber"].attrs["standart_name"] = "swath number"

    # sigma0_raw__corrected cross
    if "sigma0_raw__corrected" in xr_dataset:
        xr_dataset["owiNrcs_no_noise_correction_recalibrated"] = xr_dataset[
            "sigma0_raw__corrected"
        ].sel(pol=config["l2_params"]["copol"])
        xr_dataset.owiNrcs_no_noise_correction_recalibrated.attrs = (
            xr_dataset.sigma0_raw__corrected.attrs
        )
        xr_dataset.owiNrcs_no_noise_correction_recalibrated.attrs["units"] = "m^2 / m^2"
        xr_dataset.owiNrcs_no_noise_correction_recalibrated.attrs["long_name"] = (
            "Normalized Radar Cross Section, no noise correction applied"
        )
        xr_dataset.owiNrcs_no_noise_correction_recalibrated.attrs["comment"] = (
            "owiNrcs_no_noise_correction ; recalibrated with kersten method"
        )

        xr_dataset.owiNrcs.attrs["definition"] = (
            "owiNrcs_no_noise_correction_recalibrated - owiNesz"
        )

    if config["l2_params"]["dual_pol"]:
        if config["dsig_cr_step"] == "nrcs":
            xr_dataset = xr_dataset.rename({
                'dsig_cross': 'owiDsig_cross',
            })
        else:
            xr_dataset = xr_dataset.rename({
                'alpha': 'owiAlpha',
            })
        xr_dataset = xr_dataset.rename({
            'winddir_cross': 'owiWindDirection_cross',
            'winddir_dual': 'owiWindDirection',
            'windspeed_cross': 'owiWindSpeed_cross',
            'windspeed_dual': 'owiWindSpeed',
            'sigma0_detrend_cross': 'owiNrcs_detrend_cross',
            'nesz_cross_flattened': 'owiNesz_cross_flattened'
        })

        # nrcs cross
        xr_dataset["owiNrcs_cross"] = xr_dataset["sigma0_ocean"].sel(
            pol=config["l2_params"]["crosspol"]
        )

        xr_dataset.owiNrcs_cross.attrs["units"] = "m^2 / m^2"
        xr_dataset.owiNrcs_cross.attrs["long_name"] = "Normalized Radar Cross Section"
        xr_dataset.owiNrcs_cross.attrs["definition"] = (
            "owiNrcs_cross_no_noise_correction - owiNesz_cross"
        )

        xr_dataset["owiMask_Nrcs_cross"] = xr_dataset["sigma0_mask"].sel(
            pol=config["l2_params"]["crosspol"]
        )
        xr_dataset.owiMask_Nrcs_cross.attrs = xr_dataset.sigma0_mask.attrs

        # nesz cross
        xr_dataset = xr_dataset.assign(
            owiNesz_cross=(
                ["line", "sample"],
                xr_dataset.nesz.sel(
                    pol=config["l2_params"]["crosspol"]).values,
            )
        )  # no flattening
        xr_dataset.owiNesz_cross.attrs["units"] = "m^2 / m^2"
        xr_dataset.owiNesz_cross.attrs["long_name"] = "Noise Equivalent SigmaNaught"

        xr_dataset.owiNesz_cross_flattened.attrs["units"] = "m^2 / m^2"
        xr_dataset.owiNesz_cross_flattened.attrs["long_name"] = "Noise Equivalent SigmaNaught"

        xr_dataset["owiNrcs_cross_no_noise_correction"] = xr_dataset[
            "sigma0_ocean_raw"
        ].sel(pol=config["l2_params"]["crosspol"])

        xr_dataset.owiNrcs_cross_no_noise_correction.attrs["units"] = "m^2 / m^2"
        xr_dataset.owiNrcs_cross_no_noise_correction.attrs["long_name"] = (
            "Normalized Radar Cross Section, no noise correction applied"
        )

        #  sigma0_raw__corrected cross
        if "sigma0_raw__corrected" in xr_dataset:
            xr_dataset["owiNrcs_cross_no_noise_correction_recalibrated"] = xr_dataset[
                "sigma0_raw__corrected"
            ].sel(pol=config["l2_params"]["crosspol"])
            xr_dataset.owiNrcs_cross_no_noise_correction_recalibrated.attrs = (
                xr_dataset.sigma0_raw__corrected.attrs
            )
            xr_dataset.owiNrcs_cross_no_noise_correction_recalibrated.attrs["units"] = (
                "m^2 / m^2"
            )
            xr_dataset.owiNrcs_cross_no_noise_correction_recalibrated.attrs[
                "long_name"
            ] = "Normalized Radar Cross Section ; no noise correction applied"
            xr_dataset.owiNrcs_cross_no_noise_correction_recalibrated.attrs[
                "comment"
            ] = "owiNrcs_cross_no_noise_correction ; recalibrated with kersten method"

            xr_dataset.owiNrcs_cross.attrs["definition"] = (
                "owiNrcs_cross_no_noise_correction_recalibrated - owiNesz_cross"
            )

    if config["add_gradientsfeatures"]:
        xr_dataset = xr_dataset.rename({"heterogeneity_mask": "owiWindFilter"})
    else:
        xr_dataset["owiWindFilter"] = xr.full_like(xr_dataset.owiNrcs, 0)
        xr_dataset["owiWindFilter"].attrs[
            "long_name"
        ] = "Quality flag taking into account the local heterogeneity"
        xr_dataset["owiWindFilter"].attrs["valid_range"] = np.array([0, 3])
        xr_dataset["owiWindFilter"].attrs["flag_values"] = np.array([
                                                                    0, 1, 2, 3])
        xr_dataset["owiWindFilter"].attrs[
            "flag_meanings"
        ] = "homogeneous_NRCS, heterogeneous_from_co-polarization_NRCS, heterogeneous_from_cross-polarization_NRCS, heterogeneous_from_dual-polarization_NRCS"

    #  other variables

    xr_dataset["owiWindQuality"] = xr.full_like(xr_dataset.owiNrcs, 0)
    xr_dataset["owiWindQuality"].attrs[
        "long_name"
    ] = "Quality flag taking into account the consistency_between_wind_inverted_and_NRCS_and_Doppler_measured"
    xr_dataset["owiWindQuality"].attrs["valid_range"] = np.array([0, 3])
    xr_dataset["owiWindQuality"].attrs["flag_values"] = np.array([0, 1, 2, 3])
    xr_dataset["owiWindQuality"].attrs["flag_meanings"] = "good medium low poor"
    xr_dataset["owiWindQuality"].attrs["comment"] = "NOT COMPUTED YET"

    xr_dataset = xr_dataset.rename(
        {"line": "owiAzSize", "sample": "owiRaSize"})

    xr_dataset = xr_dataset.drop_vars(
        [
            "sigma0_ocean",
            "sigma0",
            "sigma0_ocean_raw",
            "sigma0_raw",
            "ancillary_wind",
            "nesz",
            "model_U10",
            "model_V10"

        ]
    )
    if "sigma0_raw__corrected" in xr_dataset:
        xr_dataset = xr_dataset.drop_vars(["sigma0_raw__corrected"])
    xr_dataset = xr_dataset.drop_dims(["pol"])

    table_fillValue = {
        "owiWindQuality": -1,
        "owiHeading": 9999.99,
        "owiWindDirection_IPF": -9999.0,
        "owiWindSpeed_IPF": -9999.0,
        "owiWindDirection": -9999.0,
        "owiPBright": 999.99,
        "owiWindFilter": -1,
        "owiWindSpeed": -9999.0,
        "owiWindSpeed_co": -9999.0,
        "owiWindSpeed_cross": -9999.0,
    }

    encoding = {}
    for var in list(set(xr_dataset.coords.keys()) | set(xr_dataset.keys())):
        encoding[var] = {}
        try:
            encoding[var].update({"_FillValue": table_fillValue[var]})
        except:
            if var in ["owiWindSpeed_co", "owiWindSpeed_cross", "owiWindSpeed"]:
                encoding[var].update({"_FillValue": -9999.0})
            else:
                encoding[var].update({"_FillValue": None})

    return xr_dataset, encoding


def preprocess(
    filename,
    outdir,
    config_path,
    overwrite=False,
    add_gradientsfeatures=False,
    resolution="1000m",
):
    """
    Main function to generate L2 product.

    Parameters
    ----------
    filename : str
        input filename
    outdir : str
        output folder
    config_path : str
        configuration file path
    overwrite : bool, optional
        overwrite existing file
    resolution : str, optional
        working resolution

    Returns
    -------
    xarray.Dataset
        final dataset
    """

    sensor, sensor_longname, fct_meta, fct_dataset = getSensorMetaDataset(
        filename)

    if os.path.exists(config_path):
        with open(config_path, "r") as file:
            config_base = yaml.load(file, Loader=yaml.FullLoader)
        try:
            # check if sensor is in the config
            config = config_base[sensor]
        except Exception:
            raise KeyError("sensor %s not in this config" % sensor)
    else:
        raise FileNotFoundError(
            "config_path do not exists, got %s " % config_path)

    recalibration = config["recalibration"]
    meta = fct_meta(filename)

    # Add masks to meta if configured (land, ice, lakes, etc.)
    masks_by_category = addMasks_toMeta(meta)

    # si une des deux n'est pas VV VH HH HV on ne fait rien
    if not all([pol in ["VV", "VH", "HH", "HV"] for pol in meta.pols.split(" ")]):
        raise ValueError(f"Polarisation non gérée : meta.pols =  {meta.pols}")

    overwrite_cfg = config_base.get("overwrite", None)
    # If overwrite is specified in config, it will overwrite the value given as argument (which defaults to False)
    if overwrite_cfg is not None:
        overwrite = overwrite_cfg

    no_subdir_cfg = config_base.get("no_subdir", False)
    config["no_subdir"] = no_subdir_cfg

    if "winddir_convention" in config_base:
        winddir_convention = config_base["winddir_convention"]
    else:
        winddir_convention = "meteorological"
        logging.info(
            f'Using meteorological convention because "winddir_convention" was not found in config.'
        )
    config["winddir_convention"] = winddir_convention

    if "add_gradientsfeatures" in config_base:
        add_gradientsfeatures = config_base["add_gradientsfeatures"]
    else:
        add_gradientsfeatures = False
        logging.info(f"Not computing gradients by default")
    config["add_gradientsfeatures"] = add_gradientsfeatures

    if "add_nrcs_model" in config_base:
        add_nrcs_model = config_base["add_nrcs_model"]
        add_nrcs_model = False
        logging.info(
            f"Force add_nrcs_model to be false, before fixing an issue")
    else:
        add_nrcs_model = False
        logging.info(f"Not computing nrcs from model by default")
    config["add_nrcs_model"] = add_nrcs_model

    # creating a dictionnary of parameters
    config["l2_params"] = {}

    meta_start_date = (
        meta.start_date.split(".")[0]
        .replace("-", "")
        .replace(":", "")
        .replace(" ", "t")
        .replace("Z", "")
    )
    meta_stop_date = (
        meta.stop_date.split(".")[0]
        .replace("-", "")
        .replace(":", "")
        .replace(" ", "t")
        .replace("Z", "")
    )

    out_file = getOutputName(
        filename,
        outdir,
        sensor,
        meta_start_date,
        meta_stop_date,
        subdir=not no_subdir_cfg,
    )

    if os.path.exists(out_file) and overwrite is False:
        raise FileExistsError("outfile %s already exists" % out_file)

    ancillary_name = config["ancillary"]
    map_model, ancillary_metadata = getAncillary(meta, ancillary_name)
    if map_model is None:
        raise Exception(
            f"the weather model is not set `map_model` is None -> you probably don't have access to {ancillary_name} archive"
        )
    if ancillary_metadata is None:
        raise Exception(
            f"ancillary_metadata must be defined. There is an error in getAncillary function")

    try:
        logging.info(f"recalibration = {recalibration}")
        if (recalibration) & ("SENTINEL" in sensor_longname):
            logging.info(
                f"recalibration is {recalibration} : Kersten formula is applied"
            )
            xsar_dataset = fct_dataset(
                meta, resolution=resolution, recalibration=recalibration
            )
            xr_dataset = xsar_dataset.datatree["measurement"].to_dataset()
            xr_dataset = xr_dataset.merge(
                xsar_dataset.datatree["recalibration"].to_dataset()[
                    ["swath_number", "swath_number_flag", "sigma0_raw__corrected"]
                ]
            )

        else:
            logging.info(
                f"recalibration is {recalibration} : Kersten formula is not applied"
            )
            if "SENTINEL" in sensor_longname:
                xsar_dataset = fct_dataset(
                    meta, resolution=resolution, recalibration=recalibration
                )
                xr_dataset = xsar_dataset.datatree["measurement"].to_dataset()
                xr_dataset = xr_dataset.merge(
                    xsar_dataset.datatree["recalibration"].to_dataset()[
                        ["swath_number", "swath_number_flag"]
                    ]
                )

            else:
                xsar_dataset = fct_dataset(meta, resolution=resolution)
                xr_dataset = xsar_dataset.datatree["measurement"].to_dataset()

        xr_dataset = xr_dataset.rename(map_model)
        xr_dataset.attrs = xsar_dataset.dataset.attrs

    except Exception as e:
        logging.info("%s", traceback.format_exc())
        logging.error(e)
        sys.exit(-1)

    #  add parameters in config
    config["meta"] = meta
    config["fct_dataset"] = fct_dataset
    config["map_model"] = map_model

    xr_dataset = xr_dataset.load()

    # defining dual_pol, and gmfs by channel
    if len(xr_dataset.pol.values) == 2:
        dual_pol = True
    else:
        dual_pol = False

    if "VV" in xr_dataset.pol.values:
        copol = "VV"
        crosspol = "VH"
        copol_gmf = "VV"
        crosspol_gmf = "VH"
    else:
        logging.warning(
            "inversion_rules warning : for now this processor does not support entirely HH+HV acquisitions\n "
            "it wont crash but it will use HH+VH GMF for wind inversion -> wrong hypothesis\n "
            "!! dual WIND SPEED IS NOT USABLE !! But co WIND SPEED IS USABLE !!"
        )
        config["return_status"] = 99

        copol = "HH"
        crosspol = "HV"
        copol_gmf = "HH"
        crosspol_gmf = "VH"

    if (sensor == "S1A" or sensor == "S1B" or sensor == "S1C" or sensor == "S1D") and xsar_dataset.dataset.attrs["aux_cal"] is None:
        raise ValueError(
            "aux_cal attribute is None, xsar_dataset.dataset.attrs['aux_cal'] must be set to a valid value"
        )
    cond_aux_cal = (
        (sensor == "S1A" or sensor == "S1B" or sensor == "S1C" or sensor == "S1D")
        and xsar_dataset.dataset.attrs["aux_cal"] is not None
        and xsar_dataset.dataset.attrs["aux_cal"].split("_")[-1][1:9] > "20190731"
    )

    if cond_aux_cal and xr_dataset.attrs["swath"] == "EW" and "S1_EW_calG>20190731" in config.keys():
        model_co = config["S1_EW_calG>20190731"]["GMF_" + copol_gmf + "_NAME"]
        model_cross = config["S1_EW_calG>20190731"]["GMF_" +
                                                    crosspol_gmf + "_NAME"]
        dsig_cr_name = config["S1_EW_calG>20190731"]["dsig_" +
                                                     crosspol_gmf + "_NAME"]
        apply_flattening = config["S1_EW_calG>20190731"]["apply_flattening"]
        dsig_cr_step = config["S1_EW_calG>20190731"]["dsig_cr_step"]

    else:
        model_co = config["GMF_" + copol_gmf + "_NAME"]
        model_cross = config["GMF_" + crosspol_gmf + "_NAME"]
        dsig_cr_name = config["dsig_" + crosspol_gmf + "_NAME"]
        apply_flattening = config["apply_flattening"]
        dsig_cr_step = config["dsig_cr_step"]

    # register paramaters in config
    config["l2_params"]["dual_pol"] = dual_pol
    config["l2_params"]["copol"] = copol
    config["l2_params"]["crosspol"] = crosspol
    config["l2_params"]["copol_gmf"] = copol_gmf
    config["l2_params"]["crosspol_gmf"] = crosspol_gmf
    config["l2_params"]["model_co"] = model_co
    config["l2_params"]["model_cross"] = model_cross
    config["sensor_longname"] = sensor_longname
    config["sensor"] = sensor
    config["dsig_cr_step"] = dsig_cr_step
    config["dsig_cr_name"] = dsig_cr_name
    config["apply_flattening"] = apply_flattening
    # need to load LUTs before inversion
    nc_luts = [x for x in [model_co, model_cross] if x.startswith("nc_lut")]

    if len(nc_luts) > 0:
        windspeed.register_nc_luts(getConf()["nc_luts_path"])

    if model_co == "gmf_cmod7":
        windspeed.register_cmod7(getConf()["lut_cmod7_path"])
    #  Step 2 - clean and prepare dataset

    # variables to not keep in the L2
    black_list = [
        "digital_number",
        "gamma0_raw",
        "negz",
        "azimuth_time",
        "slant_range_time",
        "velocity",
        "range_ground_spacing",
        "gamma0",
        "time",
        "nd_co",
        "nd_cr",
        "gamma0_lut",
        "sigma0_lut",
        "noise_lut_range",
        "lineSpacing",
        "sampleSpacing",
        "noise_lut",
        "noise_lut_azi",
        "nebz",
        "beta0_raw",
        "lines_flipped",
        "samples_flipped",
        "altitude",
        "beta0",
    ]
    variables = list(set(xr_dataset) - set(black_list))
    xr_dataset = xr_dataset[variables]

    #  lon/lat
    xr_dataset.longitude.attrs["units"] = "degrees_east"
    xr_dataset.longitude.attrs["long_name"] = "Longitude at wind cell center"
    xr_dataset.longitude.attrs["standard_name"] = "longitude"

    xr_dataset.latitude.attrs["units"] = "degrees_north"
    xr_dataset.latitude.attrs["long_name"] = "Latitude at wind cell center"
    xr_dataset.latitude.attrs["standard_name"] = "latitude"

    #  incidence
    xr_dataset.incidence.attrs["units"] = "degrees"
    xr_dataset.incidence.attrs["long_name"] = "Incidence angle at wind cell center"
    xr_dataset.incidence.attrs["standard_name"] = "incidence"

    #  elevation
    xr_dataset.elevation.attrs["units"] = "degrees"
    xr_dataset.elevation.attrs["long_name"] = "Elevation angle at wind cell center"
    xr_dataset.elevation.attrs["standard_name"] = "elevation"

    # offboresight
    xr_dataset.offboresight.attrs["units"] = "degrees"
    xr_dataset.offboresight.attrs["long_name"] = (
        "Offboresight angle at wind cell center"
    )
    xr_dataset.offboresight.attrs["standard_name"] = "offboresight"

    # merge land masks
    conf = getConf()
    land_mask_strategy = conf.get("LAND_MASK_STRATEGY", "merge")
    logging.info(f"land_mask_strategy = {land_mask_strategy}")

    # Store masks_by_category in config for later cleanup
    config["masks_by_category"] = masks_by_category

    merged_land_masks = None
    if land_mask_strategy == "merge" and "land" in masks_by_category:
        mergeLandMasks(xr_dataset, masks_by_category["land"])
        merged_land_masks = masks_by_category["land"]

    # Process land mask with coastal zone detection (3-level system)
    # 0 = ocean, 1 = coastal, 2 = land
    processLandMask(xr_dataset, dilation_iterations=3,
                    merged_masks=merged_land_masks)

    logging.debug("mask is a copy of land_mask")

    # Create main mask from land_mask
    # For now, mask uses the same values as land_mask
    # Can be extended later to include ice (value 3) and other categories
    xr_dataset["mask"] = xr.DataArray(xr_dataset.land_mask)
    xr_dataset.mask.attrs = {}
    xr_dataset.mask.attrs["long_name"] = "Mask of data"
    xr_dataset.mask.attrs["valid_range"] = np.array([0, 3])
    xr_dataset.mask.attrs["flag_values"] = np.array([0, 1, 2, 3])
    xr_dataset.mask.attrs["flag_meanings"] = "ocean coastal land ice"

    # ancillary
    xr_dataset["ancillary_wind_direction"] = (
        90.0 - np.rad2deg(np.arctan2(xr_dataset.model_V10,
                          xr_dataset.model_U10)) + 180
    ) % 360

    # Keep ocean (0) and coastal (1) zones for ancillary wind
    xr_dataset["ancillary_wind_direction"] = xr.where(
        xr_dataset["mask"] >= 2, np.nan, xr_dataset["ancillary_wind_direction"]
    ).transpose(*xr_dataset["ancillary_wind_direction"].dims)
    xr_dataset["ancillary_wind_direction"].attrs = {}
    xr_dataset["ancillary_wind_direction"].attrs["units"] = "degrees_north"
    xr_dataset["ancillary_wind_direction"].attrs[
        "long_name"
    ] = f"{ancillary_name} wind direction in meteorological convention (clockwise, from), ex: 0°=from north, 90°=from east"
    xr_dataset["ancillary_wind_direction"].attrs["standart_name"] = "wind_direction"

    xr_dataset["ancillary_wind_speed"] = np.sqrt(
        xr_dataset["model_U10"] ** 2 + xr_dataset["model_V10"] ** 2
    )
    xr_dataset["ancillary_wind_speed"] = xr.where(
        xr_dataset["mask"] >= 2, np.nan, xr_dataset["ancillary_wind_speed"]
    ).transpose(*xr_dataset["ancillary_wind_speed"].dims)
    xr_dataset["ancillary_wind_speed"].attrs = {}
    xr_dataset["ancillary_wind_speed"].attrs["units"] = "m s^-1"
    xr_dataset["ancillary_wind_speed"].attrs[
        "long_name"
    ] = f"{ancillary_name} wind speed"
    xr_dataset["ancillary_wind_speed"].attrs["standart_name"] = "wind_speed"

    xr_dataset["ancillary_wind"] = xr.where(
        xr_dataset["mask"] >= 2,
        np.nan,
        (
            xr_dataset.ancillary_wind_speed
            * np.exp(
                1j
                * xsarsea.dir_meteo_to_sample(
                    xr_dataset.ancillary_wind_direction, xr_dataset.ground_heading
                )
            )
        ),
    ).transpose(*xr_dataset["ancillary_wind_speed"].dims)
    xr_dataset["ancillary_wind"].attrs = {}
    xr_dataset["ancillary_wind"].attrs["long_name"] = f"{ancillary_name} wind in complex form for inversion"
    xr_dataset["ancillary_wind"].attrs[
        "description"] = "Complex wind (speed * exp(i*direction)) in antenna convention for GMF inversion"

    # Add ancillary metadata to model variables

    for attr_key, attr_value in ancillary_metadata.items():
        for var_name in ['model_U10', 'model_V10', 'ancillary_wind_speed', 'ancillary_wind_direction', 'ancillary_wind']:
            if var_name in xr_dataset:
                xr_dataset[var_name].attrs[attr_key] = attr_value

        xr_dataset.attrs[attr_key] = attr_value

    # nrcs processing
    # Keep ocean (0) and coastal (1) zones, mask out land (2) and ice (3)
    xr_dataset["sigma0_ocean"] = xr.where(
        xr_dataset["mask"] >= 2, np.nan, xr_dataset["sigma0"]
    ).transpose(*xr_dataset["sigma0"].dims)
    xr_dataset["sigma0_ocean"].attrs = xr_dataset["sigma0"].attrs
    #  we forced it to 1e-15
    xr_dataset["sigma0_ocean"].attrs[
        "comment"
    ] = "clipped, no values <=0 ; 1e-15 instread"

    xr_dataset["sigma0_ocean"] = xr.where(
        xr_dataset["sigma0_ocean"] <= 0, 1e-15, xr_dataset["sigma0_ocean"]
    )

    # add a mask for values <=0:
    xr_dataset["sigma0_mask"] = xr.where(
        xr_dataset["sigma0_ocean"] <= 0, 1, 0
    ).transpose(*xr_dataset["sigma0"].dims)
    xr_dataset.sigma0_mask.attrs["valid_range"] = np.array([0, 1])
    xr_dataset.sigma0_mask.attrs["flag_values"] = np.array([0, 1])
    xr_dataset.sigma0_mask.attrs["flag_meanings"] = "valid no_valid"

    # Keep ocean (0) and coastal (1) zones for sigma0_ocean_raw too
    xr_dataset["sigma0_ocean_raw"] = xr.where(
        xr_dataset["mask"] >= 2, np.nan, xr_dataset["sigma0_raw"]
    ).transpose(*xr_dataset["sigma0_raw"].dims)

    xr_dataset["sigma0_ocean_raw"].attrs = xr_dataset["sigma0_raw"].attrs

    xr_dataset["sigma0_detrend"] = xsarsea.sigma0_detrend(
        xr_dataset.sigma0.sel(pol=copol), xr_dataset.incidence, model=model_co
    )

    # processing
    if dual_pol:
        xr_dataset['sigma0_detrend_cross'] = xsarsea.sigma0_detrend(
            xr_dataset.sigma0.sel(pol=crosspol), xr_dataset.incidence, model=model_cross)

        try:
            xr_dataset = xr_dataset.assign(nesz_cross_flattened=(
                ['line', 'sample'], windspeed.nesz_flattening(xr_dataset.nesz.sel(pol=crosspol), xr_dataset.incidence).data))
        except Exception as e:
            if apply_flattening:
                # error
                logging.error("Error during NESZ flattening computation")
                logging.info("%s", traceback.format_exc())
                raise e
            else:
                # replace with nans
                logging.warning("nesz_flattening warning => Error during NESZ flattening computation, but apply_flattening is False, \
                                so continuing without nesz_cross_flattened and replace with NaNs\n \
                                The error comes probably from NaN in incidence angle")
                config["return_status"] = 99
                xr_dataset = xr_dataset.assign(nesz_cross_flattened=(
                    ['line', 'sample'], np.full(xr_dataset.nesz.sel(pol=crosspol).shape, np.nan)))

        xr_dataset['nesz_cross_flattened'].attrs[
            "comment"] = 'nesz has been flattened using windspeed.nesz_flattening'

        if dsig_cr_step == "nrcs":
            # dsig
            if apply_flattening:
                xr_dataset["dsig_cross"] = windspeed.get_dsig(
                    dsig_cr_name,
                    xr_dataset.incidence,
                    xr_dataset["sigma0_ocean"].sel(pol=crosspol),
                    xr_dataset.nesz_cross_flattened,
                )

                xr_dataset.dsig_cross.attrs["formula_used"] = config[
                    "dsig_" + crosspol_gmf + "_NAME"
                ]

            else:
                xr_dataset["dsig_cross"] = windspeed.get_dsig(
                    dsig_cr_name,
                    xr_dataset.incidence,
                    xr_dataset["sigma0_ocean"].sel(pol=crosspol),
                    xr_dataset.nesz.sel(pol=crosspol),
                )

            xr_dataset.dsig_cross.attrs["comment"] = (
                "variable used to ponderate copol and crosspol. this ponderation is done will combining cost functions during inversion process"
            )

            xr_dataset.dsig_cross.attrs["apply_flattening"] = str(
                apply_flattening
            )

    if (recalibration) & ("SENTINEL" in sensor_longname):
        xr_dataset.attrs["aux_cal_recal"] = xsar_dataset.datatree["recalibration"].attrs["aux_cal_new"]
        xr_dataset.attrs["aux_pp1_recal"] = xsar_dataset.datatree["recalibration"].attrs["aux_pp1_new"]

    if add_nrcs_model:
        # add timing
        phi = np.abs(
            np.rad2deg(
                xsarsea.dir_meteo_to_sample(
                    xr_dataset["ancillary_wind_direction"], xr_dataset["ground_heading"]
                )
            )
        )

        varnames = ["ancillary_nrcs"]
        gmf_names = [model_co]
        if dual_pol:
            varnames.append("ancillary_nrcs_cross")
            gmf_names.append(model_cross)

        for idx, gmf_name in enumerate(gmf_names):

            @timing(logger=root_logger.info)
            def apply_lut_to_dataset():
                lut = xsarsea.windspeed.get_model(
                    gmf_name).to_lut(unit="linear")

                def lut_selection(incidence, wspd, phi):
                    if "phi" in lut.coords:
                        return lut.sel(
                            incidence=incidence, wspd=wspd, phi=phi, method="nearest"
                        )
                    else:
                        return lut.sel(incidence=incidence, wspd=wspd, method="nearest")

                xr_dataset[varnames[idx]] = xr.apply_ufunc(
                    lut_selection,
                    xr_dataset.incidence,
                    xr_dataset.ancillary_wind_speed,
                    phi,
                    vectorize=True,
                    dask="parallelized",
                    output_dtypes=[float],
                )

            apply_lut_to_dataset()

    return xr_dataset, out_file, config


def process_gradients(xr_dataset, config):
    """
    Function to process gradients features.

    Parameters
    ----------
    xr_dataset : xarray.Dataset
        Main dataset to process.
    meta : object
        Metadata from the original dataset.
    fct_dataset : callable
        Function to load the dataset.
    map_model : dict
        Mapping model for renaming variables.
    config : dict
        Configuration dictionary.

    Returns
    -------
    tuple
        Updated xr_dataset and xr_dataset_streaks dataset.
    """
    from grdwindinversion.gradientFeatures import GradientFeatures

    meta = config["meta"]
    fct_dataset = config["fct_dataset"]
    map_model = config["map_model"]

    model_co = config["l2_params"]["model_co"]
    model_cross = config["l2_params"]["model_cross"]
    copol = config["l2_params"]["copol"]
    crosspol = config["l2_params"]["crosspol"]
    dual_pol = config["l2_params"]["dual_pol"]

    # Load the 100m dataset
    xsar_dataset_100 = fct_dataset(meta, resolution="100m")

    xr_dataset_100 = xsar_dataset_100.datatree["measurement"].to_dataset()
    xr_dataset_100 = xr_dataset_100.rename(map_model)
    # load dataset
    xr_dataset_100 = xr_dataset_100.load()

    # adding sigma0 detrend
    xr_dataset_100["sigma0_detrend"] = xsarsea.sigma0_detrend(
        xr_dataset_100.sigma0.sel(pol=copol), xr_dataset_100.incidence, model=model_co
    )

    if dual_pol:
        xr_dataset_100["sigma0_detrend_cross"] = xsarsea.sigma0_detrend(
            xr_dataset_100.sigma0.sel(pol=crosspol),
            xr_dataset_100.incidence,
            model=model_cross,
        )

        sigma0_detrend_combined = xr.concat(
            [xr_dataset_100["sigma0_detrend"],
                xr_dataset_100["sigma0_detrend_cross"]],
            dim="pol",
        )
        sigma0_detrend_combined["pol"] = [copol, crosspol]

        xr_dataset_100["sigma0_detrend"] = sigma0_detrend_combined

    # Process land mask with coastal zone detection (3-level system)
    processLandMask(xr_dataset_100, dilation_iterations=3)

    # Mask sigma0_detrend where land_mask >= 2 (land and ice)
    # Keep ocean (0) and coastal (1) zones
    xr_dataset_100["sigma0_detrend"] = xr.where(
        xr_dataset_100["land_mask"] >= 2, np.nan, xr_dataset_100["sigma0"]
    ).transpose(*xr_dataset_100["sigma0"].dims)

    xr_dataset_100["ancillary_wind"] = (
        xr_dataset_100.model_U10 + 1j * xr_dataset_100.model_V10
    ) * np.exp(1j * np.deg2rad(xr_dataset_100.ground_heading))

    downscales_factors = [1, 2, 4, 8]
    # 4 and 8 must be in downscales_factors
    assert all([x in downscales_factors for x in [4, 8]])

    gradientFeatures = GradientFeatures(
        xr_dataset=xr_dataset,
        xr_dataset_100=xr_dataset_100,
        windows_sizes=[1600, 3200],
        downscales_factors=downscales_factors,
        window_step=1,
    )

    # Compute heterogeneity mask and variables
    dataArraysHeterogeneity = gradientFeatures.get_heterogeneity_mask(config)
    xr_dataset = xr_dataset.merge(dataArraysHeterogeneity)

    # Add streaks dataset
    streaks_indiv = gradientFeatures.streaks_individual()
    if "longitude" in streaks_indiv:
        xr_dataset_streaks = xr.Dataset(
            {
                "longitude": streaks_indiv.longitude,
                "latitude": streaks_indiv.latitude,
                "dir_smooth": streaks_indiv.angle,
                "dir_mean_smooth": gradientFeatures.streaks_mean_smooth().angle,
                "dir_smooth_mean": gradientFeatures.streaks_smooth_mean().angle,
            }
        )
    else:
        root_logger.warning(
            "process_gradients warning : 'longitude' not found in streaks_indiv : there is probably an error"
        )
        xr_dataset_streaks = None

    return xr_dataset, xr_dataset_streaks


@timing(logger=root_logger.info)
def makeL2(
    filename, outdir, config_path, overwrite=False, generateCSV=True, resolution="1000m"
):
    """
    Main function to generate L2 product.

    Parameters
    ----------
    filename : str
        input filename
    outdir : str
        output folder

    config_path : str
        configuration file path
    overwrite : bool, optional
        overwrite existing file
    generateCSV : bool, optional
        generate CSV file
    resolution : str, optional
        working resolution

    Returns
    -------
    str
        output filename
    xarray.Dataset
        final dataset
    """

    xr_dataset, out_file, config = preprocess(
        filename, outdir, config_path, overwrite, resolution
    )

    # Drop only masks added from config (not internal masks like sigma0_mask, owiMask_Nrcs)
    masks_by_category = config.get("masks_by_category", {})
    masks_to_drop = []
    for category, mask_list in masks_by_category.items():
        masks_to_drop.extend(mask_list)

    # Only drop masks that actually exist in the dataset (with XSAR suffix)
    vars_to_drop = [
        m+XSAR_MASK_SUFFIX for m in masks_to_drop if (m+XSAR_MASK_SUFFIX) in xr_dataset.data_vars]
    if vars_to_drop:
        logging.info(f"Dropping external masks of dataset: {vars_to_drop}")
        xr_dataset = xr_dataset.drop_vars(vars_to_drop)

    if config["add_gradientsfeatures"]:
        xr_dataset, xr_dataset_streaks = process_gradients(xr_dataset, config)
    else:
        xr_dataset_streaks = None

    model_co = config["l2_params"]["model_co"]
    model_cross = config["l2_params"]["model_cross"]
    copol = config["l2_params"]["copol"]
    crosspol = config["l2_params"]["crosspol"]
    copol_gmf = config["l2_params"]["copol_gmf"]
    crosspol_gmf = config["l2_params"]["crosspol_gmf"]
    dual_pol = config["l2_params"]["dual_pol"]
    ancillary_name = config["ancillary"]
    sensor_longname = config["sensor_longname"]
    dsig_cr_step = config["dsig_cr_step"]
    dsig_cr_name = config["dsig_cr_name"]
    apply_flattening = config["apply_flattening"]
    if dual_pol:
        sigma0_ocean_cross = xr_dataset["sigma0_ocean"].sel(pol=crosspol)
        if dsig_cr_step == "nrcs":
            dsig_cross = xr_dataset["dsig_cross"]
        else:
            dsig_cross = 0.1
    else:
        sigma0_ocean_cross = None
        dsig_cross = 0.1  # default value set in xsarsea

    kwargs = {
        "inc_step_lr": config.pop("inc_step_lr", None),
        "wpsd_step_lr": config.pop("wspd_step_lr", None),
        "phi_step_lr": config.pop("phi_step_lr", None),
        "inc_step": config.pop("inc_step", None),
        "wpsd_step": config.pop("wspd_step", None),
        "phi_step": config.pop("phi_step", None),
        "resolution": config.pop("resolution", None),
    }

    config["return_status"] = 0  # default value SUCCESS
    logging.info("Checking incidence range within LUTS incidence range")
    inc_check_co, inc_check_cross = check_incidence_range(
        xr_dataset["incidence"], [model_co, model_cross], **kwargs
    )

    if not inc_check_co or not inc_check_cross:
        config["return_status"] = 99

    if dsig_cr_step == "nrcs":
        logging.info(
            "dsig_cr_step is nrcs : polarization are mixed at cost function step")
        wind_co, wind_dual, windspeed_cr = inverse(
            dual_pol,
            inc=xr_dataset["incidence"],
            sigma0=xr_dataset["sigma0_ocean"].sel(pol=copol),
            sigma0_dual=sigma0_ocean_cross,
            ancillary_wind=xr_dataset["ancillary_wind"],
            dsig_cr=dsig_cross,
            model_co=model_co,
            model_cross=model_cross,
            **kwargs,
        )
    elif dsig_cr_step == "wspd":
        logging.info(
            "dsig_cr_step is wspd : polarization are mixed at winds speed step")

        if apply_flattening:
            nesz_cross = xr_dataset["nesz_cross_flattened"]
        else:
            nesz_cross = xr_dataset.nesz.sel(pol=crosspol)

        wind_co, wind_dual, windspeed_cr, alpha = inverse_dsig_wspd(
            dual_pol,
            inc=xr_dataset["incidence"],
            sigma0=xr_dataset["sigma0_ocean"].sel(pol=copol),
            sigma0_dual=sigma0_ocean_cross,
            ancillary_wind=xr_dataset["ancillary_wind"],
            nesz_cr=nesz_cross,
            dsig_cr_name=dsig_cr_name,
            model_co=model_co,
            model_cross=model_cross,
            **kwargs
        )
        xr_dataset["alpha"] = xr.DataArray(
            data=alpha, dims=xr_dataset["incidence"].dims, coords=xr_dataset["incidence"].coords)
        xr_dataset["alpha"].attrs["apply_flattening"] = str(apply_flattening)
        xr_dataset["alpha"].attrs["comments"] = "alpha used to ponderate copol and crosspol. this ponderation is done will combining wind speeds."

    else:
        raise ValueError(
            f"dsig_cr_step must be 'nrcs' or 'wspd', got {dsig_cr_step}")

    # windspeed_co
    xr_dataset["windspeed_co"] = np.abs(wind_co)
    xr_dataset["windspeed_co"].attrs["units"] = "m.s⁻1"
    xr_dataset["windspeed_co"].attrs["long_name"] = (
        "Wind speed inverted from model %s (%s)" % (model_co, copol)
    )
    xr_dataset["windspeed_co"].attrs["standart_name"] = "wind_speed"
    xr_dataset["windspeed_co"].attrs["model"] = wind_co.attrs["model"]
    del xr_dataset["windspeed_co"].attrs["comment"]

    # winddir_co
    xr_dataset["winddir_co"] = transform_winddir(
        wind_co,
        xr_dataset.ground_heading,
        winddir_convention=config["winddir_convention"],
    )
    xr_dataset["winddir_co"].attrs["model"] = "%s (%s)" % (model_co, copol)

    # windspeed_dual / windspeed_cr / /winddir_dual / winddir_cr
    if dual_pol and wind_dual is not None:
        xr_dataset["windspeed_dual"] = np.abs(wind_dual)
        xr_dataset["windspeed_dual"].attrs["units"] = "m.s⁻1"
        xr_dataset["windspeed_dual"].attrs["long_name"] = (
            "Wind speed inverted from model %s (%s) & %s (%s)"
            % (model_co, copol, model_cross, crosspol)
        )
        xr_dataset["windspeed_dual"].attrs["standart_name"] = "wind_speed"
        xr_dataset["windspeed_dual"].attrs["model"] = (model_co, model_cross)
        xr_dataset["windspeed_dual"].attrs["combining_method"] = dsig_cr_step

        if "comment" in xr_dataset["windspeed_dual"].attrs:
            del xr_dataset["windspeed_dual"].attrs["comment"]

        xr_dataset["winddir_dual"] = transform_winddir(
            wind_dual,
            xr_dataset.ground_heading,
            winddir_convention=config["winddir_convention"],
        )
        xr_dataset["winddir_dual"].attrs[
            "model"
        ] = "winddir_dual is a copy of copol wind direction"

        xr_dataset = xr_dataset.assign(
            windspeed_cross=(["line", "sample"], windspeed_cr.data)
        )
        xr_dataset["windspeed_cross"].attrs["units"] = "m.s⁻1"
        xr_dataset["windspeed_cross"].attrs["long_name"] = (
            "Wind Speed inverted from model %s (%s)" % (model_cross, crosspol)
        )
        xr_dataset["windspeed_cross"].attrs["standart_name"] = "wind_speed"
        xr_dataset["windspeed_cross"].attrs["model"] = "%s" % (model_cross)

        xr_dataset["winddir_cross"] = xr_dataset["winddir_dual"].copy()
        xr_dataset["winddir_cross"].attrs = xr_dataset["winddir_dual"].attrs
        xr_dataset["winddir_cross"].attrs[
            "model"
        ] = "winddir_cross is a copy of copol wind direction"

    if config["winddir_convention"] == "oceanographic":
        attrs = xr_dataset["ancillary_wind_direction"].attrs
        xr_dataset["ancillary_wind_direction"] = xsarsea.dir_meteo_to_oceano(
            xr_dataset["ancillary_wind_direction"]
        )
        xr_dataset["ancillary_wind_direction"].attrs = attrs
        xr_dataset["ancillary_wind_direction"].attrs[
            "long_name"
        ] = f"{ancillary_name} wind direction in oceanographic convention (clockwise, to), ex: 0°=to north, 90°=to east"

    xr_dataset, encoding = makeL2asOwi(xr_dataset, config)

    xr_dataset = xr_dataset.compute()
    #  add attributes
    firstMeasurementTime = None
    lastMeasurementTime = None
    try:
        firstMeasurementTime = datetime.datetime.strptime(
            xr_dataset.attrs["start_date"], "%Y-%m-%d %H:%M:%S.%f"
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        lastMeasurementTime = datetime.datetime.strptime(
            xr_dataset.attrs["stop_date"], "%Y-%m-%d %H:%M:%S.%f"
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    except:
        firstMeasurementTime = datetime.datetime.strptime(
            xr_dataset.attrs["start_date"], "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        lastMeasurementTime = datetime.datetime.strptime(
            xr_dataset.attrs["stop_date"], "%Y-%m-%d %H:%M:%S"
        ).strftime("%Y-%m-%dT%H:%M:%SZ")

    attrs = {
        "TITLE": "Sentinel-1 OWI Component",
        "productOwner": "IFREMER",
        "sourceProduct": (
            xr_dataset.attrs["safe"]
            if "safe" in xr_dataset.attrs
            else os.path.basename(xr_dataset.attrs["product_path"])
        ),
        "sourceProduct_fullpath": xr_dataset.attrs.pop("name"),
        "missionName": sensor_longname,
        "missionPhase": "Operational",
        "polarisation": xr_dataset.attrs["pols"],
        "acquisitionStation": "",
        "xsar_version": xsar.__version__,
        "xsarsea_version": xsarsea.__version__,
        "pythonVersion": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "polarisationRatio": get_pol_ratio_name(model_co),
        "l2ProcessingUtcTime": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "processingCenter": "IFREMER",
        "firstMeasurementTime": firstMeasurementTime,
        "lastMeasurementTime": lastMeasurementTime,
        "clmSource": "/",
        "bathySource": "/",
        "oswAlgorithmName": "grdwindinversion",
        "owiAlgorithmVersion": grdwindinversion.__version__,
        "gmf": config["GMF_" + copol_gmf + "_NAME"]
        + ", "
        + config["GMF_" + crosspol_gmf + "_NAME"],
        "iceSource": "/",
        "owiNoiseCorrection": "True",
        "inversionTabGMF": config["GMF_" + copol_gmf + "_NAME"]
        + ", "
        + config["GMF_" + crosspol_gmf + "_NAME"],
        "wnf_3km_average": "False",
        "owiWindSpeedSrc": "owiWindSpeed",
        "owiWindDirectionSrc": "/",
        "ancillary_source_model": xr_dataset.attrs["ancillary_source_model"],
        "ancillary_source_path": xr_dataset.attrs["ancillary_source_path"],
        "winddir_convention": config["winddir_convention"],
        "incidence_within_lut_copol_incidence_range": str(inc_check_co),
        "incidence_within_lut_crosspol_incidence_range": str(inc_check_cross),
        "swath": xr_dataset.attrs["swath"],
        "footprint": xr_dataset.attrs["footprint"],
        "coverage": xr_dataset.attrs["coverage"],
        "cross_antimeridian": str(config["meta"].cross_antimeridian)
    }

    for recalib_attrs in ["aux_pp1_recal", "aux_pp1", "aux_cal_recal", "aux_cal"]:
        if recalib_attrs in xr_dataset.attrs:
            attrs[recalib_attrs] = xr_dataset.attrs[recalib_attrs]

    for arg in ["passDirection", "orbit_pass"]:
        if arg in xr_dataset.attrs:
            attrs["passDirection"] = xr_dataset.attrs[arg]

    _S1_added_attrs = ["ipf_version", "platform_heading"]
    _RCM_added_attrs = ["productId"]

    for sup_attr in _S1_added_attrs + _RCM_added_attrs:
        if sup_attr in xr_dataset.attrs:
            attrs[sup_attr] = xr_dataset.attrs[sup_attr]

    attrs["footprint"] = str(attrs["footprint"])

    # add in kwargs in attrs
    for key in kwargs:
        attrs["lut_params_" + key] = "/" if kwargs[key] is None else kwargs[key]

    xr_dataset.attrs = attrs

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    # Sauvegarde de xr_dataset dans le fichier de sortie final
    xr_dataset.to_netcdf(out_file, mode="w", encoding=encoding)

    # Vérifier si le dataset de streaks est présent
    if xr_dataset_streaks is not None:
        # Créer un fichier temporaire pour le dataset streaks
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp_file:
            temp_out_file = tmp_file.name

        # Écrire xr_dataset_streaks dans le fichier temporaire
        xr_dataset_streaks.to_netcdf(
            temp_out_file, mode="w", group="owiWindStreaks")

        # Charger le fichier temporaire et l'ajouter au fichier final en tant que groupe
        with xr.open_dataset(temp_out_file, group="owiWindStreaks") as ds_streaks:
            ds_streaks.to_netcdf(out_file, mode="a", group="owiWindStreaks")

        # Supprimer le fichier temporaire après l'opération
        os.remove(temp_out_file)

    if generateCSV:
        df = xr_dataset.to_dataframe()
        df = df[df.owiMask == False]

        df = df.assign(**xr_dataset.attrs)
        df.reset_index(drop=False, inplace=True)
        df.to_csv(out_file.replace(".nc", ".csv"))

    logging.info("OK for %s ", os.path.basename(filename))

    if config["add_gradientsfeatures"] and xr_dataset_streaks is None:
        config["return_status"] = 99

    return out_file, xr_dataset, config["return_status"]


def transform_winddir(wind_cpx, ground_heading, winddir_convention="meteorological"):
    """

    Parameters
    ----------
    wind_cpx : xr.DataArray | np.complex64
        complex wind, relative to antenna, anticlockwise

    ground_heading : xr.DataArray
        heading angle in degrees

    winddir_convention : str
        wind direction convention to use, either 'meteorological' or 'oceanographic'

    Returns
    -------
        xr.DataArray
        wind direction in degrees in the selected convention with appropriate long_name attribute
    """
    # to meteo winddir_convention
    dataArray = xsarsea.dir_sample_to_meteo(
        np.angle(wind_cpx, deg=True), ground_heading
    )
    long_name = "Wind direction in meteorological convention (clockwise, from), ex: 0°=from north, 90°=from east"

    if winddir_convention == "meteorological":
        # do nothing
        pass
    elif winddir_convention == "oceanographic":
        # to oceano winddir_convention
        dataArray = xsarsea.dir_meteo_to_oceano(dataArray)
        long_name = "Wind direction in oceanographic convention (clockwise, to), ex: 0°=to north, 90°=to east"
    else:
        #  warning
        logging.warning(
            f"wind direction convention {winddir_convention} is not supported, using meteorological",
        )

        long_name = "Wind direction in meteorological convention (clockwise, from), ex: 0°=from north, 90°=from east"

    dataArray = xsarsea.dir_to_360(dataArray)
    dataArray.attrs = {}
    dataArray.attrs["units"] = "degrees_north"
    dataArray.attrs["long_name"] = long_name
    dataArray.attrs["standart_name"] = "wind_direction"

    return dataArray
