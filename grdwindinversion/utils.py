import os
import time
import logging
import xsarsea


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("grdwindinversion")


mem_monitor = True
try:
    from psutil import Process
except ImportError:
    logger.warning("psutil module not found. Disabling memory monitor")
    mem_monitor = False


def convert_polarization_name(pol):
    """
    Convert polarization name to the format used in the output filename

    Parameters
    ----------
    pol : str
        polarization name

    Returns
    ------- 
    str
        polarization name in the format used in the output filename (dv/dh/sv/sh/xx)
    """
    if pol == "VV_VH":
        return "dv"
    elif pol == "HH_HV":
        return "dh"
    elif pol == "VV":
        return "sv"
    elif pol == "HH":
        return "sh"
    else:
        return "xx"


def check_incidence_range(incidence, models, **kwargs):
    """
    Check if the incidence range of the dataset is within the range of the LUT of the model.
    If not, warn the user : inversion will be approximate.

    Parameters
    ----------
    incidence : xr.DataArray
        incidence angle in degrees
    models : list of str
        list of model names

    Returns
    -------
    list of bool
        for each model,
        True if the incidence range is within the range of the LUT of the model (correct)
        False otherwise
    """
    if isinstance(models, str):
        models = [models]
    elif not isinstance(models, list):
        raise TypeError("models should be a string or a list of strings")

    rets = []
    for model_name in models:
        lut_range = xsarsea.windspeed.get_model(model_name).inc_range
        if "inc_range" in kwargs:
            logging.debug(
                f"GMF {model_name} inc_range will be changed by kwargs to {kwargs['inc_range']}"
            )
            lut_range = kwargs["inc_range"]

        inc_range = [incidence.min(), incidence.max()]
        if inc_range[0] >= lut_range[0] and inc_range[1] <= lut_range[1]:
            rets.append(True)
        else:
            logging.warning(
                f"check_incidence_range warning : incidence range {inc_range} is not within the range of the LUT of the model {model_name} {lut_range} : inversion will be approximate using LUT minmium|maximum incidences"
            )
            rets.append(False)

    return rets


def get_pol_ratio_name(model_co):
    """
    Return polarization ration name of copol model

    Parameters
    ----------
    model_co : str
        copol model name

    Returns
    -------
    str
        if pol = 'HH', return polarization ratio name ; else return '/'
    """

    model = xsarsea.windspeed.get_model(model_co)
    if model.pol == "HH":
        try:
            import re

            def check_format(s):
                pattern = r"^([a-zA-Z0-9_]+)_R(high|low)_hh_([a-zA-Z0-9_]+)$"
                match = re.match(pattern, s)
                if match:
                    vvgmf, res, polrationame = match.groups()
                    return polrationame
                else:
                    logging.warning(
                        f"String format is not correct for polarization ratio name = {s}\nReturning '/'"
                    )
                    return "/"

            get_pol_ratio_name = check_format(model_co)
            return get_pol_ratio_name
        except AttributeError:
            return "not_written_in_lut"
    else:
        return "/"


def timing(logger=logger.debug):
    """provide a @timing decorator() for functions, that log time spent in it"""

    def decorator(f):
        # @wraps(f)
        def wrapper(*args, **kwargs):
            mem_str = ""
            process = None
            if mem_monitor:
                process = Process(os.getpid())
                startrss = process.memory_info().rss
            starttime = time.time()
            result = f(*args, **kwargs)
            endtime = time.time()
            if mem_monitor:
                endrss = process.memory_info().rss
                mem_str = "mem: %+.1fMb" % ((endrss - startrss) / (1024**2))
            logger("timing %s : %.2fs. %s" %
                   (f.__name__, endtime - starttime, mem_str))
            return result

        wrapper.__doc__ = f.__doc__
        return wrapper

    return decorator


def test_config(config):
    """
    Validate configuration structure.

    Checks that the configuration contains all required fields:
    - ancillary_sources (with ecmwf or era5)
    - nc_luts_path (required)
    - lut_cmod7_path (optional - required only if using gmf_cmod7)
    - lut_ms1ahw_path (optional - required only if using gmf_cmodms1ahw)
    - masks (optional)

    Note: If you use predefined LUTs (gmf_cmod7, gmf_cmodms1ahw, etc.),
    they must be loaded with the corresponding path keywords:
    - gmf_cmod7 requires lut_cmod7_path
    - gmf_cmodms1ahw requires lut_ms1ahw_path

    Parameters
    ----------
    config : dict
        Configuration dictionary to validate

    Raises
    ------
    ValueError
        If configuration is missing required fields or has invalid structure
    """
    # Check ancillary_sources
    if 'ancillary_sources' not in config:
        raise ValueError("Configuration must contain 'ancillary_sources'")

    if not isinstance(config['ancillary_sources'], dict):
        raise ValueError("'ancillary_sources' must be a dictionary")

    if not config['ancillary_sources']:
        raise ValueError("'ancillary_sources' must not be empty")

    # Check that at least ecmwf or era5 is configured
    has_ecmwf = 'ecmwf' in config['ancillary_sources']
    has_era5 = 'era5' in config['ancillary_sources']
    if not (has_ecmwf or has_era5):
        raise ValueError(
            "'ancillary_sources' must contain at least 'ecmwf' or 'era5'")

    # Validate ancillary sources structure
    for ancillary_type, sources in config['ancillary_sources'].items():
        if not isinstance(sources, list):
            raise ValueError(
                f"'ancillary_sources.{ancillary_type}' must be a list")
        if not sources:
            raise ValueError(
                f"'ancillary_sources.{ancillary_type}' must not be empty")

        for source in sources:
            if 'name' not in source:
                raise ValueError(
                    f"Each source in 'ancillary_sources.{ancillary_type}' must have a 'name' field")
            if 'path' not in source:
                raise ValueError(
                    f"Each source in 'ancillary_sources.{ancillary_type}' must have a 'path' field")
    # Check LUT paths
    if 'nc_luts_path' not in config:
        raise ValueError("Configuration must contain 'nc_luts_path'")
    else:
        logger.debug(f"nc_luts_path found: {config['nc_luts_path']}")

    # Optional LUT paths (only needed if using specific GMFs)
    if 'lut_cmod7_path' in config:
        logger.debug(f"lut_cmod7_path found: {config['lut_cmod7_path']}")
    if 'lut_ms1ahw_path' in config:
        logger.debug(f"lut_ms1ahw_path found: {config['lut_ms1ahw_path']}")

    # Validate masks structure if present (optional)
    if 'masks' in config:
        if not isinstance(config['masks'], dict):
            raise ValueError("'masks' must be a dictionary")

        for category, mask_list in config['masks'].items():
            if not isinstance(mask_list, list):
                raise ValueError(f"'masks.{category}' must be a list")

            for mask in mask_list:
                if 'name' not in mask:
                    raise ValueError(
                        f"Each mask in 'masks.{category}' must have a 'name' field")
                if 'path' not in mask:
                    raise ValueError(
                        f"Each mask in 'masks.{category}' must have a 'path' field")

        logger.debug(f"Masks configured: {list(config['masks'].keys())}")
    else:
        logger.info("No masks configured (optional)")

    # Check which sensors are configured
    supported_sensors = ['S1A', 'S1B', 'S1C', 'S1D', 'RS2', 'RCM']
    configured_sensors = [
        sensor for sensor in supported_sensors if sensor in config]
    if configured_sensors:
        logger.info(f"Sensors configured: {', '.join(configured_sensors)}")
    else:
        logger.warning(
            "No sensors configured - at least one sensor (S1A, S1B, S1C, S1D, RS2, RCM) should be present")

    logger.info("Configuration validation passed")
