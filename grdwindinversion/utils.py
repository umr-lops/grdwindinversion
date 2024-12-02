import os
import time
import logging
import xsarsea


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('grdwindinversion')


mem_monitor = True
try:
    from psutil import Process
except ImportError:
    logger.warning("psutil module not found. Disabling memory monitor")
    mem_monitor = False


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
        if 'inc_range' in kwargs:
            logging.debug(
                f"GMF {model_name} inc_range will be changed by kwargs to {kwargs['inc_range']}")
            lut_range = kwargs['inc_range']

        inc_range = [incidence.min(), incidence.max()]
        if (inc_range[0] >= lut_range[0] and inc_range[1] <= lut_range[1]):
            rets.append(True)
        else:
            logging.warn(
                f"incidence range {inc_range} is not within the range of the LUT of the model {model_name} {lut_range} : inversion will be approximate using LUT minmium|maximum incidences")
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
    if model.pol == 'HH':
        try:
            import re

            def check_format(s):
                pattern = r'^([a-zA-Z0-9]+)_R(high|low)_hh_([a-zA-Z0-9_]+)$'
                match = re.match(pattern, s)
                if match:
                    vvgmf, res, polrationame = match.groups()
                    return polrationame
                else:
                    logging.warn(
                        f"String format is not correct for polarization ratio name = {s}\nReturning '/'")
                    return "/"
            get_pol_ratio_name = check_format(model_co)
            return get_pol_ratio_name
        except AttributeError:
            return "not_written_in_lut"
    else:
        return '/'


def timing(logger=logger.debug):
    """provide a @timing decorator() for functions, that log time spent in it"""

    def decorator(f):
        # @wraps(f)
        def wrapper(*args, **kwargs):
            mem_str = ''
            process = None
            if mem_monitor:
                process = Process(os.getpid())
                startrss = process.memory_info().rss
            starttime = time.time()
            result = f(*args, **kwargs)
            endtime = time.time()
            if mem_monitor:
                endrss = process.memory_info().rss
                mem_str = 'mem: %+.1fMb' % ((endrss - startrss) / (1024 ** 2))
            logger(
                'timing %s : %.2fs. %s' % (f.__name__, endtime - starttime, mem_str))
            return result
        wrapper.__doc__ = f.__doc__
        return wrapper
    return decorator
