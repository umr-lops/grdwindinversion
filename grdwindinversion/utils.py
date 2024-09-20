import logging
import xsarsea


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

        inc_range = [incidence.values.min(), incidence.values.max()]
        if (inc_range[0] >= lut_range[0] and inc_range[1] <= lut_range[1]):
            rets.append(True)
        else:
            logging.warn(
                f"incidence range {inc_range} is not within the range of the LUT of the model {model_name} {lut_range} : inversion will be approximate using LUT minmium|maximum incidences")
            rets.append(False)

    return rets
