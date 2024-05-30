import traceback

import xsar
import xsarsea
from xsarsea import windspeed
import grdwindinversion
import xarray as xr
import numpy as np
import sys
import datetime
import os
import yaml
from pathlib import Path
from scipy.ndimage import binary_dilation

import re
import string
import os
from grdwindinversion.load_config import getConf
# optional debug messages
import logging
logging.basicConfig()
logging.getLogger('xsarsea.windspeed').setLevel(
    logging.INFO)  # or .setLevel(logging.INFO)


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
    if ("S1A" in filename):
        return "S1A", "SENTINEL-1 A", xsar.Sentinel1Meta, xsar.Sentinel1Dataset
    elif ("S1B" in filename):
        return "S1B", "SENTINEL-1 B", xsar.Sentinel1Meta, xsar.Sentinel1Dataset
    elif ("RS2" in filename):
        return "RS2", "RADARSAT-2", xsar.RadarSat2Meta, xsar.RadarSat2Dataset
    elif ("RCM" in filename):
        return "RCM", "RADARSAT Constellation", xsar.RcmMeta, xsar.RcmDataset
    else:
        raise ValueError("must be S1A|S1B|RS2|RCM, got filename %s" % filename)


def getOutputName2(input_file, out_folder, sensor, meta):
    """
    Create output filename for L2-GRD product

    Parameters
    ----------
    input_file : str
        input filename
    out_folder : str
        output folder
    sensor : str
        sensor name
    meta : obj `xsar.BaseMeta` (one of the supported SAR mission)
        meta object

    Returns
    -------
    outfile : str
        output filename
    """
    basename = os.path.basename(input_file)
    basename_match = basename
    meta_start_date = meta.start_date.split(".")[0].replace(
        "-", "").replace(":", "").replace(" ", "t").replace("Z", "")
    meta_stop_date = meta.stop_date.split(".")[0].replace(
        "-", "").replace(":", "").replace(" ", "t").replace("Z", "")

    if sensor == 'S1A' or sensor == 'S1B':
        regex = re.compile(
            "(...)_(..)_(...)(.)_(.)(.)(..)_(........T......)_(........T......)_(......)_(......)_(....).SAFE")
        template = string.Template(
            "${MISSIONID}_${BEAM}_${PRODUCT}${RESOLUTION}_${LEVEL}${CLASS}${POL}_${STARTDATE}_${STOPDATE}_${ORBIT}_${TAKEID}_${PRODID}.SAFE")
        match = regex.match(basename_match)
        MISSIONID, BEAM, PRODUCT, RESOLUTION, LEVEL, CLASS, POL, STARTDATE, STOPDATE, ORBIT, TAKEID, PRODID = match.groups()
        new_format = f"{MISSIONID.lower()}-{BEAM.lower()}-owi-xx-{STARTDATE.lower()}-{STOPDATE.lower()}-{ORBIT}-{TAKEID}.nc"
        out_file = os.path.join(out_folder, basename, new_format)
        return out_file

    elif sensor == 'RS2':
        regex = re.compile(
            "(RS2)_OK([0-9]+)_PK([0-9]+)_DK([0-9]+)_(....)_(........)_(......)_(.._?.?.?)_(S.F)")
        template = string.Template(
            "${MISSIONID}_OK${DATA1}_PK${DATA2}_DK${DATA3}_${DATA4}_${DATE}_${TIME}_${POLARIZATION}_${LAST}")
        match = regex.match(basename_match)

        MISSIONID, DATA1, DATA2, DATA3, DATA4, DATE, TIME, POLARIZATION, LAST = match.groups()
        new_format = f"{MISSIONID.lower()}--owi-xx-{meta_start_date.lower()}-{meta_stop_date.lower()}-_____-_____.nc"
        out_file = os.path.join(out_folder, basename, new_format)
        return out_file

    elif sensor == 'RCM':
        regex = re.compile(
            "([A-Z0-9]+)_OK([0-9]+)_PK([0-9]+)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)")
        template = string.Template(
            "${MISSIONID}_OK${DATA1}_PK${DATA2}_${DATA3}_${DATA4}_${DATE}_${TIME}_${POLARIZATION1}_${POLARIZATION2}_${PRODUCT}")
        match = regex.match(basename_match)
        MISSIONID, DATA1, DATA2, DATA3, DATA4, DATE, TIME, POLARIZATION1, POLARIZATION2, LAST = match.groups()
        new_format = f"{MISSIONID.lower()}--owi-xx-{meta_start_date.lower()}-{meta_stop_date.lower()}-_____-_____.nc"
        out_file = os.path.join(out_folder, basename, new_format)
        return out_file

    else:
        raise ValueError(
            "sensor must be S1A|S1B|RS2|RCM, got sensor %s" % sensor)


def getAncillary(meta, ancillary_name='ecmwf'):
    """
    Map ancillary wind from ECMWF.
    This function is used to check if the ECMWF files are available and to map the model to the SAR data.

    Parameters
    ----------
    meta: obj `xsar.BaseMeta` (one of the supported SAR mission)

    Returns
    -------
    dict
        map model to SAR data
    """

    if ancillary_name == 'ecmwf':

        logging.debug('conf: %s', getConf())
        ec01 = getConf()['ecmwf_0100_1h']
        ec0125 = getConf()['ecmwf_0125_1h']
        logging.debug('ec01 : %s', ec01)
        meta.set_raster('ecmwf_0100_1h', ec01)
        meta.set_raster('ecmwf_0125_1h', ec0125)

        map_model = None
        # only keep best ecmwf  (FIXME: it's hacky, and xsar should provide a better method to handle this)
        for ecmwf_name in ['ecmwf_0125_1h', 'ecmwf_0100_1h']:
            ecmwf_infos = meta.rasters.loc[ecmwf_name]
            try:
                ecmwf_file = ecmwf_infos['get_function'](ecmwf_infos['resource'],
                                                         date=datetime.datetime.strptime(meta.start_date,
                                                                                         '%Y-%m-%d %H:%M:%S.%f'))[1]
            # temporary for RCM issue https://github.com/umr-lops/xarray-safe-rcm/issues/34
            except Exception as e:
                ecmwf_file = ecmwf_infos['get_function'](ecmwf_infos['resource'],
                                                         date=datetime.datetime.strptime(meta.start_date,
                                                                                         '%Y-%m-%d %H:%M:%S'))[1]
            if not os.path.isfile(ecmwf_file):
                # temporary
                # if repro does not exist we look at not repro folder (only one will exist after)
                """
                if ecmwf_name == "ecmwf_0100_1h":
                    ecmwf_infos['resource'] = ecmwf_infos['resource'].replace(
                        "netcdf_light_REPRO_tree", "netcdf_light")
                    try:
                        ecmwf_file = ecmwf_infos['get_function'](ecmwf_infos['resource'],
                                                                 date=datetime.datetime.strptime(meta.start_date,
                                                                                                 '%Y-%m-%d %H:%M:%S.%f'))[1]
                    except Exception as e:
                        ecmwf_file = ecmwf_infos['get_function'](ecmwf_infos['resource'],
                                                                 date=datetime.datetime.strptime(meta.start_date,
                                                                                                 '%Y-%m-%d %H:%M:%S'))[1]

                    if not os.path.isfile(ecmwf_file):
                        meta.rasters = meta.rasters.drop([ecmwf_name])
                    else:
                        map_model = {'%s_%s' % (ecmwf_name, uv): 'model_%s' % uv for uv in [
                            'U10', 'V10']}

                else:
                """
                meta.rasters = meta.rasters.drop([ecmwf_name])
            else:
                map_model = {'%s_%s' % (ecmwf_name, uv): 'model_%s' %
                             uv for uv in ['U10', 'V10']}

        return map_model

    else:
        raise ValueError("ancillary_name must be ecmwf, got %s" %
                         ancillary_name)


def inverse(dual_pol, inc, sigma0, sigma0_dual, ancillary_wind, dsig_cr, model_vv, model_vh):
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
            | (for example ecmwf winds), in **model convention**
    dsig_cr=: float or xarray.DataArray
        parameters used for

            | `Jsig_cr=((sigma0_gmf - sigma0) / dsig_cr) ** 2`    
    model_vv=: str 
        model to use for VV or HH polarization.
    model_vh=: str 
        model to use for VH or HV polarization.

    Returns
    -------
    xarray.DataArray or tuple
        inverted wind in **gmf convention** .

    See Also
    --------
    xsarsea documentation 
    https://cyclobs.ifremer.fr/static/sarwing_datarmor/xsarsea/examples/windspeed_inversion.html    
    """
    logging.debug("inversion")

    # add potential missing gmfs (only cmod7 & ms1ahw)

    if (model_vv == "gmf_cmod7"):
        windspeed.register_cmod7(getConf()["lut_cmod7_path"])

    if (model_vh == "sarwing_lut_cmodms1ahw"):
        windspeed.register_one_sarwing_lut(getConf()["lut_ms1ahw_path"])

    winds = windspeed.invert_from_model(
        inc,
        sigma0,
        sigma0_dual,
        ancillary_wind=-np.conj(ancillary_wind),
        dsig_cr=dsig_cr,
        model=(model_vv, model_vh))

    if dual_pol:
        wind_co, wind_dual = winds

        wind_cross = windspeed.invert_from_model(
            inc.values,
            sigma0_dual.values,
            dsig_cr=dsig_cr.values,
            model=model_vh)

        return wind_co, wind_dual, wind_cross
    else:
        wind_co = winds

    return wind_co, None, None


def makeL2asOwi(xr_dataset, dual_pol, copol, crosspol, copol_gmf, crosspol_gmf, config):
    """
    Rename xr_dataset variables and attributes to match naming convention.

    Parameters
    ----------
    xr_dataset: xarray.Dataset
        dataset to rename
    dual_pol: bool
        True if dualpol, False if singlepol
    copol: str
        copolarization name
    crosspol: str
        crosspolarization name
    copol_gmf: str
        copolarization GMF name
    crosspol_gmf: str
        crosspolarization GMF name
    config: dict
        configuration file

    Returns
    -------
    xarray.Dataset
        final dataset 
    dict
        encoding dict

    See Also
    --------
    """

    xr_dataset = xr_dataset.rename({
        'longitude': 'owiLon',
        'latitude': 'owiLat',
        'incidence': 'owiIncidenceAngle',
        'elevation': 'owiElevationAngle',
        'ground_heading': 'owiHeading',
        'land_mask': 'owiLandFlag',
        'offboresight': 'owiOffBoresightAngle',
        'mask': 'owiMask',
        'windspeed_co': 'owiWindSpeed_co',
        'windspeed_cross': 'owiWindSpeed_cross',
        'windspeed_dual': 'owiWindSpeed',
        'winddir_co': 'owiWindDirection_co',
        'winddir_cross': 'owiWindDirection_cross',
        'winddir_dual': 'owiWindDirection',
        'ancillary_wind_speed': 'owiEcmwfWindSpeed',
        'ancillary_wind_direction': 'owiEcmwfWindDirection',
    })
    xr_dataset['owiNrcs'] = xr_dataset['sigma0_ocean'].sel(pol=copol)
    xr_dataset.owiNrcs.attrs = xr_dataset.sigma0_ocean.attrs
    xr_dataset.owiNrcs.attrs['units'] = 'm^2 / m^2'
    xr_dataset.owiNrcs.attrs['long_name'] = 'Normalized Radar Cross Section'
    xr_dataset.owiNrcs.attrs['definition'] = 'owiNrcs_no_noise_correction - owiNesz'

    # NESZ & DSIG
    xr_dataset = xr_dataset.assign(
        owiNesz=(['line', 'sample'], xr_dataset.nesz.sel(pol=copol).values))
    xr_dataset.owiNesz.attrs = xr_dataset.nesz.attrs

    xr_dataset['owiNrcs_no_noise_correction'] = xr_dataset['sigma0_ocean_raw'].sel(
        pol=copol)
    xr_dataset.owiNrcs_no_noise_correction.attrs = xr_dataset.sigma0_ocean_raw.attrs
    xr_dataset.owiNrcs_no_noise_correction.attrs['units'] = 'm^2 / m^2'
    xr_dataset.owiNrcs_no_noise_correction.attrs[
        'long_name'] = 'Normalized Radar Cross Section ; no noise correction applied'
    xr_dataset.owiNrcs_no_noise_correction.attrs[
        'comment'] = 'owiNrcs_no_noise_correction ; no recalibration'

    if 'swath_number' in xr_dataset:
        xr_dataset = xr_dataset.rename({
            'swath_number': 'owiSwathNumber',
            'swath_number_flag': 'owiSwathNumberFlag'
        })

    if "sigma0_raw__corrected" in xr_dataset:
        xr_dataset['owiNrcs_no_noise_correction_recalibrated'] = xr_dataset['sigma0_raw__corrected'].sel(
            pol=copol)
        xr_dataset.owiNrcs_no_noise_correction_recalibrated.attrs = xr_dataset.sigma0_raw__corrected.attrs
        xr_dataset.owiNrcs_no_noise_correction_recalibrated.attrs['units'] = 'm^2 / m^2'
        xr_dataset.owiNrcs_no_noise_correction_recalibrated.attrs[
            'long_name'] = 'Normalized Radar Cross Section, no noise correction applied'
        xr_dataset.owiNrcs_no_noise_correction_recalibrated.attrs[
            'comment'] = 'owiNrcs_no_noise_correction ; recalibrated with kersten method'

        xr_dataset.owiNrcs.attrs['definition'] = 'owiNrcs_no_noise_correction_recalibrated - owiNesz'

    if dual_pol:

        xr_dataset = xr_dataset.rename({
            'dsig_cross': 'owiDsig_cross',
            'nesz_cross_final': 'owiNesz_cross_final'
        })

        xr_dataset['owiNrcs_cross'] = xr_dataset['sigma0_ocean'].sel(
            pol=crosspol)
        xr_dataset.owiNrcs_cross.attrs['units'] = 'm^2 / m^2'
        xr_dataset.owiNrcs_cross.attrs['long_name'] = 'Normalized Radar Cross Section'
        xr_dataset.owiNrcs_cross.attrs['definition'] = 'owiNrcs_cross_no_noise_correction - owiNesz_cross'

        xr_dataset = xr_dataset.assign(owiNesz_cross=(
            ['line', 'sample'], xr_dataset.nesz.sel(pol=crosspol).values))  # no flattening
        xr_dataset.owiNesz_cross.attrs = xr_dataset.nesz.attrs
        # unused
        xr_dataset['owiNrcs_cross_no_noise_correction'] = xr_dataset['sigma0_ocean_raw'].sel(
            pol=crosspol)

        xr_dataset.owiNrcs_cross_no_noise_correction.attrs['units'] = 'm^2 / m^2'
        xr_dataset.owiNrcs_cross_no_noise_correction.attrs[
            'long_name'] = 'Normalized Radar Cross Section, no noise correction applied'

        if "sigma0_raw__corrected" in xr_dataset:
            xr_dataset['owiNrcs_cross_no_noise_correction_recalibrated'] = xr_dataset['sigma0_raw__corrected'].sel(
                pol=crosspol)
            xr_dataset.owiNrcs_cross_no_noise_correction_recalibrated.attrs = xr_dataset.sigma0_raw__corrected.attrs
            xr_dataset.owiNrcs_cross_no_noise_correction_recalibrated.attrs['units'] = 'm^2 / m^2'
            xr_dataset.owiNrcs_cross_no_noise_correction_recalibrated.attrs[
                'long_name'] = 'Normalized Radar Cross Section ; no noise correction applied'
            xr_dataset.owiNrcs_cross_no_noise_correction_recalibrated.attrs[
                'comment'] = 'owiNrcs_cross_no_noise_correction ; recalibrated with kersten method'

            xr_dataset.owiNrcs_cross.attrs['definition'] = 'owiNrcs_cross_no_noise_correction_recalibrated - owiNesz_cross'

    xr_dataset["owiWindSpeed_co"].attrs["comment"] = xr_dataset["owiWindSpeed_co"].attrs["comment"].replace(
        "wind speed and direction", "wind speed")

    xr_dataset["owiWindDirection_co"].attrs["comment"] = "wind direction in meteorological convention, 0=North, 90=East"

    if dual_pol:
        xr_dataset["owiWindSpeed"].attrs["comment"] = xr_dataset["owiWindSpeed"].attrs["comment"].replace(
            "wind speed and direction", "wind speed")
        xr_dataset["owiWindDirection"].attrs["comment"] = "wind direction in meteorological convention, 0=North, 90=East"

        xr_dataset["owiWindSpeed_cross"].attrs['comment'] = "wind speed inverted from model %s (%s)" % (
            crosspol_gmf, crosspol)

        xr_dataset["owiWindDirection_cross"].attrs["comment"] = "wind direction in meteorological convention, 0=North, 90=East, copied from dualpol"

        xr_dataset.owiWindSpeed_cross.attrs['model'] = crosspol_gmf
        xr_dataset.owiWindSpeed_cross.attrs['units'] = 'm/s'

    xr_dataset['owiWindQuality'] = xr.full_like(xr_dataset.owiNrcs, 0)
    xr_dataset['owiWindQuality'].attrs[
        'long_name'] = "Quality flag taking into account the consistency_between_wind_inverted_and_NRCS_and_Doppler_measured"
    xr_dataset['owiWindQuality'].attrs['valid_range'] = np.array([0, 3])
    xr_dataset['owiWindQuality'].attrs['flag_values'] = np.array([
        0, 1, 2, 3])
    xr_dataset['owiWindQuality'].attrs['flag_meanings'] = "good medium low poor"
    xr_dataset['owiWindQuality'].attrs['comment'] = 'not done yet'

    xr_dataset['owiWindFilter'] = xr.full_like(xr_dataset.owiNrcs, 0)
    xr_dataset['owiWindFilter'].attrs['long_name'] = "Quality flag taking into account the local heterogeneity"
    xr_dataset['owiWindFilter'].attrs['valid_range'] = np.array([0, 3])
    xr_dataset['owiWindFilter'].attrs['flag_values'] = np.array([
        0, 1, 2, 3])
    xr_dataset['owiWindFilter'].attrs[
        'flag_meanings'] = "homogeneous_NRCS, heterogeneous_from_co-polarization_NRCS, heterogeneous_from_cross-polarization_NRCS, heterogeneous_from_dual-polarization_NRCS"
    xr_dataset['owiWindFilter'].attrs['comment'] = 'not done yet'

    xr_dataset = xr_dataset.rename(
        {"line": "owiAzSize", "sample": "owiRaSize"})

    xr_dataset = xr_dataset.drop_vars(
        ['sigma0_ocean', 'sigma0', 'sigma0_ocean_raw', 'sigma0_raw', 'ancillary_wind', 'nesz', 'spatial_ref'])
    if 'sigma0_raw__corrected' in xr_dataset:
        xr_dataset = xr_dataset.drop_vars(["sigma0_raw__corrected"])
    xr_dataset = xr_dataset.drop_dims(['pol'])

    # attrs

    xr_dataset.compute()

    for var in ['footprint', 'multidataset', 'rawDataStartTime', 'specialHandlingRequired']:
        if var in xr_dataset.attrs:
            xr_dataset.attrs[var] = str(xr_dataset.attrs[var])
        if "approx_transform" in xr_dataset.attrs:
            del xr_dataset.attrs["approx_transform"]

    xr_dataset.attrs["TITLE"] = "Sentinel-1 OWI Component"
    xr_dataset.attrs["missionPhase"] = "Test"
    xr_dataset.attrs["acquisitionStation"] = "/"
    xr_dataset.attrs["softwareVersion"] = "/"
    xr_dataset.attrs["pythonVersion"] = str(
        sys.version_info.major)+'.'+str(sys.version_info.minor)
    xr_dataset.attrs["polarisationRatio"] = "/"
    xr_dataset.attrs["l2ProcessingUtcTime"] = datetime.datetime.now().strftime(
        "%Y-%m-%dT%H:%M:%SZ")
    xr_dataset.attrs["processingCenter"] = "/"
    try:
        xr_dataset.attrs["firstMeasurementTime"] = datetime.datetime.strptime(xr_dataset.attrs['start_date'],
                                                                              "%Y-%m-%d %H:%M:%S.%f").strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        xr_dataset.attrs["lastMeasurementTime"] = datetime.datetime.strptime(xr_dataset.attrs['stop_date'],
                                                                             "%Y-%m-%d %H:%M:%S.%f").strftime(
            "%Y-%m-%dT%H:%M:%SZ")
    except:
        xr_dataset.attrs["firstMeasurementTime"] = datetime.datetime.strptime(xr_dataset.attrs['start_date'],
                                                                              "%Y-%m-%d %H:%M:%S").strftime(
            "%Y-%m-%dT%H:%M:%SZ")
        xr_dataset.attrs["lastMeasurementTime"] = datetime.datetime.strptime(xr_dataset.attrs['stop_date'],
                                                                             "%Y-%m-%d %H:%M:%S").strftime(
            "%Y-%m-%dT%H:%M:%SZ")
    xr_dataset.attrs["clmSource"] = "/"
    xr_dataset.attrs["bathySource"] = "/"
    xr_dataset.attrs['oswAlgorithmName'] = 'grdwindinversion'
    xr_dataset.attrs["owiAlgorithmVersion"] = grdwindinversion.__version__
    xr_dataset.attrs["gmf"] = config['GMF_'+copol_gmf+'_NAME'] + \
        ", " + config["GMF_"+crosspol_gmf+"_NAME"]
    xr_dataset.attrs["iceSource"] = "/"
    xr_dataset.attrs["owiNoiseCorrection"] = "False"
    xr_dataset.attrs["inversionTabGMF"] = config['GMF_'+copol_gmf +
                                                 '_NAME'] + ", " + config["GMF_"+crosspol_gmf+"_NAME"]
    xr_dataset.attrs["wnf_3km_average"] = "/"
    xr_dataset.attrs["owiWindSpeedSrc"] = "owiWindSpeed"
    xr_dataset.attrs["owiWindDirectionSrc"] = "/"

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
            encoding[var].update({'_FillValue': table_fillValue[var]})
        except:
            if (var in ["owiWindSpeed_co", "owiWindSpeed_cross", "owiWindSpeed"]):
                encoding[var].update({'_FillValue': -9999.0})
            else:
                encoding[var].update({'_FillValue': None})

    xr_dataset.attrs["xsar_version"] = xsar.__version__
    xr_dataset.attrs["xsarsea_version"] = xsarsea.__version__

    return xr_dataset, encoding


def makeL2(filename, out_folder, config_path, overwrite=False, generateCSV=True, resolution='1000m'):
    """
    Main function to generate L2 product.

    Parameters
    ----------
    filename : str
        input filename
    out_folder : str
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

    sensor, sensor_longname, fct_meta, fct_dataset = getSensorMetaDataset(
        filename)

    if Path(config_path).exists():
        config = yaml.load(
            Path(config_path).open(),
            Loader=yaml.FullLoader
        )
        try:
            config = config[sensor]
        except Exception:
            raise KeyError("sensor %s not in this config" % sensor)
    else:
        raise FileNotFoundError(
            'config_path do not exists, got %s ' % config_path)

    recalibration = config["recalibration"]
    meta = fct_meta(filename)
    out_file = getOutputName2(filename, out_folder, sensor, meta)

    if os.path.exists(out_file) and overwrite is False:
        logging.info("out_file %s exists ; returning empty Dataset" % out_file)
        return out_file, xr.Dataset()

    map_model = getAncillary(meta, ancillary_name=config["ancillary"])
    if map_model is None:
        raise Exception(
            'the weather model is not set `map_model` is None -> you probably don"t have access to ECMWF archive')

    try:
        if ((recalibration) & ("SENTINEL" in sensor_longname)):
            logging.info(
                f'recalibration is {recalibration} : Kersten formula is applied')
            xsar_dataset = fct_dataset(
                meta, resolution=resolution, recalibration=recalibration)
            xr_dataset = xsar_dataset.datatree['measurement'].to_dataset()
            xr_dataset = xr_dataset.merge(xsar_dataset.datatree["recalibration"].to_dataset()[
                                          ['swath_number', 'swath_number_flag', 'sigma0_raw__corrected']])

        else:
            logging.info(
                f'recalibration is {recalibration} : Kersten formula is not applied')
            if ("SENTINEL" in sensor_longname):
                xsar_dataset = fct_dataset(
                    meta, resolution=resolution, recalibration=recalibration)
                xr_dataset = xsar_dataset.datatree['measurement'].to_dataset()
                xr_dataset = xr_dataset.merge(xsar_dataset.datatree["recalibration"].to_dataset()[
                                              ['swath_number', 'swath_number_flag']])

            else:
                xsar_dataset = fct_dataset(meta, resolution=resolution)
                xr_dataset = xsar_dataset.datatree['measurement'].to_dataset()

        xr_dataset = xr_dataset.rename(map_model)
        # add attributes
        xr_dataset.attrs = xsar_dataset.dataset.attrs
        xr_dataset.attrs['L1_path'] = xr_dataset.attrs.pop('name')
        xr_dataset.attrs["sourceProduct"] = sensor
        xr_dataset.attrs["missionName"] = sensor_longname
        if ((recalibration) & ("SENTINEL" in sensor_longname)):
            xr_dataset.attrs["path_aux_pp1_new"] = os.path.basename(os.path.dirname(
                os.path.dirname(xsar_dataset.datatree['recalibration'].attrs['path_aux_pp1_new'])))
            xr_dataset.attrs["path_aux_cal_new"] = os.path.basename(os.path.dirname(
                os.path.dirname(xsar_dataset.datatree['recalibration'].attrs['path_aux_cal_new'])))

            xr_dataset.attrs["path_aux_pp1_old"] = os.path.basename(os.path.dirname(
                os.path.dirname(xsar_dataset.datatree['recalibration'].attrs['path_aux_pp1_old'])))
            xr_dataset.attrs["path_aux_cal_old"] = os.path.basename(os.path.dirname(
                os.path.dirname(xsar_dataset.datatree['recalibration'].attrs['path_aux_cal_old'])))

    except Exception as e:
        logging.info('%s', traceback.format_exc())
        logging.error(e)
        sys.exit(-1)

    # defining dual_pol, and gmfs by channel
    if len(xr_dataset.pol.values) == 2:
        dual_pol = True
    else:
        dual_pol = False

    if 'VV' in xr_dataset.pol.values:
        copol = 'VV'
        crosspol = 'VH'
        copol_gmf = 'VV'
        crosspol_gmf = 'VH'
    else:
        logging.warning('for now this processor does not support HH+HV acquisitions\n '
                        'it wont crash but it will use VV+VH GMF for wind inversion -> wrong hypothesis\n '
                        '!! WIND SPEED IS NOT USABLE !!')
        copol = 'HH'
        crosspol = 'HV'
        copol_gmf = 'VV'
        crosspol_gmf = 'VH'

    #  Step 2 - clean and prepare dataset

    # variables to not keep in the L2
    black_list = ['digital_number', 'gamma0_raw', 'negz',
                  'azimuth_time', 'slant_range_time', 'velocity', 'range_ground_spacing',
                  'gamma0', 'time', 'nd_co', 'nd_cr', 'gamma0_lut', 'sigma0_lut', "noise_lut_range", "lineSpacing",
                  "sampleSpacing", "noise_lut", "noise_lut_azi",
                  'nebz', 'beta0_raw', 'lines_flipped', 'samples_flipped', "altitude", "beta0"]
    variables = list(set(xr_dataset) - set(black_list))
    xr_dataset = xr_dataset[variables]

    # TODO Better land mask
    # xr_dataset.land_mask.values = cv2.dilate(xr_dataset['land_mask'].values.astype('uint8'),np.ones((3,3),np.uint8),iterations = 3)
    xr_dataset.land_mask.values = binary_dilation(xr_dataset['land_mask'].values.astype('uint8'),
                                                  structure=np.ones((3, 3), np.uint8), iterations=3)
    xr_dataset.land_mask.attrs['long_name'] = 'Mask of data'
    xr_dataset.land_mask.attrs['valid_range'] = np.array([0, 1])
    xr_dataset.land_mask.attrs['flag_values'] = np.array([0, 1])
    xr_dataset.land_mask.attrs['flag_meanings'] = 'valid no_valid'

    # MASK
    # Careful : in sarwing process sometimes there are 2 & 3. Not made here
    logging.debug("mask is a copy of land_mask")

    xr_dataset['mask'] = xr.DataArray(xr_dataset.land_mask)
    xr_dataset.mask.attrs = {}
    xr_dataset.mask.attrs['long_name'] = 'Mask of data'
    xr_dataset.mask.attrs['valid_range'] = np.array([0, 3])
    xr_dataset.mask.attrs['flag_values'] = np.array([0, 1, 2, 3])
    xr_dataset.mask.attrs['flag_meanings'] = 'valid land ice no_valid'

    # ANCILLARY
    xr_dataset['ancillary_wind_direction'] = (
        90. - np.rad2deg(np.arctan2(xr_dataset.model_V10, xr_dataset.model_U10)) + 180) % 360

    xr_dataset['ancillary_wind_direction'] = xr.where(xr_dataset['mask'], np.nan,
                                                      xr_dataset['ancillary_wind_direction'].compute()).transpose(
        *xr_dataset['ancillary_wind_direction'].dims)
    xr_dataset['ancillary_wind_direction'].attrs = {}
    xr_dataset['ancillary_wind_direction'].attrs['units'] = 'degrees_north'
    xr_dataset['ancillary_wind_direction'].attrs[
        'long_name'] = 'ECMWF Wind direction (meteorological convention)'
    xr_dataset['ancillary_wind_direction'].attrs['standart_name'] = 'wind_direction'

    xr_dataset['ancillary_wind_speed'] = np.sqrt(
        xr_dataset['model_U10']**2+xr_dataset['model_V10']**2)
    xr_dataset['ancillary_wind_speed'] = xr.where(xr_dataset['mask'], np.nan,
                                                  xr_dataset['ancillary_wind_speed'].compute()).transpose(
        *xr_dataset['ancillary_wind_speed'].dims)
    xr_dataset['ancillary_wind_speed'].attrs = {}
    xr_dataset['ancillary_wind_speed'].attrs['units'] = 'm s^-1'
    xr_dataset['ancillary_wind_speed'].attrs[
        'long_name'] = 'ECMWF Wind speed'
    xr_dataset['ancillary_wind_speed'].attrs['standart_name'] = 'wind_speed'

    xr_dataset['ancillary_wind'] = xr.where(xr_dataset['mask'], np.nan,
                                            (xr_dataset.ancillary_wind_speed * np.exp(1j * xsarsea.dir_geo_to_sample(xr_dataset.ancillary_wind_direction, xr_dataset.ground_heading))).compute()).transpose(
        *xr_dataset['ancillary_wind_speed'].dims)

    xr_dataset.attrs['ancillary_source'] = xr_dataset['model_U10'].attrs['history'].split('decoded: ')[
        1].strip()
    xr_dataset = xr_dataset.drop_vars(['model_U10', 'model_V10'])

    # NRCS & NESZ
    xr_dataset['sigma0_ocean'] = xr.where(xr_dataset['mask'], np.nan,
                                          xr_dataset['sigma0'].compute()).transpose(*xr_dataset['sigma0'].dims)
    xr_dataset['sigma0_ocean'] = xr.where(
        xr_dataset['sigma0_ocean'] <= 0, np.nan, xr_dataset['sigma0_ocean'])

    xr_dataset['sigma0_ocean'].attrs = xr_dataset['sigma0'].attrs

    xr_dataset['sigma0_ocean_raw'] = xr.where(xr_dataset['mask'], np.nan,
                                              xr_dataset['sigma0_raw'].compute()).transpose(*xr_dataset['sigma0_raw'].dims)
    xr_dataset['sigma0_ocean_raw'] = xr.where(
        xr_dataset['sigma0_ocean_raw'] <= 0, np.nan, xr_dataset['sigma0_ocean_raw'])
    xr_dataset['sigma0_ocean_raw'].attrs = xr_dataset['sigma0_raw'].attrs

    if dual_pol:

        if config["apply_flattening"]:
            xr_dataset = xr_dataset.assign(nesz_cross_final=(
                ['line', 'sample'], windspeed.nesz_flattening(xr_dataset.nesz.sel(pol=crosspol), xr_dataset.incidence)))
            xr_dataset['nesz_cross_final'].attrs[
                "comment"] = 'nesz has been flattened using windspeed.nesz_flattening'
        else:
            xr_dataset = xr_dataset.assign(
                nesz_cross_final=(['line', 'sample'], xr_dataset.nesz.sel(pol=crosspol).values))
            xr_dataset['nesz_cross_final'].attrs["comment"] = 'nesz has not been flattened'
        # dsig

        sigma0_ocean_cross = xr_dataset['sigma0_ocean'].sel(pol=crosspol)

        xr_dataset["dsig_cross"] = windspeed.get_dsig(config["dsig_"+crosspol_gmf+"_NAME"], xr_dataset.incidence,
                                                      sigma0_ocean_cross, xr_dataset.nesz_cross_final)

        xr_dataset.dsig_cross.attrs['comment'] = 'variable used to ponderate copol and crosspol'
        dsig_cross = xr_dataset.dsig_cross
    else:
        sigma0_ocean_cross = None
        dsig_cross = 0.1  # default value set in xsarsea

    wind_co, wind_dual, windspeed_cr = inverse(dual_pol,
                                               inc=xr_dataset.incidence,
                                               sigma0=xr_dataset['sigma0_ocean'].sel(
                                                   pol=copol),
                                               sigma0_dual=sigma0_ocean_cross,
                                               ancillary_wind=xr_dataset['ancillary_wind'],
                                               dsig_cr=dsig_cross,
                                               model_vv=config["GMF_" +
                                                               copol_gmf+"_NAME"],
                                               model_vh=config["GMF_"+crosspol_gmf+"_NAME"])

    #  get windspeeds
    xr_dataset['windspeed_co'] = np.abs(wind_co)
    xr_dataset['windspeed_dual'] = np.abs(wind_dual)

    if dual_pol:
        xr_dataset = xr_dataset.assign(
            windspeed_cross=(['line', 'sample'], windspeed_cr))
    else:
        xr_dataset['windspeed_cross'] = windspeed_cr

    #  get winddirections
    xr_dataset['winddir_co'] = (
        90 - (np.angle(-np.conj(wind_co), deg=True)) + xr_dataset.ground_heading) % 360

    xr_dataset['winddir_dual'] = (
        90 - (np.angle(-np.conj(wind_dual), deg=True)) + xr_dataset.ground_heading) % 360
    xr_dataset['winddir_cross'] = xr_dataset['winddir_dual'].copy()

    xr_dataset, encoding = makeL2asOwi(
        xr_dataset, dual_pol, copol, crosspol, copol_gmf, crosspol_gmf, config)

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    xr_dataset.to_netcdf(out_file, mode="w", encoding=encoding)
    if generateCSV:
        df = xr_dataset.to_dataframe()
        df = df[df.owiMask == False]

        df = df.assign(**xr_dataset.attrs)
        df.reset_index(drop=False, inplace=True)
        df.to_csv(out_file.replace(".nc", ".csv"))

    logging.info("OK for %s ", os.path.basename(filename))

    return out_file, xr_dataset
