import xsar
import xsarsea
from xsarsea import windspeed

import xarray as xr
import numpy as np

import datetime, os, yaml
from pathlib import Path
from scipy.ndimage import binary_dilation

import re, string, os
from grdwindinversion.load_config import getConf

# optional debug messages
import logging
logging.basicConfig()
logging.getLogger('xsarsea.windspeed').setLevel(logging.DEBUG) # or .setLevel(logging.INFO)


def getSensorMetaDataset(filename):
    """

    :param filename: str SAR SAFE or equivalent
    :return:
    """
    if ("S1A" in filename) :
        return "S1A","SENTINEL-1 A",xsar.Sentinel1Meta,xsar.Sentinel1Dataset
    elif ("S1B" in filename):
        return "S1B","SENTINEL-1 B",xsar.Sentinel1Meta,xsar.Sentinel1Dataset
    elif ("RS2" in filename):
        return "RS2","RADARSAT-2",xsar.RadarSat2Meta,xsar.RadarSat2Dataset
    elif ("RCM" in filename):
        return "RCM","RADARSAT Constellation",xsar.RcmMeta,xsar.RcmDataset
    else:
        raise ValueError("must be S1A|S1B|RS2|RCM, got filename %s" % filename)

def getOutputName2(input_file, out_folder,sensor,meta):
    """

    :param input_file: str
    :param out_folder: str
    :param sensor: str S1A or S1B
    :param meta: obj `xsar.Sentinel1Meta` (or any other supported SAR mission)
    :return:
    """
    basename = os.path.basename(input_file)

    meta_start_date = meta.start_date.split(".")[0].replace("-", "").replace(":", "").replace(" ", "t").replace("Z", "")
    meta_stop_date = meta.stop_date.split(".")[0].replace("-", "").replace(":", "").replace(" ", "t").replace("Z", "")

    if sensor == 'S1A' or sensor == 'S1B':
        regex = re.compile("(...)_(..)_(...)(.)_(.)(.)(..)_(........T......)_(........T......)_(......)_(......)_(....).SAFE")
        template = string.Template("${MISSIONID}_${BEAM}_${PRODUCT}${RESOLUTION}_${LEVEL}${CLASS}${POL}_${STARTDATE}_${STOPDATE}_${ORBIT}_${TAKEID}_${PRODID}.SAFE")
        match = regex.match(basename)
        MISSIONID, BEAM, PRODUCT, RESOLUTION, LEVEL, CLASS, POL, STARTDATE, STOPDATE, ORBIT, TAKEID, PRODID = match.groups()
        new_format = f"{MISSIONID.lower()}-{BEAM.lower()}-owi-xx-{STARTDATE.lower()}-{STOPDATE.lower()}-{ORBIT}-{TAKEID}.nc"
        out_file = os.path.join(out_folder,basename,new_format)
        return out_file

    elif sensor == 'RS2':
        regex = re.compile("(RS2)_OK([0-9]+)_PK([0-9]+)_DK([0-9]+)_(....)_(........)_(......)_(.._?.?.?)_(S.F)")
        template = string.Template("${MISSIONID}_OK${DATA1}_PK${DATA2}_DK${DATA3}_${DATA4}_${DATE}_${TIME}_${POLARIZATION}_${LAST}")
        match = regex.match(basename)

        MISSIONID, DATA1, DATA2, DATA3, DATA4, DATE, TIME, POLARIZATION, LAST = match.groups()
        new_format = f"{MISSIONID.lower()}--owi-xx-{meta_start_date.lower()}-{meta_stop_date.lower()}-_____-_____.nc"
        out_file = os.path.join(out_folder,basename,new_format)
        return out_file
    elif sensor == 'RCM':
        regex = re.compile("([A-Z0-9]+)_OK([0-9]+)_PK([0-9]+)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)_(.*?)")
        template = string.Template("${MISSIONID}_OK${DATA1}_PK${DATA2}_${DATA3}_${DATA4}_${DATE}_${TIME}_${POLARIZATION1}_${POLARIZATION2}_${PRODUCT}")
        match = regex.match(basename)
        MISSIONID, DATA1, DATA2, DATA3, DATA4, DATE, TIME, POLARIZATION1, POLARIZATION2, LAST = match.groups()
        new_format = f"{MISSIONID.lower()}--owi-xx-{meta_start_date.lower()}-{meta_stop_date.lower()}-_____-_____.nc"
        out_file = os.path.join(out_folder,basename,new_format)
        return out_file
    else:
        raise ValueError("sensor must be S1A|S1B|RS2|RCM, got sensor %s" % sensor)

def makeL2(filename, out_folder, config_path):
    """

    :param filename: str
    :param out_folder: str
    :param config_path: str
    :return:
     out_file: str
    """
    # 1 - Find sensor, and associated config (GMFs to use, flattening or not)
    sensor,sensor_longname, fct_meta, fct_dataset = getSensorMetaDataset(filename)
    map_model = None
    if Path(config_path).exists():
        config = yaml.load(
            Path(config_path).open(),
            Loader=yaml.FullLoader
        )
        try :
            config = config[sensor]
        except Exception:
            raise KeyError("sensor %s not in this config" %sensor)
    else :
        raise FileNotFoundError('config_path do not exists, got %s '  % config_path)




    # 2 - Add raster and load dataset at 1km resoltuion

    meta = fct_meta(filename)

    out_file = getOutputName2(filename,out_folder,sensor,meta)

    if os.path.exists(out_file):
        print("out_file %s exists" % out_file)
        return


    # land mask

    meta.set_raster(getConf['ecmwf_0100_1h'])
    meta.set_raster(getConf['ecmwf_0125_1h'])

    ## only keep best ecmwf  (FIXME: it's hacky, and xsar should provide a better method to handle this)
    for ecmwf_name in [ 'ecmwf_0125_1h', 'ecmwf_0100_1h' ]:
        ecmwf_infos = meta.rasters.loc[ecmwf_name]
        try :
            ecmwf_file = ecmwf_infos['get_function'](ecmwf_infos['resource'], date=datetime.datetime.strptime(meta.start_date, '%Y-%m-%d %H:%M:%S.%f'))[1]
        ## temporary for RCM issue https://github.com/umr-lops/xarray-safe-rcm/issues/34
        except Exception as e :
            ecmwf_file = ecmwf_infos['get_function'](ecmwf_infos['resource'], date=datetime.datetime.strptime(meta.start_date, '%Y-%m-%d %H:%M:%S'))[1]

        if not os.path.isfile(ecmwf_file):
            ## temporary
            # if repro do not exists we look at not repro folder (only one will exist after)
            if ecmwf_name == "ecmwf_0100_1h":
                ecmwf_infos['resource'] = ecmwf_infos['resource'].replace("netcdf_light_REPRO_tree","netcdf_light")
                try :
                    ecmwf_file = ecmwf_infos['get_function'](ecmwf_infos['resource'], date=datetime.datetime.strptime(meta.start_date, '%Y-%m-%d %H:%M:%S.%f'))[1]
                except Exception as e :
                    ecmwf_file = ecmwf_infos['get_function'](ecmwf_infos['resource'], date=datetime.datetime.strptime(meta.start_date, '%Y-%m-%d %H:%M:%S'))[1]

                if not os.path.isfile(ecmwf_file):
                    meta.rasters = meta.rasters.drop([ecmwf_name])
                else :
                    map_model = { '%s_%s' % (ecmwf_name, uv) : 'model_%s' % uv for uv in ['U10', 'V10'] }

            else :
                meta.rasters = meta.rasters.drop([ecmwf_name])
        else:
            map_model = { '%s_%s' % (ecmwf_name, uv) : 'model_%s' % uv for uv in ['U10', 'V10'] }
    if map_model is None:
        raise Exception('the weather model is not set `map_model` is None -> you probably don"t have access to ECMWF archive')

    try :
        xsar_obj_1000m = fct_dataset(meta, resolution='1000m')
        dataset_1000m = xsar_obj_1000m.datatree['measurement'].to_dataset()
        dataset_1000m = dataset_1000m.rename(map_model)
        #add attributes
        dataset_1000m.attrs = xsar_obj_1000m.dataset.attrs
        dataset_1000m.attrs['L1_path'] = dataset_1000m.attrs.pop('name')

    except Exception as e :
        print(e)
        return

    black_list = ['digital_number', 'gamma0_raw', 'negz',
                  'azimuth_time', 'slant_range_time', 'velocity', 'range_ground_spacing',
                  'gamma0', 'time', 'nd_co', 'nd_cr',  'gamma0_lut','sigma0_lut',"noise_lut_range","lineSpacing","sampleSpacing","noise_lut","noise_lut_azi",
                  'nebz','beta0_raw','lines_flipped','samples_flipped',"altitude","sigma0_raw","beta0"]
    variables = list(set(dataset_1000m) - set(black_list))
    # complex not allowed in netcdf
    dataset_1000m = dataset_1000m[variables]

    dataset_1000m=dataset_1000m.rename({
    'longitude':'owiLon',
    'latitude':'owiLat',
    'incidence':'owiIncidenceAngle',
    'elevation':'owiElevationAngle',
    'ground_heading': 'owiHeading',
    'land_mask':'owiLandFlag'
    })


    # TODO Would be nice to add a better land mask
    # 3 - Variables of interest
    # LAND
    #dataset_1000m.owiLandFlag.values = cv2.dilate(dataset_1000m['owiLandFlag'].values.astype('uint8'),np.ones((3,3),np.uint8),iterations = 3)
    dataset_1000m.owiLandFlag.values = binary_dilation(dataset_1000m['owiLandFlag'].values.astype('uint8'), structure=np.ones((3,3),np.uint8), iterations=3)
    dataset_1000m.owiLandFlag.attrs['long_name'] = 'Mask of data'
    dataset_1000m.owiLandFlag.attrs['valid_range'] = np.array([0, 1])
    dataset_1000m.owiLandFlag.attrs['flag_values'] = np.array([0, 1])
    dataset_1000m.owiLandFlag.attrs['flag_meanings'] = 'valid no_valid'

    # MASK
    ## Careful : in sarwing process sometimes there are 2 & 3. Not made here
    dataset_1000m['owiMask'] = xr.DataArray(dataset_1000m.owiLandFlag)
    dataset_1000m.owiMask.attrs = {}
    dataset_1000m.owiLandFlag.attrs['long_name'] = 'Mask of data'
    dataset_1000m.owiLandFlag.attrs['valid_range'] = np.array([0, 3])
    dataset_1000m.owiLandFlag.attrs['flag_values'] = np.array([0, 1, 2, 3])
    dataset_1000m.owiLandFlag.attrs['flag_meanings'] = 'valid land ice no_valid'

    ##ANCILLARY
    dataset_1000m['ancillary_wind'] = (dataset_1000m.model_U10 + 1j * dataset_1000m.model_V10) * np.exp(1j * np.deg2rad(dataset_1000m.owiHeading))
    dataset_1000m['ancillary_wind'] = xr.where(dataset_1000m['owiMask'], np.nan, dataset_1000m['ancillary_wind'].compute()).transpose(*dataset_1000m['ancillary_wind'].dims)
    dataset_1000m.attrs['ancillary_source'] = dataset_1000m['model_U10'].attrs['history'].split('decoded: ')[1].strip()
    dataset_1000m = dataset_1000m.drop_vars(['model_U10','model_V10'])


    ##NRCS

    dataset_1000m['sigma0_ocean'] = xr.where(dataset_1000m['owiMask'], np.nan, dataset_1000m['sigma0'].compute()).transpose(*dataset_1000m['sigma0'].dims)
    dataset_1000m['sigma0_ocean'] = xr.where(dataset_1000m['sigma0_ocean'] <= 0, 1e-15, dataset_1000m['sigma0_ocean'])

    dataset_1000m['owiNrcs'] = dataset_1000m['sigma0_ocean'].sel(pol='VV')
    dataset_1000m.owiNrcs.attrs['units'] = 'm^2 / m^2'
    dataset_1000m.owiNrcs.attrs['long_name'] = 'Normalized Radar Cross Section'

    dataset_1000m['owiNrcs_cross'] = dataset_1000m['sigma0_ocean'].sel(pol='VH')
    dataset_1000m.owiNrcs_cross.attrs['units'] = 'm^2 / m^2'
    dataset_1000m.owiNrcs_cross.attrs['long_name'] = 'Normalized Radar Cross Section'

    #unused
    dataset_1000m['owiNrcs_no_noise_correction'] = xr.full_like(dataset_1000m.owiNrcs, np.nan)
    dataset_1000m.owiNrcs_no_noise_correction.attrs['units'] = 'm^2 / m^2'
    dataset_1000m.owiNrcs_no_noise_correction.attrs['long_name'] = 'Normalized Radar Cross Section, no noise correction applied'
    #unused
    dataset_1000m['owiNrcs_cross_no_noise_correction'] =xr.full_like(dataset_1000m.owiNrcs_cross, np.nan)
    dataset_1000m.owiNrcs_cross_no_noise_correction.attrs['units'] = 'm^2 / m^2'
    dataset_1000m.owiNrcs_cross_no_noise_correction.attrs['long_name'] = 'Normalized Radar Cross Section, no noise correction applied'


    ## NESZ & DSIG
    dataset_1000m=dataset_1000m.assign(owiNesz=(['line','sample'],dataset_1000m.nesz.isel(pol=0).values))
    dataset_1000m=dataset_1000m.assign(owiNesz_cross=(['line','sample'],dataset_1000m.nesz.isel(pol=1).values)) #no flattening

    if config["apply_flattening"] :
        dataset_1000m=dataset_1000m.assign(owiNesz_cross_final=(['line','sample'],windspeed.nesz_flattening(dataset_1000m.owiNesz_cross, dataset_1000m.owiIncidenceAngle)))
        dataset_1000m['owiNesz_cross_final'].attrs["comment"] = 'nesz has been flattened using windspeed.nesz_flattening'
    else :
        dataset_1000m=dataset_1000m.assign(owiNesz_cross_final=(['line','sample'],dataset_1000m.owiNesz_cross.values))
        dataset_1000m['owiNesz_cross_final'].attrs["comment"] = 'nesz has not been flattened'
    ## dsig
    dataset_1000m["owiDsig_cross"] = windspeed.get_dsig(config["GMF_VH_NAME"],dataset_1000m.owiIncidenceAngle,dataset_1000m.owiNrcs_cross,dataset_1000m.owiNesz_cross_final)
    dataset_1000m.owiDsig_cross.attrs['comment'] = 'variable used to ponderate copol and crosspol'

    dataset_1000m = dataset_1000m.drop_vars(['sigma0_ocean','sigma0','nesz'])


    # 4 - Inversion
    ## 4a - co & dual inversion
    windspeed_co, windspeed_dual = windspeed.invert_from_model(
            dataset_1000m.owiIncidenceAngle,
            dataset_1000m.owiNrcs,
            dataset_1000m.owiNrcs_cross,
            #ancillary_wind=-np.conj(xsar_obj_1000m.dataset['ancillary_wind']),
            ancillary_wind=-dataset_1000m.ancillary_wind,
            dsig_cr = dataset_1000m.owiDsig_cross,
            model=(config["GMF_VV_NAME"],config["GMF_VH_NAME"]))

    dataset_1000m["owiWindSpeed_co"] = np.abs(windspeed_co)
    dataset_1000m["owiWindSpeed_co"].attrs["comment"] = dataset_1000m["owiWindSpeed_co"].attrs["comment"].replace("wind speed and direction","wind speed")
    dataset_1000m["owiWindSpeed"] = np.abs(windspeed_dual)
    dataset_1000m["owiWindSpeed"].attrs["comment"] = dataset_1000m["owiWindSpeed"].attrs["comment"].replace("wind speed and direction","wind speed")

    ## 4n - cr inversion
    """
    windspeed_cr = windspeed.invert_from_model(
        dataset_1000m.incidence,
        dataset_1000m.sigma0_ocean.isel(pol=1),
        #ancillary_wind=-np.conj(dataset_1000m['ancillary_wind']),
        dsig_cr = dsig_cr,
        model=config["GMF_VH_NAME"])

    dataset_1000m["wind_speed_cr"] = np.abs(windspeed_cr)
    """
    ## cr inversion ##TODO
    windspeed_cr = windspeed.invert_from_model(
        dataset_1000m.owiIncidenceAngle.values,
        dataset_1000m.owiNrcs_cross.values,
        #ancillary_wind=-np.conj(xsar_obj_1000m.dataset['ancillary_wind']),
        dsig_cr = dataset_1000m.owiDsig_cross.values,
        model=config["GMF_VH_NAME"])


    windspeed_cr = np.abs(windspeed_cr)
    dataset_1000m=dataset_1000m.assign(owiWindSpeed_cross=(['line','sample'],windspeed_cr))
    dataset_1000m.owiWindSpeed_cross.attrs['comment'] = "wind speed inverted from model %s (%s)" % (config["GMF_VH_NAME"], "VH")
    dataset_1000m.owiWindSpeed_cross.attrs['model'] = config["GMF_VH_NAME"]
    dataset_1000m.owiWindSpeed_cross.attrs['units'] = 'm/s'
    # 5 - saving

    dataset_1000m=dataset_1000m.assign(owiEcmwfWindSpeed=(['line','sample'],np.abs(dataset_1000m['ancillary_wind'].data)))
    dataset_1000m=dataset_1000m.assign(owiEcmwfWindDirection=(['line','sample'],np.angle(dataset_1000m['ancillary_wind'])))
    dataset_1000m['owiEcmwfWindDirection'].attrs['comment'] = 'angle in radians, anticlockwise, 0=sample'
    del dataset_1000m['ancillary_wind']


    dataset_1000m['owiWindQuality'] = xr.full_like(dataset_1000m.owiNrcs, 0)
    dataset_1000m['owiWindQuality'].attrs['long_name'] = "Quality flag taking into account the consistency_between_wind_inverted_and_NRCS_and_Doppler_measured"
    dataset_1000m['owiWindQuality'].attrs['valid_range'] = np.array([0, 3])
    dataset_1000m['owiWindQuality'].attrs['flag_values'] = np.array([0, 1, 2, 3])
    dataset_1000m['owiWindQuality'].attrs['flag_meanings'] = "good medium low poor"
    dataset_1000m['owiWindQuality'].attrs['comment'] = 'not done yet'

    dataset_1000m['owiWindFilter'] = xr.full_like(dataset_1000m.owiNrcs, 0)
    dataset_1000m['owiWindFilter'].attrs['long_name'] = "Quality flag taking into account the local heterogeneity"
    dataset_1000m['owiWindFilter'].attrs['valid_range'] = np.array([0, 3])
    dataset_1000m['owiWindFilter'].attrs['flag_values'] = np.array([0, 1, 2, 3])
    dataset_1000m['owiWindFilter'].attrs['flag_meanings'] = "homogeneous_NRCS, heterogeneous_from_co-polarization_NRCS, heterogeneous_from_cross-polarization_NRCS, heterogeneous_from_dual-polarization_NRCS"
    dataset_1000m['owiWindFilter'].attrs['comment'] = 'not done yet'

    dataset_1000m = dataset_1000m.drop_dims('pol')
    dataset_1000m = dataset_1000m.rename_dims({"line":"owiAzSize","sample":"owiRaSize"})
    dataset_1000m = dataset_1000m.rename({"line":"owiAzSize","sample":"owiRaSize"})
    #xsar_obj_1000m.recompute_attrs()
    ds_1000 = dataset_1000m.compute()

    for var in ['footprint','multidataset','rawDataStartTime','specialHandlingRequired']:
        if var in ds_1000.attrs:
            ds_1000.attrs[var] = str(ds_1000.attrs[var])
    if "approx_transform" in ds_1000.attrs:
        del ds_1000.attrs["approx_transform"]

    ds_1000.attrs["TITLE"] = "Sentinel-1 OWI Component"
    ds_1000.attrs["productOwner"] = "IFREMER"
    ds_1000.attrs["sourceProduct"] = sensor
    ds_1000.attrs["missionName"] = sensor_longname
    ds_1000.attrs["missionPhase"] = "Test"
    ds_1000.attrs["polarisation"] = "VV/VH"
    ds_1000.attrs["acquisitionStation"] = "/"
    ds_1000.attrs["softwareVersion"] = "/"
    ds_1000.attrs["pythonVersion"] = "3.1"
    ds_1000.attrs["polarisationRatio"] = "/"
    ds_1000.attrs["l2ProcessingUtcTime"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
    ds_1000.attrs["processingCenter"] = "/" ;
    try :
        ds_1000.attrs["firstMeasurementTime"] = datetime.datetime.strptime(ds_1000.attrs['start_date'], "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%dT%H:%M:%SZ")
        ds_1000.attrs["lastMeasurementTime"] = datetime.datetime.strptime(ds_1000.attrs['stop_date'], "%Y-%m-%d %H:%M:%S.%f").strftime("%Y-%m-%dT%H:%M:%SZ")
    except :
        ds_1000.attrs["firstMeasurementTime"] = datetime.datetime.strptime(ds_1000.attrs['start_date'], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%SZ")
        ds_1000.attrs["lastMeasurementTime"] = datetime.datetime.strptime(ds_1000.attrs['stop_date'], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%dT%H:%M:%SZ")
    ds_1000.attrs["clmSource"] = "/" ;
    ds_1000.attrs["bathySource"] = "/" ;
    ds_1000.attrs["owiAlgorithmVersion"] = "/" ;
    ds_1000.attrs["gmf"] = config['GMF_VV_NAME'] + ", " + config["GMF_VH_NAME"] ;
    ds_1000.attrs["iceSource"] = "/"
    ds_1000.attrs["owiNoiseCorrection"] = "False"
    ds_1000.attrs["inversionTabGMF"] = config['GMF_VV_NAME'] + ", " + config["GMF_VH_NAME"] ;
    ds_1000.attrs["wnf_3km_average"] = "/"
    ds_1000.attrs["owiWindSpeedSrc"] = "owiWindSpeed"
    ds_1000.attrs["owiWindDirectionSrc"] = "/"


    # some type like date or json must be converted to string
    #ds_1000.attrs['start_date'] = str(ds_1000.attrs['start_date'])
    #ds_1000.attrs['stop_date'] = str(ds_1000.attrs['stop_date'])
    # add ipf_version and aux_cal_stop
    #ds_1000.attrs['aux_cal_start'] = str(aux_cal_start)
    #ds_1000.attrs['aux_cal_stop'] = str(aux_cal_stop)


    # encode gcps as json string
    import json
    class JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)

    json_gcps = json.dumps(json.loads(json.dumps(ds_1000.owiAzSize.spatial_ref.gcps,cls=JSONEncoder)))
    ds_1000['owiAzSize']['spatial_ref'].attrs['gcps'] = json_gcps
    ds_1000['owiRaSize']['spatial_ref'].attrs['gcps'] = json_gcps
    ds_1000 = ds_1000.drop_vars(["owiRaSize","owiAzSize","spatial_ref"])

    # remove possible incorect values on swath border
    # for name in ['windspeed_co','windspeed_cr','windspeed_dual']:
      #  ds_1000[name].values[:,0:6] = np.nan
      #  ds_1000[name].values[:,-6::] = np.nan

    os.makedirs(os.path.dirname(out_file), exist_ok=True)

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
    "owiWindSpeed_cr": -9999.0,

    }
    encoding={}
    for var in list(set(ds_1000.coords.keys()) | set(ds_1000.keys())):
        encoding[var] = {}
        try :
            #sarwing_ds[var].attrs["_FillValue"] = table_fillValue[var]
            encoding[var].update({'_FillValue': table_fillValue[var]})
        except:
            #Nouvelles variables..
            if (var in ["owiWindSpeed_co","owiWindSpeed_cr","owiWindSpeed"]):
                #sarwing_ds[var].attrs["_FillValue"] = -9999.0
                encoding[var].update({'_FillValue': -9999.0})
            else:
                encoding[var].update({'_FillValue': None})

    ds_1000.attrs["xsar_version"] = xsar.__version__
    ds_1000.attrs["xsarsea_version"] = xsarsea.__version__

    ds_1000.to_netcdf(out_file, mode="w",encoding=encoding)

    df = ds_1000.to_dataframe()

    df = df.assign(**ds_1000.attrs)
    df.reset_index(drop=False,inplace=True)
    df.to_csv(out_file.replace(".nc",".csv"))

    ds_1000.close()

    print("OK for ",os.path.basename(filename))

    return out_file
