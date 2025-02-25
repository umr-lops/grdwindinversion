import os
import xsar

from grdwindinversion.inversion import getOutputName, getSensorMetaDataset

safes = ['S1A_IW_GRDH_1SDV_20210909T130650_20210909T130715_039605_04AE83_C34F.SAFE',
         'S1A_IW_GRDH_1SDH_20210101T102321_20210101T102346_035943_0435C4_D007.SAFE',
         'S1A_IW_GRDH_1SSV_20170105T225242_20170105T225311_014703_017ED0_584D.SAFE',
         'RCM1_OK2767220_PK2769320_1_SCLND_20230930_214014_VV_VH_GRD',
         'RCM2_OK2917789_PK2920112_1_SCLNA_20240125_195613_VV_VH_GRD',
         'RS2_OK141302_PK1242223_DK1208537_SCWA_20220904_093402_VV_VH_SGF',
         'S1A_EW_GRDM_1SDV_20230908T092521_20230908T092624_050234_060BF1_6E7A.SAFE',
         'S1A_IW_GRDH_1SDV_20150315T053621_20150315T053646_005038_006529_57CA.SAFE',
         'S1B_IW_GRDH_1SDV_20171117T164022_20171117T164047_008324_00EBB3_15F1.SAFE',
         'RCM3_OK2463574_PK2465310_1_SCLNA_20230303_063504_VV_VH_GRD',
         'RS2_OK97458_PK855025_DK787000_SCWA_20160912_212842_VV_VH_SGF',
         'RS2_OK97458_PK855025_DK787000_SCWA_20160912_212842_VV_SGF']

outfiles = ['s1a-iw-owi-dv-20210909t130650-20210909t130715-039605-04AE83.nc',
            's1a-iw-owi-dh-20210101t102321-20210101t102346-035943-0435C4.nc',
            's1a-iw-owi-sv-20170105t225242-20170105t225311-014703-017ED0.nc',
            'rcm1-sclnd-owi-dv-20230930t214011-20230930t214127-_____-_____.nc',
            'rcm2-sclna-owi-dv-20240125t195611-20240125t195726-_____-_____.nc',
            'rs2-scwa-owi-dv-20220904t093402-20220904t093518-_____-_____.nc',
            's1a-ew-owi-dv-20230908t092521-20230908t092624-050234-060BF1.nc',
            's1a-iw-owi-dv-20150315t053621-20150315t053646-005038-006529.nc',
            's1b-iw-owi-dv-20171117t164022-20171117t164047-008324-00EBB3.nc',
            'rcm3-sclna-owi-dv-20230303t063449-20230303t063629-_____-_____.nc',
            'rs2-scwa-owi-dv-20160912t212842-20160912t212958-_____-_____.nc',
            'rs2-scwa-owi-sv-20160912t212842-20160912t212958-_____-_____.nc',]

sensors = ['S1A', 'S1A', 'S1A', 'RCM', 'RCM',
           'RS2', 'S1A', 'S1A', 'S1B', 'RCM', 'RS2', 'RS2']

long_sensor_names = [
    'SENTINEL-1 A', 'SENTINEL-1 A', 'SENTINEL-1 A', 'RADARSAT Constellation 1', 'RADARSAT Constellation 2',
    'RADARSAT-2', 'SENTINEL-1 A', 'SENTINEL-1 A', 'SENTINEL-1 B', 'RADARSAT Constellation 3', 'RADARSAT-2', 'RADARSAT-2'
]
meta_functions = [
    xsar.Sentinel1Meta, xsar.Sentinel1Meta, xsar.Sentinel1Meta, xsar.RcmMeta, xsar.RcmMeta,
    xsar.RadarSat2Meta, xsar.Sentinel1Meta, xsar.Sentinel1Meta, xsar.Sentinel1Meta, xsar.RcmMeta, xsar.RadarSat2Meta, xsar.RadarSat2Meta,]
dataset_functions = [
    xsar.Sentinel1Dataset, xsar.Sentinel1Dataset, xsar.Sentinel1Dataset, xsar.RcmDataset, xsar.RcmDataset,
    xsar.RadarSat2Dataset, xsar.Sentinel1Dataset, xsar.Sentinel1Dataset, xsar.Sentinel1Dataset, xsar.RcmDataset, xsar.RadarSat2Dataset, xsar.RadarSat2Dataset
]

start_dates = ['20210909t130650',
               '20210101t102321',
               '20170105t225242',
               '20230930t214011',
               '20240125t195611',
               '20220904t093402',
               '20230908t092521',
               '20150315t053621',
               '20171117t164022',
               '20230303t063449',
               '20160912t212842',
               '20160912t212842']


stop_dates = ['20210909t130715',
              '20210101t102346',
              '20170105t225311',
              '20230930t214127',
              '20240125t195726',
              '20220904t093518',
              '20230908t092624',
              '20150315t053646',
              '20171117t164047',
              '20230303t063629',
              '20160912t212958',
              '20160912t212958']


def test_function_getSensorMetaDataset():
    """
    Test getSensorMetaDataset function for S1A/B RCM and RS2
    """

    for idx_safe, safe in enumerate(safes):
        output = (sensors[idx_safe], long_sensor_names[idx_safe],
                  meta_functions[idx_safe], dataset_functions[idx_safe])

        result = getSensorMetaDataset(safe)

        assert output == result, f"Expected {output}, got {result}"


def test_function_getOutputName():
    """
    Test getOutputName function for S1A/B RCM and RS2
    """

    for idx_safe, safe in enumerate(safes):
        sensor = sensors[idx_safe]
        start_date = start_dates[idx_safe]
        stop_date = stop_dates[idx_safe]
        output = outfiles[idx_safe]

        result = getOutputName(safe, "", sensor, start_date, stop_date, False)

        assert output == result, f"Expected {output}, got {result}"


if __name__ == '__main__':
    test_function_getSensorMetaDataset()
