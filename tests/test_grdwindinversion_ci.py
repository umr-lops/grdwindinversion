import pytest
import os
import urllib.request
from grdwindinversion.inversion import makeL2
import xsar
from grdwindinversion.load_config import getConf

# What must be done by the tests:
# - Download L1 data
# - Download needed GMFs
# - Download ECMWF data
# -
# - Setup data-config pour xsar et grdwindinversion
#
# - For recal : download auxiliary files
#

S1_path = getConf()['unit_test_s1_product']
rcm_path = getConf()['unit_test_rcm_product']
rs2_path = getConf()['unit_test_rs2_product']
print('S1_path',S1_path)
def test_makeL2_generation():
    l1_files = [
        S1_path,
        rcm_path,
        rs2_path
    ]

    # l1_files = [
    #    "/home/datawork-cersat-public/cache/project/mpc-sentinel1/data/esa/sentinel-1a/L1/IW/S1A_IW_GRDH_1S/2021/252/S1A_IW_GRDH_1SDV_20210909T130650_20210909T130715_039605_04AE83_C34F.SAFE"
    # ]

    outdir = "./out_test_data"
    os.makedirs(outdir, exist_ok=True)

    config_dir = os.path.dirname(__file__)
    config_path = os.path.join(config_dir, "config_test.yaml")

    for f in l1_files:
        # Run the makeL2 function
        print(f)
        output_nc_file, dataset = makeL2(
            filename=f,
            outdir=outdir,
            config_path=config_path,
            overwrite=True,  # Set to True to ensure a clean run
            generateCSV=False,  # Disable CSV generation for now
            resolution="1000m",
        )

        # Check if the output file (NetCDF) is generated
        assert os.path.exists(
            output_nc_file), f"NetCDF output file not created for {f}"

        # Optionally, check the dataset has content
        assert dataset is not None, f"No dataset generated for {f}"
        assert (
            "owiWindSpeed" in dataset.variables
        ), "Expected variable 'owiWindSpeed' missing in the dataset"

if __name__ == '__main__':
    test_makeL2_generation()
