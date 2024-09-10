import pytest
import os
import urllib.request
from grdwindinversion.inversion import makeL2

# What must be done by the tests:
# - Download L1 data
# - Download needed GMFs
# - Download ECMWF data
# -
# - Setup data-config pour xsar et grdwindinversion
#
# - For recal : download auxiliary files
#


def test_makeL2_generation():
    l1_files = [
        "./test_data/L1/S1A_IW_GRDH_1SDV_20210909T130650_20210909T130715_039605_04AE83_C34F.SAFE"
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
            add_streaks=False,
            resolution="1000m",
        )

        # Check if the output file (NetCDF) is generated
        assert os.path.exists(output_nc_file), f"NetCDF output file not created for {f}"

        # Optionally, check the dataset has content
        assert dataset is not None, f"No dataset generated for {f}"
        assert (
            "owiWindSpeed" in dataset.variables
        ), "Expected variable 'owiWindSpeed' missing in the dataset"
