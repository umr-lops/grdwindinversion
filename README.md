
[![Python Version](https://img.shields.io/pypi/pyversions/grdwindinversion.svg)](https://pypi.org/project/grdwindinversion/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/umr-lops/grdwindinversion/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

Package to perform Wind inversion from GRD Level-1 SAR images

-   Free software: MIT license
-   Documentation: https://grdwindinversion.readthedocs.io.

## Usage

```python

   SAR_L1-to-L2_wind_processor -h
   usage: SAR_L1-to-L2_wind_processor [-h] --input_file INPUT_FILE [--config_file CONFIG_FILE] --outputdir OUTPUTDIR [--verbose] [--overwrite]

   Perform inversion from S1(L1-GRD) SAFE, L1-RCM, L1-RS2 ; using xsar/xsarsea tools

   options:
     -h, --help            show this help message and exit
     --input_file INPUT_FILE
                           input file path
     --config_file CONFIG_FILE
                           config file path [if not provided will take config file based on input file]
     --outputdir OUTPUTDIR
     --verbose
     --overwrite           overwrite existing .nc files [default is False]
```

## Features

This Python library (based on `xarray`) allows to perform wind inversion from level-1 GRD (projected magnitude image).
Mission supported:

-   Sentinel-1
-   RCM
-   RadarSat-2
