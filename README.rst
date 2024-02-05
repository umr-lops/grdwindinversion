================
grdwindinversion
================


.. image:: https://img.shields.io/pypi/v/grdwindinversion.svg
        :target: https://pypi.python.org/pypi/grdwindinversion

.. image:: https://img.shields.io/travis/agrouaze/grdwindinversion.svg
        :target: https://travis-ci.com/agrouaze/grdwindinversion

.. image:: https://readthedocs.org/projects/grdwindinversion/badge/?version=latest
        :target: https://grdwindinversion.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status




Package to perform Wind inversion from GRD Level-1 SAR images


* Free software: MIT license
* Documentation: https://grdwindinversion.readthedocs.io.


Usage
------

 .. code-block:: python

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



Features
--------

This Python library (based on `xarray`) allows to perform wind inversion from level-1 GRD (projected magnitude image).
Mission supported:
 * Sentinel-1
 * RCM
 * RadarSat-2


Credits
-------

This package was created with Cookiecutter_.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter

