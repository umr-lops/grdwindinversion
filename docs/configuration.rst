=============
Configuration
=============

Configuration Files Implementation Guide
=========================================

Overview
--------

The ``grdwindinversion`` configuration system uses two types of YAML files:

1. **data_config.yaml**: Paths to data sources (ancillary, , LUTs, masks)
2. **config_*.yaml**: Processing parameters specific to each satellite

Configuration System Architecture
----------------------------------

Loading Hierarchy
~~~~~~~~~~~~~~~~~

The ``data_config.yaml`` file is loaded with the following priority:

.. code-block:: text

    1. ~/.grdwindinversion/data_config.yaml  (user local configuration)
    2. ./local_data_config.yaml              (project local configuration)
    3. <package>/data_config.yaml            (package default configuration)

Accessing Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from grdwindinversion.load_config import getConf

    # Get the global configuration dictionary
    config = getConf()

    # Access data paths
    ecmwf_path = config["ecmwf_0100_1h"]
    nc_luts_path = config["nc_luts_path"]

data_config.yaml Structure
---------------------------

1. Meteorological Data Paths (ancillary)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    # ECMWF at 0.1° resolution, 1 hour
    'ecmwf_0100_1h': '/path/to/ecmwf/%Y/%j/ECMWF_FORECAST_0100_%Y%m%d%H%M_10U_10V.nc'
    # ECMWF at 0.125° resolution, 1 hour
    'ecmwf_0125_1h': '/path/to/ecmwf/%Y/%j/ecmwf_%Y%m%d%H%M.nc'

**Template Format**: Uses Python datetime format codes

- ``%Y``: Year (4 digits)
- ``%j``: Day of year (001-366)
- ``%m``: Month (01-12)
- ``%d``: Day of month (01-31)
- ``%H``: Hour (00-23)
- ``%M``: Minute (00-59)

2. LUT Paths (Look-Up Tables)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    nc_luts_path: '/path/to/luts/'
    lut_cmod7_path: '/path/to/cmod7_lut.nc'

3. Masks Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    masks:
      land:
        - name: 'gshhsH'
          path: '/path/to/gshhs/GSHHS_h_L1.shp'
        - name: 'custom_land'
          path: '/path/to/custom_land.shp'
      ice:
        - name: 'iceSource'
          path: '/path/to/ice_mask.shp'

**Mask Notes**:

- The ``_mask`` suffix is automatically appended to variable names
- Multiple masks per category are supported

config_*.yaml Structure
-----------------------

Global Parameters (root level)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    no_subdir: True                          # Don't create subdirectories in output
    winddir_convention: "meteorological"     # Wind direction convention
    add_gradientsfeatures: False             # Add gradient/streak features
    add_nrcs_model: False                    # Add NRCS from model (forced to False)
    overwrite: False                         # Overwrite existing files

Recommended Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

**Use ``config_prod_v3.yaml`` as the default configuration file.**

This configuration includes:

- Support for all Sentinel-1 satellites (S1A, S1B, S1C, S1D)
- Support for RS2 and RCM satellites
- Automatic handling of S1 EW calibration changes (July 2019) with latest GMFs

Satellite-Specific Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each satellite section includes:

.. code-block:: yaml

    S1A:
      # Geophysical Model Functions
      GMF_HH_NAME: "nc_lut_gmf_cmod5n_Rhigh_hh_mouche1"
      GMF_VV_NAME: "gmf_cmod5n"
      GMF_VH_NAME: "gmf_s1_v2"
      dsig_VH_NAME: "gmf_s1_v2"
      dsig_cr_step: "nrcs"                    # Polarization mixing step

      # Automatic handling of S1 EW recalibration (after 2019-07-31)
      S1_EW_calG>20190731:
        GMF_VH_NAME: "gmf_s1_v3_ew_rec"
        dsig_VH_NAME: "dsig_wspd_s1_ew_rec_v3"
        dsig_cr_step: "wspd"

      # Processing parameters
      apply_flattening: True                  # NESZ correction
      recalibration: False                    # Kersten recalibration
      ancillary: "ecmwf"                      # Meteorological data
      inc_step: 0.1                           # Incidence angle step (°)
      wspd_step: 0.1                          # Wind speed step (m/s)
      phi_step: 1.0                           # Azimuth step (°)
      resolution: "high"

Configuration Examples
----------------------

Configuration with Recalibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

File: ``config_prod_recal.yaml``

.. code-block:: yaml

    no_subdir: True
    winddir_convention: "meteorological"
    add_gradientsfeatures: False
    add_nrcs_model: False

    S1A:
        GMF_VV_NAME: "gmf_cmod5n"
        GMF_VH_NAME: "gmf_s1_v2"
        recalibration: True  # Enable Kersten recalibration formula
        # ... other parameters ...

**Usage**: Applies recalibration correction to NRCS data before inversion.


Multi-GMFS Version Configuration (S1 v3)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

File: ``config_prod_v3.yaml``

.. code-block:: yaml

    no_subdir: True
    winddir_convention: "meteorological"
    add_gradientsfeatures: False
    add_nrcs_model: False

    S1A:
        GMF_VV_NAME: "gmf_cmod5n"
        GMF_VH_NAME: "gmf_s1_v2"
        # Standard configuration
        # ...

        # Configuration for EW products with calG after 2019-07-31
        S1_EW_calG>20190731:
            GMF_VV_NAME: "gmf_cmod5n_v3"
            GMF_VH_NAME: "gmf_s1_v3"
            # ... adapted parameters ...

**Usage**: Automatically handles different S1 calibration versions.
