=====
Usage
=====

Python API
----------

To use grdwindinversion in a project::

    import grdwindinversion


Configuration Setup
-------------------

To define the path where the ECMWF files used for wind inversion (or any supported ancillary wind)::

    cp ./grdwindinversion/data_config.yaml ./grdwindinversion/local_data_config.yaml
    # Then edit the file
    vi ./grdwindinversion/local_data_config.yaml

See :doc:`configuration` for detailed configuration options.


Command-Line Interface
----------------------

Basic Usage
~~~~~~~~~~~

Process a SAR image with wind inversion:

.. code-block:: bash

    SAR_L1-to-L2_wind_processor \
      --input_file /path/to/S1A_*.SAFE \
      --config_file /path/to/config_prod_v3.yaml \
      --outputdir /path/to/output/ \
      --resolution 1000m \
      --overwrite \
      --verbose

Command-Line Options
~~~~~~~~~~~~~~~~~~~~

``--input_file``
    Path to SAR L1 product (SAFE format for Sentinel-1)

``--config_file``
    Configuration file to use (recommended: ``config_prod_v3.yaml``)

``--outputdir``
    Output directory for L2 wind products

``--resolution``
    Output resolution (e.g., ``1000m``, ``500m``)

``--overwrite``
    Overwrite existing output files

``--verbose``
    Enable verbose output


Examples
--------

With default configuration:

.. code-block:: bash

    SAR_L1-to-L2_wind_processor \
      --input_file S1A_EW_GRDM_1SDH_*.SAFE \
      --config_file config_prod_v3.yaml \
      --outputdir ./output/ \
      --resolution 1000m

With recalibration:

.. code-block:: bash

    SAR_L1-to-L2_wind_processor \
      --input_file S1A_EW_GRDM_1SDH_*.SAFE \
      --config_file config_prod_recal.yaml \
      --outputdir ./output/ \
      --resolution 1000m

With wind streaks features:

.. code-block:: bash

    SAR_L1-to-L2_wind_processor \
      --input_file S1A_EW_GRDM_1SDH_*.SAFE \
      --config_file config_prod_streaks.yaml \
      --outputdir ./output/ \
      --resolution 1000m