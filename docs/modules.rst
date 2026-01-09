#######################################
Application Programming Interface (API)
#######################################

..
    to document functions, add them to __all__ in ../grdwindinversion/__init__.py

Core Processing
===============

Wind Inversion
--------------

.. automodule:: grdwindinversion.inversion
    :members: inverse, makeL2, makeL2asOwi, getSensorMetaDataset, getOutputName, inverse_dsig_wspd, addMasks_toMeta, mergeLandMasks, processLandMask, getAncillary, preprocess, process_gradients, transform_winddir
    :show-inheritance:

Configuration Management
------------------------

.. automodule:: grdwindinversion.load_config
    :members:
    :undoc-members:
    :show-inheritance:

Gradient Features
-----------------

.. automodule:: grdwindinversion.gradientFeatures
    :members:
    :undoc-members:
    :show-inheritance:

Utilities
=========

General Utilities
-----------------

.. automodule:: grdwindinversion.utils
    :members:
    :undoc-members:
    :show-inheritance:

Memory Management
-----------------

.. automodule:: grdwindinversion.utils_memory
    :members:
    :undoc-members:
    :show-inheritance: