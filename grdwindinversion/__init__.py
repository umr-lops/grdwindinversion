__all__ = ['degrade', 'inversion']
# try:
#     from importlib import metadata
# except ImportError: # for Python<3.8
#     import importlib_metadata as metadata
# __version__ = metadata.version('grdwindinversion')

from importlib.metadata import version
try:
    __version__ = version("grdwindinversion")
except Exception:
    __version__ = "999"
