from importlib.metadata import version
from grdwindinversion.inversion import inverse, makeL2, makeL2asOwi, getSensorMetaDataset

__all__ = [
    "inverse",
    "makeL2",
    "makeL2asOwi",
    "getSensorMetaDataset",
    "inversion",
]

try:
    __version__ = version("grdwindinversion")
except Exception:
    __version__ = "999"
