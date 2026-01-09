from importlib.metadata import version
from grdwindinversion.inversion import inverse, makeL2, makeL2asOwi, getSensorMetaDataset
from grdwindinversion.load_config import getConf

__all__ = [
    "inverse",
    "makeL2",
    "makeL2asOwi",
    "getSensorMetaDataset",
    "getConf",
    "inversion",
]

try:
    __version__ = version("grdwindinversion")
except Exception:
    __version__ = "999"
