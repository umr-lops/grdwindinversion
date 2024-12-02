from yaml import load
import logging
import os
import grdwindinversion
from yaml import CLoader as Loader
local_config_potential_path1 = os.path.expanduser(
    '~/.grdwindinversion/data_config.yaml')
local_config_potential_path2  = os.path.join(os.path.dirname(
        grdwindinversion.__file__), 'local_data_config.yaml')
if os.path.exists(local_config_potential_path1):
    config_path = local_config_potential_path1
elif os.path.exists(local_config_potential_path2):
    config_path = local_config_potential_path2
else:
    config_path = os.path.join(os.path.dirname(
        grdwindinversion.__file__), 'data_config.yaml')
logging.info('config path: %s', config_path)
stream = open(config_path, 'r')
conf = load(stream, Loader=Loader)


def getConf():
    """
    if local_config_potential_path exists it will superseed config_path
    :return:
    """
    return conf
