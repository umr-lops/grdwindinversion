from yaml import load
import logging
import os
import grdwindinversion
from yaml import CLoader as Loader
local_config_potential_path = os.path.expanduser(
    '~/.grdwindinversion/data_config.yaml')

if os.path.exists(local_config_potential_path):
    config_path = local_config_potential_path
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
