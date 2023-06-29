from yaml import load
import logging
import os
import grdwindinversion
from yaml import CLoader as Loader
local_config_pontential_path = os.path.join(os.path.dirname(grdwindinversion.__file__), 'localconfig.yml')

if os.path.exists(local_config_pontential_path):
    config_path = local_config_pontential_path
else:
    config_path = os.path.join(os.path.dirname(grdwindinversion.__file__), 'config.yml')
logging.info('config path: %s',config_path)
stream = open(config_path, 'r')
conf = load(stream, Loader=Loader)
def getConf():
    """
    if grdwindinversion/local_data_config.yaml exists it will superseed grdwindinversion/data_config.yaml
    :return:
    """
    return conf
