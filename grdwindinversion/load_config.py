from yaml import load
import logging
import os
import grdwindinversion
from yaml import CLoader as Loader
local_config_potential_path = os.path.join(os.path.dirname(grdwindinversion.__file__), 'local_data_config.yaml')

if os.path.exists(local_config_potential_path):
   config_path = local_config_potential_path
else:
   config_path = os.path.join(os.path.dirname(grdwindinversion.__file__), 'data_config.yaml')
# config_path = "./data_config.yaml"
logging.info('config path: %s',config_path)
stream = open(config_path, 'r')
conf = load(stream, Loader=Loader)
def getConf():

    """
    if grdwindinversion/local_data_config.yaml exists it will superseed grdwindinversion/data_config.yaml
    :return:
    """
    return conf
