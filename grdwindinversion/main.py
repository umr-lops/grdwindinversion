from grdwindinversion.inversion import makeL2
from grdwindinversion.utils import get_memory_usage
import grdwindinversion
import time
import logging


def processor_starting_point():
    import argparse, os
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Perform inversion from S1(L1-GRD) SAFE, L1-RCM, L1-RS2 ; using xsar/xsarsea tools')
    parser.add_argument('--input_file', help='input file path', required=True)
    parser.add_argument('--config_file',
                        help='config file path [if not provided will take config file based on input file]',required=False)
                        
    parser.add_argument('--resolution',required=False, default='1000m', help='set resolution ["full" | "1000m" | "xXxm"]')
     
    parser.add_argument('--outputdir', required=True)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite existing .nc files [default is False]', required=False)

    parser.add_argument('--no_generate_csv', action='store_false', help="En cas d'activation, désactive la génération du .csv")
    

    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    t0 = time.time()
    input_file = args.input_file.rstrip('/')
    logging.info('input file: %s', input_file)
    
    
    # if '1SDV' not in input_file and '_VV_VH' not in input_file:
    #     raise Exception('this processor only handle dual polarization acquisitions VV+VH for now.')
    # if '1SSH' in input_file or '1SDH' in input_file or '_HH_HV' in input_file:
    #     raise Exception('this processor only handle acquisitions with VV or VV+VH polarization for now.')
    
    if args.config_file is None:
        if 'S1' in input_file:
            config_file = os.path.join(os.path.dirname(grdwindinversion.__file__),'config_S1.yaml')
        elif 'RCM' in input_file:
            config_file = os.path.join(os.path.dirname(grdwindinversion.__file__),'config_RCM.yaml')
        elif 'RS2' in input_file:
            config_file = os.path.join(os.path.dirname(grdwindinversion.__file__),'config_RS2.yaml')
        elif 'hy2b' in input_file:
            config_file = os.path.join(os.path.dirname(grdwindinversion.__file__),'config_hy2b.yaml')
        else:
            raise Exception('config data file cannot be defined using the input filename')
    else:
        config_file = args.config_file
        
    
    out_folder = args.outputdir
    resolution = args.resolution
    if resolution == "full":
        resolution = None
    
    out_file,outputds = makeL2(input_file, out_folder, config_file, overwrite=args.overwrite,resolution = resolution, generateCSV = args.no_generate_csv)
    logging.info('out_file: %s', out_file)
    logging.info('current memory usage: %s ', get_memory_usage(var='current'))
    logging.info('done in %1.3f min', (time.time() - t0) / 60.)


if __name__ == '__main__':
    processor_starting_point()
