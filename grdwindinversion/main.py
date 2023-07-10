from grdwindinversion.inversion import makeL2
from grdwindinversion.degrade import *
from grdwindinversion.utils import get_memory_usage
import time
import logging

OUT_DEFAULT = "/home/datawork-cersat-public/cache/public/ftp/project/L2GRD/prod_v5"
if __name__ == '__main__':
    import argparse, os
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description='Perform inversion from S1(L1-GRD) SAFE, L1-RCM, L1-RS2 ; using xsar/xsarsea tools')
    parser.add_argument('--input_file', help='input file path', required=True)
    parser.add_argument('--config_file',
                        help='config file path [if not provided will take config file based on input file]',
                        required=False, default=None)
    parser.add_argument('--outputdir', required=False, default=OUT_DEFAULT)
    parser.add_argument('--verbose', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite existing .nc files [default is False]', required=False)

    args = parser.parse_args()
    fmt = '%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s'
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt,
                            datefmt='%d/%m/%Y %H:%M:%S', force=True)
    t0 = time.time()
    input_file = args.input_file
    if args.config_file is None:
        if 'S1' in input_file:
            config_file = 'config_S1.yaml'
        elif 'RCM' in input_file:
            config_file = 'config_RCM.yaml'
        elif 'RS2' in input_file:
            config_file = 'config_RS2.yaml'
        elif 'hy2b' in input_file:
            config_file = 'config_hy2b.yaml'
        else:
            raise Exception('config data file cannot be defined using the input filename')
    else:
        config_file = args.config_file
    out_folder = args.outputdir

    out_file = makeL2(input_file, out_folder, config_file,overwrite=args.overwrite)
    outpath = createLowerResL2(out_file, 25)
    # logging.info('%s successfully written', outpath)
    logging.info('current memory usage: %s ', get_memory_usage(var='current'))
    logging.info('done in %1.3f min', (time.time() - t0) / 60.)
