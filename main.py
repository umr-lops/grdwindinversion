from inversion import makeL2
from degrade import * 


if __name__ == '__main__':      
    import argparse, os
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description = 'Perform inversion from S1(L1-GRD) SAFE, L1-RCM, L1-RS2 ; using xsar/xsarsea tools')
    parser.add_argument('--input_file',help='input file path')
    parser.add_argument('--config_file',help='config file path')
    
    out_folder = "/home/datawork-cersat-public/cache/public/ftp/project/L2GRD/prod_v5"

    args = parser.parse_args()
    input_file = args.input_file
    config_file = args.config_file
       
    out_file = makeL2(input_file,out_folder, config_file) 
    createLowerResL2(out_file,25)
