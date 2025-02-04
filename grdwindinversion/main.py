import time
import logging
import sys
from importlib.metadata import version


def processor_starting_point():
    import argparse

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "-v", "--version", action="store_true", help="Print version"
    )
    pre_args, remaining_args = pre_parser.parse_known_args()

    # Handle the version argument right away
    if pre_args.version:
        print(version("grdwindinversion"))
        sys.exit()

    from grdwindinversion.inversion import makeL2
    from grdwindinversion.utils_memory import get_memory_usage
    from grdwindinversion.load_config import config_path
    import grdwindinversion

    parser = argparse.ArgumentParser(
        description="Perform inversion from S1(L1-GRD) SAFE, L1-RCM, L1-RS2 ; using xsar/xsarsea tools"
    )
    parser.add_argument("--input_file", help="input file path", required=True)
    parser.add_argument(
        "--config_file",
        help="config file path [if not provided will take config file based on input file]",
        required=True,
    )
    parser.add_argument(
        "--resolution",
        required=False,
        default="1000m",
        help='set resolution ["full" | "1000m" | "xXxm"]',
    )

    parser.add_argument("--outputdir", required=True)
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="overwrite existing .nc files [default is False]",
        required=False,
    )

    parser.add_argument(
        "--no_generate_csv",
        action="store_false",
        help="En cas d'activation, désactive la génération du .csv",
    )

    args = parser.parse_args()

    fmt = "%(asctime)s %(levelname)s %(filename)s(%(lineno)d) %(message)s"
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
        )
    else:
        logging.basicConfig(
            level=logging.INFO, format=fmt, datefmt="%d/%m/%Y %H:%M:%S", force=True
        )
    t0 = time.time()

    logging.info("config path: %s", config_path)

    input_file = args.input_file.rstrip("/")
    logging.info("input file: %s", input_file)

    # if '1SDV' not in input_file and '_VV_VH' not in input_file:
    #     raise Exception('this processor only handle dual polarization acquisitions VV+VH for now.')
    # if '1SSH' in input_file or '1SDH' in input_file or '_HH_HV' in input_file:
    #     raise Exception('this processor only handle acquisitions with VV or VV+VH polarization for now.')

    config_file = args.config_file
    out_folder = args.outputdir
    resolution = args.resolution
    if resolution == "full":
        resolution = None

    out_file, outputds = makeL2(
        input_file,
        out_folder,
        config_file,
        overwrite=args.overwrite,
        resolution=resolution,
        generateCSV=args.no_generate_csv,
    )

    logging.info("out_file: %s", out_file)
    logging.info("current memory usage: %s ", get_memory_usage(var="current"))
    logging.info("done in %1.3f min", (time.time() - t0) / 60.0)


if __name__ == "__main__":
    processor_starting_point()
