import os

from grdwindinversion.inversion import getOutputName


def test_function_getOutputName():
    """
    Test getOutputName function for RCM files. Checks that the function does not raise any exception and returns a string.
    """
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "listing_rcm_safe.txt")
    outdir = "./"
    sensor = "RCM"
    start_date = "2019-01-01"
    stop_date = "2019-01-02"

    with open(file_path, 'r') as f:
        listing_safe_rcm = f.readlines()

    for file in listing_safe_rcm:
        file = file.strip()
        # Appel de la fonction. Le but est juste de vérifier qu'aucune exception n'est levée.
        result = getOutputName(input_file=file, outdir=outdir, sensor=sensor,
                               meta_start_date=start_date, meta_stop_date=stop_date)

    assert isinstance(
        result, str), f"getOutputName did not return a string for file {file}, got {type(result)}"


if __name__ == '__main__':
    test_function_getOutputName()
