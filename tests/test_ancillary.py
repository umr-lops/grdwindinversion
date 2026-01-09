import pytest
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import tempfile
import os

from grdwindinversion.inversion import getAncillary


class TestGetAncillary(unittest.TestCase):
    """Test suite for getAncillary function"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_meta = Mock()
        self.mock_meta.start_date = "2023-01-01 12:00:00.000"
        # Initialize with proper columns for rasters DataFrame
        self.mock_meta.rasters = pd.DataFrame(columns=['get_function', 'resource'])

    @patch('grdwindinversion.inversion.getConf')
    def test_single_ecmwf_0100_1h_model(self, mock_get_conf):
        """
        When only ecmwf_0100_1h is configured,
        function should work without requiring ecmwf_0125_1h
        """
        mock_get_conf.return_value = {
            'ecmwf_0100_1h': '/path/to/ecmwf_0100'
        }

        try:
            getAncillary(self.mock_meta, ancillary_name='ecmwf')
        except KeyError as e:
            pytest.fail(f"Should handle single model, but raised KeyError: {e}")

    @patch('grdwindinversion.inversion.getConf')
    def test_single_ecmwf_0125_1h_model(self, mock_get_conf):
        """
        When only ecmwf_0125_1h is configured,
        function should work without requiring ecmwf_0100_1h
        """
        mock_get_conf.return_value = {
            'ecmwf_0125_1h': '/path/to/ecmwf_0125'
        }

        try:
            getAncillary(self.mock_meta, ancillary_name='ecmwf')
        except KeyError as e:
            pytest.fail(f"Should handle single model, but raised KeyError: {e}")

    @patch('grdwindinversion.inversion.getConf')
    def test_both_models_priority_ecmwf_0100_1h(self, mock_get_conf):
        """
        When both models are configured and both files exist on disk,
        ecmwf_0100_1h should be selected (more recent and precise)
        """
        # Create temporary files to simulate existing ECMWF files
        with tempfile.NamedTemporaryFile(delete=False, suffix='_0100.nc') as f_0100:
            file_0100 = f_0100.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='_0125.nc') as f_0125:
            file_0125 = f_0125.name

        try:
            # Configure both models
            mock_get_conf.return_value = {
                'ecmwf_0100_1h': '/path/to/ecmwf_0100',
                'ecmwf_0125_1h': '/path/to/ecmwf_0125'
            }

            # Mock set_raster to add entries to rasters DataFrame
            def mock_set_raster(name, path):
                def mock_get_function(resource, date):
                    if name == 'ecmwf_0100_1h':
                        return (None, file_0100)
                    else:
                        return (None, file_0125)

                self.mock_meta.rasters.loc[name] = pd.Series({
                    'get_function': mock_get_function,
                    'resource': path
                })

            self.mock_meta.set_raster = mock_set_raster

            # Call the function
            result = getAncillary(self.mock_meta, ancillary_name='ecmwf')

            # Verify that ecmwf_0100_1h is selected (not ecmwf_0125_1h)
            assert result is not None, "map_model should not be None"
            assert 'ecmwf_0100_1h_U10' in result, "Should select ecmwf_0100_1h"
            assert 'ecmwf_0100_1h_V10' in result, "Should select ecmwf_0100_1h"
            assert result['ecmwf_0100_1h_U10'] == 'model_U10'
            assert result['ecmwf_0100_1h_V10'] == 'model_V10'

        finally:
            # Clean up temporary files
            if os.path.exists(file_0100):
                os.remove(file_0100)
            if os.path.exists(file_0125):
                os.remove(file_0125)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
