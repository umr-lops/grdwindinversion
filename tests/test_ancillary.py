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
        self.mock_meta.start_date = "2023-01-01 12:00:00.123456"
        # Initialize with proper columns for rasters DataFrame
        self.mock_meta.rasters = pd.DataFrame(
            columns=['get_function', 'resource'])

    @patch('grdwindinversion.inversion.getConf')
    def test_single_ecmwf_0100_1h_model(self, mock_get_conf):
        """
        When only ecmwf_0100_1h is configured,
        function should work without requiring ecmwf_0125_1h
        """
        mock_get_conf.return_value = {
            'ancillary_sources': {
                'ecmwf': [
                    {'name': 'ecmwf_0100_1h', 'path': '/path/to/ecmwf_0100'}
                ]
            }
        }

        # Mock set_raster before calling getAncillary
        def mock_set_raster(name, path):
            mock_get_func = Mock(return_value=(None, '/nonexistent/file.nc'))
            self.mock_meta.rasters.loc[name] = pd.Series({
                'get_function': mock_get_func,
                'resource': path
            })
        self.mock_meta.set_raster = mock_set_raster

        # Mock drop to handle when file doesn't exist
        self.mock_meta.rasters.drop = Mock(return_value=self.mock_meta.rasters)

        try:
            map_model, metadata = getAncillary(
                self.mock_meta, ancillary_name='ecmwf')
            # Should not raise error
            assert True
        except (KeyError, ValueError) as e:
            pytest.fail(f"Should handle single model, but raised error: {e}")

    @patch('grdwindinversion.inversion.getConf')
    def test_single_ecmwf_0125_1h_model(self, mock_get_conf):
        """
        When only ecmwf_0125_1h is configured,
        function should work without requiring ecmwf_0100_1h
        """
        mock_get_conf.return_value = {
            'ancillary_sources': {
                'ecmwf': [
                    {'name': 'ecmwf_0125_1h', 'path': '/path/to/ecmwf_0125'}
                ]
            }
        }

        # Mock set_raster before calling getAncillary
        def mock_set_raster(name, path):
            mock_get_func = Mock(return_value=(None, '/nonexistent/file.nc'))
            self.mock_meta.rasters.loc[name] = pd.Series({
                'get_function': mock_get_func,
                'resource': path
            })
        self.mock_meta.set_raster = mock_set_raster

        # Mock drop to handle when file doesn't exist
        self.mock_meta.rasters.drop = Mock(return_value=self.mock_meta.rasters)

        try:
            map_model, metadata = getAncillary(
                self.mock_meta, ancillary_name='ecmwf')
            # Should not raise error
            assert True
        except (KeyError, ValueError) as e:
            pytest.fail(f"Should handle single model, but raised error: {e}")

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
                'ancillary_sources': {
                    'ecmwf': [
                        {'name': 'ecmwf_0100_1h', 'path': '/path/to/ecmwf_0100'},
                        {'name': 'ecmwf_0125_1h', 'path': '/path/to/ecmwf_0125'}
                    ]
                }
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
            map_model, metadata = getAncillary(
                self.mock_meta, ancillary_name='ecmwf')

            # Verify that ecmwf_0100_1h is selected (not ecmwf_0125_1h)
            assert map_model is not None, "map_model should not be None"
            assert 'ecmwf_0100_1h_U10' in map_model, "Should select ecmwf_0100_1h"
            assert 'ecmwf_0100_1h_V10' in map_model, "Should select ecmwf_0100_1h"
            assert map_model['ecmwf_0100_1h_U10'] == 'model_U10'
            assert map_model['ecmwf_0100_1h_V10'] == 'model_V10'

            # Verify metadata
            assert metadata is not None, "metadata should not be None"
            assert metadata['ancillary_source_model'] == 'ecmwf_0100_1h'
            assert 'ancillary_source_path' in metadata

        finally:
            # Clean up temporary files
            if os.path.exists(file_0100):
                os.remove(file_0100)
            if os.path.exists(file_0125):
                os.remove(file_0125)

    @patch('grdwindinversion.inversion.getConf')
    def test_era5_model(self, mock_get_conf):
        """
        When era5_0250_1h is configured,
        function should work correctly
        """
        mock_get_conf.return_value = {
            'ancillary_sources': {
                'era5': [
                    {'name': 'era5_0250_1h', 'path': '/path/to/era5_0250'}
                ]
            }
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='_era5.nc') as f_era5:
            file_era5 = f_era5.name

        try:
            # Mock set_raster
            def mock_set_raster(name, path):
                mock_get_func = Mock(return_value=(None, file_era5))
                self.mock_meta.rasters.loc[name] = pd.Series({
                    'get_function': mock_get_func,
                    'resource': path
                })
            self.mock_meta.set_raster = mock_set_raster

            # Call the function
            map_model, metadata = getAncillary(
                self.mock_meta, ancillary_name='era5')

            # Verify ERA5 is selected
            assert map_model is not None, "map_model should not be None"
            assert 'era5_0250_1h_U10' in map_model, "Should select era5_0250_1h"
            assert 'era5_0250_1h_V10' in map_model, "Should select era5_0250_1h"
            assert map_model['era5_0250_1h_U10'] == 'model_U10'
            assert map_model['era5_0250_1h_V10'] == 'model_V10'

            # Verify metadata
            assert metadata is not None, "metadata should not be None"
            assert metadata['ancillary_source_model'] == 'era5_0250_1h'
            assert 'ancillary_source_path' in metadata

        finally:
            # Clean up temporary file
            if os.path.exists(file_era5):
                os.remove(file_era5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
