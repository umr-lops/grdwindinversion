"""
Lightweight test suite for mask management functions.
Focuses on core functionality and critical error cases.
"""
import pytest
import numpy as np
import xarray as xr
from unittest.mock import Mock, patch
from grdwindinversion.inversion import mergeLandMasks, addMasks_toMeta, processLandMask


class TestMergeLandMasks:
    """Core tests for mergeLandMasks function"""

    def test_merge_single_mask(self):
        """Test the main use case: merging one additional mask"""
        default_mask = np.array([[0, 1], [1, 0]], dtype="uint8")
        additional_mask = np.array([[1, 0], [0, 1]], dtype="uint8")
        expected_merged = np.array([[1, 1], [1, 1]], dtype="uint8")

        ds = xr.Dataset({
            "land_mask": (["y", "x"], default_mask),
            "gshhsH_mask": (["y", "x"], additional_mask)
        })

        result = mergeLandMasks(ds, ["gshhsH"])

        np.testing.assert_array_equal(result.land_mask.values, expected_merged)
        assert result is ds  # Verify it returns the same object

    def test_merge_missing_land_mask_raises_error(self):
        """Test validation: dataset must have land_mask"""
        ds = xr.Dataset({
            "some_other_var": (["y", "x"], np.zeros((2, 2)))
        })

        with pytest.raises(ValueError, match="Dataset must contain a 'land_mask' variable"):
            mergeLandMasks(ds, ["gshhsH"])

    def test_merge_with_missing_mask_is_skipped(self):
        """Test resilience: missing masks are skipped without crashing"""
        default_mask = np.array([[0, 1], [1, 0]], dtype="uint8")
        ds = xr.Dataset({
            "land_mask": (["y", "x"], default_mask)
        })

        result = mergeLandMasks(ds, ["nonexistent"])

        # Should not crash and land_mask should remain unchanged
        np.testing.assert_array_equal(result.land_mask.values, default_mask)


class TestAddMasksToMeta:
    """Core tests for addMasks_toMeta function"""

    @patch('grdwindinversion.inversion.getConf')
    def test_add_single_land_mask(self, mock_get_config):
        """Test the main use case: adding one mask"""
        mock_get_config.return_value = {
            'masks': {
                'land': [
                    {'name': 'gshhsH', 'path': '/path/to/mask.shp'}
                ]
            }
        }
        mock_meta = Mock()

        result = addMasks_toMeta(mock_meta)

        assert result == {'land': ['gshhsH']}
        mock_meta.set_mask_feature.assert_called_once_with(
            'gshhsH', '/path/to/mask.shp')

    def test_add_mask_without_set_mask_feature_method(self):
        """Test validation: meta must have set_mask_feature method"""
        mock_meta = Mock(spec=[])

        with pytest.raises(AttributeError, match="must have a 'set_mask_feature' method"):
            addMasks_toMeta(mock_meta)

    @patch('grdwindinversion.inversion.getConf')
    def test_add_mask_with_file_error(self, mock_get_config):
        """Test resilience: file errors are caught without crashing"""
        mock_get_config.return_value = {
            'masks': {
                'land': [
                    {'name': 'bad_mask', 'path': '/nonexistent.shp'}
                ]
            }
        }
        mock_meta = Mock()
        mock_meta.set_mask_feature.side_effect = FileNotFoundError(
            "File not found")

        result = addMasks_toMeta(mock_meta)

        # Should not crash, just skip the failed mask
        assert result == {'land': []}


class TestProcessLandMask:
    """Core tests for processLandMask function"""

    def test_create_three_level_mask(self):
        """Test that processLandMask creates 3-level mask (ocean=0, coastal=1, land=2)"""
        # Create a simple land mask: land in center, ocean around
        land_mask = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype="uint8")

        ds = xr.Dataset({
            "land_mask": (["y", "x"], land_mask)
        })

        processLandMask(ds, dilation_iterations=1)

        result = ds.land_mask.values

        # Check that we have 3 levels
        assert 0 in result  # Ocean
        assert 1 in result  # Coastal (around the land pixel)
        assert 2 in result  # Land (original land pixel becomes 2)

        # Original land pixel should now be 2
        assert result[2, 2] == 2

        # Pixels around land should be coastal (1)
        assert result[1, 2] == 1  # Above
        assert result[3, 2] == 1  # Below
        assert result[2, 1] == 1  # Left
        assert result[2, 3] == 1  # Right


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
