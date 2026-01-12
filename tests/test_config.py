import unittest
import yaml
import os
from grdwindinversion.load_config import getConf


class TestConfigStructure(unittest.TestCase):
    """Test configuration file structure and required fields."""

    def setUp(self):
        """Load configuration for testing."""
        self.config = getConf()

    def test_ancillary_sources_exists(self):
        """Test that ancillary_sources exists in configuration (mandatory)."""
        self.assertIn('ancillary_sources', self.config,
                      "Configuration must contain 'ancillary_sources'")

    def test_ancillary_sources_not_empty(self):
        """Test that ancillary_sources is not empty."""
        self.assertTrue(self.config['ancillary_sources'],
                        "ancillary_sources should not be empty")

    def test_ancillary_sources_has_ecmwf_or_era5(self):
        """Test that ancillary_sources contains at least ecmwf or era5."""
        ancillary_sources = self.config['ancillary_sources']
        has_ecmwf = 'ecmwf' in ancillary_sources
        has_era5 = 'era5' in ancillary_sources
        self.assertTrue(has_ecmwf or has_era5,
                        "ancillary_sources should contain at least 'ecmwf' or 'era5'")

    def test_ancillary_sources_structure(self):
        """Test that ancillary sources have correct structure (name and path)."""
        ancillary_sources = self.config['ancillary_sources']

        # Test each ancillary type (ecmwf, era5, etc.)
        for ancillary_type, sources in ancillary_sources.items():
            with self.subTest(ancillary_type=ancillary_type):
                self.assertIsInstance(sources, list,
                                      f"{ancillary_type} sources should be a list")
                self.assertGreater(len(sources), 0,
                                   f"{ancillary_type} sources should not be empty")

                for source in sources:
                    self.assertIn('name', source,
                                  f"Each {ancillary_type} source must have a 'name' field")
                    self.assertIn('path', source,
                                  f"Each {ancillary_type} source must have a 'path' field")
                    self.assertIsInstance(source['name'], str,
                                          f"{ancillary_type} source name must be a string")
                    self.assertIsInstance(source['path'], str,
                                          f"{ancillary_type} source path must be a string")

    def test_masks_optional(self):
        """Test that masks field is optional."""
        # This test passes if masks doesn't exist or if it exists and is valid
        if 'masks' in self.config:
            # If masks exists, it should be a dict
            self.assertIsInstance(self.config['masks'], dict,
                                  "If masks exists, it should be a dict")

    def test_masks_structure_if_present(self):
        """Test masks structure if present (should have categories with list of sources)."""
        if 'masks' not in self.config:
            self.skipTest("masks not configured (optional)")

        masks = self.config['masks']

        # Each category should be a list of sources
        for category, sources in masks.items():
            with self.subTest(category=category):
                self.assertIsInstance(sources, list,
                                      f"Mask category '{category}' should be a list")

                for source in sources:
                    self.assertIn('name', source,
                                  f"Each source in mask category '{category}' must have a 'name' field")
                    self.assertIn('path', source,
                                  f"Each source in mask category '{category}' must have a 'path' field")
                    self.assertIsInstance(source['name'], str,
                                          f"Mask source name in '{category}' must be a string")
                    self.assertIsInstance(source['path'], str,
                                          f"Mask source path in '{category}' must be a string")

    def test_default_config_file_exists(self):
        """Test that the default data_config.yaml file exists in the package."""
        import grdwindinversion
        config_path = os.path.join(
            os.path.dirname(grdwindinversion.__file__),
            'data_config.yaml'
        )
        self.assertTrue(os.path.exists(config_path),
                        f"Default config file should exist at {config_path}")

    def test_default_config_valid_yaml(self):
        """Test that the default config file is valid YAML."""
        import grdwindinversion
        config_path = os.path.join(
            os.path.dirname(grdwindinversion.__file__),
            'data_config.yaml'
        )

        with open(config_path, 'r') as f:
            try:
                yaml.safe_load(f)
            except yaml.YAMLError as e:
                self.fail(f"Default config file is not valid YAML: {e}")


if __name__ == '__main__':
    unittest.main()
