import pytest
import numpy as np
import logging
from unittest.mock import patch

class TestCheckPerplexities:
    
    def setup_method(self):
        """Test class setup with a dummy parent class"""
        class DummyParent:
            def check_perplexities(self, perplexities, n_samples):
                if isinstance(perplexities, np.ndarray):
                    if perplexities.size == 0:
                        raise ValueError("`perplexities` must be non-empty")
                elif not perplexities:
                    raise ValueError("`perplexities` must be non-empty")

                perplexities = np.asarray(perplexities, dtype=int)
                if perplexities.size > 1:
                    perplexities = np.unique(perplexities)

                if np.any(perplexities <= 0):
                    raise ValueError("All perplexity values must be positive")
                
                max_allowed_perplexity = (n_samples - 1) // 3

                perplexities_above_max = perplexities > max_allowed_perplexity
                
                if np.any(perplexities_above_max):
                    logging.warning(f"Excessively high perplexity values {perplexities[perplexities_above_max]} were detected and clipped to {max_allowed_perplexity}")
                    perplexities = np.unique(np.clip(perplexities, None, max_allowed_perplexity))

                logging.info("Perplexity values have been successfully validated.")
                return perplexities.astype(int)
        
        self.obj = DummyParent()
    
    def test_single_valid_perplexity(self):
        assert np.array_equal(self.obj.check_perplexities([30], 100), [30])
    
    def test_multiple_valid_perplexities(self):
        result = self.obj.check_perplexities([30, 10, 20], 100)
        assert np.array_equal(result, [10, 20, 30])  # Sorted and deduplicated
    
    def test_duplicate_perplexities(self):
        result = self.obj.check_perplexities([10, 20, 10], 100)
        assert np.array_equal(result, [10, 20])
    
    # Test edge cases
    def test_max_allowed_perplexity(self):
        n_samples = 100
        max_allowed = (n_samples - 1) // 3
        result = self.obj.check_perplexities([max_allowed], n_samples)
        assert np.array_equal(result, [max_allowed])
    
    def test_above_max_perplexity(self):
        n_samples = 100
        max_allowed = (n_samples - 1) // 3
        result = self.obj.check_perplexities([max_allowed + 10], n_samples)
        assert np.array_equal(result, [max_allowed])
    
    def test_multiple_above_max(self):
        n_samples = 100
        max_allowed = (n_samples - 1) // 3
        result = self.obj.check_perplexities([max_allowed + 5, max_allowed + 10], n_samples)
        assert np.array_equal(result, [max_allowed])
    
    # Test invalid inputs
    def test_empty_input(self):
        with pytest.raises(ValueError, match="`perplexities` must be non-empty"):
            self.obj.check_perplexities([], 100)
    
    def test_zero_perplexity(self):
        with pytest.raises(ValueError, match="All perplexity values must be positive"):
            self.obj.check_perplexities([0, 30], 100)
    
    def test_negative_perplexity(self):
        with pytest.raises(ValueError, match="All perplexity values must be positive"):
            self.obj.check_perplexities([-5, 30], 100)
    
    def test_non_integer_perplexity(self):
        result = self.obj.check_perplexities([30.5, 10.2], 100)
        assert np.array_equal(result, [10, 30])  # Should be converted to int
    
    # Test logging
    def test_warning_on_high_perplexity(self, caplog):
        n_samples = 100
        max_allowed = (n_samples - 1) // 3
        with caplog.at_level(logging.WARNING):
            self.obj.check_perplexities([max_allowed + 10], n_samples)
            assert "Excessively high perplexity values" in caplog.text
    
    def test_info_on_success(self, caplog):
        with caplog.at_level(logging.INFO):
            self.obj.check_perplexities([30], 100)
            assert "successfully validated" in caplog.text
    
    # Test numpy array inputs
    def test_numpy_array_input(self):
        result = self.obj.check_perplexities(np.array([30, 20]), 100)
        assert np.array_equal(result, [20, 30])
    
    def test_single_element_numpy_array(self):
        result = self.obj.check_perplexities(np.array([30]), 100)
        assert np.array_equal(result, [30])