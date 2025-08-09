import numpy as np
import pandas as pd
import pytest

from src.bias.stbc import STBC


class TestSTBC:
    """Test suite for Short-term Bias Compensation (STBC) class."""
    
    def create_test_dataframe(self, close_prices, predictions, volume=None, market_cap=None):
        """Helper method to create properly structured DataFrame for STBC testing."""
        n = len(close_prices)
        
        # Default values if not provided
        if volume is None:
            volume = [1000 + i * 100 for i in range(n)]
        if market_cap is None:
            market_cap = [50000 + i * 1000 for i in range(n)]
        
        # Create DataFrame with exact structure expected by STBC
        df = pd.DataFrame({
            'close': close_prices,          # Column 0: actual prices (STBC reads from here)
            'volume': volume,               # Column 1: volume  
            'marketCap': market_cap,        # Column 2: market cap
            'Predictions': predictions,     # Column 3: predictions (STBC reads from here)
            'diff': [p - c for p, c in zip(predictions, close_prices)],                    # Column 4: difference
            'diff_percentage': [(p - c) / c for p, c in zip(predictions, close_prices)]    # Column 5: percentage diff
        })
        
        return df
    
    def test_stbc_initialization(self):
        """Test STBC initialization with valid parameters."""
        df = self.create_test_dataframe(
            close_prices=[100.0, 105.0, 102.0],
            predictions=[98.0, 107.0, 104.0]
        )
        
        stbc = STBC(df, calibration_threshold=0.05)
        
        # Check that DataFrame is copied
        assert not stbc.df is df  # Should be a copy
        assert stbc.df.equals(df)
        
        # Check initial values
        assert stbc.previous_calibration == 0
        assert stbc.calibration_threshold == 0.05
    
    def test_stbc_calibrate_basic(self):
        """Test basic calibration functionality."""
        df = self.create_test_dataframe(
            close_prices=[100.0, 105.0, 102.0, 108.0],
            predictions=[98.0, 107.0, 104.0, 106.0]
        )
        
        stbc = STBC(df, calibration_threshold=0.05)
        stbc.calibrate()
        
        # Should have added two new columns at positions 6 and 7
        assert 'STBC Predictions' in stbc.df.columns
        assert 'STBC Predictions diff' in stbc.df.columns
        assert stbc.df.shape[1] == df.shape[1] + 2
        
        # Check that calibrated predictions column was inserted at position 6
        assert stbc.df.columns[6] == 'STBC Predictions'
        assert stbc.df.columns[7] == 'STBC Predictions diff'
    
    def test_stbc_calibrate_with_threshold(self):
        """Test calibration behavior with threshold logic."""
        df = self.create_test_dataframe(
            close_prices=[100.0, 105.0],
            predictions=[95.0, 100.0]  # Significant errors to trigger calibration
        )
        
        stbc = STBC(df, calibration_threshold=0.03)  # 3% threshold
        stbc.calibrate()
        
        # First row: calibrated_price = 95.0 + 0 = 95.0
        # calibration_today = 100.0 - 95.0 = 5.0
        # 5.0 > 100.0 * 0.03 = 3.0, so previous_calibration = 5.0
        
        first_calibrated = stbc.df.iloc[0, 6]  # STBC Predictions column
        assert np.isclose(first_calibrated, 95.0)
        
        # Second row should use the updated calibration
        second_calibrated = stbc.df.iloc[1, 6]
        expected_second = 100.0 + 5.0  # prediction + previous_calibration
        assert np.isclose(second_calibrated, expected_second)
    
    def test_stbc_evaluate_accuracy(self):
        """Test accuracy evaluation method."""
        df = self.create_test_dataframe(
            close_prices=[100.0, 110.0],
            predictions=[95.0, 105.0]
        )
        
        stbc = STBC(df, calibration_threshold=0.05)
        stbc.calibrate()
        
        accuracy = stbc.evaluate_accuracy()
        
        # Should return mean absolute percentage difference
        assert isinstance(accuracy, (float, np.floating))
        assert accuracy >= 0  # Accuracy should be non-negative
    
    def test_stbc_evaluate_mse(self):
        """Test MSE evaluation method."""
        df = self.create_test_dataframe(
            close_prices=[100.0, 110.0, 120.0],
            predictions=[98.0, 108.0, 118.0]
        )
        
        stbc = STBC(df, calibration_threshold=0.05)
        stbc.calibrate()
        
        mse = stbc.evaluate_mse()
        
        # Should return mean squared error
        assert isinstance(mse, (float, np.floating))
        assert mse >= 0  # MSE should be non-negative
    
    def test_stbc_perfect_predictions(self):
        """Test STBC with perfect predictions (should have minimal adjustment)."""
        df = self.create_test_dataframe(
            close_prices=[100.0, 105.0, 102.0],
            predictions=[100.0, 105.0, 102.0]  # Perfect predictions
        )
        
        stbc = STBC(df, calibration_threshold=0.01)  # Very strict threshold
        stbc.calibrate()
        
        accuracy = stbc.evaluate_accuracy()
        mse = stbc.evaluate_mse()
        
        # With perfect predictions, calibration should be minimal
        assert accuracy < 0.1  # Very low error
        assert mse < 1.0  # Very low MSE
    
    def test_stbc_small_dataframe(self):
        """Test STBC with minimal data."""
        df = self.create_test_dataframe(
            close_prices=[100.0],
            predictions=[95.0]
        )
        
        stbc = STBC(df, calibration_threshold=0.05)
        stbc.calibrate()
        
        # Should work with single row
        assert stbc.df.shape[0] == 1
        assert 'STBC Predictions' in stbc.df.columns
        assert 'STBC Predictions diff' in stbc.df.columns
    
    def test_stbc_threshold_edge_cases(self):
        """Test STBC behavior at threshold boundaries."""
        df = self.create_test_dataframe(
            close_prices=[100.0, 100.0],
            predictions=[97.0, 98.0]  # 3% and 2% errors
        )
        
        stbc = STBC(df, calibration_threshold=0.025)  # 2.5% threshold
        stbc.calibrate()
        
        # First prediction: 3% error > 2.5% threshold, should calibrate
        # Second prediction: 2% error < 2.5% threshold, should not calibrate much
        
        accuracy = stbc.evaluate_accuracy()
        assert isinstance(accuracy, (float, np.floating))
    
    def test_stbc_negative_values(self):
        """Test STBC with negative values in data."""
        df = self.create_test_dataframe(
            close_prices=[-10.0, 5.0, 15.0],
            predictions=[-8.0, 7.0, 12.0],
            volume=[1000, 1100, 1200],
            market_cap=[50000, 55000, 60000]
        )
        
        stbc = STBC(df, calibration_threshold=0.1)
        
        # Should not crash with negative values
        stbc.calibrate()
        accuracy = stbc.evaluate_accuracy()
        mse = stbc.evaluate_mse()
        
        assert isinstance(accuracy, (float, np.floating))
        assert isinstance(mse, (float, np.floating))
    
    def test_stbc_zero_threshold(self):
        """Test STBC with zero threshold (should always calibrate)."""
        df = self.create_test_dataframe(
            close_prices=[100.0, 105.0],
            predictions=[99.0, 104.0]
        )
        
        stbc = STBC(df, calibration_threshold=0.0)
        stbc.calibrate()
        
        # With zero threshold, any error should trigger calibration
        accuracy = stbc.evaluate_accuracy()
        assert isinstance(accuracy, (float, np.floating))
    
    def test_stbc_high_threshold(self):
        """Test STBC with very high threshold (should rarely calibrate)."""
        df = self.create_test_dataframe(
            close_prices=[100.0, 105.0, 102.0],
            predictions=[90.0, 95.0, 92.0]  # Large errors
        )
        
        stbc = STBC(df, calibration_threshold=1.0)  # 100% threshold
        stbc.calibrate()
        
        # With very high threshold, should rarely trigger calibration
        accuracy = stbc.evaluate_accuracy()
        assert isinstance(accuracy, (float, np.floating))
    
    def test_stbc_dataframe_copy_independence(self):
        """Test that STBC operates on a copy and doesn't modify original data."""
        original_df = self.create_test_dataframe(
            close_prices=[100.0, 105.0],
            predictions=[98.0, 103.0]
        )
        
        original_shape = original_df.shape
        original_columns = original_df.columns.tolist()
        
        stbc = STBC(original_df, calibration_threshold=0.05)
        stbc.calibrate()
        
        # Original DataFrame should be unchanged
        assert original_df.shape == original_shape
        assert original_df.columns.tolist() == original_columns
        assert 'STBC Predictions' not in original_df.columns
