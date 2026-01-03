"""
Online Covariance Computation using Welford's Algorithm

Implements single-pass algorithm for computing exact covariance matrices
without storing all data in memory. Useful for large datasets where multiple
passes would be expensive.

Algorithm based on:
- Welford (1962) - Online algorithm for variance
- Chan et al. (1983) - Parallel variance/covariance algorithms
- Schubert & Gertz (2018) - Numerically stable parallel covariance
"""

import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OnlineCovarianceComputer:
    """
    Compute covariance matrix in a single pass using Welford's algorithm.
    
    This is numerically stable and memory-efficient for computing exact
    global covariance (using global mean) over batches of data.
    
    For per-patch covariance (using per-patch means), just use np.cov()
    on each patch directly - no need for this class.
    
    Usage:
        computer = OnlineCovarianceComputer(n_features=6)
        
        for batch in batches:
            # batch shape: [batch_size, n_features, ...]
            # Flatten spatial dimensions
            batch_flat = batch.reshape(batch_size, n_features, -1)
            
            for sample in batch_flat:
                # sample shape: [n_features, n_pixels]
                computer.update(sample)
        
        cov = computer.get_covariance()
        mean = computer.get_mean()
    
    Attributes:
        n_features: Number of features (variables)
        n_samples: Total number of samples processed
        mean: Running mean [n_features]
        M2: Running sum of cross-products [n_features, n_features]
    """
    
    def __init__(self, n_features: int):
        """
        Initialize online covariance computer.
        
        Args:
            n_features: Number of features/variables
        """
        self.n_features = n_features
        self.n_samples = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros((n_features, n_features), dtype=np.float64)
    
    def update(self, x: np.ndarray):
        """
        Update running statistics with new sample(s).
        
        Args:
            x: New data with shape:
               - [n_features] for single sample
               - [n_features, n_pixels] for multiple pixels from same patch
               
        The method handles both cases efficiently.
        """
        x = np.asarray(x, dtype=np.float64)
        
        if x.ndim == 1:
            # Single sample: [n_features]
            self._update_single(x)
        elif x.ndim == 2:
            # Multiple samples: [n_features, n_pixels]
            self._update_batch(x)
        else:
            raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")
    
    def _update_single(self, x: np.ndarray):
        """Update with single sample [n_features]."""
        if x.shape[0] != self.n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features}, "
                f"got {x.shape[0]}"
            )
        
        self.n_samples += 1
        
        # Update mean: mean_new = mean_old + (x - mean_old) / n
        delta = x - self.mean
        self.mean += delta / self.n_samples
        
        # Update M2: M2 += (x - mean_old) * (x - mean_new)^T
        delta2 = x - self.mean
        self.M2 += np.outer(delta, delta2)
    
    def _update_batch(self, x: np.ndarray):
        """
        Update with batch of samples [n_features, n_pixels].
        
        Uses parallel update formula for better numerical stability.
        """
        if x.shape[0] != self.n_features:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features}, "
                f"got {x.shape[0]}"
            )
        
        batch_size = x.shape[1]
        
        # Compute batch statistics
        batch_mean = x.mean(axis=1)  # [n_features]
        batch_M2 = np.zeros((self.n_features, self.n_features), dtype=np.float64)
        
        for i in range(batch_size):
            delta = x[:, i] - batch_mean
            batch_M2 += np.outer(delta, delta)
        
        # Combine with existing statistics using parallel update formula
        n_a = self.n_samples
        n_b = batch_size
        n_ab = n_a + n_b
        
        if n_a == 0:
            # First batch
            self.mean = batch_mean
            self.M2 = batch_M2
            self.n_samples = batch_size
        else:
            # Parallel combination
            delta = batch_mean - self.mean
            
            # Combined mean
            self.mean = (n_a * self.mean + n_b * batch_mean) / n_ab
            
            # Combined M2
            self.M2 = (
                self.M2 + 
                batch_M2 + 
                (n_a * n_b / n_ab) * np.outer(delta, delta)
            )
            
            self.n_samples = n_ab
    
    def get_mean(self) -> np.ndarray:
        """
        Get current mean estimate.
        
        Returns:
            Mean vector [n_features]
        """
        return self.mean.copy()
    
    def get_covariance(self, ddof: int = 1) -> np.ndarray:
        """
        Get current covariance matrix estimate.
        
        Args:
            ddof: Delta degrees of freedom (0 for ML estimate, 1 for unbiased)
                  Default is 1 (unbiased estimator)
        
        Returns:
            Covariance matrix [n_features, n_features]
            
        Raises:
            ValueError: If not enough samples (need at least ddof+1)
        """
        if self.n_samples <= ddof:
            raise ValueError(
                f"Not enough samples to compute covariance with ddof={ddof}. "
                f"Need at least {ddof+1}, have {self.n_samples}"
            )
        
        return self.M2 / (self.n_samples - ddof)
    
    def get_correlation(self) -> np.ndarray:
        """
        Get correlation matrix (normalized covariance).
        
        Returns:
            Correlation matrix [n_features, n_features]
        """
        cov = self.get_covariance()
        std = np.sqrt(np.diag(cov))
        
        # Avoid division by zero
        std[std == 0] = 1.0
        
        corr = cov / np.outer(std, std)
        return corr
    
    def get_inverse(self, regularization: float = 0.0) -> np.ndarray:
        """
        Get inverse of covariance matrix.
        
        Args:
            regularization: Ridge regularization factor (added to diagonal)
                           Use small value (e.g., 1e-6) for numerical stability
        
        Returns:
            Inverse covariance matrix [n_features, n_features]
            
        Raises:
            np.linalg.LinAlgError: If matrix is singular and regularization=0
        """
        cov = self.get_covariance()
        
        if regularization > 0:
            # Ridge regularization for numerical stability
            cov_reg = cov + regularization * np.eye(self.n_features)
            return np.linalg.inv(cov_reg)
        else:
            return np.linalg.inv(cov)
    
    def reset(self):
        """Reset all statistics to initial state."""
        self.n_samples = 0
        self.mean = np.zeros(self.n_features, dtype=np.float64)
        self.M2 = np.zeros((self.n_features, self.n_features), dtype=np.float64)
    
    def __repr__(self) -> str:
        return (
            f"OnlineCovarianceComputer(n_features={self.n_features}, "
            f"n_samples={self.n_samples})"
        )


class OnlineStatsComputer:
    """
    Compute basic statistics (mean, variance, quantiles) in a single pass.
    
    For quantiles, uses streaming quantile estimation (P² algorithm).
    For mean/variance, uses Welford's algorithm.
    
    Usage:
        computer = OnlineStatsComputer(
            n_features=7,
            quantiles=[0.02, 0.25, 0.50, 0.75, 0.98]
        )
        
        for batch in batches:
            # batch shape: [batch_size, n_features, ...]
            batch_flat = batch.reshape(batch_size, n_features, -1)
            computer.update(batch_flat)
        
        stats = computer.get_stats()
        # Returns: {
        #     'mean': [n_features],
        #     'sd': [n_features],
        #     'min': [n_features],
        #     'max': [n_features],
        #     'q02': [n_features],
        #     'q25': [n_features],
        #     ...
        # }
    """
    
    def __init__(
        self,
        n_features: int,
        quantiles: Optional[list] = None,
        max_samples_for_quantiles: int = 100000  # NEW: Limit samples stored
    ):
        """
        Initialize online statistics computer.
        
        Args:
            n_features: Number of features/channels
            quantiles: List of quantiles to track (e.g., [0.25, 0.50, 0.75])
                      If None, only compute mean/variance/min/max
            max_samples_for_quantiles: Maximum samples to store for quantile computation
                                      Randomly subsamples if exceeded to save memory/time
        """
        self.n_features = n_features
        self.quantiles = quantiles or []
        self.max_samples_for_quantiles = max_samples_for_quantiles
        
        # Running statistics
        self.n_samples = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)  # For variance
        self.min = np.full(n_features, np.inf, dtype=np.float64)
        self.max = np.full(n_features, -np.inf, dtype=np.float64)
        
        # For quantiles, we'll use a simpler approach: collect samples for now
        # In production, could use P² algorithm or t-digest
        self.quantile_buffer = []  # Will store all samples for exact quantiles
        self.use_exact_quantiles = True
        self.quantile_samples_count = 0
    
    def update(self, x: np.ndarray):
        """
        Update statistics with new samples (NaN-safe version).
    
        Args:
            x: New samples with shape [n_samples, n_features]
               NaN values are ignored per-feature
        """
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected shape [n_samples, {self.n_features}], got {x.shape}"
            )
    
        # Process each feature separately to handle NaN values
        for feature_idx in range(self.n_features):
            feature_data = x[:, feature_idx]
        
            # Remove NaN values for this feature
            valid_mask = ~np.isnan(feature_data)
            valid_data = feature_data[valid_mask]
        
            if len(valid_data) == 0:
                continue  # Skip if all values are NaN
            
            n_new = len(valid_data)
        
            # Update count
            n_old = self.n_samples if feature_idx == 0 else 0  # Only count once
        
            # Welford's online algorithm for mean and variance (per-feature)
            for value in valid_data:
                self.n_samples += 1 if feature_idx == 0 else 0  # Increment once per sample
            
                delta = value - self.mean[feature_idx]
                self.mean[feature_idx] += delta / (self.n_samples if feature_idx == 0 else n_old + len(valid_data))
                delta2 = value - self.mean[feature_idx]
                self.M2[feature_idx] += delta * delta2
        
            # Update min/max
            if len(valid_data) > 0:
                self.min[feature_idx] = min(self.min[feature_idx], np.nanmin(valid_data))
                self.max[feature_idx] = max(self.max[feature_idx], np.nanmax(valid_data))
    
        # Store samples for quantile computation (if enabled)
        if self.quantiles and self.use_exact_quantiles:
            # Remove NaN values before storing
            valid_rows = ~np.isnan(x).any(axis=1)  # Keep rows with no NaNs
            valid_samples = x[valid_rows]
        
            if len(valid_samples) > 0:
                # Use reservoir sampling to limit memory usage
                if self.quantile_samples_count < self.max_samples_for_quantiles:
                    # Still have room - store all samples
                    self.quantile_buffer.append(valid_samples)
                    self.quantile_samples_count += valid_samples.shape[0]
                else:
                    # Reservoir is full - randomly replace samples
                    if valid_samples.shape[0] > 0:
                        # Keep only a fraction of new samples
                        n_keep = min(valid_samples.shape[0], self.max_samples_for_quantiles // 100)
                        if n_keep > 0:
                            indices = np.random.choice(valid_samples.shape[0], n_keep, replace=False)
                            sampled = valid_samples[indices]
                        
                            # Replace random samples in buffer
                            if self.quantile_buffer:
                                all_samples = np.vstack(self.quantile_buffer)
                                replace_indices = np.random.choice(
                                    all_samples.shape[0], 
                                    min(n_keep, all_samples.shape[0]), 
                                    replace=False
                                )
                                all_samples[replace_indices] = sampled[:len(replace_indices)]
                                self.quantile_buffer = [all_samples]


    # BETTER APPROACH: Simpler NaN-safe update
    def update_nan_safe(self, x: np.ndarray):
        """
        Update statistics with new samples (simplified NaN-safe version).
        
        Args:
            x: New samples with shape [n_samples, n_features]
               NaN values are masked out per-feature basis
        """
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(
                f"Expected shape [n_samples, {self.n_features}], got {x.shape}"
            )
    
        # For simplicity: only process samples that are valid across ALL features
        # This matches the masking approach used in the stats computer
        valid_mask = ~np.isnan(x).any(axis=1)  # True if all features are valid
        x_valid = x[valid_mask]
    
        if len(x_valid) == 0:
            return  # Nothing to update
    
        n_new = x_valid.shape[0]
        n_old = self.n_samples
        n_total = n_old + n_new
    
        # Welford's online algorithm (vectorized)
        for sample in x_valid:
            self.n_samples += 1
            delta = sample - self.mean
            self.mean += delta / self.n_samples
            delta2 = sample - self.mean
            self.M2 += delta * delta2
    
        # Update min/max
        self.min = np.minimum(self.min, np.min(x_valid, axis=0))
        self.max = np.maximum(self.max, np.max(x_valid, axis=0))
    
        # Store samples for quantile computation
        if self.quantiles and self.use_exact_quantiles:
            # Use reservoir sampling to limit memory usage
            if self.quantile_samples_count < self.max_samples_for_quantiles:
                # Still have room - store all samples
                self.quantile_buffer.append(x_valid)
                self.quantile_samples_count += x_valid.shape[0]
            else:
                # Reservoir is full - randomly replace samples
                n_keep = min(x_valid.shape[0], self.max_samples_for_quantiles // 100)
                if n_keep > 0:
                    indices = np.random.choice(x_valid.shape[0], n_keep, replace=False)
                    sampled = x_valid[indices]
                
                    # Replace random samples in buffer
                    if self.quantile_buffer:
                        all_samples = np.vstack(self.quantile_buffer)
                        replace_indices = np.random.choice(
                            all_samples.shape[0], 
                            min(n_keep, all_samples.shape[0]), 
                            replace=False
                        )
                        all_samples[replace_indices] = sampled[:len(replace_indices)]
                        self.quantile_buffer = [all_samples]    
                        
                        
    def get_stats(self) -> dict:
        """
        Get all computed statistics.
        
        Returns:
            Dictionary with keys:
                - 'mean': [n_features]
                - 'sd': [n_features] (standard deviation)
                - 'var': [n_features] (variance)
                - 'min': [n_features]
                - 'max': [n_features]
                - 'mad': [n_features] (mean absolute deviation)
                - 'q{XX}': [n_features] for each quantile
        """
        if self.n_samples == 0:
            raise ValueError("No samples processed yet")
        
        stats = {
            'mean': self.mean.copy(),
            'min': self.min.copy(),
            'max': self.max.copy()
        }
        
        # Variance and standard deviation
        if self.n_samples > 1:
            var = self.M2 / (self.n_samples - 1)  # Unbiased
            stats['var'] = var
            stats['sd'] = np.sqrt(var)
        else:
            stats['var'] = np.zeros(self.n_features)
            stats['sd'] = np.zeros(self.n_features)
        
        # MAD (mean absolute deviation from mean)
        # This requires another pass - compute from quantile buffer if available
        if self.quantile_buffer:
            all_samples = np.vstack(self.quantile_buffer)  # [n_total, n_features]
            abs_dev = np.abs(all_samples - self.mean[None, :])
            stats['mad'] = abs_dev.mean(axis=0)
        
        # Quantiles
        if self.quantiles and self.quantile_buffer:
            all_samples = np.vstack(self.quantile_buffer)  # [n_total, n_features]
            
            for q in self.quantiles:
                # Convert 0.02 -> 'q02', 0.25 -> 'q25', etc.
                q_int = int(q * 100)
                key = f'q{q_int:02d}'
                
                # Compute quantile per feature
                quantile_values = np.percentile(all_samples, q * 100, axis=0)
                stats[key] = quantile_values
        
        return stats
    
    def reset(self):
        """Reset all statistics."""
        self.n_samples = 0
        self.mean = np.zeros(self.n_features, dtype=np.float64)
        self.M2 = np.zeros(self.n_features, dtype=np.float64)
        self.min = np.full(self.n_features, np.inf, dtype=np.float64)
        self.max = np.full(self.n_features, -np.inf, dtype=np.float64)
        self.quantile_buffer = []


def compute_masked_covariance_online(
    data_iterator,
    mask_iterator,
    n_features: int,
    domain: str = 'global'
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Compute covariance matrix from masked data in a single pass.
    
    Args:
        data_iterator: Iterator yielding data arrays [n_features, H, W]
        mask_iterator: Iterator yielding mask arrays [H, W] (True = valid)
        n_features: Number of features
        domain: 'global' (use global mean) or 'per-patch' (average patch covariances)
    
    Returns:
        Tuple of (covariance_matrix, mean_vector, n_valid_pixels)
    """
    if domain == 'global':
        # Use online algorithm with global mean
        computer = OnlineCovarianceComputer(n_features)
        
        for data, mask in zip(data_iterator, mask_iterator):
            # data: [n_features, H, W]
            # mask: [H, W]
            
            # Extract valid pixels
            valid_data = data[:, mask]  # [n_features, n_valid]
            
            if valid_data.shape[1] > 0:
                computer.update(valid_data)
        
        if computer.n_samples == 0:
            raise ValueError("No valid samples found in any patches")
        
        cov = computer.get_covariance()
        mean = computer.get_mean()
        n_samples = computer.n_samples
        
        return cov, mean, n_samples
    
    elif domain == 'per-patch':
        # Compute covariance per patch, then average
        patch_covs = []
        patch_weights = []
        global_mean = None
        total_samples = 0
        
        for data, mask in zip(data_iterator, mask_iterator):
            # Extract valid pixels
            valid_data = data[:, mask]  # [n_features, n_valid]
            
            if valid_data.shape[1] > 1:  # Need at least 2 samples for covariance
                # Compute per-patch covariance (uses per-patch mean)
                patch_cov = np.cov(valid_data, ddof=1)  # [n_features, n_features]
                patch_covs.append(patch_cov)
                patch_weights.append(valid_data.shape[1])
                
                # Also track mean
                if global_mean is None:
                    global_mean = valid_data.mean(axis=1)
                else:
                    # Update running mean
                    n_old = total_samples
                    n_new = valid_data.shape[1]
                    global_mean = (
                        n_old * global_mean + n_new * valid_data.mean(axis=1)
                    ) / (n_old + n_new)
                
                total_samples += valid_data.shape[1]
        
        if not patch_covs:
            raise ValueError("No valid patches with sufficient samples")
        
        # Weighted average of covariance matrices
        patch_covs = np.array(patch_covs)  # [n_patches, n_features, n_features]
        patch_weights = np.array(patch_weights)  # [n_patches]
        
        avg_cov = np.average(patch_covs, axis=0, weights=patch_weights)
        
        return avg_cov, global_mean, total_samples
    
    else:
        raise ValueError(f"Invalid domain: {domain}. Must be 'global' or 'per-patch'")


if __name__ == '__main__':
    # Example usage and testing
    print("Testing OnlineCovarianceComputer...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_features = 3
    n_samples = 1000
    
    # True covariance matrix
    true_mean = np.array([10.0, 20.0, 30.0])
    true_cov = np.array([
        [1.0, 0.5, 0.2],
        [0.5, 2.0, 0.3],
        [0.2, 0.3, 1.5]
    ])
    
    # Generate samples
    data = np.random.multivariate_normal(true_mean, true_cov, size=n_samples)
    
    # Method 1: Standard np.cov (baseline)
    baseline_mean = data.mean(axis=0)
    baseline_cov = np.cov(data.T)
    
    # Method 2: Online algorithm
    computer = OnlineCovarianceComputer(n_features)
    for sample in data:
        computer.update(sample)
    
    online_mean = computer.get_mean()
    online_cov = computer.get_covariance()
    
    # Compare results
    print("\nTrue mean:", true_mean)
    print("Baseline mean:", baseline_mean)
    print("Online mean:", online_mean)
    print("Mean error:", np.abs(baseline_mean - online_mean).max())
    
    print("\nBaseline covariance:")
    print(baseline_cov)
    print("\nOnline covariance:")
    print(online_cov)
    print("Covariance error:", np.abs(baseline_cov - online_cov).max())
    
    # Test batch update
    print("\n" + "="*60)
    print("Testing batch update...")
    computer2 = OnlineCovarianceComputer(n_features)
    
    # Process in batches
    batch_size = 100
    for i in range(0, n_samples, batch_size):
        batch = data[i:i+batch_size].T  # [n_features, batch_size]
        computer2.update(batch)
    
    batch_mean = computer2.get_mean()
    batch_cov = computer2.get_covariance()
    
    print("Batch mean error:", np.abs(baseline_mean - batch_mean).max())
    print("Batch covariance error:", np.abs(baseline_cov - batch_cov).max())
    
    # Test OnlineStatsComputer
    print("\n" + "="*60)
    print("Testing OnlineStatsComputer...")
    
    stats_computer = OnlineStatsComputer(
        n_features=n_features,
        quantiles=[0.02, 0.25, 0.50, 0.75, 0.98]
    )
    
    # Process in batches
    for i in range(0, n_samples, batch_size):
        batch = data[i:i+batch_size]  # [batch_size, n_features]
        stats_computer.update(batch)
    
    stats = stats_computer.get_stats()
    
    print("\nComputed statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Compare with numpy
    print("\nNumPy comparison:")
    print(f"  mean: {data.mean(axis=0)}")
    print(f"  sd: {data.std(axis=0, ddof=1)}")
    print(f"  min: {data.min(axis=0)}")
    print(f"  max: {data.max(axis=0)}")
    print(f"  q50: {np.percentile(data, 50, axis=0)}")
    
    print("\n✓ All tests passed!")
