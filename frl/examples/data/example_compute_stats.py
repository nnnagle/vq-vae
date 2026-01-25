#!/usr/bin/env python3
"""
Example script to compute statistics for forest dataset features.

This script demonstrates how to use the StatsCalculator to compute
univariate statistics and covariance matrices for features defined
in the bindings configuration.
"""

import logging
from pathlib import Path

from data.stats import compute_stats_from_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Compute statistics from bindings configuration."""

    # Path to bindings YAML
    bindings_path = "config/frl_binding_v1.yaml"

    logger.info(f"Computing stats from: {bindings_path}")

    # Compute stats (will respect 'compute: if-not-exists' setting)
    compute_stats_from_config(bindings_path, force=False)

    logger.info("Done!")


if __name__ == "__main__":
    main()
