"""
Pre-normalization transforms for feature channels.

This module provides a registry of element-wise mathematical transforms
(log, sqrt, etc.) that can be applied to raw channel data *before*
normalization.  The same transform is applied both when computing
statistics (``StatsCalculator``) and when building features at training
time (``FeatureBuilder``), so that the stats match the transformed
data distribution.

Supported transforms
--------------------

======== ============================== ==========================
Name     Function                       Domain
======== ============================== ==========================
log      ``np.log(x)``                  x > 0
log1p    ``np.log1p(x)``  (= ln(1+x))  x > -1  (safe for x >= 0)
log10    ``np.log10(x)``                x > 0
sqrt     ``np.sqrt(x)``                 x >= 0
cbrt     ``np.cbrt(x)``                 all reals
======== ============================== ==========================

Values outside a transform's domain produce ``NaN``, which is then
caught by the existing NaN masking in both the stats and feature
builder pipelines.  If unexpected NaNs appear, check that the raw
channel values are within the transform's domain.

Usage
-----

.. code-block:: python

    from data.loaders.transforms import apply_transform, TRANSFORMS

    # Apply a named transform
    transformed = apply_transform(raw_data, 'log1p')

    # Check if a name is valid
    assert 'sqrt' in TRANSFORMS

Configuration (YAML)
--------------------

Transforms are specified per-channel in the ``features`` section of the
bindings YAML:

.. code-block:: yaml

    features:
      ccdc_history:
        dim: [C, H, W]
        channels:
          static.spectral_distance_per_decade: {transform: log1p, norm: robust_iqr}
          static.variance_ndvi: {transform: sqrt, norm: robust_iqr}
          static.num_segments: {norm: robust_iqr}   # no transform
"""

import numpy as np
from typing import Callable, Dict, List, Optional

# ---------------------------------------------------------------------------
# Transform registry
# ---------------------------------------------------------------------------

#: Registry mapping transform names to numpy element-wise functions.
#: Each function accepts and returns an ``np.ndarray``.
TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'log':   np.log,
    'log1p': np.log1p,
    'log10': np.log10,
    'sqrt':  np.sqrt,
    'cbrt':  np.cbrt,
}


def get_transform_names() -> List[str]:
    """Return the sorted list of available transform names.

    Returns:
        List of registered transform name strings.
    """
    return sorted(TRANSFORMS.keys())


def validate_transform(name: str) -> None:
    """Validate that *name* is a registered transform.

    Args:
        name: Transform name to check.

    Raises:
        ValueError: If *name* is not in the registry.
    """
    if name not in TRANSFORMS:
        raise ValueError(
            f"Unknown transform '{name}'. "
            f"Available transforms: {get_transform_names()}"
        )


def apply_transform(
    data: np.ndarray,
    transform_name: Optional[str],
) -> np.ndarray:
    """Apply a named transform to *data*, returning a new array.

    If *transform_name* is ``None`` the data is returned unchanged
    (no copy).

    Values outside the mathematical domain of the transform (e.g.
    ``log`` of a negative number) will become ``NaN``.  Downstream
    masking handles these automatically.

    Args:
        data: Input array of any shape.
        transform_name: Name of a registered transform, or ``None``
            to skip.

    Returns:
        Transformed array (new allocation when a transform is applied).

    Raises:
        ValueError: If *transform_name* is not ``None`` and not
            registered.
    """
    if transform_name is None:
        return data

    validate_transform(transform_name)
    fn = TRANSFORMS[transform_name]
    return fn(data)
