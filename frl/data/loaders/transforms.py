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
log      ``np.log(x + epsilon)``        x > -epsilon  (default eps=1)
log1p    ``np.log1p(x)``  (= ln(1+x))  x > -1  (safe for x >= 0)
log10    ``np.log10(x)``                x > 0
sqrt     ``np.sqrt(x)``                 x >= 0
cbrt     ``np.cbrt(x)``                 all reals
======== ============================== ==========================

Values outside a transform's domain produce ``NaN``, which is then
caught by the existing NaN masking in both the stats and feature
builder pipelines.  If unexpected NaNs appear, check that the raw
channel values are within the transform's domain.

Parameterized transforms
------------------------

Transforms can be specified as a plain string or as a dict with
parameters:

.. code-block:: yaml

    features:
      ccdc_history:
        dim: [C, H, W]
        channels:
          # Simple string — log with default epsilon=1
          static.spectral_distance: {transform: log, norm: robust_iqr}

          # Dict with explicit epsilon
          static.treecover: {transform: {name: log, epsilon: 0.5}, norm: robust_iqr}

          # Non-parameterized transforms still work as plain strings
          static.variance_ndvi: {transform: sqrt, norm: robust_iqr}

Usage
-----

.. code-block:: python

    from data.loaders.transforms import apply_transform, TRANSFORMS

    # Apply a named transform (plain string)
    transformed = apply_transform(raw_data, 'log1p')

    # Apply a parameterized transform (dict)
    transformed = apply_transform(raw_data, {'name': 'log', 'epsilon': 0.5})

    # Check if a name is valid
    assert 'sqrt' in TRANSFORMS
"""

import numpy as np
from typing import Callable, Dict, List, Optional, Union

# Type alias: a transform spec is either a string name or a dict
TransformSpec = Optional[Union[str, Dict]]

# ---------------------------------------------------------------------------
# Transform registry — simple (non-parameterized) transforms
# ---------------------------------------------------------------------------

#: Registry mapping transform names to numpy element-wise functions.
#: Each function accepts and returns an ``np.ndarray``.
TRANSFORMS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    'log1p': np.log1p,
    'log10': np.log10,
    'sqrt':  np.sqrt,
    'cbrt':  np.cbrt,
}

# ---------------------------------------------------------------------------
# Parameterized transforms
# ---------------------------------------------------------------------------

#: Default epsilon for the ``log`` transform.
LOG_DEFAULT_EPSILON: float = 1.0

#: Set of transform names that accept parameters via dict syntax.
PARAMETERIZED_TRANSFORMS = {'log'}

#: All valid transform names (simple + parameterized).
ALL_TRANSFORM_NAMES = set(TRANSFORMS.keys()) | PARAMETERIZED_TRANSFORMS


def get_transform_names() -> List[str]:
    """Return the sorted list of available transform names.

    Returns:
        List of registered transform name strings.
    """
    return sorted(ALL_TRANSFORM_NAMES)


def _parse_transform_spec(spec: TransformSpec):
    """Parse a transform spec into (name, params).

    Args:
        spec: A string name, a dict ``{name: ..., **params}``, or None.

    Returns:
        ``(name, params)`` tuple.  *name* is ``None`` when *spec* is
        ``None``.

    Raises:
        ValueError: If *spec* is a dict without a ``name`` key.
    """
    if spec is None:
        return None, {}
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict):
        spec = dict(spec)  # shallow copy
        name = spec.pop('name', None)
        if name is None:
            raise ValueError(
                f"Parameterized transform dict must include a 'name' key, "
                f"got {spec}"
            )
        return name, spec
    raise TypeError(
        f"Transform spec must be str, dict, or None — got {type(spec).__name__}"
    )


def validate_transform(spec: TransformSpec) -> None:
    """Validate that *spec* refers to a registered transform.

    Accepts both plain string names and parameterized dicts::

        validate_transform('log1p')
        validate_transform({'name': 'log', 'epsilon': 0.5})

    Args:
        spec: Transform specification to check.

    Raises:
        ValueError: If the transform name is not registered.
    """
    name, _params = _parse_transform_spec(spec)
    if name is None:
        return
    if name not in ALL_TRANSFORM_NAMES:
        raise ValueError(
            f"Unknown transform '{name}'. "
            f"Available transforms: {get_transform_names()}"
        )


def apply_transform(
    data: np.ndarray,
    spec: TransformSpec,
) -> np.ndarray:
    """Apply a named transform to *data*, returning a new array.

    If *spec* is ``None`` the data is returned unchanged (no copy).

    Accepts both plain string names and parameterized dicts::

        apply_transform(arr, 'log1p')
        apply_transform(arr, {'name': 'log', 'epsilon': 0.5})

    For the ``log`` transform, computes ``np.log(x + epsilon)`` where
    *epsilon* defaults to ``LOG_DEFAULT_EPSILON`` (1.0).

    Args:
        data: Input array of any shape.
        spec: Transform specification — a string name, a dict
            ``{name: ..., **params}``, or ``None`` to skip.

    Returns:
        Transformed array (new allocation when a transform is applied).

    Raises:
        ValueError: If the transform name is not registered.
    """
    name, params = _parse_transform_spec(spec)
    if name is None:
        return data

    # Simple (non-parameterized) transforms
    if name in TRANSFORMS:
        fn = TRANSFORMS[name]
        return fn(data)

    # Parameterized: log with epsilon
    if name == 'log':
        epsilon = params.get('epsilon', LOG_DEFAULT_EPSILON)
        return np.log(data + epsilon)

    raise ValueError(
        f"Unknown transform '{name}'. "
        f"Available transforms: {get_transform_names()}"
    )


def inverse_transform(
    data: np.ndarray,
    spec: TransformSpec,
) -> np.ndarray:
    """Apply the inverse of a named transform, returning a new array.

    If *spec* is ``None`` the data is returned unchanged.

    Args:
        data: Input array of any shape.
        spec: Transform specification (same format as
            :func:`apply_transform`).

    Returns:
        Inverse-transformed array.

    Raises:
        ValueError: If the transform has no known inverse or the name
            is not registered.
    """
    name, params = _parse_transform_spec(spec)
    if name is None:
        return data

    if name == 'log':
        epsilon = params.get('epsilon', LOG_DEFAULT_EPSILON)
        return np.exp(data) - epsilon
    if name == 'log1p':
        return np.expm1(data)
    if name == 'log10':
        return np.power(10.0, data)
    if name == 'sqrt':
        return np.square(data)
    if name == 'cbrt':
        return np.power(data, 3)

    raise ValueError(
        f"Unknown transform '{name}'. "
        f"Available transforms: {get_transform_names()}"
    )
