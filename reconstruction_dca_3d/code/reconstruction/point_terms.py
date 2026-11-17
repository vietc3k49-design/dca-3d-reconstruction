"""Observation uncertainty and Cauchy robust IRLS loss for 2D reprojection residuals.

Default behavior (when cauchy_scale_pixels and point_weights are None) preserves
exact squared distance outside the observation disk for backward compatibility.
"""
import numpy as np


def _effective_residuals(residual, width, tolerance_pixels=0.):
    squared = np.sum(residual**2, axis=1)
    distance = np.sqrt(squared)
    if tolerance_pixels == 0.:
        return distance, squared, distance, residual
    tol = tolerance_pixels / width
    effective = np.maximum(distance - tol, 0.)
    r_hat = residual * (effective / np.maximum(distance, 1e-15))[:, None]
    return distance, squared, effective, r_hat


def cauchy_irls_weights(residual, width, cauchy_scale_pixels, tolerance_pixels=0.):
    """Compute DCA/IRLS weights w_i = 1 / (1 + (e_i / c)^2) in normalized units."""
    if width <= 0 or tolerance_pixels < 0 or cauchy_scale_pixels <= 0:
        raise ValueError('Invalid width, tolerance, or Cauchy scale.')
    residual = np.asarray(residual, float)
    if len(residual) == 0:
        return np.empty(0, float)
    _, _, effective, _ = _effective_residuals(residual, width, tolerance_pixels)
    c = float(cauchy_scale_pixels) / width
    return 1.0 / (1.0 + (effective * effective) / (c * c))


def reprojection_term(residual, width, tolerance_pixels=0., cauchy_scale_pixels=None, point_weights=None):
    if width <= 0 or tolerance_pixels < 0:
        raise ValueError('Invalid width or tolerance.')
    if cauchy_scale_pixels is not None and cauchy_scale_pixels <= 0:
        raise ValueError('Invalid Cauchy scale.')
    residual = np.asarray(residual, float)
    if residual.ndim != 2 or residual.shape[1] != 2:
        raise ValueError('Expected N x 2 residuals.')
    n = len(residual)
    if not n:
        return 0., np.zeros_like(residual), dict(observations=0, outside_tolerance=0, rms_px=0., max_px=0.)

    distance, squared, effective, r_hat = _effective_residuals(residual, width, tolerance_pixels)
    if point_weights is not None:
        w = np.asarray(point_weights, float)
        if w.shape != (n,):
            raise ValueError('Expected point_weights of shape (N,).')
        value = float(np.mean(w * effective**2))
        gradient = 2 * w[:, None] * r_hat / n
    elif cauchy_scale_pixels is not None:
        c = float(cauchy_scale_pixels) / width
        c2 = c * c
        ratio2 = (effective**2) / c2
        w = 1.0 / (1.0 + ratio2)
        value = float(np.mean(c2 * np.log1p(ratio2)))
        gradient = 2 * w[:, None] * r_hat / n
    else:
        w = None
        if tolerance_pixels == 0:
            value = float(np.mean(squared))
            gradient = 2 * residual / n
        else:
            value = float(np.mean(effective**2))
            gradient = 2 * r_hat / n

    stats = dict(
        observations=n,
        outside_tolerance=int(np.sum(distance > tolerance_pixels / width)),
        rms_px=float(np.sqrt(np.mean(squared)) * width),
        max_px=float(np.max(distance) * width),
    )
    if w is not None:
        stats.update(
            cauchy_weight_min=float(np.min(w)),
            cauchy_weight_median=float(np.median(w)),
            cauchy_effective_inliers=int(np.sum(w >= 0.5)),
        )
    return value, gradient, stats
