from typing import Tuple, List, Union, Any, Optional, Dict, Set, Literal, Callable
import os
import sys
from pathlib import Path
import functools
from functools import partial
import time

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int, PRNGKeyArray, UInt8
import pandas as pd


############################################
### bin by sorting then taking quantiles ###
############################################


def subsample_data_if_big(seed: PRNGKeyArray, data: Array, sample_size: int = 100_000):
    """
    Sample a subset of the data uniformly without replacement if it exceeds the sample size.
    
    Args:
        seed: JAX PRNG key for random sampling.
        data: JAX array of shape (N, ...).
        sample_size: Number of samples to draw from the data.

    Returns:
        A subsampled array of shape (sample_size, ...).
    """
    N = data.shape[0]
    if N > sample_size:
        idx = jax.random.choice(seed, N, shape=(sample_size,), replace=False)
        return data[idx]
    return data


def create_quantile_bin_edges_1d(seed: PRNGKeyArray, data: Float[Array, "N"], num_bins: int = 256, 
                                 bin_construct_sample_cnt : int = 100_000):
    """
    Compute quantile bin edges from a subsample of `data`.

    Args:
        seed: JAX PRNG key for random sampling.
        data: 1D array of shape (N,) containing feature values.
        num_bins: Number of bins to produce.
        subsample: Maximum number of values to use for computing quantiles.

    Returns:
        An array of bin edges of shape (num_bins,).
    """
    # Compute quantile boundaries
    data = subsample_data_if_big(seed, data, bin_construct_sample_cnt)
    bin_edges = jnp.quantile(data, jnp.linspace(0.0, 1.0, num_bins + 1)[1:-1])
    return bin_edges


def create_quantile_bin_edges(seed: PRNGKeyArray, X: Float[Array, "N D"], num_bins: int = 256, 
                              bin_construct_sample_cnt: int = 100_000,):
    """
    Vectorized creation of quantile bins across features.

    Args:
        seed: JAX PRNG key.
        X: (N, D) array.
        num_bins: Number of bins.
        bin_construct_sample_cnt: Max subsample size per feature.

    Returns:
        (D, num_bins) bin edge matrix.
    """
    N, D = X.shape
    seeds = jax.random.split(seed, D)

    # vmap over features (columns)
    bin_edges = jax.vmap(
        lambda s, x: create_quantile_bin_edges_1d(s, x, num_bins, bin_construct_sample_cnt),
        in_axes=(0, 1), out_axes=0
    )(seeds, X)
    return bin_edges  # (D, num_bins)


def map_cont_to_bins(X: Float[Array, "N D"], bin_edges: Float[Array, "D num_bins"]):
    """
    Maps continuous input X to bin indices using precomputed bin edges.

    Args:
        X: (N, D) array of float32 input features.
        bin_edges: (D, num_bins) array of bin edges per feature.

    Returns:
        X_binned: (N, D) uint8 array of bin indices.
    """

    # vmap over each feature dimension
    X_binned = jax.vmap(
        lambda e, x: jnp.searchsorted(e, x, side='right').astype(jnp.uint8),
        in_axes=(0, 1), out_axes=1   
    )(bin_edges, X)
    return X_binned  # (N, D) uint8 array of bin indices