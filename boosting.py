
from typing import Tuple, List, Union, Any, Optional, Dict, Set, Literal, Callable
import os
import sys
from pathlib import Path
import functools
from functools import partial
import time
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, Int, UInt8, PRNGKeyArray
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from binning import create_quantile_bin_edges, map_cont_to_bins
from losses import LossGradHess, mse_grad_hess, bce_grad_hess, cce_grad_hess, charbonnier_grad_hess


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DTEnsemble: #Decision Tree Ensemble
    nodewise_dims: Int[Array, "n_estimators  max_depth  max_num_nodes"]
    nodewise_edges: UInt8[Array, "n_estimators  max_depth  max_num_nodes"]
    leaf_values: Float[Array, "n_estimators  2**max_depth  p"]
    history: Float #: Float[Array] # TODO



#################################################
#### Shared Decision Tree Boosting functions ####
#################################################



def update_sample_to_node_routing_1_depth(
        sample_to_node: Int[Array, "N"],
        X_bin_idxs: UInt8[Array, "N D"],
        nodewise_dims: Int[Array, "max_num_nodes"], 
        nodewise_bins: UInt8[Array, "max_num_nodes"],
    ):
    "Updates sample_to_node indices based on the current node splits."
    N = X_bin_idxs.shape[0]
    dim_idxs = nodewise_dims[sample_to_node]
    edge_idxs = nodewise_bins[sample_to_node]
    sample_bin = X_bin_idxs[jnp.arange(N), dim_idxs]
    go_right = (sample_bin > edge_idxs)
    sample_to_node = 2 * sample_to_node + go_right
    return sample_to_node
    


def get_starting_prediction(initial_yhat: Float | Float[Array, "p"], N:int):
    yhat = jnp.tile(initial_yhat, (N,1))
    return yhat



def build_decision_tree_layer(
    depth: int,
    max_depth: int,
    sample_to_node: Int[Array, "N"],
    X_bin_idxs: UInt8[Array, "N D"],
    gh: Float[Array, "N p 2"],
    num_bins: int, 
):
    """Builds a single layer of the tree, updating sample_to_node indices and node split info."""
    max_num_nodes = 2**(max_depth-1)
    max_num_segments = max_num_nodes * num_bins
    p = gh.shape[1] # p is the number of classes/outputs

    def per_feature_gain(bin_idxs: UInt8[Array, "N"]):
        """Calculates the best split for each node for a single feature."""
        # 1. calculate histograms of gradients and hessians
        segment_ids = sample_to_node * num_bins + bin_idxs  # shape (N,)
        gh_hist = jax.ops.segment_sum(gh, segment_ids, max_num_segments) # shape (max_num_segments, p, 2)
        gh_cumsum = jnp.cumsum(gh_hist.reshape((max_num_nodes, num_bins, p, 2)), axis=1) # shape (max_num_nodes, num_bins, p, 2)
        g_cumsum = gh_cumsum[..., 0]  # shape (max_num_nodes, num_bins, p)
        h_cumsum = gh_cumsum[..., 1]  # shape (max_num_nodes, num_bins, p)
        total_g = g_cumsum[:, -1, :]  # shape (max_num_nodes, p)
        total_h = h_cumsum[:, -1, :]  # shape (max_num_nodes, p)

        # 2. Calculate gain. The formula is applied element-wise over the 'p' dimension,
        #    and then summed to get a single scalar gain value for each split.
        eps = 1e-7
        left_gain = jnp.sum(g_cumsum**2 / (h_cumsum + eps), axis=-1) # shape (max_num_nodes, num_bins)
        right_g = total_g[:, None, :] - g_cumsum
        right_h = total_h[:, None, :] - h_cumsum
        right_gain = jnp.sum(right_g**2 / (right_h + eps), axis=-1) # shape (max_num_nodes, num_bins)
        dont_split_gain = jnp.sum(total_g**2 / (total_h + eps), axis=-1) # shape (max_num_nodes,)
        gain = left_gain + right_gain - dont_split_gain[:, None] # shape (max_num_nodes, num_bins)
        
        # 3. Get the best gain value and its corresponding bin index for each node.
        best_gain_per_node = jnp.max(gain, axis=1) # shape (max_num_nodes,)
        best_bins = jnp.argmax(gain, axis=1) # shape (max_num_nodes,)
        return best_bins, best_gain_per_node
    
    # Map over all features to compute gains, with shapes (D, max_num_nodes)
    best_bins, best_gain_per_node = lax.map(per_feature_gain, X_bin_idxs.T)
    
    # Find the best feature (dim) and overall best gain for each node, and bin index
    nodewise_gains = jnp.max(best_gain_per_node, axis=0)      # shape (max_num_nodes,)
    nodewise_dims = jnp.argmax(best_gain_per_node, axis=0)    # shape (max_num_nodes,)
    nodewise_bins = best_bins[nodewise_dims, jnp.arange(best_bins.shape[1])]  # shape (max_num_nodes,)
    
    # Update sample routing based on the best splits found.
    sample_to_node = update_sample_to_node_routing_1_depth(
        sample_to_node, X_bin_idxs, nodewise_dims, nodewise_bins
    )
    return sample_to_node, nodewise_dims, nodewise_bins, nodewise_gains



def get_grad_hess(
    y: Float[Array, "N"],
    y_hat: Float[Array, "N p"],
    grad_hess_fn: Callable[[Float[Array, "N"], Float[Array, "N p"]], LossGradHess],
    boosting_method: Literal["newton", "gradient", "gradient_momentum", "newton_momentum"],
):
    grad_hess_obj = grad_hess_fn(y, y_hat)
    grad = grad_hess_obj.grad
    hess = grad_hess_obj.hess
    
    if "gradient" in boosting_method:
        hess = jnp.ones_like(grad)
    return grad, hess



def hilbert_norm(g: Float[Array, "N p"]) -> float:
    innerprod_g = jnp.sum(g**2, axis=1)  # shape (N,)
    g_norm = jnp.sqrt(jnp.mean(innerprod_g))  # float
    return g_norm


def T_norm(x: Float[Array, "N p"], h_reg: Float[Array, "N p"]) -> float:
    return hilbert_norm( x * jnp.sqrt(h_reg))


history_order = {
    "g_norm": 0,
    "lambda_t": 1,
    "hilbert_norm_f_weak": 2,
    "hilbert_norm_f_exact": 3,
    "t_norm_weak": 4,
    "t_norm_exact": 5,
    "hilbert_cosangle": 6,
    "t_cosangle": 7,
    "weak_g_edge": 8,
    "1 - weak_g_edge^2": 9,
}

def fit_decision_tree(
    grad: Float[Array, "N p"],
    hess: Float[Array, "N p"],
    X_bin_idxs: UInt8[Array, "N D"],
    max_depth: int = 6,
    l2_reg: float = 1.0,
    grad_regularized_l2: float = 0.0,
    num_bins: int = 255,
    include_history: bool = False,
):
    """
    Fits a single decision tree and returns its structure and predictions.

    Returns:
        pred: The predictions of this single tree for each sample, shape (N, p).
        node_dims: The feature indices for each split, shape (max_depth, max_num_nodes).
        node_edges: The bin thresholds for each split, shape (max_depth, max_num_nodes).
        leaf_values: The output vector for each leaf node, shape (2**max_depth, p).
    """
    # initialize carry
    N, D = X_bin_idxs.shape
    sample_to_node = jnp.zeros((N,), dtype=jnp.int32)
    
    # Apply gradient regularization if specified (default no change since grad_regularized_l2=0)
    g_norm = hilbert_norm(grad)  # float
    lambda_t = jnp.sqrt(grad_regularized_l2 * g_norm)   # float
    _l2_reg = l2_reg + lambda_t
    
    # stack g with h_reg
    hess_reg = hess + _l2_reg # NOTE: this differs slightly from XGBoost's l2 regularization. Ours depends on the number of leafs in the split
    gh = jnp.stack([grad, hess_reg], axis=-1)  # shape (N, p, 2)

    # loop over depth [0, max_depth) to build tree
    def scan_body_build_one_decision_tree_layer(carry, depth):
        sample_to_node = carry
        sample_to_node, nodewise_dims, nodewise_bins, nodewise_gains = build_decision_tree_layer(
            depth=depth,
            max_depth=max_depth,
            sample_to_node=sample_to_node,
            X_bin_idxs=X_bin_idxs,
            gh=gh,
            num_bins=num_bins,
        )
        return sample_to_node, (nodewise_dims, nodewise_bins, nodewise_gains)
    

    # After the scan, sample_to_node maps each of N samples to a leaf index [0, 2**max_depth - 1].
    # The other returned values describe the tree's split structure.
    sample_to_node, (node_dims, node_edges, node_gains) = lax.scan(
        scan_body_build_one_decision_tree_layer,
        sample_to_node,
        jnp.arange(max_depth),
    )
    
    # NOTE Currently does not include any pruning.
    eps = 1e-7
    leaf_gh = jax.ops.segment_sum(gh, sample_to_node, num_segments=2**max_depth) # shape (2**max_depth, p, 2).
    leaf_grad = leaf_gh[..., 0] # shape (2**max_depth, p)
    leaf_hess = leaf_gh[..., 1] # shape (2**max_depth, p)
    leaf_values = -leaf_grad / (leaf_hess + eps) # shape (2**max_depth, p)
    
    # Compute this tree's prediction for each sample by looking up its leaf value.
    pred = leaf_values[sample_to_node] # shape (N, p)

    # Save useful metrics to history if specified
    history = None
    if include_history:
        g_norm = hilbert_norm(grad)
        f_weak = pred
        f_exact = - grad /(hess_reg)
        hilbert_norm_f_weak = hilbert_norm(f_weak)
        hilbert_norm_f_exact = hilbert_norm(f_exact)
        t_norm_weak = T_norm(f_weak, hess_reg)
        t_norm_exact = T_norm(f_exact, hess_reg)
        hilbert_cosangle = jnp.mean(f_weak * f_exact) / (hilbert_norm_f_weak * hilbert_norm_f_exact)
        t_cosangle = jnp.mean(f_weak * f_exact * (hess_reg)) / (t_norm_weak * t_norm_exact)
        g_weak = - (hess_reg) * f_weak
        one_minus_edge_sq = hilbert_norm(g_weak-grad)**2 / hilbert_norm(grad)**2
        weak_g_edge = jnp.sqrt(jnp.maximum(0.0, 1.0 - one_minus_edge_sq))
        history = (g_norm, lambda_t, hilbert_norm_f_weak, hilbert_norm_f_exact, 
                   t_norm_weak, t_norm_exact, hilbert_cosangle, t_cosangle,
                   weak_g_edge, one_minus_edge_sq)
    
    return pred, node_dims, node_edges, leaf_values, history



def decision_tree_predict(
    X_bin_idxs: UInt8[Array, "N D"],
    tree_dims: Int[Array, "max_depth  max_num_nodes"],
    tree_edges: UInt8[Array, "max_depth  max_num_nodes"],
    leaf_values: Float[Array, "2**max_depth  p"],
    ) -> Float[Array, "N p"]:
    """Returns the predictions of a single decision tree for batch input X."""
    
    def route_one_depth(sample_to_node, depth_data):
        nodewise_dims, nodewise_bins = depth_data
        return update_sample_to_node_routing_1_depth(
            sample_to_node, X_bin_idxs, nodewise_dims, nodewise_bins
        ), None

    N = X_bin_idxs.shape[0]
    sample_to_node = jnp.zeros((N,), dtype=jnp.int32)
    sample_to_node, _ = lax.scan(route_one_depth, sample_to_node, (tree_dims, tree_edges))
    return leaf_values[sample_to_node]




###################################################
#### Newton Boosting with hessian and gradient ####
###################################################



def get_decision_tree_ensemble_output(
    starting_yhat: Float[Array, "N p"],
    X_bin_idxs: UInt8[Array, "N D"],
    dt_ensemble: DTEnsemble,
) -> Float[Array, "N p"]:
    """
    Predict logit for input X using the trained ensemble.
    """
    N = X_bin_idxs.shape[0]
    def ensemble_predict_body(carry, tree_data):
        yhat = carry
        depth_nodewise_dims, depth_nodewise_bins, leaf_values = tree_data

        yhat += decision_tree_predict(
            X_bin_idxs,
            depth_nodewise_dims,
            depth_nodewise_bins,
            leaf_values
        )
        return yhat, None
        
    y_hat, _ = lax.scan(
        ensemble_predict_body,
        starting_yhat,
        (dt_ensemble.nodewise_dims, 
         dt_ensemble.nodewise_edges, 
         dt_ensemble.leaf_values)
    )
    return (y_hat,)

    
    
def newton_boosting(
    grad_hess_fn: Callable[[Float[Array, "N"], Float[Array, "N p"]], LossGradHess], # (y, y_hat) -> LossGradHess
    initial_yhat: Float[Array, "N p"],
    boosting_method: Literal["newton", "gradient", "gradient_momentum", "newton_momentum"],
    X_bin_idxs: UInt8[Array, "N D"],
    y: Float[Array, "N"],
    n_estimators: int = 100,
    lr: float = 0.1,
    max_depth: int = 6,
    l2_reg: float = 1.0,
    grad_regularized_l2: float = 0.0,
    num_bins: int = 255,
    include_history: bool = False,
    ):
    """performs newton boosting given grad and hess function"""
    def scan_body_boosting(y_hat, t):
        grad, hess = get_grad_hess(y, y_hat, grad_hess_fn, boosting_method)
        
        (tree_pred, 
         depth_nodewise_dims, 
         depth_nodewise_edges, 
         leaf_values,
         history) = fit_decision_tree(
            grad=grad,
            hess=hess,
            X_bin_idxs=X_bin_idxs,
            max_depth=max_depth,
            l2_reg=l2_reg,
            grad_regularized_l2=grad_regularized_l2,
            num_bins=num_bins,
            include_history=include_history,
        )
        #update ensemble pred with new tree
        y_hat += lr * tree_pred
        return y_hat, (depth_nodewise_dims, depth_nodewise_edges, lr*leaf_values, history)

    y_hat = get_starting_prediction(initial_yhat, y.shape[0])
    y_hat, (ensemble_depth_nodewise_dims, 
            ensemble_depth_nodewise_edges, 
            ensemble_leaf_values,
            history) = jax.lax.scan(
        scan_body_boosting,
        y_hat,
        jnp.arange(n_estimators)
    )

    return y_hat, DTEnsemble(
        nodewise_dims=ensemble_depth_nodewise_dims,
        nodewise_edges=ensemble_depth_nodewise_edges,
        leaf_values=ensemble_leaf_values,
        history=history
    )



##########################
#### Classifier Class ####
##########################



class GBDT(BaseEstimator):
    def __init__(self,
                 loss: str = "cls:bce", # ["cls:bce", "cls:cce", "reg:mse", "reg:charbonnier"]
                 boosting_method: str = "newton", # ["gradient", "newton"]
                 num_bins: int = 255,
                 n_estimators: int = 100,
                 lr: float = 0.1,
                 max_depth: int = 6,
                 n_classes: int|None = None,
                 l2_reg: float = 1.0,
                 grad_regularized_l2: float = 0.0,
                 verbose: int = 0,
                 seed: int = 42,
                 include_history: bool = True,
                 charbonnier_epsilon: float = 1.0,
                 ):
        self.loss= loss
        self.boosting_method = boosting_method
        self.num_bins = num_bins
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.l2_reg = l2_reg
        self.grad_regularized_l2 = grad_regularized_l2
        self.verbose = verbose
        self.seed = seed
        self.include_history = include_history
        self.tree_ensemble: DTEnsemble | None = None
        self.initial_yhat: Float | Float[Array, "p"] | None = None
        
        if num_bins < 2 or num_bins > 255:
            raise ValueError("num_bins must be in [2, 255]")
        
        if self.loss == "reg:mse":
            self.grad_hess_fn = mse_grad_hess
            self.initial_yhat_fn = lambda y: jnp.mean(y)
            self.raw_to_pred_fn = lambda yhat: yhat  # Identity for regression
        elif self.loss == "reg:charbonnier":
            self.grad_hess_fn = partial(charbonnier_grad_hess, epsilon=charbonnier_epsilon)
            self.initial_yhat_fn = lambda y: jnp.mean(y)
            self.raw_to_pred_fn = lambda yhat: yhat
        elif self.loss == "cls:bce":
            self.grad_hess_fn = bce_grad_hess
            def initial_yhat_bce(y): y_mean = jnp.mean(y); return jnp.log(y_mean) - jnp.log(1 - y_mean)
            self.initial_yhat_fn = initial_yhat_bce
            self.raw_to_pred_fn = lambda yhat: (yhat > 0.0).astype(jnp.int32)
            def raw_to_proba_fn(yhat): p = lax.logistic(yhat); return jnp.concat([1-p, p], axis=1)
            self.raw_to_proba_fn = raw_to_proba_fn
        elif self.loss == "cls:cce":
            self.grad_hess_fn = cce_grad_hess
            self.initial_yhat_fn = lambda y: jnp.mean(jax.nn.one_hot(y, n_classes), axis=0)
            self.raw_to_pred_fn = lambda yhat: jnp.argmax(yhat, axis=1) 
            self.raw_to_proba_fn = lambda yhat: jax.nn.softmax(yhat, axis=1)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")
        

    def _fit_predict_raw(self, X: Float[Array, "N D"], y: Float[Array, "N"]):
        """
        Fit the model to the data (X, y) and return raw predictions 
        (i.e. log-odds for classification).
        
        Returns:
            y_hat: Float[Array, "N p"]
        """
        # binning
        seed = jax.random.PRNGKey(self.seed)
        self.edges = create_quantile_bin_edges(seed, X, num_bins=self.num_bins)
        X_bin_idxs = map_cont_to_bins(X, self.edges)
        
        # fit boosting model
        self.initial_yhat = self.initial_yhat_fn(y)
        
        (y_hat, self.tree_ensemble) = newton_boosting(
            grad_hess_fn=self.grad_hess_fn,
            initial_yhat=self.initial_yhat,
            boosting_method=self.boosting_method,
            X_bin_idxs=X_bin_idxs,
            y=y,
            n_estimators=self.n_estimators,
            lr=self.lr,
            max_depth=self.max_depth,
            l2_reg=self.l2_reg,
            grad_regularized_l2=self.grad_regularized_l2,
            num_bins=self.num_bins,
            include_history=self.include_history,
        )
        return y_hat
    
    
    def fit(self, X: Float[Array, "N D"], y: Float[Array, "N"]):
        self._fit_predict_raw(X, y)
        return self
    
    
    def fit_predict(self, X: Float[Array, "N D"], y: Float[Array, "N"], raw=False):
        y_hat = self._fit_predict_raw(X, y)
        if raw:
            return y_hat
        else:
            return self.raw_to_pred_fn(y_hat)


    def fit_predict_proba(self, X: Float[Array, "N D"], y: Float[Array, "N"]):
        y_hat = self._fit_predict_raw(X, y)
        return self.raw_to_proba_fn(y_hat)
    
    
    def _predict_raw(self, X: Float[Array, "N D"]) -> Float[Array, "N p"]:
        if not self.tree_ensemble:
            raise ValueError("Model is not fitted yet. Please call 'fit' before 'predict'.")
        # Map X to bins using stored edges
        X_bin_idxs = map_cont_to_bins(X, self.edges)
        initial = get_starting_prediction(self.initial_yhat, X.shape[0])
        y_hat = get_decision_tree_ensemble_output(
            initial,
            X_bin_idxs,
            self.tree_ensemble,
        )[0]
        return y_hat


    def predict_proba(self, X: Float[Array, "N D"]) -> Array:
        if self.loss not in ["cls:bce", "cls:cce"]:
            raise ValueError("predict_proba is only available for classification tasks with 'cls:bce' or 'cls:cce' loss.")
        
        y_hat = self._predict_raw(X)
        return self.raw_to_proba_fn(y_hat)
    

    def predict(self, X: Float[Array, "N D"], raw=False) -> Array:
        y_hat = self._predict_raw(X)
        if raw:
            return y_hat
        else:
            return self.raw_to_pred_fn(y_hat)