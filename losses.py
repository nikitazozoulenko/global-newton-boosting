from typing import Tuple, List, Union, Any, Optional, Dict, Set, Literal, Callable
from functools import partial
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.lax as lax
from jaxtyping import Array, Float, UInt8, Int, PRNGKeyArray


EPS = 1e-5


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class LossGradHess:
    grad: Float[Array, "N p"]
    hess: Float[Array, "N p"]
    loss: Callable[float, float] | None = None # Optional loss function for reference
    
    
###################################################################################
##### Each function returns a LossGradHess, with grad and (diagonal) hessians #####
###################################################################################


def mse_grad_hess(y: Float[Array, "N"], y_hat: Float[Array, "N 1"]) -> LossGradHess:
    """
    Compute gradients and hessians for Mean Squared Error (MSE) loss.

    Args:
        y: True targets of shape (N).
        y_hat: Predicted values of shape (N, 1).

    Returns:
        A LossGradHess object containing gradients and hessians.
    """
    grad = y_hat - y.reshape(y_hat.shape)  # Gradient of MSE
    hess = jnp.ones_like(y_hat)  # Hessian of MSE is constant
    loss_lambda = lambda y_true, y_pred: jnp.mean((y_true - y_pred) ** 2)
    return LossGradHess(grad=grad, hess=hess, loss=loss_lambda)



def bce_grad_hess(y: Float[Array, "N"], y_hat: Float[Array, "N 1"]) -> LossGradHess:
    """
    Compute gradients and hessians for Binary Cross-Entropy (BCE) loss.

    Args:
        y: True 0/1 binary labels of shape (N).
        y_hat: Predicted logits of shape (N, 1).

    Returns:
        A LossGradHess object containing gradients and hessians.
    """
    p = lax.logistic(y_hat)  # Sigmoid to get probabilities
    grad = p - y.reshape(p.shape)
    hess = (p * (1 - p)).clip(min=EPS)
    loss_lambda = lambda y_true, y_pred: -jnp.mean(y_true * jnp.log(lax.logistic(y_pred)) + (1 - y_true) * jnp.log(1 - lax.logistic(y_pred)))
    return LossGradHess(grad=grad, hess=hess, loss=loss_lambda)



def cce_grad_hess(y: Float[Array, "N"], y_hat: Float[Array, "N p"]) -> LossGradHess:
    """
    Compute gradients and hessians for Categorical Cross-Entropy (CCE) loss.

    Args:
        y: True 0,...,p-1 labels of shape (N).
        y_hat: Predicted logits of shape (N, p).

    Returns:
        A LossGradHess object containing gradients and hessians.
    """
    p = jax.nn.softmax(y_hat, axis=1)  # Softmax to get probabilities
    grad = p - jax.nn.one_hot(y, y_hat.size(1))  # Gradient of CCE
    hess = (p * (1 - p)).clip(min=EPS)  # Hessian of CCE (diagonal approximation)
    loss_lambda = lambda y_true, y_pred: -jnp.mean(jnp.sum(jax.nn.one_hot(y_true, y_pred.shape[1]) * jnp.log(jax.nn.softmax(y_pred, axis=1)), axis=1))
    return LossGradHess(grad=grad, hess=hess, loss=loss_lambda)


def charbonnier_grad_hess(
    y: Float[Array, "N"], 
    y_hat: Float[Array, "N 1"], 
    epsilon: float = 1
) -> LossGradHess:
    """
    Compute gradients and hessians for Charbonnier loss.

    Args:
        y: True targets of shape (N).
        y_hat: Predicted values of shape (N, 1).
        epsilon: Small constant for numerical stability.
    Returns:
        A LossGradHess object containing gradients and hessians.
    """
    diff = y_hat - y.reshape(y_hat.shape)
    abs_diff = jnp.sqrt(diff ** 2 + epsilon ** 2)
    grad = diff / abs_diff  # Gradient of Charbonnier loss
    hess = (epsilon ** 2) / (abs_diff ** 3).clip(min=EPS)  # Hessian of Charbonnier loss
    loss_lambda = lambda y_true, y_pred: jnp.mean(jnp.sqrt((y_pred - y_true) ** 2 + epsilon ** 2)-epsilon)
    return LossGradHess(grad=grad, hess=hess, loss=loss_lambda)