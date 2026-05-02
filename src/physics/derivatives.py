"""
src/physics/derivatives.py

Core autograd utilities for Physics-Informed Neural Networks.

These functions wrap PyTorch's ``torch.autograd.grad`` to compute first- and
second-order partial derivatives of a scalar or batched output tensor with
respect to an input tensor.  They are the building blocks used later by the
PDE residual loss (e.g. Heat Equation, Navier-Stokes).
"""

import torch
from torch import Tensor


def compute_gradient(output: Tensor, input_tensor: Tensor) -> Tensor:
    """Compute the first-order partial derivative ∂output/∂input_tensor.

    Parameters
    ----------
    output:
        The scalar (or summed) output of a computation graph.  If *output*
        has more than one element, it is summed automatically so that
        ``torch.autograd.grad`` receives a scalar loss-like value.
    input_tensor:
        The leaf tensor with respect to which the derivative is computed.
        Must have been created with ``requires_grad=True``.

    Returns
    -------
    Tensor
        Gradient tensor with the same shape as *input_tensor*.
    """
    if output.numel() > 1:
        output = output.sum()

    (grad,) = torch.autograd.grad(
        outputs=output,
        inputs=input_tensor,
        create_graph=True,   # keep graph so higher-order derivatives work
        retain_graph=True,
    )
    return grad


def compute_second_derivative(output: Tensor, input_tensor: Tensor) -> Tensor:
    """Compute the second-order partial derivative ∂²output/∂input_tensor².

    Internally calls :func:`compute_gradient` twice:
    first to obtain ∂output/∂x, then to differentiate that result again.

    Parameters
    ----------
    output:
        The scalar (or summed) output of a computation graph.
    input_tensor:
        The leaf tensor (``requires_grad=True``) to differentiate with respect
        to.

    Returns
    -------
    Tensor
        Second-derivative tensor with the same shape as *input_tensor*.
    """
    first_grad = compute_gradient(output, input_tensor)

    # If the first gradient has no grad_fn it is a constant with respect to
    # input_tensor (e.g. ∂(ax+b)/∂x = a, a constant), so the second
    # derivative is identically zero.
    if first_grad.grad_fn is None:
        return torch.zeros_like(first_grad)

    second_grad = compute_gradient(first_grad, input_tensor)
    return second_grad
