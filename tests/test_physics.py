"""
tests/test_physics.py

TDD – Red phase: unit tests for the partial-derivative calculator in
src/physics/derivatives.py.

These tests are written BEFORE the implementation so that running them
initially produces failures ("Red").  Once src/physics/derivatives.py is
implemented they must all pass ("Green").

Determinism is guaranteed by fixing PyTorch and NumPy seeds at the top of
every test that generates random tensors, as required by the engineering
guidelines in CLAUDE.MD.
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Helper: import the module under test
# ---------------------------------------------------------------------------

from src.physics.derivatives import compute_gradient, compute_second_derivative


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def fix_seed():
    """Fix random seeds before every test for full reproducibility."""
    torch.manual_seed(42)
    yield


# ---------------------------------------------------------------------------
# Tests for compute_gradient
# ---------------------------------------------------------------------------

class TestComputeGradient:
    """Verify that compute_gradient returns ∂output/∂input correctly."""

    def test_gradient_of_square_is_two_x(self):
        """∂(x²)/∂x = 2x at x = 3.0 → expected 6.0."""
        x = torch.tensor([3.0], requires_grad=True)
        u = x ** 2
        grad = compute_gradient(u, x)
        assert torch.isclose(grad, torch.tensor([6.0])), (
            f"Expected 6.0, got {grad.item()}"
        )

    def test_gradient_of_linear_is_constant(self):
        """∂(5x)/∂x = 5 for any x."""
        x = torch.tensor([7.0], requires_grad=True)
        u = 5.0 * x
        grad = compute_gradient(u, x)
        assert torch.isclose(grad, torch.tensor([5.0])), (
            f"Expected 5.0, got {grad.item()}"
        )

    def test_gradient_of_sine(self):
        """∂(sin(x))/∂x = cos(x); check at x = π/4."""
        x = torch.tensor([torch.pi / 4], requires_grad=True)
        u = torch.sin(x)
        grad = compute_gradient(u, x)
        expected = torch.cos(x.detach())
        assert torch.isclose(grad, expected, atol=1e-6), (
            f"Expected {expected.item():.6f}, got {grad.item():.6f}"
        )

    def test_gradient_wrt_time(self):
        """∂(x·t)/∂t = x; spatial variable x treated as constant."""
        torch.manual_seed(42)
        x_val = torch.tensor([2.0])
        t = torch.tensor([5.0], requires_grad=True)
        u = x_val * t          # u = x·t, ∂u/∂t = x = 2.0
        grad = compute_gradient(u, t)
        assert torch.isclose(grad, torch.tensor([2.0])), (
            f"Expected 2.0, got {grad.item()}"
        )

    def test_gradient_shape_matches_input(self):
        """Gradient tensor must have the same shape as the input tensor."""
        torch.manual_seed(42)
        x = torch.linspace(0.0, 1.0, steps=10, requires_grad=True)
        u = x ** 3
        grad = compute_gradient(u.sum(), x)
        assert grad.shape == x.shape, (
            f"Shape mismatch: expected {x.shape}, got {grad.shape}"
        )


# ---------------------------------------------------------------------------
# Tests for compute_second_derivative
# ---------------------------------------------------------------------------

class TestComputeSecondDerivative:
    """Verify ∂²output/∂input² using compute_second_derivative."""

    def test_second_derivative_of_square_is_two(self):
        """∂²(x²)/∂x² = 2 (constant)."""
        x = torch.tensor([5.0], requires_grad=True)
        u = x ** 2
        d2u = compute_second_derivative(u, x)
        assert torch.isclose(d2u, torch.tensor([2.0])), (
            f"Expected 2.0, got {d2u.item()}"
        )

    def test_second_derivative_of_cubic(self):
        """∂²(x³)/∂x² = 6x; at x = 2.0 → 12.0."""
        x = torch.tensor([2.0], requires_grad=True)
        u = x ** 3
        d2u = compute_second_derivative(u, x)
        assert torch.isclose(d2u, torch.tensor([12.0])), (
            f"Expected 12.0, got {d2u.item()}"
        )

    def test_second_derivative_of_sine(self):
        """∂²(sin(x))/∂x² = -sin(x); check at x = π/3."""
        x = torch.tensor([torch.pi / 3], requires_grad=True)
        u = torch.sin(x)
        d2u = compute_second_derivative(u, x)
        expected = -torch.sin(x.detach())
        assert torch.isclose(d2u, expected, atol=1e-5), (
            f"Expected {expected.item():.6f}, got {d2u.item():.6f}"
        )

    def test_second_derivative_of_linear_is_zero(self):
        """∂²(ax + b)/∂x² = 0."""
        x = torch.tensor([3.0], requires_grad=True)
        u = 4.0 * x + 7.0
        d2u = compute_second_derivative(u, x)
        assert torch.isclose(d2u, torch.tensor([0.0]), atol=1e-6), (
            f"Expected 0.0, got {d2u.item()}"
        )
