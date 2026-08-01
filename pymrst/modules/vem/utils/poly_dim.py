"""Dimension of a polynomial space (MRST ``vem/utils/polyDim.m``).

The number of monomials of degree at most `k` in `dim` variables, which is how
many moments a VEM projection has to carry. Everything in the module that sizes
a local matrix asks this first.

Negative `k` returns 0 rather than raising. That is load-bearing rather than
defensive: the assembly code computes `polyDim(k - 2, ...)` to count the
interior and face moment degrees of freedom, and at `k = 1` there are none, so
the negative case *is* the first-order element.
"""

from math import comb


def poly_dim(k: int, dim: int) -> int:
    """Dimension of the polynomials of degree <= ``k`` in ``dim`` variables."""
    return 0 if k < 0 else comb(k + dim, k)


polyDim = poly_dim


if __name__ == "__main__":

    def test_degree_zero_is_the_constants() -> None:
        assert poly_dim(0, 2) == 1 and poly_dim(0, 3) == 1
        print("ok test_degree_zero_is_the_constants")

    def test_linear_in_two_dimensions_is_one_x_y() -> None:
        assert poly_dim(1, 2) == 3
        print("ok test_linear_in_two_dimensions_is_one_x_y")

    def test_quadratic_in_two_dimensions_adds_three_terms() -> None:
        assert poly_dim(2, 2) == 6            # + x^2, xy, y^2
        print("ok test_quadratic_in_two_dimensions_adds_three_terms")

    def test_linear_in_three_dimensions_is_one_x_y_z() -> None:
        assert poly_dim(1, 3) == 4
        print("ok test_linear_in_three_dimensions_is_one_x_y_z")

    def test_negative_degree_is_empty_not_an_error() -> None:
        """k-2 at k=1 is -1: the first-order element has no interior moments."""
        assert poly_dim(-1, 2) == 0
        assert poly_dim(-2, 3) == 0
        print("ok test_negative_degree_is_empty_not_an_error")

    def test_it_grows_with_both_arguments() -> None:
        assert poly_dim(2, 3) > poly_dim(1, 3) > poly_dim(0, 3)
        assert poly_dim(2, 3) > poly_dim(2, 2)
        print("ok test_it_grows_with_both_arguments")

    test_degree_zero_is_the_constants()
    test_linear_in_two_dimensions_is_one_x_y()
    test_quadratic_in_two_dimensions_adds_three_terms()
    test_linear_in_three_dimensions_is_one_x_y_z()
    test_negative_degree_is_empty_not_an_error()
    test_it_grows_with_both_arguments()
    print("\nAll tests passed!")
