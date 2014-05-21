import numpy as np
from codim1.core.basis_funcs import basis_from_nodes, basis_from_degree,\
                                    get_equispaced_nodes
from codim1.fast_lib import SingleFunctionBasis
from codim1.core.mesh_gen import circular_mesh

def my_assert(a, b):
    np.testing.assert_almost_equal(np.array(a), np.array(b))

def test_degree_zero_nodes():
    nodes = get_equispaced_nodes(0)
    assert(nodes[0] == 0.5)
    assert(len(nodes) == 1)

def test_degree_zero_eval():
    bf = basis_from_degree(0)
    assert(bf.n_fncs == 1)
    for x in np.linspace(0.0, 1.0, 10):
        my_assert(bf.evaluate(0, x, [x, x]), 1.0)
        deriv = bf.get_gradient_basis().evaluate(0, x, [x, x])
        my_assert(deriv, 0.0)

def test_from_degree():
    nodes = get_equispaced_nodes(2)
    assert(nodes[0] == 0.0)
    assert(nodes[1] == 0.5)
    assert(nodes[2] == 1.0)

def test_basis_nodes():
    bf = basis_from_nodes([-1.0, 0.0, 1.0])
    # Check to make sure the nodes are right
    x = [1.0, 1.0]
    my_assert(bf.evaluate(0, -1.0, x), 1.0)
    my_assert(bf.evaluate(0, 0.0, x), 0.0)
    my_assert(bf.evaluate(0, 1.0, x), 0.0)
    my_assert(bf.evaluate(1, 0.0, x), 1.0)
    my_assert(bf.evaluate(2, 1.0, x), 1.0)


def test_basis_2nd_order():
    bf = basis_from_nodes([-1.0, 0.0, 1.0])
    # Test for 2nd degree polynomials
    x_hat = 0.3
    x = [x_hat, x_hat]
    y_exact1 = 0.5 * x_hat * (x_hat - 1)
    y_exact2 = (1 - x_hat) * (1 + x_hat)
    y_exact3 = 0.5 * x_hat * (1 + x_hat)
    y_est1 = bf.evaluate(0, x_hat, x)
    y_est2 = bf.evaluate(1, x_hat, x)
    y_est3 = bf.evaluate(2, x_hat, x)
    my_assert(y_exact1, y_est1)
    my_assert(y_exact2, y_est2)
    my_assert(y_exact3, y_est3)

def test_basis_derivative():
    bf = basis_from_nodes([-1.0, 0.0, 1.0])
    x_hat = 0.3
    x = [x_hat, x_hat]
    yd_exact1 = x_hat - 0.5
    yd_exact2 = -2 * x_hat
    yd_exact3 = x_hat + 0.5
    yd_est1 = bf.get_gradient_basis().evaluate(0, x_hat, x)
    yd_est2 = bf.get_gradient_basis().evaluate(1, x_hat, x)
    yd_est3 = bf.get_gradient_basis().evaluate(2, x_hat, x)
    my_assert(yd_exact1, yd_est1)
    my_assert(yd_exact2, yd_est2)
    my_assert(yd_exact3, yd_est3)

def test_function():
    f = lambda x,d: x[d]
    F = SingleFunctionBasis(f)
    f_val = F.evaluate(0, 1.0, (0.0, 11.1))

    assert(f_val[0] == 0.0)
    assert(f_val[1] == 11.1)

def test_gradient():
    msh = circular_mesh(200, 1.0)
    bf = basis_from_nodes([0.0, 1.0])
    gradient = bf.get_gradient_basis()
    e = msh.elements[57]
    chain_rule = gradient.chain_rule(e.mapping.get_jacobian(0.0))
    value = np.array(gradient.evaluate(0, 0.5, [0.0,0.0]))
    np.testing.assert_almost_equal(chain_rule * value,
                                   -31.83229765 * np.ones(2))

if __name__ == "__main__":
    test_basis_nodes()
    test_gradient()
