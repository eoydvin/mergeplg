import numpy as np
import pytest

from mergeplg.radolan import idw


# fmt: off
def test_idw_standard_method():
    # simple example in 1D
    x = np.array([[0, ], [1, ]])
    y = np.array([1, 0])
    interpolator = idw.Invdisttree(x)

    np.testing.assert_almost_equal(
        np.array([1. , 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.]),
        interpolator(q=np.linspace([0, ], [1, ], 11), z=y, p=1)
    )
    np.testing.assert_almost_equal(
        np.array([1. , 0.98780488, 0.94117647, 0.84482759, 0.69230769, 0.5 ,
                  0.30769231, 0.15517241, 0.05882353, 0.01219512, 0.]),
        interpolator(q=np.linspace([0, ], [1, ], 11), z=y, p=2)
    )

    # some examples in 2D
    x = np.array([[1, -1], [1, 1], [-1, -1], [-1, 1]])
    y = np.array([1, 0, 0, 1])
    interpolator = idw.Invdisttree(x)

    # test with increasing number of points. We do this because there
    # was a bug if only 1 to 3 or so interpolation points were supplied.
    xi = np.array([[0, 0], [0.5, 0.5], [1, 0], [2, 2], [1, 1.1], [0.2, 1], [1, 0.3]])
    for i in range(len(xi)):
        np.testing.assert_almost_equal(
            interpolator(q=xi[:i + 1], z=y, idw_method='standard', p=2),
            np.array(
                [0.5, 0.26470588, 0.5, 0.26470588, 0.00473317, 0.34256927, 0.26870145]
                )[:i + 1]
        )

    # test with max_distance
    xi = np.array([[-0.5, 1], [-0.4, 1], [0.5, 1], [0.4, 1], [1, 0.4]])
    np.testing.assert_almost_equal(
        interpolator(q=xi, z=y, idw_method='standard', p=2, max_distance=1.5),
        np.array([1. , 0.84482759, 0. , 0.15517241, 0.15517241])
    )
    xi = np.array([[-0.5, 1], [-0.4, 1], [0.5, 1], [0.4, 1], [1, 0.4], [0, 0]])
    np.testing.assert_almost_equal(
        interpolator(q=xi, z=y, idw_method='standard', p=2, max_distance=1.3),
        np.array([1. , 1., 0., 0., 0., np.nan])
    )


def test_idw_radolan_method():
    # simple example in 1D
    x = np.array([[0, ], [1, ]])
    y = np.array([1, 0])
    interpolator = idw.Invdisttree(x)

    np.testing.assert_almost_equal(
        np.array([1.00000000e+00, 9.99988556e-01, 9.99771167e-01, 9.96169281e-01,
                  9.41379310e-01, 5.00000000e-01, 5.86206897e-02, 3.83071872e-03,
                  2.28832952e-04, 1.14440045e-05, 0.00000000e+00]),
        interpolator(q=np.linspace([0, ], [1, ], 11),
                     z=y, idw_method='radolan', max_distance=1)
    )

    # example in 2D
    x = np.array([[1, -1], [1, 1], [-1, -1], [-1, 1]])
    y = np.array([1, 0, 0, 1])
    interpolator = idw.Invdisttree(x)

    # test with max_distance
    xi = np.array([[-0.5, 1], [-0.4, 1], [0.5, 1], [0.4, 1], [1, 0.4]])
    np.testing.assert_almost_equal(
        interpolator(q=xi, z=y, idw_method='radolan', p=2, max_distance=1.5),
        np.array([1.00000000e+00, 9.99387581e-01,
                  0.00000000e+00, 6.12418902e-04, 6.12418902e-04])
    )
    xi = np.array([[-0.5, 1], [-0.4, 1], [0.5, 1], [0.4, 1], [1, 0.4], [0, 0]])
    np.testing.assert_almost_equal(
        interpolator(q=xi, z=y, idw_method='radolan', p=2, max_distance=1.3),
        np.array([1. , 1., 0., 0., 0., np.nan])
    )


def test_idw_raise():
    x = np.array([[0, ], [1, ]])
    y = np.array([1, 0])
    interpolator = idw.Invdisttree(x)

    # fail when nnear is not
    with pytest.raises(ValueError, match="`nnear` must be greater than 1"):
        interpolator(q=np.linspace([0, ], [1, ], 11), z=y, nnear=1)

    with pytest.raises(ValueError, match="IDW method foobar not supported"):
        interpolator(q=np.linspace([0, ], [1, ], 11), z=y, idw_method='foobar')

    with pytest.raises(ValueError, match='`q` must be at least have 2 dimension'):
        interpolator(q=np.linspace(0, 1, 11), z=y)

# fmt: on
