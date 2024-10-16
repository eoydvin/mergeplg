import numpy as np
import numpy.testing as npt

import mergeplg as mrg


def test_load_and_transform_openmrg_data():
    (
        ds_rad,
        ds_cmls,
        ds_gauges,
        ds_gauges_smhi,
    ) = mrg.io.load_and_transform_openmrg_data()

    npt.assert_almost_equal(
        ds_rad.x.data[:3], np.array([646133.41558064, 648111.35727356, 650089.36933527])
    )
    npt.assert_almost_equal(
        ds_rad.y.data[:3],
        np.array([6346085.11915342, 6348058.22706042, 6350031.7601296]),
    )

    npt.assert_almost_equal(
        ds_cmls.x.data[:3],
        np.array([678357.52489917, 677298.84120483, 677450.08175986]),
    )
    npt.assert_almost_equal(
        ds_cmls.y.data[:3],
        np.array([6399335.39433133, 6401892.33059722, 6398459.30374539]),
    )

    npt.assert_almost_equal(
        ds_gauges.x.data[:3],
        np.array([675647.15106274, 680799.29807671, 682881.40333814]),
    )
    npt.assert_almost_equal(
        ds_gauges.y.data[:3],
        np.array([6393119.07429812, 6401433.82411907, 6405152.76685568]),
    )

    npt.assert_almost_equal(ds_gauges_smhi.x.data, np.array([678243.792424]))
    npt.assert_almost_equal(ds_gauges_smhi.y.data, np.array([6400984.17275754]))
