import numpy as np
import numpy.testing as npt

import mergeplg as mrg


def test_transform_openmrg_data_for_old_radolan_code():
    (
        ds_rad,
        ds_cmls,
        ds_gauges,
        ds_gauges_smhi,
    ) = mrg.io.load_and_transform_openmrg_data()
    df_cmls = mrg.radolan.io.transform_openmrg_data_for_old_radolan_code(ds_cmls)

    assert df_cmls.sensor_type.iloc[0] == "cml_ericsson"

    npt.assert_equal(
        df_cmls.station_id.iloc[30:32].to_numpy(), np.array([10031, 10032])
    )

    npt.assert_equal(
        df_cmls.index[357:361].to_numpy(),
        np.array(
            [
                "2015-07-25T12:30:00.00000000",
                "2015-07-25T12:30:00.00000000",
                "2015-07-25T12:35:00.00000000",
                "2015-07-25T12:35:00.00000000",
            ],
            dtype="datetime64[ns]",
        ),
    )
