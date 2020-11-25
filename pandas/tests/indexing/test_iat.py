import numpy as np

import pandas as pd


def test_iat(float_frame):

    for i, row in enumerate(float_frame.index):
        for j, col in enumerate(float_frame.columns):
            result = float_frame.iat[i, j]
            expected = float_frame.at[row, col]
            assert result == expected


def test_iat_duplicate_columns():
    # https://github.com/pandas-dev/pandas/issues/11754
    df = pd.DataFrame([[1, 2]], columns=["x", "x"])
    assert df.iat[0, 0] == 1

def test_iat_correct_upcast():
    # GH: 37692
    # Initial DataFrame is int64
    df = pd.DataFrame(index=['A','B','C'])
    df['D'] = 0
    assert df['D'].dtypes == np.dtype(np.int64)

    # Test upcasting from int64 to float64
    df_iat_copy = df.copy()
    df_iat_copy.iat[1, 0] = 44.5
    assert df_iat_copy['D'].dtypes == np.dtype(np.float64)

    # Test upcasting from int32 to float64
    df_iat_copy = df.astype('int32')
    df_iat_copy.iat[1, 0] = 44.5
    assert df_iat_copy['D'].dtypes == np.dtype(np.float64)

    # Test upcasting from int64 to object
    df_iat_copy = df.copy()
    df_iat_copy.iat[1, 0] = "hello"
    assert df_iat_copy['D'].dtypes == np.dtype(np.object)