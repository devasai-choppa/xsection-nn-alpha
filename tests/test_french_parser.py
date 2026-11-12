"""Ken French CSV parsing, without the network."""

from __future__ import annotations

import pandas as pd

from xsnn.data import _extract_close_volume, _parse_french_csv


def test_parse_monthly_french_block() -> None:
    text = (
        "Header line that should be ignored\n"
        "\n"
        ",Mkt-RF,SMB,HML,RMW,CMA,RF\n"
        "200501, 1.50, -0.20, 0.30, 0.10, -0.10, 0.15\n"
        "200502, -2.00, 0.40, 0.00, 0.00, 0.20, 0.16\n"
        "\n"
        " Annual Factors\n"
        "2005, 10, 1, 1, 1, 1, 2\n"
    )
    frame = _parse_french_csv(text)
    assert list(frame.columns) == ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    assert frame.index[0] == pd.Timestamp("2005-01-31")
    assert frame.loc[pd.Timestamp("2005-01-31"), "Mkt-RF"] == 1.5
    assert len(frame) == 2


def test_yfinance_multiindex_layouts() -> None:
    index = pd.bdate_range("2020-01-02", periods=2)
    by_ticker = pd.MultiIndex.from_product([["AAA"], ["Close", "Volume"]])
    frame = pd.DataFrame([[10.0, 1000.0], [11.0, 1100.0]], index=index, columns=by_ticker)
    extracted = _extract_close_volume(frame, ["AAA"])
    assert extracted["AAA"]["close"].iloc[-1] == 11.0
    assert extracted["AAA"]["volume"].iloc[-1] == 1100.0

    by_field = pd.MultiIndex.from_product([["Close", "Volume"], ["BBB"]])
    other = pd.DataFrame([[5.0, 50.0], [6.0, 60.0]], index=index, columns=by_field)
    extracted = _extract_close_volume(other, ["BBB"])
    assert extracted["BBB"]["close"].iloc[0] == 5.0
