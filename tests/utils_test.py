from phenocam_snow.utils import get_site_dates
import pytest


def test_get_site_dates():
    dates = get_site_dates("canadaojp")
    assert dates == ("2015-12-31", "2020-12-31")
