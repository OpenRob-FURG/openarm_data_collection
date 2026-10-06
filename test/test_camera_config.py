"""Regras de câmera do launch: serial no formato do driver e guarda contra RealSense trocada."""
import pytest

from openarm_data_collection.camera_config import (
    check_driver,
    check_realsense_serials,
    realsense_serial,
)


def test_serial_gets_the_underscore_the_driver_compares_against():
    assert realsense_serial("123622270736") == "_123622270736"
    assert realsense_serial("_123622270736") == "_123622270736"     # idempotente
    assert realsense_serial("  123622270736 ") == "_123622270736"
    assert realsense_serial(123622270736) == "_123622270736"        # yaml leu como int


@pytest.mark.parametrize("empty", ["", "  ", None])
def test_empty_serial_stays_empty(empty):
    assert realsense_serial(empty) == ""


def test_single_realsense_may_have_no_serial():
    check_realsense_serials({})
    check_realsense_serials({"chest": ""})


def test_two_realsense_need_serials_on_both():
    check_realsense_serials({"chest": "_1", "right_wrist": "_2"})
    with pytest.raises(ValueError, match="right_wrist"):
        check_realsense_serials({"chest": "_1", "right_wrist": ""})
    with pytest.raises(ValueError, match="chest"):
        check_realsense_serials({"chest": "", "right_wrist": "_2"})


def test_error_message_says_how_to_list_serials():
    with pytest.raises(ValueError, match="rs-enumerate-devices"):
        check_realsense_serials({"chest": "", "left_wrist": "", "right_wrist": ""})


def test_same_serial_twice_is_rejected():
    with pytest.raises(ValueError, match="mesmo serial"):
        check_realsense_serials({"chest": "_1", "right_wrist": "_1"})


def test_driver_check():
    assert check_driver("realsense") == "realsense"
    assert check_driver("v4l2") == "v4l2"
    with pytest.raises(ValueError):
        check_driver("usb_cam", "pulso direito")
