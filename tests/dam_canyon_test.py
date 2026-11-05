import numpy as np
import pytest

from pystrata.dam import (
    RectangularCanyon,
    canyon_freq_ratio,
    canyon_rect_natural_freqs,
    canyon_rect_rigid_base_tf,
    wedge_natural_freqs,
    wedge_rigid_base_tf,
    wedge_truncated_rigid_base_tf,
)

H, VS = 50.0, 300.0


def test_invalid_length():
    with pytest.raises(ValueError):
        RectangularCanyon(0)


def test_long_canyon_recovers_2d():
    freqs = np.array([0.5, 1.0, 2.0])
    length = 5000.0
    tf = canyon_rect_rigid_base_tf(freqs, length / 2, H, VS, length, 0.05, n_modes=2000)
    ref = wedge_rigid_base_tf(freqs, H, VS, damping=0.05)
    np.testing.assert_allclose(tf, ref, rtol=2e-3)


def test_long_truncated_canyon_recovers_2d():
    freqs = np.array([0.5, 1.0, 2.0])
    length, r0 = 5000.0, 10.0
    tf = canyon_rect_rigid_base_tf(
        freqs, length / 2, H, VS, length, 0.05, crest_offset=r0, n_modes=2000
    )
    ref = wedge_truncated_rigid_base_tf(freqs, H, VS, r0, damping=0.05)
    np.testing.assert_allclose(tf, ref, rtol=2e-3)


@pytest.mark.parametrize("r0", [0.0, 10.0])
def test_abutment_and_zero_freq(r0):
    tf = canyon_rect_rigid_base_tf([0.0, 3.0], 0.0, H, VS, 200.0, 0.05, crest_offset=r0)
    np.testing.assert_allclose(tf, 1.0, atol=1e-12)
    tf0 = canyon_rect_rigid_base_tf([0.0], 100.0, H, VS, 200.0, 0.05, crest_offset=r0)
    np.testing.assert_allclose(tf0, 1.0)


def test_symmetric():
    f = np.linspace(0.2, 4, 30)
    a = canyon_rect_rigid_base_tf(f, 60.0, H, VS, 200.0, 0.05)
    b = canyon_rect_rigid_base_tf(f, 140.0, H, VS, 200.0, 0.05)
    np.testing.assert_allclose(a, b, atol=1e-10)


def test_peak_at_natural_frequency():
    length = 150.0
    f_n = canyon_rect_natural_freqs(H, VS, length, n=1)[0]
    freqs = np.linspace(0.5 * f_n, 1.5 * f_n, 3001)
    amp = np.abs(canyon_rect_rigid_base_tf(freqs, length / 2, H, VS, length, 0.005))
    assert freqs[np.argmax(amp)] == pytest.approx(f_n, rel=2e-3)


def test_freq_ratio():
    length = 100.0
    expected = np.sqrt(1 + (np.pi * H / (2.4048 * length)) ** 2)
    assert canyon_freq_ratio(H, VS, length) == pytest.approx(expected, rel=1e-4)
    assert canyon_freq_ratio(H, VS, 1e6) == pytest.approx(1.0, abs=1e-6)
    f2d = wedge_natural_freqs(H, VS, 0.0, 1)[0]
    f3d = canyon_rect_natural_freqs(H, VS, length, n=1)[0]
    assert f3d > f2d
