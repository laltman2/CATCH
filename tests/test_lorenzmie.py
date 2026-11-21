'''Model profiles of the Lorenz-Mie theory, against pylorenzmie'''
import unittest
from importlib.util import find_spec

import numpy as np
import torch

from CATCH.lib.lorenzmie import RadialModel, _step, coefficients, orders

HAVE_PYLORENZMIE = find_spec('pylorenzmie') is not None
K_UM = 2 * np.pi * 1.34 / 0.447
RADII = 2. * np.arange(1, 641)[::16]


def reference_profiles(z, a, n):
    '''The same profiles from pylorenzmie's hologram'''
    from pylorenzmie.theory import LorenzMie, Sphere
    zero = np.zeros_like(RADII)
    model = LorenzMie()
    model.instrument.properties = dict(wavelength=0.447, magnification=0.048,
                                       n_m=1.34)
    model.coordinates = np.vstack([np.r_[RADII, zero], np.r_[zero, RADII],
                                   np.r_[zero, zero]])
    sphere = model.single(Sphere)
    sphere.x_p = sphere.y_p = 0.
    out = []
    for zi, ai, ni in zip(z, a, n, strict=True):
        sphere.z_p, sphere.a_p, sphere.n_p = zi, ai, ni
        h = np.asarray(model.hologram()).ravel()
        out.append((h[:len(RADII)] + h[len(RADII):]) / 2)
    return np.array(out)


def spheres(count, seed=0):
    rng = np.random.default_rng(seed)
    a = np.exp(rng.uniform(np.log(0.25), np.log(5.), count))
    n = rng.uniform(1.38, 2.5, count)
    z = np.maximum(np.exp(rng.uniform(np.log(200.), np.log(6000.), count)),
                   1.5 * a / 0.048)
    return z, a, n


class TestCoefficients(unittest.TestCase):

    def test_orders(self) -> None:
        self.assertEqual(orders(np.array([5.])).tolist(),
                         [int(5 + 4 * 5 ** (1 / 3) + 1)])
        self.assertEqual(orders(np.array([50.])).tolist(),
                         [int(50 + 4.05 * 50 ** (1 / 3) + 2)])

    def test_shape_and_zeros_above_the_orders(self) -> None:
        x = K_UM * np.array([0.3, 3.])
        a, b = coefficients(x, np.array([1.2, 1.1]))
        self.assertEqual(a.shape, b.shape)
        self.assertEqual(a.shape[1], orders(x).max() + 1)
        self.assertEqual(a[0, 0], 0)
        self.assertTrue(np.all(a[0, orders(x)[0] + 1:] == 0))

    def test_empty(self) -> None:
        a, b = coefficients(np.zeros(0), np.zeros(0))
        self.assertEqual(a.shape, (0, 1))

    @unittest.skipUnless(HAVE_PYLORENZMIE, 'pylorenzmie is not installed')
    def test_agree_with_pylorenzmie(self) -> None:
        from pylorenzmie.theory import Sphere
        from pylorenzmie.theory._mie import mie_ab
        rng = np.random.default_rng(1)
        a_p = np.exp(rng.uniform(np.log(0.2), np.log(5.5), 40))
        n_p = rng.uniform(1.38, 2.6, 40)    # high indexes start the
        x, m = K_UM * a_p, n_p / 1.34       # recurrence above |m x|
        a, b = coefficients(x, m)
        for i in range(len(x)):
            ab = mie_ab(x[i], m[i] + 0j, Sphere.wiscombe_yang(x[i], m[i]))
            last = orders(x[i:i + 1])[0] + 1
            np.testing.assert_allclose(a[i, 1:last], ab[1:last, 0],
                                       atol=1e-10)
            np.testing.assert_allclose(b[i, 1:last], ab[1:last, 1],
                                       atol=1e-10)


@unittest.skipUnless(HAVE_PYLORENZMIE, 'pylorenzmie is not installed')
class TestRadialModel(unittest.TestCase):

    def test_double_precision_agrees_with_pylorenzmie(self) -> None:
        z, a, n = spheres(12)
        model = RadialModel(RADII, 0.447, 0.048, 1.34, compile=False)
        self.assertEqual(model.dtype, torch.float64)
        h = model(z, a, n).numpy()
        np.testing.assert_allclose(h, reference_profiles(z, a, n), atol=1e-5)

    def test_single_precision_is_close(self) -> None:
        z, a, n = spheres(12, seed=2)
        model = RadialModel(RADII, 0.447, 0.048, 1.34, dtype=torch.float32,
                            compile=False)
        h = model(z, a, n).numpy()
        np.testing.assert_allclose(h, reference_profiles(z, a, n), atol=2e-3)

    def test_chunks_and_padding_give_the_same_profiles(self) -> None:
        z, a, n = spheres(7, seed=3)
        whole = RadialModel(RADII, 0.447, 0.048, 1.34, compile=False)
        parts = RadialModel(RADII, 0.447, 0.048, 1.34, chunk=3,
                            compile=False)
        parts.step = lambda *args: _step(*args)   # as if it were compiled,
        np.testing.assert_allclose(parts(z, a, n).numpy(),  # so it pads
                                   whole(z, a, n).numpy())


class TestEmpty(unittest.TestCase):

    def test_no_spheres(self) -> None:
        model = RadialModel(RADII, 0.447, 0.048, 1.34, compile=False)
        h = model(np.zeros(0), np.zeros(0), np.zeros(0))
        self.assertEqual(tuple(h.shape), (0, len(RADII)))

    def test_device_and_precision(self) -> None:
        model = RadialModel(RADII, 0.447, 0.048, 1.34, device='cpu',
                            compile=False)
        self.assertEqual(model.device.type, 'cpu')
        h = model(np.array([1000.]), np.array([1.]), np.array([1.5]))
        self.assertEqual(h.dtype, torch.float64)
        self.assertTrue(torch.isfinite(h).all())


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
