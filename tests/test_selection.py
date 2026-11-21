'''Choosing among the fits of a particle, and the ambiguity'''
import unittest

import numpy as np

from CATCH.selection import COLUMNS, distinct, select


def fit(a, n, redchi, z=100., x=10., y=20.):
    return dict(x_p=x, y_p=y, z_p=z, a_p=a, n_p=n, redchi=redchi)


class TestSelect(unittest.TestCase):

    def test_no_fits(self) -> None:
        out = select([fit(1., 1.5, np.nan)])
        self.assertEqual(out['flag'], 'no fit')
        self.assertEqual(out['nsolutions'], 0)
        self.assertEqual(set(out), set(COLUMNS))

    def test_the_standard_range_wins_a_close_call(self) -> None:
        out = select([fit(0.5, 1.65, 1.10), fit(0.65, 2.3, 1.09)])
        self.assertEqual(out['a_fit'], 0.5)
        self.assertEqual(out['flag'], '')
        self.assertEqual(out['alt_a'], 0.65)

    def test_a_much_better_high_index_fit_wins(self) -> None:
        out = select([fit(0.5, 1.65, 1.10), fit(0.65, 2.3, 0.9)])
        self.assertEqual(out['a_fit'], 0.65)
        self.assertEqual(out['flag'], 'high index')

    def test_only_high_index_fits(self) -> None:
        out = select([fit(0.65, 2.3, 1.)])
        self.assertEqual(out['flag'], 'high index')

    def test_prior(self) -> None:
        fits = [fit(1.5, 1.62, 1.01), fit(1.3, 1.75, 1.00),
                fit(1.4, 1.45, 1.20)]
        out = select(fits, (1.55, 1.70))
        self.assertEqual(out['a_fit'], 1.5)
        self.assertEqual(out['nsolutions'], 1)     # inside the prior only
        out = select(fits, (1.0, 1.2))
        self.assertEqual(out['flag'], 'outside prior')
        self.assertEqual(out['a_fit'], 1.3)

    def test_ambiguity(self) -> None:
        fits = [fit(1.50, 1.62, 1.000, z=200.), fit(1.30, 1.66, 1.003,
                                                     z=203.),
                fit(1.502, 1.62, 1.001), fit(1.1, 1.7, 1.2)]
        out = select(fits)
        self.assertEqual(out['nsolutions'], 2)
        self.assertAlmostEqual(out['a_spread'], 0.2)
        self.assertAlmostEqual(out['n_spread'], 0.04)
        self.assertAlmostEqual(out['z_spread'], 3.)

    def test_distinct(self) -> None:
        fits = [fit(1.0, 1.5, 2.), fit(1.001, 1.5, 1.), fit(1.2, 1.5, 3.)]
        out = distinct(fits)
        self.assertEqual([f['a_p'] for f in out], [1.001, 1.2])


if __name__ == '__main__':  # pragma: no cover
    unittest.main()
