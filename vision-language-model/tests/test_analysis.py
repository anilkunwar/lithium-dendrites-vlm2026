import unittest
import numpy as np
from analyze_results import metrics
from plot_batch import interpolate, associations, summarize


class MetricsTests(unittest.TestCase):
    def test_planar_front_excludes_detached_island(self):
        a = np.zeros((5, 11, 3))
        a[:, :4, 0] = 1
        a[2, 9, 0] = 1
        result = metrics(a, [0, 10, 0, 4])
        self.assertEqual(result['tip_x'], 3)
        self.assertEqual(result['front_std_x'], 0)
        self.assertAlmostEqual(result['attached_grid_fraction'], 4/11)

    def test_empty_phase_and_nonfinite_values(self):
        a = np.zeros((5, 11, 3))
        a[0, 0, 1] = np.nan
        result = metrics(a, [0, 10, 0, 4])
        self.assertIsNone(result['tip_x'])
        self.assertEqual(result['fields']['c']['nonfinite_count'], 1)

    def test_channel_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            metrics(np.zeros((5, 5, 2)), [0, 1, 0, 1])

    def test_common_time_interpolation(self):
        records = [{'time': 0, 'tip_x': 2}, {'time': 4, 'tip_x': 10}]
        self.assertEqual(interpolate(records, 'tip_x', 1), 4)
        with self.assertRaises(ValueError):
            interpolate(records, 'tip_x', 5)

    def test_no_association_from_one_case(self):
        rows = [dict(parameters={'voltage': -1}, tip_displacement=2,
                     front_std_x=1, attached_grid_fraction=.2)]
        self.assertTrue(all(r['pearson_r'] is None for r in associations(rows)))

    def test_comparison_uses_shared_time(self):
        cases = []
        for name, end in [('short', 2), ('long', 4)]:
            records = [dict(time=t, tip_x=3 + 2*t, front_std_x=t,
                            attached_grid_fraction=.1*t) for t in [0, end]]
            cases.append(dict(id=name, records=records, parameters={}))
        rows = summarize(cases, 2)
        self.assertEqual([r['tip_displacement'] for r in rows], [4, 4])
        self.assertEqual([r['mean_tip_speed'] for r in rows], [2, 2])


if __name__ == '__main__':
    unittest.main()
