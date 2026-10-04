import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from synthesize_results import build_evidence, render_report, Synthesis, validate_references


class SynthesisTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(case_id=f'case_{i}', tip_displacement=float(i), mean_tip_speed=i/10,
                          front_std_x=float(10-i), attached_grid_fraction=i/10,
                          parameters={'voltage': -i}) for i in range(6)]
        self.run = dict(comparison_time=10, units='simulation units', limitations=[])
        self.packet = build_evidence(self.rows, [], self.run)

    def test_all_cases_contribute_to_distribution(self):
        distribution = self.packet['evidence']['distribution:tip_displacement']
        self.assertEqual(distribution['n'], 6)
        self.assertEqual(distribution['quantiles']['median'], 2.5)
        self.assertEqual(distribution['maximum_example'], 'case_5')

    def test_unknown_evidence_is_rejected(self):
        report = Synthesis(cross_case_patterns=[dict(title='Unsupported', interpretation='Test',
                           evidence_ids=['invented'], limitations='Test')],
                           parameter_associations=[], anomalies=[], follow_up_checks=[])
        with self.assertRaises(ValueError):
            validate_references(report, self.packet)

    def test_report_contains_traceable_evidence(self):
        report = Synthesis(cross_case_patterns=[dict(title='Growth contrast', interpretation='Test',
                           evidence_ids=['distribution:tip_displacement'], limitations='Synthetic data')],
                           parameter_associations=[], anomalies=[], follow_up_checks=[])
        validate_references(report, self.packet)
        with TemporaryDirectory() as directory:
            render_report(Path(directory), self.packet, report)
            html = (Path(directory) / 'report.html').read_text()
            self.assertIn('Growth contrast', html)
            self.assertIn('distribution:tip_displacement', html)
            self.assertIn('2.5', html)

    def test_missing_values_are_not_zero(self):
        self.rows[0]['tip_displacement'] = None
        packet = build_evidence(self.rows, [], self.run)
        distribution = packet['evidence']['distribution:tip_displacement']
        self.assertEqual(distribution['missing'], 1)
        self.assertEqual(distribution['quantiles']['min'], 1)


if __name__ == '__main__':
    unittest.main()
