"""Standard-library tests. Synthetic examples are not scientific observations."""
import unittest
from pathlib import Path
from diagnose import diagnose,read,flag

class DiagnosticTests(unittest.TestCase):
    def test_all_states(self):
        root=Path(__file__).parent/'example';tables=diagnose(root)
        self.assertEqual({r['query_id']:r['state'] for r in tables['per_query']},
                         {r['query_id']:r['state'] for r in read(root/'expected_states.tsv')})
        self.assertEqual(sum(r['queries'] for r in tables['state_counts']),10)
    def test_acceptance_denominator(self):
        s=diagnose(Path(__file__).parent/'example')['output_summary'][0]
        self.assertEqual((s['queries'],s['evaluable_queries'],s['selected_queries']),(10,9,9))
        self.assertEqual((s['accepted_queries'],s['evaluable_accepted'],s['accepted_concordant'],s['accepted_disagreements']),(5,4,1,3))
    def test_missing_acceptance_is_not_rejection(self):
        self.assertIsNone(flag('',optional=True))
        self.assertFalse(flag('0'))
        with self.assertRaises(ValueError):flag('unknown')

if __name__=='__main__':unittest.main()
