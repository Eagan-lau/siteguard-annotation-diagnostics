import unittest
from diagnostic_core import trace

class CoreTests(unittest.TestCase):
    def test_states(self):
        pool=[('a',{'A'},2),('b',{'B'},1)]
        cases=[([],{'A','B'},pool,{'A'},2,'NO_RECORDED_ACTIVITY'),
            (['C'],{'A','B'},pool,{'A'},2,'NO_LIBRARY_SUPPORT'),
            (['C'],{'A','B','C'},pool,{'A'},2,'ABSENT_FROM_RETURNED_CANDIDATES'),
            (['B'],{'A','B'},pool,{'A'},1,'RETENTION_LOSS'),
            (['B'],{'A','B'},pool,{'A'},2,'SELECTION_LOSS'),
            (['B'],{'A','B'},pool,{'A','B'},2,'CONCORDANT_OUTPUT')]
        for *args,state in cases:self.assertEqual(trace(*args)['state'],state)
    def test_no_hits(self):
        r=trace({'A'},{'A'},[],set(),50)
        self.assertEqual(r['state'],'ABSENT_FROM_RETURNED_CANDIDATES');self.assertEqual(r['has_output'],0)
    def test_missing_truth(self):self.assertIsNone(trace([],{'A'},[],[],50)['hit'])
    def test_multilabel(self):
        r=trace({'B'},{'A','B'},[('a',{'A','B'},3)],{'A','B'},50)
        self.assertEqual((r['hit'],r['output_count'],r['matched_output_count']),(1,2,1))
    def test_tie_respects_native_order(self):
        r=trace({'B'},{'A','B'},[('a',{'A'},3),('b',{'B'},3)],{'A'},10)
        self.assertEqual(r['state'],'SELECTION_LOSS')
    def test_invalid(self):
        with self.assertRaises(ValueError):trace({'A'},{'A'},[('x',{'B'},3)],{'B'},10)
        with self.assertRaises(ValueError):trace({'A'},{'A'},[('x',{'A'},3),('x',{'A'},2)],{'A'},10)
        with self.assertRaises(ValueError):trace({'A'},{'A','B'},[('x',{'A'},3)],{'B'},10)

if __name__=='__main__':unittest.main()
