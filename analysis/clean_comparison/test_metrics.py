import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from metrics_core import counts,measures,ci

class Tests(unittest.TestCase):
    def test_single(self):
        self.assertEqual(counts(['1.1.1.1'],['1.1.1.1']),[1,1,1,1,1,1,1,1,1])
    def test_multilabel_not_single_precision(self):
        values=measures(counts(['a','b'],['a']))
        self.assertEqual(values[2],1);self.assertEqual(values[4],.5)
        self.assertEqual(values[3],0)
    def test_abstention_kept(self):
        values=measures(counts([],['a']));self.assertEqual(values[2],0);self.assertEqual(values[0],0)
        self.assertTrue(np.isnan(values[4]));self.assertEqual(values[6],0)
    def test_unknown_truth(self):
        values=measures(counts(['a'],[]));self.assertEqual(values[0],1);self.assertTrue(np.isnan(values[2]))
    def test_invalid_draw_suppressed(self):
        lo,hi,n=ci([1,np.nan]);self.assertTrue(np.isnan(lo));self.assertTrue(np.isnan(hi));self.assertEqual(n,1)
    def test_ratio_of_sums(self):
        value=measures(np.array(counts(['a'],['a']))+np.array(counts(['b','c','d'],['z'])))
        self.assertEqual(value[4],.25);self.assertEqual(value[2],.5)

if __name__=='__main__':unittest.main()
