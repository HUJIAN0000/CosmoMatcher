"""Run the original main block; only add deterministic RNG and output capture."""
from pathlib import Path
import os,sys,runpy
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['MPLBACKEND']='Agg'
os.environ['PYTHONIOENCODING']='utf-8'
import numpy as np
import emcee
R=Path(__file__).resolve().parent

if __name__=='__main__':
    case=sys.argv[1]
    assert case in ('with_systematics','without_systematics')
    out=R/'runs'/case
    np.random.seed(20261019)
    original_run=emcee.EnsembleSampler.run_mcmc
    def capture(self,*args,**kwargs):
        # The calculation, moves and step count are those of cddv301.py.
        result=original_run(self,*args,**kwargs)
        np.save(out/'chain.npy',self.get_chain())
        np.save(out/'log_probability.npy',self.get_log_prob())
        np.save(out/'acceptance.npy',self.acceptance_fraction)
        return result
    original_get_chain=emcee.EnsembleSampler.get_chain
    def capture_chain(self,*args,**kwargs):
        result=original_get_chain(self,*args,**kwargs)
        if kwargs.get('flat',False):np.save(out/'flat_samples.npy',result)
        return result
    emcee.EnsembleSampler.run_mcmc=capture
    emcee.EnsembleSampler.get_chain=capture_chain
    runpy.run_path(str(out/'cddv301.py'),run_name='__main__')
