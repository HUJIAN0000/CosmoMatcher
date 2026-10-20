"""Verify frozen inputs; optionally rerun exactly the two requested cases."""
from pathlib import Path
import subprocess,sys,argparse,os
R=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONIOENCODING']='utf-8';os.environ['MPLBACKEND']='Agg'

def main():
    p=argparse.ArgumentParser();p.add_argument('--sample',action='store_true');p.add_argument('--full-inference',dest='sample',action='store_true',help='Alias for --sample, retained for the former entry point.');args=p.parse_args()
    subprocess.run([sys.executable,str(R/'verify_inputs.py')],check=True,cwd=R)
    if args.sample:
        subprocess.run([sys.executable,str(R/'prepare.py')],check=True,cwd=R)
        for case in ('with_systematics','without_systematics'):
            with (R/'runs'/case/'run.log').open('w',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-u',str(R/'run_case.py'),case],check=True,cwd=R,stdout=log,stderr=subprocess.STDOUT)
        subprocess.run([sys.executable,str(R/'make_outputs.py')],check=True,cwd=R)
    print('Current protocol: cddv301.py original settings, with/without systematics only.')

if __name__=='__main__':main()
