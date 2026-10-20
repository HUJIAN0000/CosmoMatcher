"""Freeze author inputs; statistical-only script differs by one assignment."""
from pathlib import Path
import json,hashlib,shutil,ast
R=Path(__file__).resolve().parent
AUTHOR=R/'inputs'  # Portable frozen inputs supplied with this release.

def prepare():
    inputs=R/'inputs';inputs.mkdir(exist_ok=True)
    names=['cddv301.py','result_final_min_select_handle','cov_sgls_milp_dd_pantheon.npz',
           'CosmoMatcherv1.43.py','Ancillary_C11_jla.csv','Ancillary_C11.FITRES',
           'sys_full_long_C11.txt','sgls_143_fiduciary_ss3.csv']
    manifest={}
    for name in names:
        source=AUTHOR/name;target=inputs/name
        if source.resolve()!=target.resolve():shutil.copy2(source,target)
        manifest[name]={'source':str(source),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    fig=AUTHOR/'cddv301_getdist.pdf'
    if not fig.exists():fig=inputs/'author_cddv301_getdist.pdf'
    target=inputs/'author_cddv301_getdist.pdf'
    if fig.resolve()!=target.resolve():shutil.copy2(fig,target)
    manifest['author_cddv301_getdist.pdf']={'source':str(fig),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    original=(inputs/'cddv301.py').read_text(encoding='utf-8')
    anchor='# Pre-bundle data for clean argument passing'
    assert original.count(anchor)==1
    stat=original.replace(anchor,'_C_DELTA_MU = None  # Only comparison switch: omit all C11 systematics.\n\n'+anchor)
    for case,source in [('with_systematics',original),('without_systematics',stat)]:
        folder=R/'runs'/case;folder.mkdir(parents=True,exist_ok=True)
        if case=='with_systematics':shutil.copy2(inputs/'cddv301.py',folder/'cddv301.py')
        else:(folder/'cddv301.py').write_text(source,encoding='utf-8')
        for name in names[1:3]:shutil.copy2(inputs/name,folder/name)
    # Prove the one functional difference rather than trusting a prose description.
    first=ast.parse(original);second=ast.parse(stat)
    added=[n for n in second.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='_C_DELTA_MU' and isinstance(n.value,ast.Constant) and n.value.value is None]
    assert len(added)==1
    second.body.remove(added[0])
    assert ast.dump(first,include_attributes=False)==ast.dump(second,include_attributes=False)
    manifest['comparison_change']='Only _C_DELTA_MU=None before _DATA is bundled; lnlike uses original diagonal branch.'
    manifest['sampling']='Author Nelder-Mead initialization, 140 walkers, 1000 steps, default emcee move, 8 processes, original burn/thin selection.'
    manifest['host_errors']='Unchanged, including error=99; no filtering or special-case transformation.'
    (R/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Author inputs frozen; one-assignment statistical-only comparison verified.')

if __name__=='__main__':prepare()
