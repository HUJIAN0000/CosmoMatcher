"""Check catalogue identity, parent covariance indices and untouched author inputs."""
from pathlib import Path
import json,hashlib,numpy as np,pandas as pd
R=Path(__file__).resolve().parent

def main():
    inputs=R/'inputs';manifest=json.loads((R/'manifest.json').read_text())
    for name,item in manifest.items():
        if isinstance(item,dict):assert hashlib.sha256((inputs/name).read_bytes()).hexdigest()==item['sha256'],name
    frame=pd.read_csv(R/'reference_matching/matched_sgls_milp_dd_pantheon.csv',dtype={'sn_lens_name':str,'sn_source_name':str})
    sn=pd.read_csv(inputs/'Ancillary_C11_jla.csv',dtype={'name':str})
    assert len(frame)==99 and len(sn)==1048
    li=frame.sn_lens_original_index.to_numpy(int);si=frame.sn_source_original_index.to_numpy(int)
    assert sn.iloc[li]['name'].tolist()==frame.sn_lens_name.tolist()
    assert sn.iloc[si]['name'].tolist()==frame.sn_source_name.tolist()
    assert len(set(np.r_[li,si]))==198
    values=np.loadtxt(inputs/'sys_full_long_C11.txt');n=int(values[0]);c=values[1:].reshape(n,n);c=(c+c.T)/2
    cov=np.load(inputs/'cov_sgls_milp_dd_pantheon.npz')
    differences={}
    for key,i,j in [('lens',li,li),('source',si,si),('cross',li,si)]:
        differences[key]=float(abs(cov[key]-c[np.ix_(i,j)]).max());assert differences[key]==0
    data=np.loadtxt(inputs/'result_final_min_select_handle')
    for name,column in [('sn_lens_3rdvar',17),('sn_lens_d3rdvar',18),('sn_source_3rdvar',19),('sn_source_d3rdvar',20)]:
        expected=frame[name].to_numpy(float)
        if column in (18,20):expected=np.where(expected<=0,.1,expected)
        assert np.allclose(expected,data[:,column],rtol=5e-9,atol=1e-10)
    result={'verified':True,'matched_systems':99,'distinct_sn':198,'parent_sn':1048,
        'maximum_covariance_difference':differences,'host_error_99_entries_kept':int((data[:,18]==99).sum()+(data[:,20]==99).sum()),
        'note':'No special handling of host errors; same data for both inference cases.'}
    (R/'input_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
