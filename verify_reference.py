#!/usr/bin/env python3
"""Independently reconstruct the published compact model and SN covariance.

No CosmoMatcher imports. Uses direct quadrature, enumeration, SciPy MILP and
parent-matrix indexing. This checks the supplied fixed 5% reference case.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.integrate import quad
from scipy.optimize import milp, Bounds, LinearConstraint
from threadpoolctl import threadpool_limits


def verify(data, reference):
    sn = pd.read_csv(data/'Pantheon+SH0ES.dat', sep=r'\s+', dtype={'CID': str})
    try:
        targets = pd.read_csv(data/'161sgls.csv', encoding='utf-8-sig')
    except UnicodeDecodeError:
        targets = pd.read_csv(data/'161sgls.csv', encoding='gb18030')
    selected = pd.read_csv(reference/'matched_sgls_milp_dd_pantheon.csv',
                           dtype={'sn_lens_CID': str, 'sn_source_CID': str})
    index = pd.read_csv(reference/'index_map_sgls_milp_dd_pantheon.csv')
    z = sn.zHD.to_numpy(); zl = targets.zl.to_numpy(); zs = targets.zs.to_numpy()
    nl, n = len(targets), len(sn)
    raw = np.loadtxt(data/'Pantheon+SH0ES_STAT+SYS.cov').ravel()
    assert raw.size == n*n+1 and raw[0] in (n, n*n)
    parent = raw[1:].reshape(n,n); parent = (parent+parent.T)/2
    ii = index.covariance_row.to_numpy(int)
    c = parent[np.ix_(ii,ii)]
    with np.load(reference/'cov_sgls_milp_dd_pantheon.npz') as saved:
        np.testing.assert_allclose(c,saved['full'],rtol=0,atol=1e-14)
        k=len(selected); jl=selected.sn_lens_original_index.to_numpy(int)
        js=selected.sn_source_original_index.to_numpy(int)
        np.testing.assert_array_equal(ii,np.r_[jl,js])
        assert len(set(sn.CID.iloc[np.r_[jl,js]]))==2*k
        for role,indices in [('lens',jl),('source',js)]:
            np.testing.assert_array_equal(sn.CID.iloc[indices],selected[f'sn_{role}_CID'])
            np.testing.assert_allclose(z[indices],selected[f'sn_{role}_redshift'],atol=1e-14)
            np.testing.assert_allclose(sn.m_b_corr.iloc[indices],selected[f'sn_{role}_m_b_corr'],atol=1e-12)
        np.testing.assert_allclose(c[:k,:k],saved['lens'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(c[k:,k:],saved['source'],rtol=0,atol=1e-14)
        np.testing.assert_allclose(c[:k,k:],saved['cross'],rtol=0,atol=1e-14)
        m=sn.m_b_corr.to_numpy();kap=np.log(10)/5
        q=(1+z[js])/(1+z[jl])*np.exp(kap*(m[jl]-m[js]))
        j=np.zeros((k,2*k));j[np.arange(k),np.arange(k)]=-kap*q
        j[np.arange(k),k+np.arange(k)]=kap*q
        cr=j@c@j.T
        np.testing.assert_allclose(cr,saved['ratio'],rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(1-q,selected.R_flat,rtol=1e-12,atol=1e-14)
        covariance_residual=float(np.max(np.abs(cr-saved['ratio'])))
    def chi(value):
        return quad(lambda x: 1/np.sqrt(.3*(1+x)**3+.7),0,float(value),
                    epsabs=1e-12,epsrel=1e-12)[0]
    d=np.array([chi(v) for v in z]);dl=np.array([chi(v) for v in zl]);ds=np.array([chi(v) for v in zs])
    el=np.abs(d[None,:]-dl[:,None])/dl[:,None]
    es=np.abs(d[None,:]-ds[:,None])/ds[:,None]
    _,groups=np.unique(sn.CID.str.strip(),return_inverse=True);ng=int(groups.max()+1)
    rows=[];cols=[];values=[];cost=[0.]*nl;edges={};ntriplets=0
    for target in range(nl):
        for role in range(2):rows.append(2*target+role);cols.append(target);values.append(-1.)
        left=np.flatnonzero(el[target]<=.05);right=np.flatnonzero(es[target]<=.05)
        distinct=groups[left,None]!=groups[right][None,:]
        ordered=z[left,None]<z[right][None,:]
        ntriplets+=int((distinct&ordered).sum())
        # This reference catalogue needs no extra redshift-order incompatibility rows.
        assert not np.any(distinct&~ordered)
        for role,candidates,errors in [(0,left,el),(1,right,es)]:
            for row in candidates:
                column=len(cost);cost.append(float(errors[target,row]**2))
                edges[(target,role,int(row))]=column
                rows.extend([2*target+role,2*nl+int(groups[row])])
                cols.extend([column,column]);values.extend([1.,1.])
    a=sparse.csc_matrix((values,(rows,cols)),shape=(2*nl+ng,len(cost)))
    active=np.asarray(a.getnnz(axis=1)).ravel()>0
    saved_a=sparse.load_npz(reference/'constraint_matrix.npz')
    assert a[active].shape==saved_a.shape and (a[active]-saved_a).nnz==0
    choice=np.zeros(len(cost));choice[selected.obj_index.to_numpy(int)]=1
    for record in selected.to_dict('records'):
        for role,label in [(0,'lens'),(1,'source')]:
            choice[edges[(int(record['obj_index']),role,int(record[f'sn_{label}_original_index']))]]=1
    used=a@choice;assert np.max(abs(used[:2*nl]))==0 and np.max(used[2*nl:])<=1
    constraint=LinearConstraint(a,np.r_[np.zeros(2*nl),np.full(ng,-np.inf)],
                                np.r_[np.zeros(2*nl),np.ones(ng)])
    coverage=np.r_[np.ones(nl),np.zeros(len(cost)-nl)]
    options={'mip_rel_gap':0.0}
    first=milp(-coverage,integrality=np.ones(len(cost)),bounds=Bounds(0,1),constraints=constraint,options=options)
    assert first.status==0
    maximum=int(round(-first.fun));assert maximum==len(selected)
    cost=np.asarray(cost)
    second=milp(cost/cost.max(),integrality=np.ones(len(cost)),bounds=Bounds(0,1),
                constraints=[constraint,LinearConstraint(sparse.csc_matrix(coverage[None,:]),maximum,maximum)],options=options)
    assert second.status==0
    best=float(cost@np.rint(second.x));reference_cost=float(cost@choice)
    assert abs(best-reference_cost)<1e-12
    return {'complete_systems':maximum,'triplets':ntriplets,'variables':len(cost),
            'active_constraint_rows':int(active.sum()),'nonzeros':int(a.nnz),
            'minimum_cost':best,'reference_cost':reference_cost,
            'ratio_covariance_max_residual':covariance_residual,
            'backend_imported':False,'status':'passed'}


if __name__=='__main__':
    root=Path(__file__).resolve().parent
    parser=argparse.ArgumentParser(description=__doc__)
    default_data=root/'data' if (root/'data').exists() else root.parent/'data'
    parser.add_argument('--data',type=Path,default=default_data)
    parser.add_argument('--reference',type=Path,default=root/'reference_results')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):result=verify(args.data,args.reference)
    text=json.dumps(result,indent=2)
    print(text)
    if args.report:args.report.write_text(text+'\n')
