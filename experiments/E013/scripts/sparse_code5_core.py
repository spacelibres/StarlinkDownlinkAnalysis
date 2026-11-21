"""Exact low-weight dual search and synchronous normalized min-sum decoding."""
from pathlib import Path
import sys
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'project.json').is_file())
sys.path.insert(0,str(ROOT));import project_paths
from code5_core import recover_generator


def _integer_rows(matrix):
    return [int.from_bytes(row.tobytes(),'little') for row in np.packbits(matrix,axis=1,bitorder='little')]


def _positions(value):
    positions=[]
    while value:
        bit=value&-value;positions.append(bit.bit_length()-1);value^=bit
    return positions


def _shift(support,amount,lift):
    return tuple(sorted((p//lift)*lift+(p+amount)%lift for p in support))


def binary_rank(matrix):
    basis={}
    for value in _integer_rows(np.asarray(matrix,dtype=np.uint8)):
        while value:
            pivot=(value&-value).bit_length()-1
            if pivot in basis:value^=basis[pivot]
            else:basis[pivot]=value;break
    return len(basis)


def find_sparse_checks(generator,lift=128,max_weight=6,progress=None):
    p=np.asarray(generator,dtype=np.uint8);n=p.shape[0]
    if p.shape!=(n,n) or n%lift:raise ValueError('Search requires aligned square parity generator')
    inverse=recover_generator(p,np.eye(n,dtype=np.uint8))
    matrices=[('parity',p)]
    if inverse['generator'] is not None:matrices.append(('systematic',inverse['generator']))
    seeds=set();counts={};hits={}
    for side,matrix in matrices:
        vectors=_integer_rows(matrix.T);counts[side]={1:0,2:0,3:0};hits[side]=0
        def consider(image,support):
            if image.bit_count()+len(support)>max_weight:return
            source=_positions(image)
            positions=source+[n+j for j in support] if side=='parity' else list(support)+[n+j for j in source]
            canonical=min(_shift(positions,j,lift) for j in range(lift))
            seeds.add(canonical);hits[side]+=1
        for anchor in range(0,n,lift):
            counts[side][1]+=1;consider(vectors[anchor],(anchor,))
            for second in range(anchor+1,n):
                pair=vectors[anchor]^vectors[second];counts[side][2]+=1;consider(pair,(anchor,second))
                if max_weight>=3:
                    for third in range(second+1,n):
                        counts[side][3]+=1
                        image=pair^vectors[third]
                        if image.bit_count()<=max_weight-3:consider(image,(anchor,second,third))
            if progress:progress(side,anchor,len(seeds))
    return {'seeds':sorted(seeds,key=lambda x:(len(x),x)),'tested_supports':counts,'full_dimension_hits':hits,
            'parity_generator_rank':inverse['rank'],'inverse_available':inverse['generator'] is not None,
            'search_scope':'QC-normalized support weights1..3 in each available pivot basis; total row weight<=cap, exact full-dimensional check'}


def independent_checks(seeds,target_rank,length,lift):
    basis={};supports=[];used=[]
    for index,seed in enumerate(seeds):
        added=0
        for shift in range(lift):
            support=_shift(seed,shift,lift);original=sum(1<<j for j in support);value=original
            while value:
                pivot=(value&-value).bit_length()-1
                if pivot in basis:value^=basis[pivot]
                else:
                    basis[pivot]=value;supports.append(support);added+=1;break
            if len(basis)==target_rank:break
        if added:used.append({'seed_index':index,'independent_rows':added})
        if len(basis)==target_rank:break
    matrix=np.zeros((len(supports),length),dtype=np.uint8)
    for row,support in enumerate(supports):matrix[row,list(support)]=1
    return {'matrix':matrix,'rank':len(basis),'used_orbits':used}


def syndrome(words,checks):
    words=np.asarray(words,dtype=np.uint8);checks=np.asarray(checks,dtype=np.uint8)
    result=np.empty((len(words),len(checks)),dtype=np.uint8)
    for row,h in enumerate(checks):result[:,row]=words[:,np.flatnonzero(h)].sum(axis=1)%2
    return result


def normalized_min_sum(llr,checks,eta=.8,max_iterations=60):
    llr=np.asarray(llr,dtype=float);checks=np.asarray(checks,dtype=np.uint8)
    rows,columns=np.nonzero(checks);degree=checks.sum(axis=1).astype(int)
    if llr.shape!=(checks.shape[1],) or not np.isfinite(llr).all() or np.any(degree<2):
        raise ValueError('Expected finite LLR vector and checks with degree>=2')
    starts=np.r_[0,np.cumsum(degree)[:-1]]
    def failures(bits):return int(np.count_nonzero(np.add.reduceat(bits[columns],starts)%2))
    bits=(llr<0).astype(np.uint8);initial=failures(bits)
    if not initial:return {'bits':bits,'converged':True,'iterations':0,'initial_failed_checks':0,'final_failed_checks':0}
    q=llr[columns].copy()
    for iteration in range(1,max_iterations+1):
        absolute=abs(q);minimum=np.minimum.reduceat(absolute,starts);is_min=absolute==minimum[rows]
        count=np.add.reduceat(is_min,starts)
        second=np.minimum.reduceat(np.where(is_min,np.inf,absolute),starts)
        second=np.where(count>1,minimum,second)
        extrinsic=np.where(is_min,second[rows],minimum[rows])
        sign=np.where(q>=0,1.,-1.);total=np.multiply.reduceat(sign,starts)
        messages=eta*total[rows]*sign*extrinsic
        posterior=llr+np.bincount(columns,weights=messages,minlength=len(llr))
        bits=(posterior<0).astype(np.uint8);remaining=failures(bits)
        if not remaining:break
        q=posterior[columns]-messages
    return {'bits':bits,'converged':remaining==0,'iterations':iteration,
            'initial_failed_checks':initial,'final_failed_checks':remaining}
