"""Exact finite constraints on an unknown constellation labeling."""

import numpy as np


def nullspace(equations, width):
    """Packed GF(2) row reduction and a basis of the homogeneous solutions."""
    pivots={}
    for equation in equations:
        row=int(equation)
        while row:
            pivot=row.bit_length()-1
            if pivot in pivots:
                row^=pivots[pivot]
            else:
                pivots[pivot]=row
                break
    basis=[]
    ordered=sorted(pivots.items())
    for free in range(width):
        if free in pivots:
            continue
        x=1<<free
        for pivot,row in ordered:
            if (row & x).bit_count()%2:
                x|=1<<pivot
        basis.append(x)
    return basis,len(pivots)


def forced_collisions(basis, points=16, bits=4):
    collisions=[]
    for a in range(points):
        for b in range(a+1,points):
            difference=[(1<<(p*points+a)) ^ (1<<(p*points+b)) for p in range(bits)]
            if all((q & v).bit_count()%2==0 for q in basis for v in difference):
                collisions.append((a,b))
    return collisions


def contract_checks(labels, supports, points=16, bits=4):
    """Evaluate row supports symbolically in the constellation-label variables."""
    indices=(supports%bits)*points+labels[supports//bits]
    return np.bitwise_xor.reduce(np.left_shift(np.uint64(1),indices.astype(np.uint64)),axis=1)


def projection_fit(u,v,lift,short=16):
    from code21_ring import pack_blocks,recover,reduce_polynomial
    uu=[[reduce_polynomial(b,short) for b in row] for row in pack_blocks(u,lift)]
    vv=[[reduce_polynomial(b,short) for b in row] for row in pack_blocks(v,lift)]
    return recover(uu,vv,short)
