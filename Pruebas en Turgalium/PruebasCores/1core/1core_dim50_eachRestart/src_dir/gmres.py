#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Based on this question (and responses) from stackoverflow:
# https://stackoverflow.com/questions/37962271/whats-wrong-with-my-gmres-implementation


import numpy as np
import scipy as sp

from scipy.linalg         import get_blas_funcs, get_lapack_funcs
# from scipy.sparse.sputils import upcast

from .util import matmul_a, cidx, mrange

# Cambiado **
def upcast(*args):
    '''
    Use the numpy function upcast instead of the scipy function
    '''
    return np.find_common_type(args, [])

#
# EDITOR'S NOTE: In order to better check this against literature, I used the
# the cidx (turn a 1..N -- fortran-style -- index to a 0..N-1 -- c-style --
# index), and the `mrange` (math-style range for loops: 1..N instead of 0..N-1)
#


def update_solution(x, y, q):
    '''
    Auxiliar function to do Q*y
    '''
    g = np.zeros_like(x)

    q2=np.asarray(q)
    g2=np.dot(q2.transpose(),y)


    for i, iy in enumerate(y):
        # g += q[i]*iy  # This line causes a reduction in precision when x0 input of GMRES comes from the NN. 
        g    = g + q[i]*iy
    return g



def GMRES(A, *args, **kwargs):
    if isinstance(A, np.matrix):
        # construct closure  => linear operator
        lin_op = lambda x: matmul_a(A, x)
        return GMRES_op(lin_op, *args, **kwargs)
    else:
        return GMRES_op(A, *args, **kwargs)



def GMRES_op(A, b, x0, e, nmax_iter, restart=None, debug=False):
    """
    Quick and dirty GMRES -- TODO: optimize going to larger
    systems.

    - A: Laplace operator
    - b: random matrix with dipole (dim x dim)
    - x0: Initial guess matrix (dim x dim)
    - e: relative tolerance
    - nmax_iter: dimension of the Krylov's subspace
    - restart: option of restart the GMRES algorithm until it converges
    - debug: option to put in the return the results of the inner-loop
    """

    # TODO: you can use this to make the problem agnostic to complex numbers
    # # Defining xtype as dtype of the problem, to decide which BLAS functions
    # # import.
    # xtype = upcast(x0.dtype, b.dtype)

    # Defining dimension
    dimen = len(x0)

    # TODO: use BLAS functions
    # # Get fast access to underlying BLAS routines
    # [lartg] = get_lapack_funcs(['lartg'], [x0] )
    # if np.iscomplexobj(np.zeros((1,), dtype=xtype)):
    #     [axpy, dotu, dotc, scal] =\
    #         get_blas_funcs(['axpy', 'dotu', 'dotc', 'scal'], [x0])
    # else:
    #     # real type
    #     [axpy, dotu, dotc, scal] =\
    #         get_blas_funcs(['axpy', 'dot', 'dot', 'scal'], [xO])

    # TODOs for this function:
    # 1. list -> numpy.array <= better memory access
    # 2. don't append to lists -> prealoc and slice
    # 3. add documentation -- this will probably never happen :P
    
    normb = np.linalg.norm(b)
    # Avoid ||b|| = 0
    if normb == 0.0:
        normb = 1.0

    # TODO: is the old code (below) faster?
    # r = b - np.asarray(np.dot(A, x0)).reshape(-1)
    # r = b - matmul_a(A, x0)
    r = b - A(x0) # residue

    # Set number of outer loops based on the value of `restart`
    n_outer = 1 
    if restart is not None:
        n_outer = int(restart) # restart

    x     = [x0] # convert matrix x0 to a tensor
    x_sol = x0 # auxiliar variable to update the sol

    for l in mrange(n_outer): # restart loop
        q    = [np.zeros_like(x0)] * (nmax_iter) # define Q
        q[cidx(1)] = r / np.linalg.norm(r) # q_0 = r / ||r||

        h = np.zeros((nmax_iter + 1, nmax_iter)) # define H tilde

        for k in mrange(min(nmax_iter, dimen**2)): # inner-loop (dimension of krylov's subespace) cambiado
            
            # TODO: is the old code (below) faster?
            # y = np.asarray(np.dot(A, q[k])).reshape(-1)
            # y = matmul_a(A, q[cidx(k)])
            y = A(q[cidx(k)]) # y = A*q_k

            # Modified Grahm-Schmidt
            for j in range(1, k+1): # loop 1...k 
                # use flatten -> enable N-D dot product
                h[cidx(j), cidx(k)] = np.dot(q[cidx(j)].flatten(), y.flatten()) # h_j,k = q_j^T * y
                y = y - h[cidx(j), cidx(k)] * q[cidx(j)] # y = y - h_jk * q_j

            h[cidx(k + 1), cidx(k)] = np.linalg.norm(y) # h_k+1,k = ||y||

            if (h[cidx(k + 1), cidx(k)] != 0 and k != nmax_iter):
                q[cidx(k + 1)] = y / h[cidx(k + 1), cidx(k)] # q_k+1 = y / h_k+1,k

            beta    = np.zeros(nmax_iter + 1)
            beta[0] = np.linalg.norm(r)
            y       = np.linalg.lstsq(h, beta, rcond=None)[0]
            # g       = np.dot(np.asarray(q[:cidx(k)]).transpose(), y[:cidx(k)])
            g       = update_solution(x_sol, y[:cidx(k)], q[:cidx(k)])
            #x.append(x_sol + g)
            normr = np.linalg.norm(b-A(x_sol+g))
            if normr/normb < e:
                return x


        beta    = np.zeros(nmax_iter + 1) # vector m+1
        beta[0] = np.linalg.norm(r) # beta[0] = ||r||, e1
        y       = np.linalg.lstsq(h, beta, rcond=None)[0] # y = Least-squares of H tilde * y - ||b||*e1
        # g       = np.dot(np.asarray(q).transpose(), y)
        g       = update_solution(x_sol, y, q) # x = Q*y
        
        x_sol   = x_sol + g # update solution
        x.append(x_sol) # save iteration solution 
 
        # r = b - matmul_a(A, x_sol)
        r = b - A(x_sol)

        # Break out if the residual is lower than threshold
        if np.linalg.norm(r)/normb < e:
            break
    return x



def apply_givens(Q, v, k):
    """
    Apply the first k Givens rotations in Q to the vector v.
    Arguments
    ---------
        Q: list, list of consecutive 2x2 Givens rotations
        v: array, vector to apply the rotations to
        k: int, number of rotations to apply
    Returns
    -------
        v: array, that is changed in place.
    """

    for j in range(k):
        Qloc = Q[j]
        # TODO: why sp.dot and not np.dot?
        v[j:j+2] = sp.dot(Qloc, v[j:j+2])



def GMRES_R(A, b, x0, tol, max_outer, max_inner, restart=None):
    """
    Quick and dirty GMRES -- TODO: optimize mem footprint when going to larger
    systems.
    """

    X = x0

    # Defining xtype as dtype of the problem, to decide which BLAS functions
    # import.
    xtype = upcast(X.dtype, b.dtype)

    # Get fast access to underlying BLAS routines
    # dotc is the conjugate dot, dotu does no conjugation

    [lartg] = get_lapack_funcs(['lartg'], [X] )
    if np.iscomplexobj(np.zeros((1,), dtype=xtype)):
        [axpy, dotu, dotc, scal] =\
            get_blas_funcs(['axpy', 'dotu', 'dotc', 'scal'], [X])
    else:
        # real type
        [axpy, dotu, dotc, scal] =\
            get_blas_funcs(['axpy', 'dot', 'dot', 'scal'], [X])

    # Make full use of direct access to BLAS by defining own norm
    def norm(z):
        return np.sqrt(np.real(dotc(z, z)))

    # Defining dimension
    dimen = len(X)


    # TODOs for this function:
    # 1. list -> numpy.array <= better memory access
    # 2. don't append to lists -> prealoc and slice
    # 3. lapack replacemnt for matmul_a?
    # 4. clean up this function!
    # 3. add documentation -- this will probably never happen :P

    r = b - matmul_a(A, x0)

    normr = norm(r)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0

    iteration = 0
    x = list()

    # Here start the GMRES
    for outer in range(max_outer):
        # Preallocate for Givens Rotations, Hessenberg matrix and Krylov Space
        # Space required is O(dimen*max_inner).
        # NOTE:  We are dealing with row-major matrices, so we traverse in a
        #        row-major fashion,
        #        i.e., H and V's transpose is what we store.

        Q = []  # Initialzing Givens Rotations
        # Upper Hessenberg matrix, which is then
        # converted to upper triagonal with Givens Rotations

        H = np.zeros((max_inner + 1, max_inner + 1), dtype=xtype)
        V = np.zeros((max_inner + 1, dimen), dtype=xtype)  # Krylov space

        # vs store the pointers to each column of V.
        # This saves a considerable amount of time.
        vs = []

        # v = r/normr
        V[0, :] = scal(1.0/normr, r)  # scal wrapper of dscal --> x = a*x
        vs.append(V[0, :])

        # Saving initial residual to be used to calculate the rel_resid
        if iteration == 0:
            res_0 = normb
 
        # RHS vector in the Krylov space
        g    = np.zeros((dimen, ), dtype=xtype)
        g[0] = normr

        for inner in range(max_inner):
            # New search direction
            v    = V[inner+1, :]  # pointer!
            v[:] = matmul_a(A, vs[-1])
            vs.append(v)

            # Modified Gram Schmidt
            for k in range(inner+1):
                vk          = vs[k]
                alpha       = dotc(vk, v)
                H[inner, k] = alpha
                v[:]        = axpy(vk, v, dimen, -alpha)  # y := a*x + y
                #axpy is a wrapper for daxpy (blas function)

            normv = norm(v)
            H[inner, inner + 1] = normv

            # Check for breakdown
            if H[inner, inner + 1] != 0.0:
                v[:] = scal(1.0/H[inner, inner + 1], v)

            # Apply for Givens rotations to H
            if inner > 0:
                apply_givens(Q, H[inner, :], inner)

            # Calculate and apply next complex-valued Givens rotations

            # If max_inner = dimen, we don't need to calculate, this
            # is unnecessary for the last inner iteration when inner = dimen -1

            if inner != dimen - 1:
                if H[inner, inner + 1] != 0:
                    # lartg is a lapack function that computes the parameters
                    # for a Givens rotation
                    [c, s, _] = lartg(H[inner, inner], H[inner, inner + 1])
                    Qblock = np.array([[c, s], [-np.conjugate(s),c]], dtype=xtype)
                    Q.append(Qblock)

                    #Apply Givens Rotations to RHS for the linear system in
                    # the krylov space. TODO: why sp.dot and not np.dot?
                    g[inner:inner+2] = sp.dot(Qblock, g[inner:inner+2])

                    #Apply Givens rotations to H
                    H[inner, inner] = dotu(Qblock[0,:], H[inner, inner:inner+2])
                    H[inner, inner+1] = 0.0

            iteration+= 1

            if inner < max_inner-1:
                normr = abs(g[inner+1])
                rel_resid = normr/res_0

                if rel_resid < tol:
                    break

        # end inner loop, back to outer loop

        # Find best update to X in Krylov Space V.  Solve inner X inner system.
        y      = sp.linalg.solve(H[0:inner+1, 0:inner+1].T, g[0:inner+1])
        update = np.ravel(sp.mat(V[:inner+1, :]).T.dot(y.reshape(-1,1)))
        X      = X + update
        aux    = matmul_a(A, X)
        r      = b - aux
        
        normr = norm(r)
        rel_resid = normr/res_0

        x.append(X)

        # test for convergence
        if rel_resid < tol:
            print('GMRES solve')
            print(f'Converged after {iteration} iterations to a residual of {rel_resid}')
            return x

    #end outer loop

    return x
