import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from functools import partial
from typing import Callable

from jeigsh import jeigsh

def DiffusionMap(R_ix, k=2, c:int=10, σ2=2., α=0.5, seed:int=777, NLSA:bool=False):

    key=jax.random.PRNGKey(seed)
    κ=jnp.ones(c)
    N=R_ix.shape[0]
    s=1
    c=κ.size

    y           = int((N - c) // s + 1) ## result size y (vert.)
    left_over   = (N-c) % s ## on the left-side
    window_size = 1 + (y-1)*s ## to stride over

    end   = N - left_over
    start = end - window_size

    D  = (jnp.ones((R_ix.shape[0]-c+1))) #**(-0.5)
    #@jax.jit
    def mv(v, s=s, start=start, end=end):
        v *= D

        B     = κ[0] * (R_ix[ start:end:s , :] @ (R_ix.T[: , start:end:s] @ v))
        for i in range(1, c):
            start-=1
            end  -=1
            B += κ[i]* (R_ix[ start:end:s , :] @ (R_ix.T[: , start:end:s] @ v))

        v *= len(κ) ## the identity
        v += 2*B

        v *= D
        return v

    D += mv(D)**(-α)
    def MV(v, s=s, start=start, end=end):
        v *= D

        B     = κ[0] * (R_ix[ start:end:s , :] @ ( R_ix.T[: , start:end:s] @ v))
        for i in range(1, c):
            start-=1
            end  -=1
            B += κ[i]* (R_ix[ start:end:s , :] @ ( R_ix.T[: , start:end:s] @ v))

        v *= len(κ) ## the identity
        v += 2*B

        v *= D
        return v

    #eigenvalues, eigenvectors = jeigsh(Δ_LO, k=6, N=R_ix.shape[0])
    v_0  = jax.random.uniform(key, (R_ix.shape[0]-c+1,), dtype=jnp.float64)
    eigenvalues, eigenvectors = jeigsh(MV, v_0, k=k) #, N=(R_ix.shape[0]-c+1))
    idx          = jnp.argsort(eigenvalues)[::-1] ## sort the eigenvalues
    eigenvalues  = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    embedding    = eigenvectors[:, 1:(1+k)]

    if NLSA:
        def SVD(v, s=s, start=start, end=end):
            B     = embedding.T @ (R_ix[ start:end:s , :] @ ( R_ix.T[: , start:end:s] @ (embedding @ v)))
            for i in range(1, c):
                start-=1
                end  -=1
                B += embedding.T @ (R_ix[ start:end:s , :] @ ( R_ix.T[: , start:end:s] @ (embedding @ v)))
            return B

        v_00  = jax.random.uniform(key, (embedding.shape[1],), dtype=jnp.float64)
        eigenvalues_, eigenvectors_ = jeigsh(SVD, v_00, k=v_00.shape[0])

        return embedding, [eigenvalues_, eigenvectors_]

    return embedding