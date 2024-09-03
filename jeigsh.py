import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(0,1))
def jeigsh(LO, k=6, δ=1.E-14, Largest=True, key=jax.random.PRNGKey(0), DTYPE=jnp.float64):
    N=LO.shape[0]     ## matrix-size
    m = int((N)**(k/N)) + (1-k//N)*k*(N.bit_length()-1)
    #c = jax.lax.select(Largest, -1., 1.)

    ##########################################
    @jax.jit
    def Gram_Schmidt_(i, Data):
        V_, K = Data
        v     = K[i, :]
        V_   -= (jnp.conj(v) @ V_) * v
        return [ V_, K ]

    @jax.jit
    def Lanczos_( Data ):
        #### inputs
        V_, K, α, β, _, δ, i = Data

        #### compute
        V_  = V_.reshape(-1)
        β_  = jnp.sqrt( V_ @ V_ )
        V_ /= β_

        ### GRAM-SCHMIDT
        V_, K = jax.lax.fori_loop(0, i, Gram_Schmidt_, [V_, K])

        AV  = LO(V_)
        #AV *= c
        AV  = AV.reshape(-1)
        α_  = jnp.conj(V_) @ AV
        AV -= V_ * α_
        AV -= K[i-1,:] * β_

        K   = K.at[ i,: ].set(V_)
        β   = β.at[ i-1 ].set(β_)
        α   = α.at[ i-1 ].set(α_)

        return [ AV, K, α, β, β_, δ, i+1 ]

    @jax.jit
    def if_( Data ):
        """
        i (iteration)
        max_i (maxiteration)
        """
        _, K, _, _, β_, δ, i = Data

        def check_(check_values):
            β_, δ = check_values
            return jax.lax.cond(β_ < δ, False, lambda x:x, True, lambda x:x)
        return jax.lax.cond(i < K.shape[0], [β_, δ], check_, False, lambda x:x)

    ##########################################

    #### initialize data required for calculation
    v_0  = jax.random.uniform(key, (N,), dtype=DTYPE)  ## initial guess eigenvector
    kV   = jnp.zeros((m+1, N), dtype=DTYPE)            ## Krylov Vectors
    α    = jnp.zeros(m, dtype=DTYPE)                   ## diag of Lanczos Matrix
    β    = jnp.zeros(m, dtype=DTYPE)                   ## off-diag of Lanczos Matrix

    Data        = [v_0, kV, α, β, 1.0, δ, 1]
    v_f, kV, α, β, _, _, iter = jax.lax.while_loop(if_, Lanczos_, Data)
    kV          = kV.at[iter,:].set(v_f)
    kV          = kV[1:,:]

    #### Oh use LinearOperator for Tridaigonal??? restarted Lanczos!
    A_tridiag   = jnp.diag(α)
    A_tridiag  += jnp.diag( β[1:], 1)
    A_tridiag  += jnp.diag(jnp.conj(β[1:]), -1)

    #### get eigen-values/vectors & Unitary matrix (m x m) matrix, N -> m -> k?
    eigvals, U  = jnp.linalg.eigh(A_tridiag)       ## ~m^3, only need k
    eigvals     = eigvals.astype(A_tridiag.dtype)
    #eigvals    *= c
    #### get eigenvectors
    #eigvecs     = (U.T[:k,:] @ kV)
    eigvecs     = (U.T[-k:,:] @ kV)

    #return eigvals[:k], eigvecs
    return (eigvals[-k:]), eigvecs
