import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from functools import partial
from typing import Callable

from jeigsh import jeigsh
from uwmanifeld import DiffusionMap


##################################
from tqdm import tqdm
import numpy as np
import time

def get_noise(f):
    f_ω = np.fft.fft(f[:,0])
    P_ω = (f_ω*jnp.conj(f_ω)).real

    f_Nyquist = f.shape[0]//2
    f_signal  = f_Nyquist//int(100000 * (f.shape[0]/2999872))
    noise     = jnp.sum(P_ω[f_signal:f_Nyquist]) / jnp.sum( P_ω[:f_Nyquist] )
    return noise

slices = np.logspace(2, 6.7781513, 100).astype(int)

c = 2**1 ## 2**10 #2**12
N = 100000

#R_ix = jax.numpy.load("saxs_highq_N6000000_D300.npy")
#R_ix = jax.numpy.load("saxs_highq_N600000_D300.npy")
R_ix = jax.numpy.load("SAXS_nS600k_D996.npy")[:,600:900]


n    = R_ix.shape[0]//N
R_ix = R_ix[0::n,:]

#R_ix = jnp.concatenate((R_ix, R_ix[:2500000,:]), axis=0)
print(R_ix.shape)
#R_ix = jax.numpy.load("SAXS_nS600k_D996.npy")[:,600:900]

#out_str = "_c" + str(c) + "_" + str(embedding.shape[0]) + "_" + str(embedding.shape[1])
#jax.numpy.save("embedding" + out_str, embedding)
 
cs = jnp.logspace(1, 4, 50, base=2).astype(int)
cs = jnp.array([2**2, 2**6, 2**12])
noise=[]
for c in tqdm(cs):
    embed = DiffusionMap(R_ix, k=8, c=c)
    noise.append( get_noise(embed) )
    jax.numpy.save("embed_E8_" + str(c), embed)