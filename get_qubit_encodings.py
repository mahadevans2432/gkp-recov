import multiprocessing as mp
import numpy as np
import qutip as q
import sparse
from choi import *
from helper_functions import *
from choi_helper import *
from gaussian_helper import *
from gkp_helper import *


def task(delta):
    N = 100
    gkp = gkp_encoding(N,delta)
    sparse.save_npz(f'gkp_encoding_{N}_{delta}.npz', gkp.four_tensor)

deltas = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
with mp.Pool(mp.cpu_count()) as pp:
    pp.map(task,deltas)