import multiprocessing as mp
import numpy as np
import qutip as q
import sparse
from choi import *
from helper_functions import *
from choi_helper import *
from gaussian_helper import *
from gkp_helper import *


def task(gamma):
    N = 100
    loss = boson_loss_channel(gamma, N)
    sparse.save_npz(f'pure_loss_noise/loss_{N}_{gamma}.npz', loss.four_tensor)

gammas = [0.01, 0.02, 0.03,0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
          0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
          0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28,
          0.29, 0.3]
with mp.Pool(mp.cpu_count()) as pp:
    pp.map(task,gammas)