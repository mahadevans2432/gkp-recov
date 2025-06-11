import numpy as np
import matplotlib.pyplot as plt
import qutip as q
from choi import *
from helper_functions import *
from choi_helper import *

def gkp_qunaught_state(N, delta):
    """
    Constructs an approximate GKP qunaught state in Fock basis.
    Uses equation (4) from PhysRevA.93.012315
    
    Parameters:
        N_cutoff (int): Photon number cutoff for the Hilbert space.
        delta (float): Envelope width (squeezing parameter).
    
    Returns:
        qutip.Qobj: The GKP qunaught density matrix in the Fock basis
    """
    state = 0*q.fock(N, 0)
    m = int(2/(delta*np.sqrt(np.pi)))
    r = -np.log(delta)

    for i in range(-m,m+1):
        # Shifted position eigenstate
        sq_state = np.exp(-np.pi*delta*delta*i*i)*q.displace(N,np.sqrt(np.pi)*i)*q.squeeze(N,r)*q.fock(N, 0)
        state += sq_state

    state = (state).unit()
    return state*state.dag()


def gkp_encoding(N, delta):
    """
    Constructs an approximate GKP encoding in Fock basis
    Uses equation (4) from PhysRevA.93.012315
    
    Parameters:
        N_cutoff (int): Photon number cutoff for the Hilbert space.
        delta (float): Envelope width (squeezing parameter).
    
    Returns:
        Choi (Choi object): The GKP encoding channel in the Fock basis
    """
    ket0 = 0*q.fock(N, 0)
    ket1 = 0*q.fock(N, 0)
    m = int(4/(delta*np.sqrt(np.pi)))
    r = -np.log(delta)

    for i in range(-m,m+1):
        # Shifted position eigenstate
        if i%2==0:
            ket0 += np.exp(-np.pi*delta*delta*i*i/2)*q.displace(N,np.sqrt(np.pi/2)*i)*q.squeeze(N,r)*q.fock(N, 0)
        else:
            ket1 += np.exp(-np.pi*delta*delta*i*i/2)*q.displace(N,np.sqrt(np.pi/2)*i)*q.squeeze(N,r)*q.fock(N, 0)

    ket0 = (ket0).unit()
    ket1 = (ket1 - ket0*(ket0.dag()*ket1)).unit()
    U_enc = ket0*(q.fock(2, 0).dag()) + ket1*(q.fock(2, 1).dag())
    encoding = Choi(2, N)
    encoding.Kraus_construct([U_enc.full()])
    return encoding