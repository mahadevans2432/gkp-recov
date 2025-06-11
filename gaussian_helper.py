import numpy as np
import qutip as q
from choi import *
from helper_functions import *
from choi_helper import *
import sparse

def beamsplitter(N,eta):
    """
    Unitary for a beamsplitter with transmissivity eta.
    Defined by transformation a1 -> sqrt(eta)*a1 + sqrt(1-eta)*a2, a2 -> -sqrt(1-eta)*a1 + sqrt(eta)*a2

    Parameters:
        N (int): Photon number cutoff for Hilbert space
        eta (float): Transmissivity of the beamsplitter (0 <= eta <= 1)
    
    Returns:
        qutip.Qobj: Unitary operator representing the beamsplitter
    """
    a = q.destroy(N)
    return (np.arcsin(np.sqrt(eta))*(q.tensor(a,a.dag()) - q.tensor(a.dag(),a))).expm()

def twoModeSq(N, r):
    """
    Unitary for a two-mode squeezing operation with squeezing parameter r.
    Defined by transformation x1+x2 -> exp(r)*(x1+x2), p1+p2 -> exp(-r)*(p1+p2)

    Parameters:
        N (int): Photon number cutoff for Hilbert space
        r (float): Squeezing paramter (complex number)
    
    Returns:
        qutip.Qobj: Unitary operator representing the 2 mode squeezing operation
    """

    a = q.destroy(N)
    return(r*q.tensor(a,a) - np.conj(r)*q.tensor(a.dag(),a.dag())).expm()

def boson_loss_channel(gamma,photon_cutoff):
    a = q.destroy(photon_cutoff)
    ad = q.create(photon_cutoff)
    U = (np.arcsin(np.sqrt(gamma))*(q.tensor(a,ad) - q.tensor(ad,a))).expm()
    N = Choi(photon_cutoff, photon_cutoff)
    env = q.fock_dm(photon_cutoff,0)
    for i in range(photon_cutoff):
        for j in range(i+1):
            ij = q.fock(photon_cutoff,i)*q.fock(photon_cutoff,j).dag()
            add = sparse.kron(sparse.asarray((q.ptrace(U*q.tensor(ij,env)*U.dag(),0)).full()),sparse.asarray(ij.full()))/4
            N.matrix += add + add.T.conj()
    N.four_tensor = N.matrix.reshape((photon_cutoff,photon_cutoff,photon_cutoff,photon_cutoff))
    return N

def boson_loss_channel_alt(gamma,photon_cutoff):
    a = q.destroy(photon_cutoff)
    ad = q.create(photon_cutoff)
    U = (np.arcsin(np.sqrt(gamma))*(q.tensor(a,ad) - q.tensor(ad,a))).expm()
    N = Choi(photon_cutoff, photon_cutoff)
    vac = q.fock(photon_cutoff,0)
    for i in range(photon_cutoff):
        kraus = q.tensor(q.identity(photon_cutoff),q.fock(photon_cutoff,i).dag())*U*q.tensor(q.identity(photon_cutoff),vac)
        vec_kraus = sparse.asarray(kraus.full().reshape((photon_cutoff*photon_cutoff,1)))
        N.matrix += vec_kraus@vec_kraus.T.conj()
    N.four_tensor = N.matrix.reshape((photon_cutoff,photon_cutoff,photon_cutoff,photon_cutoff))
    return N

def boson_deph_channel(gammad,photon_cutoff):
    a = q.destroy(photon_cutoff)
    ad = q.create(photon_cutoff)
    U = (np.sqrt(gammad)*q.tensor(ad*a,ad-a)).expm()
    N = Choi(photon_cutoff, photon_cutoff)
    env = q.fock_dm(photon_cutoff,0)
    for i in range(photon_cutoff):
        for j in range(i+1):
            ij = q.fock(photon_cutoff,i)*q.fock(photon_cutoff,j).dag()
            add = sparse.kron(sparse.asarray((q.ptrace(U*q.tensor(ij,env)*U.dag(),0)).full()),sparse.asarray(ij.full()))/4
            N.matrix += add + add.T.conj()
    N.four_tensor = N.matrix.reshape((photon_cutoff,photon_cutoff,photon_cutoff,photon_cutoff))
    return N

def boson_deph_channel_alt(gammad,photon_cutoff):
    a = q.destroy(photon_cutoff)
    ad = q.create(photon_cutoff)
    U = (np.sqrt(gammad)*q.tensor(ad*a,ad-a)).expm()
    N = Choi(photon_cutoff, photon_cutoff)
    vac = q.fock(photon_cutoff,0)
    for i in range(photon_cutoff):
        kraus = q.tensor(q.identity(photon_cutoff),q.fock(photon_cutoff,i).dag())*U*q.tensor(q.identity(photon_cutoff),vac)
        vec_kraus = sparse.asarray(kraus.full().reshape((photon_cutoff*photon_cutoff,1)))
        N.matrix += vec_kraus@vec_kraus.T.conj()
    N.four_tensor = N.matrix.reshape((photon_cutoff,photon_cutoff,photon_cutoff,photon_cutoff))
    return N