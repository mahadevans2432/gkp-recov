from choi import Choi
import numpy as np
import scipy
import sparse
import qutip as q

def choi_compound(chi1, chi2):
    """
    Returns the choi matrix chi2*chi1
    """
    chi_c = Choi(input_dim = chi1.input_dim, output_dim = chi2.output_dim)
    chi_c.four_tensor = sparse.einsum('ijkl,jmln->imkn',chi2.four_tensor,chi1.four_tensor)
    chi_c.matrix = chi_c.four_tensor.reshape((chi_c.input_dim*chi_c.output_dim,chi_c.input_dim*chi_c.output_dim))
    #
    return chi_c


def entanglement_fidelity(P, Q):
    #Finds entanglement fidelity between P and Q
    return np.real(sparse.einsum('ii',P._kraus_transposed().matrix@Q.matrix)/(Q.input_dim**2))

def get_random_encoding(d, cutoff):
        E = scipy.stats.unitary_group.rvs(cutoff)[:, :d]
        E_map = Choi(d,cutoff)
        E_map.Kraus_construct([E])
        return E_map

def get_random_encoding_lower(d, photon_cutoff, Delta):
        E = scipy.stats.unitary_group.rvs(photon_cutoff)[:, :d]
        decay = np.diagflat(np.exp([-i*Delta for i in range(0,photon_cutoff)]))
        E = decay@E
        U = scipy.linalg.cholesky(E.T.conj()@E)
        E = E@np.linalg.inv(U)
        E_map = Choi(d,photon_cutoff)
        E_map.Kraus_construct([E])
        return E_map

