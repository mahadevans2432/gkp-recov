import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import qutip as q
#from process_QECmat_SCS import process_QECmat
plt.rcParams['font.size'] = 45
plt.rcParams['legend.fontsize'] = 40
plt.rcParams['font.family'] = 'serif'
plt.rcParams["figure.figsize"] = (36,18)
plt.rcParams['hist.bins'] = 500
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['mathtext.fontset'] = 'stix'
# plt.style.use('seaborn-pastel')
tol = 1e-16

# The vec function as documented in api/cones
def vec(S):
    n = S.shape[0]
    S = np.copy(S)
    S *= np.sqrt(2)
    S[range(n), range(n)] /= np.sqrt(2)
    return S[np.triu_indices(n)]

def vec_alt(S):
    #Function for just the constraints
    n = S.shape[0]
    S = np.copy(S)
    S *= 1/np.sqrt(2)
    S[range(n), range(n)] /= 1/np.sqrt(2)
    return S[np.triu_indices(n)]


# The mat function as documented in api/cones
def mat(s):
    n = int((np.sqrt(8 * len(s) + 1) - 1) / 2)
    S = np.zeros((n, n))
    S[np.triu_indices(n)] = s / np.sqrt(2)
    S = S + S.T
    S[range(n), range(n)] /= np.sqrt(2)
    return S

def complex_vec(S):
    return np.real(vec(np.block([[np.real(S),-np.imag(S)],
                         [np.imag(S), np.real(S)]]))/np.sqrt(2))

def complex_mat(S):
    block = mat(S)
    n = block.shape[0]//2
    X = np.zeros((n,n),dtype=np.complex128)
    X = block[:n,:n] + 1j*block[n:,:n]
    return np.sqrt(2)*X
  
def np_fock(dim,ind):
    fock = np.zeros((dim,1),dtype=np.complex128)
    fock[ind] = 1
    return fock

def log(x):
    if abs(x) < 1e-16:
        return 0
    return np.log(x)


def np_coherent(dim,alpha):
    coh = np.zeros((dim,1),dtype=np.complex128)
    factorial = 1.0
    alpha_pow = 1.0
    for i in range(dim):
        coh[i] = alpha_pow/np.sqrt(factorial)
        alpha_pow *= alpha
        factorial *= i+1
    return coh/(tol+np.linalg.norm(coh))

def wigner_plot(rho):
    xvec = np.linspace(-8, 8, 500)
    W = q.wigner(rho/np.trace(rho.full()), xvec, xvec)
    #wmap = q.wigner_cmap(W)  # Generate Wigner colormap
    nrm = mpl.colors.Normalize(-0.25, 0.25)
    fig, axes = plt.subplots(1, 1, figsize=(30, 24))
    plt1 = axes.contourf(xvec, xvec, W, 100, cmap='RdBu', norm=nrm)
    cb1 = fig.colorbar(plt1, ax=axes)
    fig.tight_layout()
    plt.show()


def support(M, tol=1e-11):
    u,s,vh = np.linalg.svd(M)
    s = np.diagflat(s)
    return u@s@np.linalg.pinv(s,rcond=tol)@u.T.conj()


def transform_dual_rec(y_in, T_in, T_out, d = 2):
    #only works if T_in and T_out have same shape
    n = T_in.shape[0]
    constraints = 2*(n**2)
    Ys_in = complex_mat(y_in[constraints:])
    T = (T_out@T_in.T.conj()).conj()
    Y = T@((y_in[[i for i in range(0,constraints,2)]] + 
         1j*y_in[[i for i in range(1,constraints,2)]]).reshape((n,n)))@T.T.conj()
    transform = np.kron(np.identity(d),T)
    Ys_out = transform@Ys_in@transform.T.conj()
    Y_flat = []
    for i in range(n):
        for j in range(n):
            Y_flat.append(np.real(Y[i,j]))
            Y_flat.append(np.imag(Y[i,j]))
    y_out = np.append(np.array(Y_flat),complex_vec(Ys_out))
    return y_out

def transform_dual_enc(y_in, T_in, T_out, d = 2):
    constraints = 2*(d**2) + 1 #These stay the same
    Ys_in = complex_mat(y_in[constraints:])
    transform = np.kron((T_out@T_in.T.conj()),np.identity(d))
    Ys_out = transform@Ys_in@transform.T.conj()
    y_out = np.append(y_in[:constraints],complex_vec(Ys_out))
    return y_out