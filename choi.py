import numpy as np
import sparse
import scipy
from helper_functions import *
class Choi:
    """
    Data structure for Choi matrices
    Stores the rank-4 tensor form and matrix form
    """

    def __init__(self, input_dim, output_dim):
        self.four_tensor = sparse.asarray(np.zeros((output_dim,input_dim,output_dim,input_dim),dtype=np.complex128))
        self.matrix = sparse.asarray(np.zeros((output_dim*input_dim,output_dim*input_dim),dtype=np.complex128))
        self.input_dim = input_dim
        self.output_dim = output_dim

    def load_npz(self, path):
        output_dim = self.output_dim
        input_dim = self.input_dim
        self.matrix = sparse.load_npz(path).reshape((output_dim*input_dim,output_dim*input_dim))
        self.four_tensor = self.matrix.reshape((output_dim,input_dim,output_dim,input_dim))

    def load_npy(self, path):
        output_dim = self.output_dim
        input_dim = self.input_dim
        self.matrix = sparse.asarray(np.load(path).reshape((output_dim*input_dim,output_dim*input_dim)))
        self.four_tensor = self.matrix.reshape((output_dim,input_dim,output_dim,input_dim))

    def act_on(self, rho):
        return sparse.asnumpy(sparse.einsum('ijkl,jl->ik',self.four_tensor,sparse.asarray(rho)))
    
    def trace_check(self):
        #print(np.linalg.eigvalsh(sparse.asnumpy(sparse.einsum('ijik->jk',self.four_tensor))))
        return np.linalg.norm(sparse.asnumpy(sparse.einsum('ijik->jk',self.four_tensor)) - np.identity(self.input_dim))

    def Kraus_construct(self, Kraus):
        self.matrix = sparse.asarray(np.zeros((self.output_dim*self.input_dim,self.output_dim*self.input_dim),dtype=np.complex128))
        #kdagk = np.zeros((self.input_dim,self.input_dim),dtype=np.complex128)
        for K in Kraus:
            Kvec = K.reshape((K.shape[0]*K.shape[1],1))
            #kdagk += K.T.conj()@K
            self.matrix += sparse.asarray(Kvec@Kvec.T.conj())
        self.four_tensor = self.matrix.reshape((self.output_dim,self.input_dim,self.output_dim,self.input_dim))
        #print(np.linalg.norm(np.trace(self.four_tensor,axis1=0,axis2=2) - kdagk))
    
    def basis_change_output(self, T):
        trans_choi = Choi(self.input_dim,T.shape[0])
        trans_choi.matrix = sparse.kron(T,sparse.eye(self.input_dim))@self.matrix@(sparse.kron(T,sparse.eye(self.input_dim)).T.conj())
        trans_choi.four_tensor = trans_choi.matrix.reshape(shape=trans_choi.four_tensor.shape)
        #trans_choi._renormalize()
        return trans_choi

    def basis_change_input(self, T):
        trans_choi = Choi(T.shape[0],self.output_dim)
        trans_choi.matrix = sparse.kron(sparse.eye(self.output_dim),T.conj())@self.matrix@(sparse.kron(sparse.eye(self.output_dim),T.conj()).T.conj())
        trans_choi.four_tensor = trans_choi.matrix.reshape(shape=trans_choi.four_tensor.shape)
        #trans_choi._renormalize()
        return trans_choi
    
    def _kraus_transposed(self):
        #Returns the choi matrix with Kraus operators being transposed
        transposed = Choi(self.output_dim,self.input_dim)
        transposed.four_tensor = sparse.einsum('ijkl->lkji',self.four_tensor)
        transposed.matrix = transposed.four_tensor.reshape((self.input_dim*self.output_dim,self.input_dim*self.output_dim))
        return transposed
    
    def transpose(self,tol=1e-7):
        #Rework this function
        trans_choi = self._kraus_transposed()
        projection = sparse.asnumpy(sparse.einsum('ijkj->ik',self.four_tensor))
        projection = (projection + projection.T.conj())/2
        # For finding inverse of projection on its own support
        s, u = np.linalg.eigh(projection)
        s0 = np.amax(s)
        support1 = u@np.diagflat([0j+bool(abs(i)>=tol*s0) for i in s])@u.T.conj()
        inverse = u@np.diagflat([0j+bool(abs(i)>=tol*s0)*np.sqrt(1/max(abs(i),1e-16)) for i in s])@u.T.conj()
        #np.linalg.pinv(eig@np.diagflat(np.sqrt(lam+0j))@eig.T.conj(),rcond=1e-10,hermitian=True)
        normalization = sparse.kron(sparse.eye(trans_choi.output_dim),inverse.conj())
        trans_choi.matrix = normalization@trans_choi.matrix@normalization.T.conj()
        trans_choi.matrix += sparse.kron(sparse.eye(self.input_dim),(sparse.eye(self.output_dim) - sparse.asarray(support1.conj())))/self.input_dim
        trans_choi.four_tensor = trans_choi.matrix.reshape((trans_choi.output_dim,trans_choi.input_dim,trans_choi.output_dim,trans_choi.input_dim))
        return trans_choi
    
    def transpose_choi(self):
        dimL = self.dimL
        d = self.d
        trans_choi = np.matrix(np.zeros([dimL*d*d,dimL*d*d]))
        for i in range(dimL):
            for mu in range(d):
                for mup in range(d):
                    trans_choi[mu*d*dimL+i*d+mu,mup*d*dimL+i*d+mup] = 1
        return trans_choi

    
    def approx_isometry(self, return_isometry = False):
        #Takes the most prominent Kraus operator and assumes it to be isometric
        isomet = Choi(self.input_dim,self.output_dim)
        _, eig = scipy.sparse.linalg.eigsh(self.matrix.tocsr(),k=2)
        n = self.input_dim*self.output_dim
        if return_isometry:
            return eig[:,0].reshape((self.output_dim,self.input_dim))
        isomet.matrix = sparse.asarray(2*(eig[:,0].reshape((n,1)))@(eig[:,0].reshape((n,1))).T.conj())
        isomet.four_tensor = isomet.matrix.reshape((self.output_dim,self.input_dim,self.output_dim,self.input_dim))
        return isomet
    
    def force_isometry(self):
        U_enc = self.approx_isometry(True)
        U = scipy.linalg.cholesky(U_enc.T.conj()@U_enc)
        E = U_enc@np.linalg.inv(U)
        isomet = Choi(self.input_dim,self.output_dim)
        isomet.Kraus_construct([E])
        return isomet


    def _renormalize(self):
        #Renormalization assuming partial trace of C is idempotent but not identity
        partial_trace = sparse.einsum('ijil->jl',self.four_tensor)
        lam, eig = np.linalg.eigh(sparse.asnumpy(partial_trace))
        remainder = 0*self.matrix
        for i in range(len(lam)):
            vec = (np_fock(self.output_dim,0)@eig[:,i,np.newaxis].T.conj()).reshape((self.input_dim*self.output_dim,1))
            remainder += (1-lam[i])*sparse.asarray(vec@vec.T.conj())
        self.matrix += remainder
        self.four_tensor = self.matrix.reshape((self.output_dim,self.input_dim,self.output_dim,self.input_dim))

    def _renormalize_alt(self):
        #Works only for same input output
        partial_trace = sparse.einsum('ijil->jl',self.four_tensor)
        lam, eig = np.linalg.eigh(sparse.asnumpy(partial_trace))
        print(lam)
        remainder = 0*self.matrix
        for i in range(len(lam)):
            vec = (eig[:,i,np.newaxis]@eig[:,i,np.newaxis].T.conj()).reshape((self.input_dim*self.output_dim,1))
            remainder += (1-lam[i])*sparse.asarray(vec@vec.T.conj())
        self.matrix += remainder
        self.four_tensor = self.matrix.reshape((self.output_dim,self.input_dim,self.output_dim,self.input_dim))
            
    def get_transform_kraus(self, L, inv_backaction):
        d = self.input_dim
        n = self.output_dim
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(self.matrix.tocsr(),k=L)
        Kraus = []
        for i in range(0,L):
            if abs(eigenvalues[-i]) <= 1e-7:
                break
            Kraus.append(inv_backaction@eigenvectors[:,i].reshape((n,d))*np.sqrt(eigenvalues[i]))
        L = len(Kraus)
        M = np.zeros((d*L,d*L),dtype=np.complex128)
        for l1 in range(L):
            for l2 in range(L):
                l1l2 = np.zeros((L,L))
                l1l2[l2,l1] = 1
                M += np.kron(l1l2,Kraus[l2].T.conj()@Kraus[l1])
        
        U, S, Vh = scipy.linalg.svd(M)
        T_tilde = np.diagflat(1/np.sqrt(S+0j))@Vh
        kraus_states = []
        for i in range(L):
            for mu in range(d):
                kraus_states.append((Kraus[i]@np_fock(d,mu)).reshape(n,).conj())
        kraus_states = np.vstack(tuple(kraus_states))
        T = T_tilde@kraus_states
        return (T, M, d*L)
    
    def get_transform(self, backaction, add_proj=0, rcond=1e-11, force_rank=None):
        inv_backaction = np.linalg.pinv(backaction)
        mixed_state = add_proj + inv_backaction@sparse.asnumpy(sparse.einsum('ijkj->ik',self.four_tensor))@inv_backaction.T.conj()
        u, s, vh = np.linalg.svd(mixed_state)
        if force_rank != None:
            rank = min(force_rank,np.linalg.matrix_rank(np.diagflat(s)/s[0], tol=rcond))
        else:
            rank = np.linalg.matrix_rank(np.diagflat(s)/s[0], tol=rcond)
        T = u[:,:rank].T.conj()
        proj = T.T.conj()@T
        return T, proj, rank
    
    def get_transform_backaction(self, N, projector, backaction, tol):
        inv_backaction = np.linalg.pinv(backaction)
        mixed_state = sparse.asnumpy(sparse.einsum('ijkj->ik',self.four_tensor))
        u, s, vh = np.linalg.svd(mixed_state)
        try_rank = np.linalg.matrix_rank(projector)
        lower = 0
        higher = try_rank
        try_rank = try_rank//2
        while True:
            old_try_rank = try_rank
            u1, s1, _ = np.linalg.svd(inv_backaction@u[:,:try_rank]@u[:,:try_rank].T.conj()@inv_backaction.T.conj())
            u11, _, _ = np.linalg.svd(inv_backaction@u[:,:try_rank+1]@u[:,:try_rank+1].T.conj()@inv_backaction.T.conj())
            P1 = u1[:,:try_rank]@u1[:,:try_rank].T.conj()
            P11 = u11[:,:try_rank+1]@u11[:,:try_rank+1].T.conj()
            NP1 = N.act_on(P1)
            NP11 = N.act_on(P11)
            dif1 = np.linalg.norm(projector@NP1 - NP1)
            dif2 = np.linalg.norm(projector@NP11 - NP11)
            print(try_rank, dif1, dif2)
            if dif1 <= tol and dif2 >= tol:
                break

            if dif2 <= tol:
                try_rank = (higher + try_rank)//2
                lower = old_try_rank

            if dif1 >= tol:
                try_rank = (lower + try_rank)//2
                higher = old_try_rank

            if try_rank == 1 or lower == higher:
                break
        T = u[:,:try_rank].T.conj()
        return T, T.T.conj()@T, try_rank
    
    def get_transform_limited(self, L, gamma, channel='loss',inv_backaction=False, rcond=1e-11, force_rank = None):
        d = self.input_dim
        n = self.output_dim
        Pe = self.act_on(np.identity(d))
        state = 0*Pe
        if channel == 'loss':
            N0 = np.diagflat([(1-gamma)**(i/2) for i in range(n)])
            K = q.destroy(n).full()
        if channel == 'deph':
            N0 = np.diagflat([np.exp(-gamma*i*i/2) for i in range(n)])
            K = np.diagflat([i for i in range(n)])
        for i in range(L):
            if inv_backaction:
                operator = np.linalg.matrix_power(K,i)
            else:
                operator = N0@np.linalg.matrix_power(K,i)
            state += (np.power(gamma,i)/scipy.special.factorial(i))*operator@Pe@operator.T.conj()
        U, S, Vh = scipy.linalg.svd(state)
        S_sqrtinv = np.sqrt(np.linalg.pinv(np.diagflat(S),rcond=rcond,hermitian=True))
        rank = np.linalg.matrix_rank(S_sqrtinv/np.amax(S_sqrtinv))
        if force_rank != None:
            rank = min(rank,force_rank)
        T = U[:rank,:]
        return (T, state, rank)

    def reverse_output_basis_change(self, T):
        reverse = Choi(self.input_dim,T.shape[1])
        T = sparse.asarray(T)
        reverse.matrix = sparse.kron(T.T.conj(),sparse.eye(self.input_dim))@self.matrix@(sparse.kron(T.T.conj(),sparse.eye(self.input_dim)).T.conj())
        reverse.four_tensor = reverse.matrix.reshape(reverse.four_tensor.shape)
        return reverse
    
    def reverse_input_basis_change(self, T, normalize=True):
        reverse = Choi(T.shape[1],self.output_dim)
        T = sparse.asarray(T)
        reverse.matrix = sparse.kron(sparse.eye(self.output_dim),T.T)@self.matrix@(sparse.kron(sparse.eye(self.output_dim),T.T).T.conj())
        if normalize:
            reverse.matrix += sparse.kron(sparse.eye(self.output_dim),
                                          (sparse.eye(T.shape[1]) - T.T.conj()@T).conj())/self.output_dim
        reverse.four_tensor = reverse.matrix.reshape(reverse.four_tensor.shape)
        return reverse
    
    def rank(self, tol=1e-11):
        proj = sparse.asnumpy(sparse.einsum('ijkj->ik',self.four_tensor))
        return np.linalg.matrix_rank(proj,tol=tol)
    