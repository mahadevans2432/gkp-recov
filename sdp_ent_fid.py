import numpy as np
import scipy
import sparse
import cvxpy as cp
import scs
from choi import Choi
from choi_helper import *
from helper_functions import *

class SDP_ent_fid:
    """
    Finds the optimal choi matrix of channel P
    given choi matrix of channel Q as input.
    P = argmax_P F_e(P◦Q)
    Q_choi is a scipy.sparse matrix
    """

    def __init__(self, Q_choi, n, d, T, n_constraint, scs_init=True, verbose = False):
        self.Q_choi = Q_choi
        self.n = n    # Q output dimension
        self.d = d # Q input dimension
        self.scs_solver = None
        self.T = T
        self.verbose = verbose
        self.n_constraint = n_constraint
        self.cost_choi = self.Q_choi._kraus_transposed()
        #self.SDP_set_prob()
        if scs_init:
            self.update_SCSprob(fresh_start=True)
            #self.update_SCSprob_old(fresh_start=True)
    

    # compute transpose fidelity with equation 
    def transpose_infid(self,M,m):
        dimL = M.shape[0]/self.d
        d = self.d
        u, s, vh = np.linalg.svd(m, full_matrices=True)
        U = np.diag(np.array(s)**(-0.5))@vh
        U = np.kron(np.eye(self.l_cut), U)
        M = U@M@U.transpose()
        Msqrt = scipy.linalg.sqrtm(M)
        ptrMsqrt = Msqrt.reshape([dimL,d,dimL,d])
        ptrMsqrt = np.matrix([[ptrMsqrt[i,0,j,0]+ptrMsqrt[i,1,j,1] for j in range(dimL)] for i in range(dimL)])
        fid = (1/self.d**2)*np.trace(ptrMsqrt@ptrMsqrt.transpose())
        return 1-fid
    
    # find an optimized recovery with SDP
    # use phonon number basis
    def SDP_set_prob(self):
        n = self.n
        d = self.d
        self.R = cp.Variable((d*n,d*n), complex=True)
        Q = cp.Constant(sparse.asnumpy(self.cost_choi.matrix))

        # compute partial trace of R
        constraints = [self.R >> 0, (self.R == self.R.H)]
        #partial_trace = cp.partial_trace(self.R,(d,n),axis=0)
        
        identMat = np.eye(n)
        #constraints.append((partial_trace == cp.Constant(identMat)))
        for i in range(n):
            for j in range(n):
                cons = (np_fock(n,i)@np_fock(n,j).T)
                cons = np.kron(np.eye(d),cons)
                constraints += [cp.trace(cp.transpose(cons)@self.R)==identMat[i,j]] 
        
        if n < d:
            n_op = self.T@np.diagflat([i for i in range(self.T.shape[1])])@self.T.T.conj()
            for i in range(n):
                constraints.append(cp.real(cp.trace(cp.kron(n_op,cp.Constant(np_fock(n,i)@np_fock(n,i).T.conj()))@self.R)) <= self.n_constraint)
        self.prob = cp.Problem((1/min(d,n)**2)*cp.Maximize(cp.real(cp.trace(Q@self.R))),constraints)
        return
    # generate choi matrix for transpose channel
    
    def change_choi(self, Q_choi):
        self.Q_choi = Q_choi
        self.cost_choi = self.Q_choi._kraus_transposed()
        self.update_SCSprob(fresh_start=False)
        
    
    def update_SCSprob(self, fresh_start = False, tol = 1.0e-7):
        n = int(self.n)
        d = int(self.d)
        flat = n*d*(2*n*d + 1)

        if (fresh_start or (self.scs_solver == None)):
            A = []
            b = []
            # compute partial trace of R
            identMat = np.eye(n)
            for i in range(n):
                for j in range(n):
                    cons_real = (np_fock(n,i)@np_fock(n,j).T + np_fock(n,j)@np_fock(n,i).T)/2
                    cons_imag = 1j*(np_fock(n,i)@np_fock(n,j).T - np_fock(n,j)@np_fock(n,i).T)/2
                    A.append(scipy.sparse.csr_array(complex_vec(np.kron(np.eye(d),cons_real)).reshape((1,flat))))
                    b.append(identMat[i,j])
                    A.append(scipy.sparse.csr_array(complex_vec(np.kron(np.eye(d),cons_imag)).reshape((1,flat))))
                    b.append(0)
            if n<d:
                number_constraint = self.T@np.diagflat([i for i in range(self.T.shape[1])])@self.T.T.conj()
                A.append(scipy.sparse.csr_array(complex_vec(np.kron(number_constraint,np.identity(n))).reshape((1,flat))))
                b.append(n*self.n_constraint)
                # for i in range(n):
                #     A.append(complex_vec(np.kron(number_constraint,np_fock(n,i)@np_fock(n,i).T.conj())).reshape((1,flat)))
                #     b.append(self.n_constraint)
            # solve SDP
            self.c = complex_vec(-sparse.asnumpy(self.cost_choi.matrix))/(min(d,n)**2)
            self.b = np.real(np.array(b + [0]*len(self.c)))
            self.A = scipy.sparse.vstack([scipy.sparse.vstack(A, format='csc'),
                                   -scipy.sparse.eye(len(self.c),format="csc")],
                                    format="csc")
            if n>d:
                self.warm_x = complex_vec(sparse.asnumpy(self.Q_choi.transpose().matrix)) #Primal for transpose recovery
                self.warm_s = self.b - self.A@self.warm_x
                self.warm_y = scipy.sparse.linalg.lsqr(scipy.sparse.vstack([self.A.T,(self.warm_s.T).reshape(1,len(self.warm_s))]), np.append(-self.c,[0]))[0] #Dual for transpose recovery
            data = dict(A=self.A,b=self.b,c=self.c)
            if n<d:
                cone = dict(z=len(self.b)-len(self.c)-1,l=1,s=2*n*d)
            else:
                cone = dict(z=len(self.b)-len(self.c),s=2*n*d)
            self.scs_solver = scs.SCS(data=data, cone=cone, verbose=self.verbose, eps_abs=tol, eps_rel=tol)
        
        else:
            c_new = complex_vec(-sparse.asnumpy(self.cost_choi.matrix))/(min(d,n)**2)
            self.c = c_new
            self.scs_solver.update(c=c_new)


    def solve(self,warm=True):
        #Does the solving of the SCS problem
        if warm:
            sol = self.scs_solver.solve(warm_start = True, x = self.warm_x, y = self.warm_y, s = self.warm_s)
        else:
            sol = self.scs_solver.solve()
        self.P_choi = Choi(self.n,self.d)
        self.P_choi.matrix = sparse.asarray(complex_mat(sol['x']))
        self.P_choi.four_tensor = self.P_choi.matrix.reshape((self.d,self.n,self.d,self.n))
        return sol
    
    def solve_cvx(self):
        self.prob.solve(solver='MOSEK')
        self.P_choi = Choi(self.n,self.d)
        self.P_choi.matrix = self.R.value
        self.P_choi.four_tensor = self.P_choi.matrix.reshape(self.d,self.n,self.d,self.n)


class SDP_ent_fid_real:
    """
    Finds the optimal choi matrix of channel P
    given choi matrix of channel Q as input.
    P = argmax_P F_e(P◦Q)
    Q_choi is a scipy.sparse matrix
    """

    def __init__(self, Q_choi, n, d, n_constraint, scs_init=True, verbose = False):
        self.Q_choi = Q_choi
        self.n = n    # Q output dimension
        self.d = d # Q input dimension
        self.scs_solver = None
        self.verbose = verbose
        self.n_constraint = n_constraint
        self.cost_choi = self.Q_choi._kraus_transposed()
        if scs_init:
            self.update_SCSprob(fresh_start=True)
    

    # generate choi matrix for transpose channel
    
    def change_choi(self, Q_choi):
        self.Q_choi = Q_choi
        self.cost_choi = self.Q_choi._kraus_transposed()
        self.update_SCSprob(fresh_start=False)
        
    
    def update_SCSprob(self, fresh_start = False, tol = 1.0e-7):
        n = int(self.n)
        d = int(self.d)
        flat = (n*d*(n*d + 1))//2

        if (fresh_start or (self.scs_solver == None)):
            A = []
            b = []
            # compute partial trace of R
            identMat = np.eye(n)
            for i in range(n):
                for j in range(n):
                    cons_real = (np_fock(n,i)@np_fock(n,j).T + np_fock(n,j)@np_fock(n,i).T)/2
                    A.append(scipy.sparse.csr_array(vec(np.kron(np.eye(d),cons_real)).reshape((1,flat))))
                    b.append(identMat[i,j])
            if n<d:
                number_constraint = np.diagflat([i for i in range(max(d,n))])
                A.append(scipy.sparse.csr_array(vec(np.kron(number_constraint,np.identity(n))).reshape((1,flat))))
                b.append(n*self.n_constraint)

            # solve SDP
            self.c = np.real(vec(-sparse.asnumpy(self.cost_choi.matrix))/(min(d,n)**2))
            self.b = np.real(np.array(b + [0]*len(self.c)))
            self.A = scipy.sparse.vstack([scipy.sparse.vstack(A, format='csc', dtype=np.float64),
                                   -scipy.sparse.eye(len(self.c),format="csc")],
                                    format="csc")
            data = dict(A=self.A,b=self.b,c=self.c)
            if n<d:
                cone = dict(z=len(self.b)-len(self.c)-1,l=1,s=n*d)
            else:
                cone = dict(z=len(self.b)-len(self.c),s=n*d)
            self.scs_solver = scs.SCS(data=data, cone=cone, verbose=self.verbose, eps_abs=tol, eps_rel=tol)
        
        else:
            c_new = vec(-sparse.asnumpy(self.cost_choi.matrix))/(min(d,n)**2)
            self.c = np.real(c_new)
            self.scs_solver.update(c=self.c)


    def solve(self,warm=True):
        #Does the solving of the SCS problem
        if warm:
            sol = self.scs_solver.solve(warm_start = True)
        else:
            sol = self.scs_solver.solve()
        self.P_choi = Choi(self.n,self.d)
        self.P_choi.matrix = sparse.asarray(mat(sol['x']))
        self.P_choi.four_tensor = self.P_choi.matrix.reshape((self.d,self.n,self.d,self.n))
        return sol