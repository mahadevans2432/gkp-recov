from sdp_ent_fid import *
from choi import Choi
from helper_functions import *
from choi_helper import *


def biconvex_limited(N, Ntype, gam, E, d, L, L1, n_constraint, iterations = 100, stop = 1, dipping = 4, store = True, lam = 1, message = ''):
    """
    Takes the noise channel and initial encoding as inputs
    Gives an optimized encoding and decoding as output
    Does a reduction of the space using isometry T
    """
    E_prev = E
    all_R = []
    all_E = []
    all_Nk = []
    all_T = []
    infid1 = []
    infid2 = []
    norm_diff = []
    energies = []
    counter = 0
    photon_cutoff = N.input_dim
    if Ntype=='loss':
        backaction = np.diagflat([np.power(1-gam,lam*i/2) for i in range(photon_cutoff)])
    else:
        backaction = scipy.linalg.fractional_matrix_power(N.approx_isometry(True),lam)

    for k in range(iterations):

        #Optimization of recovery
        Q_k = choi_compound(E_prev, N)
        T,M,n = Q_k.get_transform(np.identity(N.output_dim),force_rank=d*L) #Output transform
        T1,M1,n1 = Q_k.get_transform(backaction,add_proj=E_prev.act_on(np.identity(d)/d),force_rank=d*L1) #Input transform
        P = T.T.conj()@T
        P1 = T1.T.conj()@T1
        Np = N.act_on(P1)
        Q_k_red = Q_k.basis_change_output(T)
        sdp1 = SDP_ent_fid(Q_k_red, n, d, T, n_constraint, verbose=False)
        flag = False
        if k > 0 and (T_old.shape[0] == T.shape[0]): #Warm start condition
            sdp1.warm_y = transform_dual_rec(y_old, T_old, T, d)
            flag = True
        sol1 = sdp1.solve(warm=flag)
        y_old = sol1['y']
        T_old = T
        R_k = sdp1.P_choi
        R_k._renormalize() #Removes numerical errors due to faulty tolerance of SDP
        infid1.append(1-entanglement_fidelity(R_k,Q_k_red))
        
        #Optimization of encoding
        N_k = (N.basis_change_input(T1)).basis_change_output(T)
        #print(N_k.trace_check(), np.amin(np.linalg.eigvalsh(sparse.asnumpy(N_k.matrix))))
        #print(1, R_k.trace_check(), np.amin(np.linalg.eigvalsh(sparse.asnumpy(R_k.matrix))))
        #N_k._renormalize_alt()
        Q_kk = choi_compound(N_k, R_k)
        sdp2 = SDP_ent_fid(Q_kk, d, n1, T1, n_constraint, verbose=False)
        if k>0:
            sdp2.warm_x = complex_vec(sparse.asnumpy(E_prev.basis_change_output(T1).matrix))
            sdp2.warm_s = sdp2.b - sdp2.A@sdp2.warm_x
            sdp2.warm_y = transform_dual_enc(y_old1, T_old1, T1, d)
        sol2 = sdp2.solve(warm=(k>0))
        y_old1 = sol2['y']
        T_old1 = T1
        E_k = sdp2.P_choi.force_isometry()
        #print(2, E_k.trace_check(), np.amin(np.linalg.eigvalsh(sparse.asnumpy(E_k.matrix))))
        E_kk = E_k.reverse_output_basis_change(T1)
        #norm_diff.append(np.linalg.norm(sparse.asnumpy(E_kk.matrix - E_prev.matrix),ord='nuc'))
        norm_diff.append(np.linalg.norm(Np@P - Np,ord='nuc'))
        E_prev = E_kk
        energies.append(np.trace(np.diagflat([i for i in range(photon_cutoff)])@E_prev.act_on(np.identity(d)/d)))
        infid2.append(1-entanglement_fidelity(choi_compound(N,R_k.reverse_input_basis_change(T,True)),E_prev))
        
        if True:
            print("Lambda = "+str(lam)+"\tforcing isometry error = "+str(np.linalg.norm(sparse.asnumpy(E_k.matrix - sdp2.P_choi.matrix))))
            print(message+"Iteration "+str(k+1)+"\t Infidelity = "+str(infid1[-1])+", "+str(infid2[-1])+", "
                  +str(1-entanglement_fidelity(Q_kk,E_k))+
                  "\tIterations = "+str(sol1['info']['iter'])+", "+str(sol2['info']['iter'])+
                  "\tSizes = "+str(n)+", "+str(n1)+"\t Energy = "+str(energies[-1]),flush=True)
            
        if E_prev.trace_check() > 1e-4:
            print("-------Error too big to proceed!--------")
            break
        
        if store:
            all_E.append(E_prev)
            all_Nk.append(N_k)
            all_R.append(R_k)
            all_T.append(T)
        
        if infid2[-1] <= 1-stop:
            print("Convergence by value ",infid2[-1])
            break

        if len(infid2) >= 2:

            if np.min(infid2[:-1]) < infid2[-1]:
                counter += 1
                if counter == dipping:
                    print("Convergence by getting worse",infid2[-1])
                    break
            else:
                R_opt = R_k
                E_opt = E_prev
                T_opt = (T, T1)
                counter = 0
    
    if store:
        return all_R, all_E, all_Nk, all_T, infid1, infid2, norm_diff, energies
    else:
        return R_opt, E_opt, T_opt, infid1, infid2, norm_diff, energies
    
def biconvex_no_reduce(N, E, d, n_constraint, iterations = 100, message=''):
    """
    Takes the noise channel and initial encoding as inputs
    Gives an optimized encoding and decoding as output
    """
    E_prev = E
    n = N.four_tensor.shape[0]
    all_R = []
    all_E = []
    infid1 = []
    infid2 = []
    norm_diff = []
    counter = 0
    for k in range(iterations):
        Q_k = choi_compound(E_prev, N)
        T = np.identity(n,dtype=np.complex128)
        if k == 0:
            sdp1 = SDP_ent_fid(Q_k, n, d, n_constraint)
        if k>0:
            sdp1.change_choi(Q_k)
        sol1 = sdp1.solve(True)
        infid1.append(1+sol1['info']['pobj'])
        R_k = sdp1.P_choi
        all_R.append(R_k)
        Q_kk = choi_compound(N, R_k)
        if k == 0:
            sdp2 = SDP_ent_fid(Q_kk, d, n, n_constraint)
        else:
            sdp2.change_choi(Q_kk)
        sol2 = sdp2.solve(True)
        infid2.append(1+sol2['info']['pobj'])
        E_k = sdp2.P_choi#.force_isometry()
        all_E.append(E_k)
        norm_diff.append(np.linalg.norm(sparse.asnumpy(E_k.matrix - E_prev.matrix),ord='nuc'))
        E_prev = E_k
        flag1 = True#(k%20==0)
        print(message+"Iteration "+str(k+1)+"\tInfidelity = "+str(infid1[-1])+", "+str(infid2[-1])+
              "\tIterations = "+str(sol1['info']['iter'])+", "+str(sol2['info']['iter']),flush=flag1)
    return all_R, all_E, infid1, infid2, norm_diff