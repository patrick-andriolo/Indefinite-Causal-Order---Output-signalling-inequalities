"Indefinite causality with signalling of inputs or outputs"

### Final function for see-saw:
    
# SeeSaw(d, inequality_index, n, m, signalling, Fams_output)
    
# d always equals 4 in this work;
# inequality index - same ordering as appearing in Appendix A;
# n - number of see-saws between process matrix - instrument A - instrument B;
# m - number of see-saws between density matrix and POVM within each instrument optimization;
# signalling - string "input" or "output".
# Fams_output - array containing inequalities defined in the beginning.

import numpy as np
import warnings
import cvxpy as cp

# dimension of (input) x (output) space of each party
d = 4  

# families of output signalling inequalities:
Fams_output = np.array([[0,  0,  1,  1,  1,  2,  0,  2,  0,  1,  1,  1,  1,  1,  1,  0],
                 [0,  0,  1,  1,  1,  2,  0,  2,  1,  1,  0,  1,  1,  0,  1,  1],
                 [0,  0,  1,  1,  2,  1,  2,  0,  1,  0,  1,  1,  1,  1,  0,  1],
                 [0,  0,  1,  1,  2,  1,  2,  0,  1,  1,  1,  0,  0,  1,  1,  1],
                 [0,  0,  1,  1,  2,  2,  0,  1,  1,  1,  1,  0,  2,  0,  2,  0],
                 [0,  0,  1,  1,  2,  2,  0,  2,  1,  1,  0,  1,  1,  0,  1,  0],
                 [0,  0,  1,  1,  2,  2,  2,  0,  1,  1,  1,  0,  0,  1,  0,  1],
                 [0,  0,  2,  2,  1,  2,  0,  2,  0,  1,  1,  1,  1,  0,  1,  0],
                 [0,  0,  2,  2,  2,  1,  2,  0,  1,  0,  1,  1,  0,  1,  0,  1],
                 [0,  1,  1,  1,  1,  0,  1,  0,  1,  1,  0,  0,  0,  2,  2,  2],
                 [0,  1,  1,  1,  1,  0,  1,  0,  1,  1,  1,  0,  0,  1,  2,  2],
                 [0,  1,  1,  1,  1,  1,  1,  0,  1,  1,  1,  0,  0,  1,  1,  1],
                 [0,  1,  1,  1,  1,  1,  1,  0,  2,  0,  2,  1,  1,  1,  0,  0],
                 [0,  1,  2,  2,  1,  1,  0,  0,  2,  0,  2,  0,  1,  0,  1,  1],
                 [0,  1,  2,  2,  1,  1,  0,  1,  1,  0,  1,  0,  1,  0,  1,  1],
                 [1,  0,  1,  0,  1,  1,  0,  1,  2,  2,  0,  1,  1,  0,  1,  1],
                 [1,  0,  1,  1,  1,  1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  1],
                 [0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  0,  1],
                 [0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  1,  1,  0],
                 [0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  1,  1,  1,  1,  1,  0],
                 [0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  1,  1,  1,  1,  0,  0],
                 [0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  0,  1,  0,  1,  1],
                 [0,  0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  1,  0,  0,  1,  1],
                 [0,  0,  0,  0,  0,  1,  1,  1,  0,  1,  1,  1,  1,  0,  0,  0],
                 [0,  0,  0,  0,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  0,  0],
                 [0,  0,  0,  0,  1,  0,  1,  0,  1,  1,  0,  0,  0,  1,  1,  1],
                 [0,  0,  0,  0,  1,  0,  1,  0,  1,  1,  1,  0,  0,  0,  1,  1],
                 [0,  0,  0,  0,  1,  0,  1,  1,  1,  0,  1,  1,  0,  1,  0,  0],
                 [0,  0,  0,  0,  1,  1,  1,  0,  1,  1,  1,  0,  0,  0,  0,  1],
                 [0,  0,  0,  1,  1,  1,  0,  0,  1,  0,  1,  0,  1,  1,  1,  0],
                 [0,  0,  0,  1,  1,  1,  0,  0,  1,  1,  1,  0,  1,  0,  1,  0],
                 [0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  0,  1,  0,  1,  1],
                 [0,  0,  1,  1,  0,  1,  0,  0,  1,  0,  1,  1,  1,  0,  1,  0],
                 [0,  0,  1,  1,  0,  1,  0,  1,  0,  0,  1,  1,  1,  0,  1,  0],
                 [0,  0,  1,  1,  0,  1,  0,  1,  0,  1,  1,  1,  1,  0,  0,  0],
                 [0,  0,  1,  1,  1,  0,  0,  0,  0,  1,  1,  1,  0,  1,  0,  1],
                 [0,  0,  1,  1,  1,  0,  1,  0,  1,  0,  1,  1,  0,  1,  0,  0],
                 [0,  0,  1,  1,  1,  1,  0,  0,  1,  0,  1,  0,  1,  0,  1,  0],
                 [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1],
                 [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0],
                 [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0]])



# the function below is the trace-and-replace composition of Eq. (22)

def Composition(W, d, i):

    subsys_dim = int(np.sqrt(d)) 

    Z = np.reshape(np.copy(W),[2,2,2,2,2,2,2,2])

    # trace Ai
    if(i == 0):
        
        Z_traced = np.einsum('akmpalnq -> kmplnq', np.copy(Z)) 
        
        for u in range(subsys_dim):
            for v in range(subsys_dim):
                if u == v:
                    Z[u,:,:,:,v,:,:,:] = 0.5 * Z_traced
                else:
                    Z[u,:,:,:,v,:,:,:] = np.zeros((2,2))

    # trace Ao
    elif(i == 1):
        
        Z_traced = np.einsum('akmpbknq -> ampbnq', np.copy(Z)) 

        for u in range(subsys_dim):
            for v in range(subsys_dim):
                if u == v:
                    Z[:,u,:,:,:,v,:,:] = 0.5 * Z_traced
                else:
                    Z[:,u,:,:,:,v,:,:] = np.zeros((2,2))

    # trace Bi
    elif(i == 2):
        
        Z_traced = np.einsum('akmpblmq -> akpblq', np.copy(Z)) 

        for u in range(subsys_dim):
            for v in range(subsys_dim):
                if u == v:
                    Z[:,:,u,:,:,:,v,:] = 0.5 * Z_traced
                else:
                    Z[:,:,u,:,:,:,v,:] = np.zeros((2,2))

    # trace Bo
    elif(i == 3):
        
        Z_traced = np.einsum('akmpblnp -> akmbln', np.copy(Z)) 
        
        for u in range(subsys_dim):
            for v in range(subsys_dim):
                if u == v:
                    Z[:,:,:,u,:,:,:,v] = 0.5 * Z_traced
                else:
                    Z[:,:,:,u,:,:,:,v] = np.zeros((2,2))

    else:
        print('No subsystem corresponding to index ' + str(i))

    Z = np.reshape(Z, [16,16])
    
    return(Z)


    
    
# random instruments to start optimization in input signalling scenario:
    
def Random_Instruments_InputSignalling(d):
     
     alpha = 0.99

     H = np.random.randn(d, d) + 1j * np.random.randn(d, d)
     H = H @ H.conj().T
     H = H / np.max(np.linalg.eigvalsh(H)).real
     H = (H + H.conj().T) / 2
     
     M1 = 0.5 * alpha * H
     M2 = 0.5 * np.eye(d) - M1
     M2 = (M2 + M2.conj().T) / 2
 
     return M1, M2





# random instruments to start optimization in output signalling scenario:
    
def Random_Instruments_OutputSignalling(d):
    
    d_povm = int(np.sqrt(d))
    d_rho = int(np.sqrt(d))

    # random rho_0
    rho0 = np.random.randn(d_rho, d_rho) + 1j * np.random.randn(d_rho, d_rho)
    rho0 = rho0 @ rho0.conj().T
    rho0 = rho0 / np.trace(rho0)

    # random rho_1
    rho1 = np.random.randn(d_rho, d_rho) + 1j * np.random.randn(d_rho, d_rho)
    rho1 = rho1 @ rho1.conj().T
    rho1 = rho1 / np.trace(rho1)

    rho = [rho0, rho1]

    # random POVM for x = 0
    povm = np.random.randn(d_povm, d_povm) + 1j * np.random.randn(d_povm, d_povm)
    E0 = povm @ povm.conj().T
    E0 = E0 / np.trace(E0)
    E1 = np.eye(d_povm) - E0

    # random POVM for x = 1
    povm = np.random.randn(d_povm, d_povm) + 1j * np.random.randn(d_povm, d_povm)
    E2 = povm @ povm.conj().T
    E2 = E2 / np.trace(E2)
    E3 = np.eye(d_povm) - E2

    E = [E0, E1, E2, E3]

    instrument = [
        np.kron(E[0], rho[0].T),
        np.kron(E[1], rho[1].T),
        np.kron(E[2], rho[0].T),
        np.kron(E[3], rho[1].T)
    ]

    return instrument, E, rho



# Function to start optimization with set of instruments compatible with each scenario

def Initial_Instruments(d, signalling):
    
    if(signalling == "input"):
    
        instrument = [Random_Instruments_InputSignalling(d), Random_Instruments_InputSignalling(d)]
        instrument = [instrument[0][0], instrument[0][1], instrument[1][0], instrument[1][1]] 
        return instrument
    
    elif(signalling == "output"):
        
        instrument, E, rho = Random_Instruments_OutputSignalling(d)
        return instrument


# SDP to optimize process matrix:

def Optimize_ProcessMatrix(d, A, B, inequality_index, Fams_output): 

    global Inequality_value
    

    subsys_dim = int(np.sqrt(d))

    Wfixed = cp.Variable([d**2, d**2], hermitian=True)
    
    W = np.full((d**2, d**2), None, dtype='object')
    
    for i in range(d**2):
        for j in range(d**2):
            W[i, j] = Wfixed[i, j]
    
    p0000 = np.trace((np.kron(A[0], B[0])) @ W)
    p0100 = np.trace((np.kron(A[0], B[1])) @ W) 
    p1000 = np.trace((np.kron(A[1], B[0])) @ W) 
    p1100 = np.trace((np.kron(A[1], B[1])) @ W) 
    p0001 = np.trace((np.kron(A[0], B[2])) @ W) 
    p0101 = np.trace((np.kron(A[0], B[3])) @ W) 
    p1001 = np.trace((np.kron(A[1], B[2])) @ W) 
    p1101 = np.trace((np.kron(A[1], B[3])) @ W) 
    p0010 = np.trace((np.kron(A[2], B[0])) @ W) 
    p0110 = np.trace((np.kron(A[2], B[1])) @ W) 
    p1010 = np.trace((np.kron(A[3], B[0])) @ W) 
    p1110 = np.trace((np.kron(A[3], B[1])) @ W) 
    p0011 = np.trace((np.kron(A[2], B[2])) @ W) 
    p0111 = np.trace((np.kron(A[2], B[3])) @ W) 
    p1011 = np.trace((np.kron(A[3], B[2])) @ W) 
    p1111 = np.trace((np.kron(A[3], B[3])) @ W)

    behaviour = np.array([p0000, p0100, p1000, p1100, p0001, p0101, p1001, p1101, p0010, p0110, p1010, p1110, p0011, p0111, p1011, p1111])
    
    # objective function - Eq. (30a)
    obj = cp.real(np.sum(np.multiply(Fams_output[inequality_index - 1], behaviour)))
    
    #### constraints of the process matrix:
        
    # positive semidefinite condition - Eq. (30b)
    constraints = [Wfixed >> 0] 

    # normalization - Eq. (30c)
    constraints += [np.trace(W) == subsys_dim**2]  

    # Eq. (30d) 
    constraints += [eW == eI for eW, eI in zip(Composition(Composition(W, d, 3), d, 2).flatten(), Composition(Composition(Composition(W, d, 3), d, 2), d, 1).flatten())]
    
    # Eq. (30e) 
    constraints += [eW == eI for eW, eI in zip(Composition(Composition(W, d, 1), d, 0).flatten(), Composition(Composition(Composition(W, d, 3), d, 1), d, 0).flatten())]

    # Eq. (30f) 
    constraints += [eW == eI for eW, eI in zip( W.flatten(), ( Composition(W, d, 3) + Composition(W, d, 1) - Composition(Composition(W, d, 3), d, 1) ).flatten())]  
    
    ### sanity checks:
    
    # definite causal order from A to B  W^{A prec B} = {B_O}_W^{A \prec B}. If the line below is uncommented, we shouldn't see violations
    #constraints += [eW == eI for eW, eI in zip( W.flatten(), Composition(W, d, 3).flatten())]
    
    # definite causal order from B to A  W^{B prec A} = {A_O}_W^{B \prec A}. If the line below is uncommented, we shouldn't see violations
    #constraints += [eW == eI for eW, eI in zip( W.flatten(), Composition(W, d, 1).flatten())]  
    
    #constraints += Input_Signalling_Constraints(behaviour)
    
    # solve SDP:
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose = False)

    Inequality_value = obj.value

    return(Wfixed.value)




# SDP to optimize instruments in the unconstrained scenario (Sec. III.A)
    
def Instruments_input_signalling(d, W, variable_instrument, fixed_instrument, inequality_index, Fams_output):

    global Inequality_value

    # 2 pairs of instruments
    M = [cp.Variable((4, 4), hermitian=True), cp.Variable((4, 4), hermitian=True), cp.Variable((4, 4), hermitian=True), cp.Variable((4, 4), hermitian=True)]

    # PSD condition - Eq. (19)
    constraints = [M[i] >> 0 for i in range(len(M))]
    
    # traced-out output matrices sum up to identity - Eq. (19)
    constraints += [cp.partial_trace(M[0] + M[1], [2,2], axis = 1) == np.eye(2)]
    constraints += [cp.partial_trace(M[2] + M[3], [2,2], axis = 1) == np.eye(2)]

    if variable_instrument == 'A':

        A = M
        B = fixed_instrument
        
    else:

        B = M
        A = fixed_instrument

    p0000 = cp.trace((cp.kron(A[0], B[0])) @ W) 
    p0100 = cp.trace((cp.kron(A[0], B[1])) @ W) 
    p1000 = cp.trace((cp.kron(A[1], B[0])) @ W) 
    p1100 = cp.trace((cp.kron(A[1], B[1])) @ W) 
    p0001 = cp.trace((cp.kron(A[0], B[2])) @ W) 
    p0101 = cp.trace((cp.kron(A[0], B[3])) @ W) 
    p1001 = cp.trace((cp.kron(A[1], B[2])) @ W) 
    p1101 = cp.trace((cp.kron(A[1], B[3])) @ W) 
    p0010 = cp.trace((cp.kron(A[2], B[0])) @ W)
    p0110 = cp.trace((cp.kron(A[2], B[1])) @ W)
    p1010 = cp.trace((cp.kron(A[3], B[0])) @ W)
    p1110 = cp.trace((cp.kron(A[3], B[1])) @ W)
    p0011 = cp.trace((cp.kron(A[2], B[2])) @ W) 
    p0111 = cp.trace((cp.kron(A[2], B[3])) @ W) 
    p1011 = cp.trace((cp.kron(A[3], B[2])) @ W) 
    p1111 = cp.trace((cp.kron(A[3], B[3])) @ W)

    behaviour = np.array([p0000, p0100, p1000, p1100, p0001, p0101, p1001, p1101, p0010, p0110, p1010, p1110, p0011, p0111, p1011, p1111])
    
    obj = cp.real(np.sum(np.multiply(Fams_output[inequality_index - 1], behaviour)))
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose = False)
    
    if variable_instrument == 'A':
        Inequality_value = obj.value
        return [a.value for a in M]
    else:
        Inequality_value = obj.value
        return [b.value for b in M]






### SDP to optimize instruments for output signalling scenario:
# we need to split into optimization of POVM's and states. We start with randomly generated states

# function for random density matrices:
def Random_Density_Matrix():
    
      # random 2x2 complex matrix
      A = np.random.randn(2, 2) + 1j * np.random.randn(2, 2)
      
      # positive semidefinite matrix by A*A^dagger
      rho = A @ A.conj().T
      
      # normalization
      rho = rho / np.trace(rho)
      
      return rho
 
# SDP to optimize POVM's:
def Optimize_POVM(d, W, variable_instrument, fixed_instrument, inequality_index, rho, Fams_output):

    global Inequality_value
    
    # 2 pairs of POVM's
    E = [cp.Variable((2, 2), hermitian=True), cp.Variable((2, 2), hermitian=True), cp.Variable((2, 2), hermitian=True), cp.Variable((2, 2), hermitian=True)]

    # PSD of POVM's
    constraints = [E[i] >> 0 for i in range(len(E))]

    if variable_instrument == 'A':

        A = [cp.kron(E[0],(rho[0]).T) , cp.kron(E[1],(rho[1]).T), cp.kron(E[2],(rho[0]).T), cp.kron(E[3],(rho[1]).T)]
        
        # each pairs sums up to identity
        constraints += [cp.partial_trace(A[0] + A[1], [2,2], 1) == np.eye(2)]
        constraints += [cp.partial_trace(A[2] + A[3], [2,2], 1) == np.eye(2)]
        
        B = fixed_instrument
        
    else:

        B = [cp.kron(E[0],(rho[0]).T) , cp.kron(E[1],(rho[1]).T), cp.kron(E[2],(rho[0]).T), cp.kron(E[3],(rho[1]).T)]

        # each pairs sums up to identity
        constraints += [cp.partial_trace(B[0] + B[1], [2,2], 1) == np.eye(2)]
        constraints += [cp.partial_trace(B[2] + B[3], [2,2], 1) == np.eye(2)]
        
        A = fixed_instrument

    p0000 = cp.trace((cp.kron(A[0], B[0])) @ W) 
    p0100 = cp.trace((cp.kron(A[0], B[1])) @ W) 
    p1000 = cp.trace((cp.kron(A[1], B[0])) @ W) 
    p1100 = cp.trace((cp.kron(A[1], B[1])) @ W) 
    p0001 = cp.trace((cp.kron(A[0], B[2])) @ W) 
    p0101 = cp.trace((cp.kron(A[0], B[3])) @ W) 
    p1001 = cp.trace((cp.kron(A[1], B[2])) @ W) 
    p1101 = cp.trace((cp.kron(A[1], B[3])) @ W) 
    p0010 = cp.trace((cp.kron(A[2], B[0])) @ W)
    p0110 = cp.trace((cp.kron(A[2], B[1])) @ W)
    p1010 = cp.trace((cp.kron(A[3], B[0])) @ W)
    p1110 = cp.trace((cp.kron(A[3], B[1])) @ W)
    p0011 = cp.trace((cp.kron(A[2], B[2])) @ W) 
    p0111 = cp.trace((cp.kron(A[2], B[3])) @ W) 
    p1011 = cp.trace((cp.kron(A[3], B[2])) @ W) 
    p1111 = cp.trace((cp.kron(A[3], B[3])) @ W)

    behaviour = np.array([p0000, p0100, p1000, p1100, p0001, p0101, p1001, p1101, p0010, p0110, p1010, p1110, p0011, p0111, p1011, p1111])

    #constraints += Input_Signalling_Constraints(behaviour)

    obj = cp.real(np.sum(np.multiply(Fams_output[inequality_index - 1], behaviour)))
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose = False)

    if variable_instrument == 'A':
        Inequality_value = obj.value
        return [a.value for a in E]
    else:
        
        Inequality_value = obj.value
        return [b.value for b in E]
    

# SDP to optimize density matrices:
def Optimize_Density_Matrix(d, W, variable_instrument, fixed_instrument, inequality_index, E, Fams_output):

    global Inequality_value

    rho = [cp.Variable((2, 2), hermitian=True), cp.Variable((2, 2), hermitian=True)]

    # normalization of DM
    constraints = [cp.trace(rho[i]) == 1 for i in range(len(rho))]
    
    # PSD of DM
    constraints += [rho[i] >> 0 for i in range(len(rho))]

    if variable_instrument == 'A':

        A = [cp.kron(E[0],(rho[0]).T) , cp.kron(E[1],(rho[1]).T), cp.kron(E[2],(rho[0]).T), cp.kron(E[3],(rho[1]).T)]
        B = fixed_instrument
        
    else:
        
        B = [cp.kron(E[0],(rho[0]).T) , cp.kron(E[1],(rho[1]).T), cp.kron(E[2],(rho[0]).T), cp.kron(E[3],(rho[1]).T)]
        A = fixed_instrument

    p0000 = cp.trace((cp.kron(A[0], B[0])) @ W) 
    p0100 = cp.trace((cp.kron(A[0], B[1])) @ W) 
    p1000 = cp.trace((cp.kron(A[1], B[0])) @ W) 
    p1100 = cp.trace((cp.kron(A[1], B[1])) @ W) 
    p0001 = cp.trace((cp.kron(A[0], B[2])) @ W) 
    p0101 = cp.trace((cp.kron(A[0], B[3])) @ W) 
    p1001 = cp.trace((cp.kron(A[1], B[2])) @ W) 
    p1101 = cp.trace((cp.kron(A[1], B[3])) @ W) 
    p0010 = cp.trace((cp.kron(A[2], B[0])) @ W)
    p0110 = cp.trace((cp.kron(A[2], B[1])) @ W)
    p1010 = cp.trace((cp.kron(A[3], B[0])) @ W)
    p1110 = cp.trace((cp.kron(A[3], B[1])) @ W)
    p0011 = cp.trace((cp.kron(A[2], B[2])) @ W) 
    p0111 = cp.trace((cp.kron(A[2], B[3])) @ W) 
    p1011 = cp.trace((cp.kron(A[3], B[2])) @ W) 
    p1111 = cp.trace((cp.kron(A[3], B[3])) @ W)

    behaviour = np.array([p0000, p0100, p1000, p1100, p0001, p0101, p1001, p1101, p0010, p0110, p1010, p1110, p0011, p0111, p1011, p1111])

    #constraints += Input_Signalling_Constraints(behaviour)

    obj = cp.real(np.sum(np.multiply(Fams_output[inequality_index - 1], behaviour)))
    
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(verbose = False)

    if variable_instrument == 'A':

        Inequality_value = obj.value
        print(Inequality_value)

        return [a.value for a in rho]
    else:

        Inequality_value = obj.value
        print(Inequality_value)
        return [b.value for b in rho]
    
    
# loop to perform see-saw between DM's and POVM's:
def Optimize_Instrument(d, W, variable_instrument, fixed_instrument, inequality_index, signalling, m, Fams_output, current_E=None, current_rho=None):
    
    if(signalling == 'output'):

        E = [e.copy() for e in current_E]
        rho = [r.copy() for r in current_rho]
    
        for _ in range(m):
    
            E = Optimize_POVM(
                d,
                W,
                variable_instrument,
                fixed_instrument,
                inequality_index,
                rho,
                Fams_output
            )
    
            rho = Optimize_Density_Matrix(
                d,
                W,
                variable_instrument,
                fixed_instrument,
                inequality_index,
                E,
                Fams_output
            )
    
        instrument = [
            np.kron(E[0], rho[0].T),
            np.kron(E[1], rho[1].T),
            np.kron(E[2], rho[0].T),
            np.kron(E[3], rho[1].T)
        ]
    
        return instrument, E, rho
        
    elif(signalling == 'input'):
        
        E = Instruments_input_signalling(d, W, variable_instrument, fixed_instrument, inequality_index, Fams_output)
        
        if variable_instrument == 'A':
            
            A = E
            return [a for a in A]
        
        else:
            
            B = E
            return [b for b in B]
        
        
    
### Final function for see-saw:
    
# SeeSaw(d, inequality_index, n, m, signalling, Fams_output)
    
# d always equals 4 in this work;
# inequality index - same ordering as appearing in Appendix A;
# n - number of see-saws between process matrix - instrument A - instrument B;
# m - number of see-saws between density matrix and POVM within each instrument optimization;
# signalling - string "input" or "output".
# Fams_output - array containing inequalities defined in the beginning.

def SeeSaw(d, inequality_index, n, m, signalling, Fams_output):
    
    import itertools

    # define Pauli matrices to print matrix decomposition later

    pauli_tol = 1e-2        # minimum value of coefficient to be printed
    pauli_decimals = 6      # precision of the coefficient
    tensor_symbol = " ⊗ "

    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    Paulis = {
        "I": I,
        "X": X,
        "Y": Y,
        "Z": Z
    }
    
    # more helper functions to print Paulis:

    def kron_all(mats):
        out = np.array([[1]], dtype=complex)
        for M in mats:
            out = np.kron(out, M)
        return out

    def clean_coeff(c):
        r = np.real(c)
        im = np.imag(c)

        if abs(r) < pauli_tol / 10:
            r = 0.0
        if abs(im) < pauli_tol / 10:
            im = 0.0

        if im == 0.0:
            return f"{r:.{pauli_decimals}g}"
        elif r == 0.0:
            return f"{im:.{pauli_decimals}g}j"
        else:
            sign = "+" if im >= 0 else "-"
            return f"{r:.{pauli_decimals}g} {sign} {abs(im):.{pauli_decimals}g}j"

    def pauli_decomposition(M, tol=pauli_tol):
        M = np.array(M, dtype=complex)

        if M.shape[0] != M.shape[1]:
            raise ValueError("Matrix must be square.")

        N = M.shape[0]
        n_qubits = int(round(np.log2(N)))

        if 2**n_qubits != N:
            raise ValueError("Matrix dimension must be a power of 2.")

        terms = []

        for labels in itertools.product(["I", "X", "Y", "Z"], repeat=n_qubits):
            P = kron_all([Paulis[label] for label in labels])
            coeff = np.trace(P.conj().T @ M) / N

            if abs(coeff) > tol:
                terms.append(("".join(labels), coeff))

        terms.sort(key=lambda x: abs(x[1]), reverse=True)

        return terms

    def print_pauli_decomposition(name, M, subsystem_labels=None):
        M = np.array(M, dtype=complex)
        N = M.shape[0]
        n_qubits = int(round(np.log2(N)))

        terms = pauli_decomposition(M, tol=pauli_tol)

        print(name)
        print("-" * len(name))
        print(f"dimension: {N} x {N}")
        print(f"number of qubits: {n_qubits}")

        print(f"terms shown with |coefficient| > {pauli_tol}")
        print("")

        if len(terms) == 0:
            print("0")
        else:
            for pauli_string, coeff in terms:
                pauli_tensor = tensor_symbol.join(pauli_string)
                print(f"{clean_coeff(coeff)} * {pauli_tensor}")

        print("\n")

    def print_instrument_decomposition(party_name, instrument):
        labels = [
            f"{party_name}_M_0|0",
            f"{party_name}_M_1|0",
            f"{party_name}_M_0|1",
            f"{party_name}_M_1|1"
        ]

        for label, M in zip(labels, instrument):
            print_pauli_decomposition(
                label,
                M,
                subsystem_labels=[f"{party_name}_I", f"{party_name}_O"]
            )

    # See-saw (process matrix - instrument A - instrument B) function

    global Inequality_value
    
    i = 0

    if signalling == 'output':

        A, A_E, A_rho = Random_Instruments_OutputSignalling(d)
        B, B_E, B_rho = Random_Instruments_OutputSignalling(d)
        #B, B_E, B_rho = A, A_E, A_rho

    else:
    
        A = Initial_Instruments(d, signalling)
        B = Initial_Instruments(d, signalling)
        #B = A
        
    while i < n:
    
        # calling the process matrix optimization. To use a fixed W
        # comment this line and insert the desiderd process matrix. For instance
        
        #theta = 0.25*np.pi
        
        #W = 0.25 * (
        #        np.kron(np.kron(np.kron(I,I),I),I)
        #        + np.sin(theta) * np.kron(np.kron(np.kron(Y,X),Y),I)
        #       + np.cos(theta) * np.kron(np.kron(np.kron(Z,I),Y),X)
        #   )
        
        W = Optimize_ProcessMatrix(d, A, B, inequality_index, Fams_output)
    
        if signalling == 'output':
    
            A, A_E, A_rho = Optimize_Instrument(
                d,
                W,
                'A',
                B,
                inequality_index,
                signalling,
                m,
                Fams_output,
                current_E=A_E,
                current_rho=A_rho
            )
            print(Inequality_value)
    
            B, B_E, B_rho = Optimize_Instrument(
                d,
                W,
                'B',
                A,
                inequality_index,
                signalling,
                m,
                Fams_output,
                current_E=B_E,
                current_rho=B_rho
            )
            print(Inequality_value)
    
        else:
    
            A = Optimize_Instrument(
                d,
                W,
                'A',
                B,
                inequality_index,
                signalling,
                m,
                Fams_output
            )
            print(Inequality_value)
    
            B = Optimize_Instrument(
                d,
                W,
                'B',
                A,
                inequality_index,
                signalling,
                m,
                Fams_output
            )
            print(Inequality_value)

        i += 1

    # print all final results:

    print("\n")
    print("============================================================")
    print("Final result")
    print("============================================================")
    print("Inequality index:", inequality_index)
    print("Final inequality value:", Inequality_value)
    print("Signalling scenario:", signalling)
    print("Outer iterations n:", n)
    print("Inner iterations m:", m)
    print("Pauli coefficient tolerance:", pauli_tol)
    print("============================================================")
    print("\n")

    print("A instruments in Pauli basis")
    print("============================")
    print_instrument_decomposition("A", A)

    print("B instruments in Pauli basis")
    print("============================")
    print_instrument_decomposition("B", B)

    print("Process matrix W in Pauli basis")
    print("===============================")
    print_pauli_decomposition(
        "W",
        W,
        subsystem_labels=["A_I", "A_O", "B_I", "B_O"]
    )

    if signalling == 'output':

        print("Output-signalling decomposition of A")
        print("====================================")
        for j, E in enumerate(A_E):
            print_pauli_decomposition(f"A_E_{j}", E, subsystem_labels=["A_I"])
        for j, rho in enumerate(A_rho):
            print_pauli_decomposition(f"A_rho_{j}", rho, subsystem_labels=["A_O"])

        print("Output-signalling decomposition of B")
        print("====================================")
        for j, E in enumerate(B_E):
            print_pauli_decomposition(f"B_E_{j}", E, subsystem_labels=["B_I"])
        for j, rho in enumerate(B_rho):
            print_pauli_decomposition(f"B_rho_{j}", rho, subsystem_labels=["B_O"])
    
    print("Inequality value:")
    print()
    print(Inequality_value)

    return A, B, W, Inequality_value


warnings.filterwarnings("ignore", category=UserWarning)



### In order to add input signalling inequalities being respected, 
### use the families and the function encoding their constrains defined below.
### In each step of the optimization, one just needs to add the line:
    
###     constraints += Input_Signalling_Constraints(behaviour)

# 32 nontrivial input-signalling causal inequalities.
# Each row is [c0,c1,...,c16] and represents
# c0 + sum_i c_i p_i >= 0, with probabilities ordered as in behaviour.
Inequalities_input = np.array([
    [ 0,  1,  0,  1,  0,  0,  0,  0,  0, -1,  0,  0,  0,  1,  1,  0,  0],
    [ 0, -1,  0,  0,  0,  1,  1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  1,  0,  1,  0,  1,  1,  0,  0, -1,  0,  0,  0],
    [ 0,  1,  1,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  1,  0],
    [ 0, -1,  0,  0,  0,  1,  1,  1,  0,  1,  1,  1,  0, -1,  0,  0,  0],
    [ 0,  1,  1,  1,  0, -1,  0,  0,  0, -1,  0,  0,  0,  1,  1,  1,  0],
    [ 1, -1,  0,  0,  0,  0,  0, -1,  0,  1,  1,  1,  0,  0, -1,  0,  0],
    [ 1,  0,  0,  0,  0, -1,  0, -1,  0,  1,  1,  0,  0,  0, -1,  0,  0],
    [ 1,  1,  1,  1,  0,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0, -1,  0],
    [ 1,  1,  1,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0, -1,  0, -1,  0],
    [ 1,  0, -1,  0,  0,  1,  1,  1,  0,  0,  0, -1,  0, -1,  0,  0,  0],
    [ 1,  0,  0, -1,  0, -1,  0,  0,  0,  0, -1,  0,  0,  1,  1,  1,  0],
    [ 1,  0, -1,  0,  0,  1,  1,  0,  0, -1,  0, -1,  0,  0,  0,  0,  0],
    [ 1, -1,  0, -1,  0,  0,  0,  0,  0,  0, -1,  0,  0,  1,  1,  0,  0],
    [ 1,  0,  0,  0,  0,  1,  0,  1,  0, -1, -1,  0,  0,  0,  0, -1,  0],
    [ 1, -1,  0,  0,  0,  1,  1,  1,  0,  0, -1,  0,  0,  0,  0, -1,  0],
    [ 1,  0,  0,  0,  0, -1,  0, -1,  0, -1, -1,  0,  0,  1,  1,  1,  0],
    [ 1, -1,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0,  0,  1,  1,  1,  0],
    [ 1,  1,  1,  1,  0,  0, -1,  0,  0,  0,  0, -1,  0, -1,  0,  0,  0],
    [ 1,  0,  0, -1,  0,  0, -1,  0,  0,  1,  1,  1,  0, -1,  0,  0,  0],
    [ 1,  1,  1,  1,  0, -1, -1,  0,  0, -1,  0, -1,  0,  0,  0,  0,  0],
    [ 1,  0,  0, -1,  0, -1, -1,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0],
    [ 1,  0, -1,  0,  0,  1,  1,  1,  0, -1,  0,  0,  0,  0,  0, -1,  0],
    [ 1,  0, -1,  0,  0,  0,  0, -1,  0, -1,  0,  0,  0,  1,  1,  1,  0],
    [ 1, -1, -1,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0, -1,  0, -1,  0],
    [ 1, -1, -1,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  1,  0,  1,  0],
    [ 2,  0, -1,  0,  0,  0,  0, -1,  0,  0,  0, -1,  0,  0, -1,  0,  0],
    [ 2,  0,  0, -1,  0,  0, -1,  0,  0,  0, -1,  0,  0,  0,  0, -1,  0],
    [ 1,  1,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1, -1,  0,  0],
    [ 1, -1,  0, -1,  0,  0,  0,  0,  0,  1,  1,  1,  0, -1, -1,  0,  0],
    [ 1,  1,  1,  1,  0, -1,  0,  0,  0,  0,  0, -1,  0,  0, -1,  0,  0],
    [ 1,  0,  0, -1,  0, -1,  0,  0,  0,  1,  1,  1,  0,  0, -1,  0,  0]
])

def Input_Signalling_Constraints(behaviour):
    
    behaviour_vector = cp.hstack(list(behaviour))
    inequality_values = (Inequalities_input[:, 0] + Inequalities_input[:, 1:] @ behaviour_vector)
    return [cp.real(inequality_values) >= 0]




