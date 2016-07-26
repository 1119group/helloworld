import numpy as np

def init(S):
    '''Initialize calculations by computing Sx, Sy and Sz.'''
    
    # Create the raising and lowering operators.
    S_plus = np.zeros([int(2*S+1),int(2*S+1)],float)
    for i in np.arange(-S,S,1):
        m = -i-1
        S_plus[i+S,i+S+1] = np.sqrt(S*(S+1)-m*(m+1))
    
    S_minus = np.transpose(S_plus)
    
    # Create Sx, Sy and Sz.
    Sx = 0.5*(S_plus + S_minus)
    Sy = -0.5j*(S_plus - S_minus)
    
    Sz = np.zeros([int(2*S+1),int(2*S+1)],float)
    for i in np.arange(S,-S-1,-1):
        Sz[-(i-S),-(i-S)] = i
        
    return Sx, Sy, Sz

def get_expected(A,psi):
    A_exp = np.dot(np.dot(np.conjugate(psi),A),psi)
    
    return A_exp

def get_uncertainty(A,psi):
    A_exp = get_expected(A,psi)
    A_uncert = get_expected(np.dot(A,A),psi) - np.dot(A_exp,A_exp)
    
    return A_uncert
    
    
S = 100
Sx, Sy, Sz = init(S)

# Create psi
psi = np.zeros(int(2*S+1),float)
psi[0] = 1

# Expectation values of Sx, Sy and Sz.
print("Expectation values for Sx, Sy and Sz:")
for i in [Sx,Sy,Sz]:
    i_exp = get_expected(i,psi)
    print(i_exp)

# Uncertainty for Sx, Sy and Sz.
print("\nUncertainties for Sx, Sy and Sz:")
for i in [Sx,Sy,Sz]:
    i_uncert = get_uncertainty(i,psi)
    print(i_uncert)