import numpy as np

def init(S):
    '''Initialize calculations by computing Sx, Sy and Sz.'''
    
    # Create the raising and lowering operators.
    S_plus = np.zeros([int(2*S+1),int(2*S+1)],float)
    for i in np.arange(-S,S,1):
        m = -i-1
        S_plus[int(i+S),int(i+S+1)] = np.sqrt(S*(S+1)-m*(m+1))
    
    S_minus = np.transpose(S_plus)
    
    # Create Sx, Sy and Sz.
    Sx = 0.5*(S_plus + S_minus)
    Sy = -0.5j*(S_plus - S_minus)
    
    Sz = np.zeros([int(2*S+1),int(2*S+1)],float)
    for i in np.arange(S,-S-1,-1):
        Sz[-int(i-S),-int(i-S)] = i
        
    return Sx, Sy, Sz
    

I = 3
J = 5
Ix, Iy, Iz = init(I)
Jx, Jy, Jz = init(J)

IJ = np.kron(Ix,Jx) + np.kron(Iy,Jy) + np.kron(Iz,Jz)
I2 = np.kron(np.dot(Ix,Ix),np.eye(2*J+1)) + np.kron(np.dot(Iy,Iy),np.eye(2*J+1)) + np.kron(np.dot(Iz,Iz),np.eye(2*J+1))
J2 = np.kron(np.eye(2*I+1),np.dot(Jx,Jx)) + np.kron(np.eye(2*I+1),np.dot(Jy,Jy)) + np.kron(np.eye(2*I+1),np.dot(Jz,Jz))

F2 = I2 + J2 + 2*IJ

E = np.linalg.eigvals(F2)

print(E)