import numpy as np

def epg_mgrad(FpFmZ, noadd=0):
    """
  	Propagate EPG states through a *NEGATIVE* "unit" gradient.
	(States move the opposite direction to epg_grad!)
  
    Arguments:
    - `FpFmZ`: 3xN vector of F+, F- and Z states.
    - `noadd`: 1 to NOT add any higher-order states - assume
			that they just go to zero.  Be careful - this
			speeds up simulations, but may compromise accuracy!
    """
    
    #% Gradient does not affect the Z states.
    if noadd == 0.:
        FpFmZ = np.array(np.hstack((FpFmZ, np.array(np.vstack((np.hstack((0.)), \
                                    np.hstack((0.)), np.hstack((0.))))))))
        #% Add higher dephased state.
    
    FpFmZTop = np.roll(FpFmZ,  -1, axis=1)
    FpFmZBottom  = np.roll(FpFmZ,  1, axis=1)

    FpFmZ[0,:] = FpFmZTop[0,:]    #% Shift Fp states.
    FpFmZ[1,:] = FpFmZBottom[1,:]    #% Shift Fm states.


    FpFmZ[0,int(0)-1] = 0.                          #% Zero highest Fp state.
    FpFmZ[0,0] = np.conj(FpFmZ[1,0])                #% Fill in lowest Fm state.
    return FpFmZ

def epg_grad(FpFmZ, noadd=0):
    """
    Propagate EPG states through a "unit" gradient.

    Arguments:
    - `FpFmZ`: 3xN vector of F+, F- and Z states.
    - `noadd`: 1 to NOT add any higher-order states - assume
               that they just go to zero.  Be careful - this
               speeds up simulations, but may compromise accuracy!

    """
    
    #% Gradient does not affect the Z states.
    if noadd == 0.:
        FpFmZ = np.array(np.hstack((FpFmZ, np.array([[0.],[0.],[0.]]))))
        #% Add higher dephased state.



    FpFmZTop = np.roll(FpFmZ,  1, axis=1)
    FpFmZBottom  = np.roll(FpFmZ,  -1, axis=1)

    FpFmZ[0,:] = FpFmZTop[0,:]    #% Shift Fp states.
    FpFmZ[1,:] = FpFmZBottom[1,:]    #% Shift Fm states.

    FpFmZ[1,int(0)-1] = 0.    #% Zero highest Fm state.
    FpFmZ[0,0] = np.conjugate(FpFmZ[1,0])    #% Fill in lowest Fp state.

    return FpFmZ

def epg_relax(FpFmZ, T1, T2, T):
    """
  	Propagate EPG states through a period of relaxation over an 
        interval T	
  
    Arguments:
    - `FpFmZ`: 3xN vector of F+, F- and Z states.
    - `T1`: Relaxation times (s)
    - `T2`: Relaxation times (s)
    - `T`: Time interval (s)
    - `kg`: k-space traversal due to gradient (rad/m) for diffusion
    - `D`: Diffusion coefficient (m^2/s)
    - `Gon`: 0 if no gradient on, 1 if gradient on
    - `noadd`: 1 to not add higher-order states - see epg_grad
    """
    
    # set up the relaxation
    E2 = np.exp(-T / T2)
    E1 = np.exp(-T / T1)
    EE = np.diag(np.array(np.hstack((E2, E2, E1))))

    #% Decay of states due to relaxation alone.
    RR = 1. - E1
    #% Mz Recovery, affects only Z0 state, as 
    #% recovered magnetization is not dephased.
    FpFmZ = np.dot(EE, FpFmZ)


    #% Apply Relaxation
    FpFmZ[2,0] = FpFmZ[2,0]+RR

    return FpFmZ

def epg_grelax(FpFmZ, T1, T2, T, kg, D, Gon=1, noadd=0):
    """
  	Propagate EPG states through a period of relaxation, and
	diffusion over an interval T, with or without a gradient.
	Leave last 3 blank to exclude diffusion effects.
  
    Arguments:
    - `FpFmZ`: 3xN vector of F+, F- and Z states.
    - `T1`: Relaxation times (s)
    - `T2`: Relaxation times (s)
    - `T`: Time interval (s)
    - `kg`: k-space traversal due to gradient (rad/m) for diffusion
    - `D`: Diffusion coefficient (m^2/s)
    - `Gon`: 0 if no gradient on, 1 if gradient on
    - `noadd`: 1 to not add higher-order states - see epg_grad
    """
    
    # set up the relaxation
    E2 = np.exp(-T / T2)
    E1 = np.exp(-T / T1)
    EE = np.diag(np.array(np.hstack((E2, E2, E1))))

    #% Decay of states due to relaxation alone.
    RR = 1. - E1
    #% Mz Recovery, affects only Z0 state, as 
    #% recovered magnetization is not dephased.
    FpFmZ = np.dot(EE, FpFmZ)


    #% Apply Relaxation
    FpFmZ[2,0] = FpFmZ[2,0]+RR

    #% Recovery  ( here applied before diffusion,
    #% but could be after or split.)
    
    #% Model Diffusion Effects
    Findex = np.arange(0., len(FpFmZ[0,:]))
    #% index of states, 0...N-1
    bvalZ = (np.square(Findex * kg)) * T
    #% diffusion  for Z states, assumes that
    #% the Z-state has to be refocused, so
    #% this models "time between gradients"
    #% For F states, the following models the additional diffusion time
    #% (Findex) and the fact that the state will change if the gradient is
    #% on (0.5*Gon), then the additional diffusion *during* the gradient, 
    #% ... Gon*kg^2/12 term.
    bvalp = (np.square(( Findex + 0.5*Gon)*kg) + Gon*(kg**2)/12)*T    #% for F+ states
    bvalm = (np.square((-Findex + 0.5*Gon)*kg) + Gon*(kg**2)/12)*T    #% for F- states

    #% for F- states
    FpFmZ[0,:] = FpFmZ[0,:]*np.exp(-bvalp * D)    #% diffusion on F+ states
    FpFmZ[1,:] = FpFmZ[1,:]*np.exp(-bvalm * D)    #% diffusion on F- states
    FpFmZ[2,:] = FpFmZ[2,:]*np.exp(-bvalZ * D)    #% diffusion of Z states.



    BV = np.array(np.vstack((np.hstack((bvalp)), np.hstack((bvalm)), np.hstack((bvalZ)))))
    #% For output. 
    
    if Gon == 1.:
        if kg >= 0.:
            FpFmZ = epg_grad(FpFmZ, noadd)
            #% Advance states.
        else:
            FpFmZ = epg_mgrad(FpFmZ, noadd)
            #% Advance states by negative gradient.

    return FpFmZ

def epg_rf(FpFmZ, alpha, phi=(-np.pi/2)):
    """
    Propogate EPG states through RF pulse.
    Arguments:
    - `FpFmZ`: 3xN vector of F+, F- and Z states.
    - `alpha`:
    - `phi`:
    """
    

    RR = np.array([[np.cos(alpha/2.)**2, np.exp(2*1j*phi)*(np.sin(alpha/2))**2, -1j*np.exp(1j*phi)*np.sin(alpha)],\
                   [np.exp(-2*1j*phi)*(np.sin(alpha/2))**2, (np.cos(alpha/2))**2, 1j*np.exp(-1j*phi)*np.sin(alpha)], \
                   [-1j/2*np.exp(-1j*phi)*np.sin(alpha), 1j/2*np.exp(1j*phi)*np.sin(alpha), np.cos(alpha)]])

    FpFmZ = np.dot(RR, FpFmZ)
    return FpFmZ

def epg_FZ2spins(FpFmZ, N=0, frac=0):
    """Function returns a 3xN array of the magnetization vectors [Mx My Mz]' that 
       is represented by the EPG state FpFmZ.  Note that the magnetization assumes 
       that a "voxel" is divided into N states, as is necessary to make the conversion
       N must be at least as large as twice the number of F+ states minus one.
          
    Arguments:
    - `FpFmZ`: F,Z states, 3xQ vector where there are Q states, where rows represent F+, F-,
               and Z states starting at 0.
    - `N`:  number of spins used to represent the states.  Default is minumum.
    - `frac`: (optional) fraction of state to increment/decrement so if frac=1, this is 
              equivalent to first passing FpFmZ through epg_grad().  This is mostly just
              to make plots
    """

    if FpFmZ.shape[0] is not 3:
        sys.exit("Error: must be 3xQ array")

    Ns = FpFmZ.shape[1]

    if N is 0:
        N = 2*Ns - 1

    # use a matrix for FFT to support arbitrary N
    x = np.asarray(range(N),dtype=float)/N
    x = x.reshape(-1,1)

    NsPlusFrac = np.asarray(np.arange(-(Ns-1),Ns)+frac,dtype=float)
    NsPlusFrac = NsPlusFrac.reshape(-1,1)

    ph = np.exp(1j*2*np.pi*x.dot(NsPlusFrac.T))

    
    Fstates = np.hstack((np.flipud(FpFmZ[1,1:]).conj(), FpFmZ[0,:]))

    Mxy = ph.dot(Fstates) # Fourier transform to Mxy

    # find Mz
    NsArray = np.asarray(range(Ns))
    NsArray = NsArray.reshape(-1,1)
    ph = np.exp(1j*2*np.pi*x.dot(NsArray.T))
    FpFmZ[2,0] /= 2
    Mz = 2*np.real(ph.dot(FpFmZ[2,:]))
    M = np.vstack((np.real(Mxy),
                   np.imag(Mxy),
                   Mz))/N

    return M


def epg_cpmg(flipangle, etl, T1, T2, esp):
    """
	EPG Simulation of CPMG sequence.  First flip angle
	is 90 about y axis, and others by default are about
	x-axis (make flipangle complex to change that).

	flipangle = refocusing flip angle or list (radians)
	etl = echo train length, if single flipangle is to be repeated.
	T1,T2,esp = relaxation times and echo spacing (arb. units).

	Note that if refoc flip angle is constant, less than pi, and etl > 1 the
	first refocusing flip angle is the average of 90 and the desired
	refocusing flip angle, as proposed by Hennig.

	All states are kept, for demo purposes, though this 
	is not really necessary.
    
    Arguments:
    - `flipangle`: the flipangle, can be scalar or vector
    - `etl`: the echo train length
    - `T1`: relaxation time
    - `T2`: relaxation time
    - `esp`: echo spacing
    """

    try:
        if (len(flipangle) == 1):
            scalarCheck = True
        else:
            scalarCheck = False
    except TypeError:
        scalarCheck = True
        
    if scalarCheck:
        flipangle = flipangle*np.ones((etl),dtype=complex)
        if (etl > 1) and (np.absolute(flipangle[0]) < np.pi):
            flipangle[0] = (np.pi * np.exp(1j*np.angle(flipangle[1])) + flipangle[1])/2        


    if (etl > len(flipangle)):
        flipangleTemp = np.zeros((etl),dtype=complex)
        flipangleTemp[0:len(flipangle)] = flipangle
        flipangleTemp[len(flipangle):] = flipangle[-1]
        flipangle = flipangleTemp
        
        

    P = np.zeros((3, (2*etl)),dtype=complex)    #% Allocate all known states, 2 per echo.
    P[2,0] = 1.                                   #% Initial condition/equilibrium.
    
    Pstore = np.zeros(((4*etl), etl),dtype=complex)    #% Store F,F* states, with 2 gradients per echo
    Zstore = np.zeros(((2*etl), etl),dtype=complex)    #% Store Z states, with 2 gradients per echo

    #% -- 90 excitation
    P = epg_rf(P, (np.pi/2.), (np.pi/2.))    #% Do 90 tip.

    s = np.zeros((etl),dtype=complex)           #% Allocate signal vector to store.
    
    for ech in range(etl):

        P = epg_grelax(P, T1, T2, (esp/2.), 1., 0., 1., 1.)                 #% -- Left crusher
        P = epg_rf(P, np.abs(flipangle[ech]), np.angle(flipangle[ech]))        #% -- Refoc. RF
        P = epg_grelax(P, T1, T2, (esp/2.), 1., 0., 1., 1.)                 #% -- Right crusher
        
        s[ech] = P[0,0]                                                           #% Signal is F0 state.
        Pstore[2*etl-1:4*etl-1, ech] = P[1,:].T                                 #% Put in negative states
        Pstore[0:2*etl,ech] = np.flipud(P[0,:].T)                              #% Put in positive, overwrite center.
        Zstore[:,ech] = P[2,:].T

    return s

def main():
    """main function
    """

    flipangle = 160./180*np.pi
    etl = 70
    T1 = 50. # ms
    T2 = 300. # ms
    esp = 3.5 # ms
    angle = 90./180*np.pi
    
    signal =  epg_cpmg(flipangle, etl, T1, T2, esp)

    print "signal for:"
    for ii in range(len(signal)):
        print "\techo[%i] = %f" % (ii, abs(signal[ii]))

if __name__ == "__main__":
    main()
