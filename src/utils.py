# load packages

import numpy as np
import scipy.special

import astropy.units as u
from astropy.coordinates import Distance
from astropy.cosmology import Planck15 as cosmo

from gwbench import network

# define constants

G = 6.67e-8
c = 2.998e10
Msol = 1.99e33
H0 = cosmo.H(0).value # 67.8 for Planck 2015

# define population models

def get_mass_model(POPMOD):

    if POPMOD == 'unif':

        def NS_MASS_PDF(m,params): # NS mass distribution probability density function

          mmin, mmax = params

          if np.isscalar(m): m = np.array([m])
          else: m = np.array(m)
          z = np.zeros(len(m))

          p = np.full(len(m),1./(mmax-mmin)) # uniform mass distribution

          return np.where((m > mmax) | (m < mmin), z, p) # this enforces the mmin and mmax cutoffs

    elif POPMOD == 'bimod':

        def NS_MASS_PDF(m,params): # NS mass distribution probability density function

          def gaussian(m,lambdaa):

            mu, sigma = lambdaa[:2]

            if np.isscalar(m): m = np.array([m])
            else: m = np.array(m)

            p = np.exp(-((m-mu)/(np.sqrt(2)*sigma))**2)/(sigma*np.sqrt(2*np.pi))

            return p

          mu1, sigma1, mu2, sigma2, w, mmin, mmax = params

          if np.isscalar(m): m = np.array([m])
          else: m = np.array(m)
          z = np.zeros(len(m))

          norm1 = 0.5*(scipy.special.erf((mmax-mu1)/(np.sqrt(2)*sigma1))-scipy.special.erf((mmin-mu1)/(np.sqrt(2)*sigma1)))
          norm2 = 0.5*(scipy.special.erf((mmax-mu2)/(np.sqrt(2)*sigma2))-scipy.special.erf((mmin-mu2)/(np.sqrt(2)*sigma2)))
          p = w*gaussian(m,(mu1,sigma1))/norm1 + (1.-w)*gaussian(m,(mu2,sigma2))/norm2 # bimodal mass distribution

          return np.where((m > mmax) | (m < mmin), z, p) # this enforces the mmin and mmax cutoffs
    
    else: 
        
        assert(POPMOD == 'unif' or POPMOD == 'bimod')

    def BNS_MASS_PDF(m1,m2,params): # BNS mass distribution probability density function

        if np.isscalar(m1): m1 = np.array([m1])
        else: m1 = np.array(m1)
        if np.isscalar(m2): m2 = np.array([m2])
        else: m2 = np.array(m2)
        z = np.zeros(len(m1))

        p = NS_MASS_PDF(m1,params)*NS_MASS_PDF(m2,params) # random pairing of m1, m2 drawn from same underlying NS mass distribution

        return np.where(m1 < m2, z, p) # this enforces m2 < m1
        
    return BNS_MASS_PDF

def get_redshift_model(ZMIN,ZMAX):
    
    dlmin = Distance(z=ZMIN, unit=u.Mpc).value
    dlmax = Distance(z=ZMAX, unit=u.Mpc).value

    def SFR(z): # star formation rate as a function of redshift

        if np.isscalar(z): z = np.array([z])
        else: z = np.array(z)

        return  (1.+1./2.9**5.6)*(1.+z)**2.7/(1.+((1.+z)/2.9)**5.6) # Madau-Dickinson SFR, up to normalization

    def DL_PDF(dl,params): # luminosity distance distribution probability density function

      dlmin, dlmax = params

      if np.isscalar(dl): dl = np.array([dl])
      else: dl = np.array(dl)
      Z = np.zeros(len(dl))

      z = np.array([Distance(d,unit=u.Mpc).compute_z(cosmology=cosmo) for d in dl])
      p = 4.*np.pi*dl**2*SFR(z)/(cosmo.H(z).value*(1.+z)**3) # uniform in comoving volume distribution, with Madau-Dickinson SFR; see https://arxiv.org/abs/1505.05607, https://arxiv.org/abs/1805.10270

      return np.where((dl > dlmax) | (dl < dlmin), Z, p) # this enforces the dlmin and dlmax cutoffs

    return DL_PDF, (dlmin,dlmax)

# define transformed observables

def mchirp_from_mass1_mass2(mass1, mass2):
    """Returns the chirp mass from mass1 and mass2."""
    return eta_from_mass1_mass2(mass1, mass2)**(3./5) * (mass1+mass2)

def eta_from_mass1_mass2(mass1, mass2):
    """Returns the symmetric mass ratio from mass1 and mass2."""
    return mass1*mass2 / (mass1+mass2)**2.

def q_from_mass1_mass2(mass1, mass2):
    """Returns the chirp mass from mass1 and mass2."""
    return mass2/mass1

def lambdatilde(lambda1,lambda2,m1,m2): 
    mtotal = m1+m2
    m1fac=m1+12*m2
    m2fac=m2+12*m1
    return 16./13*(m1fac*m1**(4)*lambda1+m2fac*m2**(4)*lambda2)/mtotal**(5)

def deltalambdatilde(lambda1,lambda2,q):
    oneplusq6 = np.power(1+q,6)
    m1fac=(1319.-7996.*q-11005.*q*q)/1319./oneplusq6
    m2fac=np.power(q,4)*(1319.*q*q-7996.*q-11005.)/1319./oneplusq6
    return (m1fac*lambda1-m2fac*lambda2)

# define wrapper function for gwbench call

def get_snrs(net_key,nets_dict,wf_model_name,fmin,deriv_symbs_string,conv_cos,conv_log,use_rot,inj_params):
       
    mc,eta,Lt,dLt,dl,z,dec,thetajn,ra,psi,phase = inj_params['mc'], inj_params['eta'], inj_params['Lambda_tilde'], inj_params['deltaLambda_tilde'], inj_params['dL'], inj_params['z'], inj_params['dec'], inj_params['thetajn'], inj_params['ra'], inj_params['psi'], inj_params['phase']
    
    if eta == 0.25: eta = 0.24999 # hack to fix bug with exactly equal-mass binaries

    injection_parameters = {
        'Mc':    mc*(1.+z), # detector frame chirp mass in Msun
        'eta':   eta,
        'lam_t': Lt,
        'delta_lam_t': dLt,
        'chi1z': 0,
        'chi2z': 0,
        'DL':    dl, # Mpc
        'tc':    0,
        'phic':  phase,
        'iota':  thetajn, # iota = thetajn for nonprecessing binaries
        'ra':    ra,
        'dec':   dec,
        'psi':   psi,
        'gmst0': 0
        }
    
    fcontact = np.sqrt(G*(mc/eta**(3./5.))*(1.+z)*Msol/(24.*1e5)**3)/np.pi # approx 1.5 kHz
    f = np.arange(fmin,fcontact,2**-4) # frequency range

    network_spec = nets_dict[net_key]
    net = network.Network(network_spec)
    net.set_wf_vars(wf_model_name=wf_model_name)
    net.set_net_vars(
        f=f, inj_params=injection_parameters,
        deriv_symbs_string=deriv_symbs_string,
        conv_cos=conv_cos, conv_log=conv_log,
        use_rot=use_rot
        )
    net.set_logger_level('ERROR')

    net.calc_wf_polarizations()
    net.calc_wf_polarizations_derivs_num()
    net.setup_ant_pat_lpf_psds()
    net.calc_det_responses()
    net.calc_det_responses_derivs_num()
    net.calc_snrs()
    net.calc_errors()

    snr = net.snr
    errs = net.errs
    dlogmc, dlogdl, deta, dLt = errs['log_Mc'], errs['log_DL'], errs['eta'], errs['lam_t']
    ddl = dl*dlogdl
    dz = H0*1e5*ddl/c
    dmcdet = mc*(1.+z)*dlogmc # detector-frame chirp mass error
    dmc = np.sqrt(dmcdet**2/(1.+z)**2 + mc**2*dz**2/(1.+z)**2) # source-frame chirp mass error (dominated by z error at cosmological distances)

    return snr, dmc, deta, dLt