#!/usr/bin/env python3
# coding: utf-8


'FISHERMATRIXBNSPOP.PY -- simulate binary neutron star merger population with Fisher matrix parameter uncertainties'
__usage__ = 'FisherMatrixBNSpop.py'
__author__ = 'Philippe Landry (pgjlandry@gmail.com)'
__date__ = '07-2024'


# import packages

from argparse import ArgumentParser
import configparser
import numpy as np
import pandas as pd
import math
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

import astropy.units as u
from astropy.coordinates import Distance
from astropy.cosmology import Planck15 as cosmo

from src.utils import *


# parse user input

parser = ArgumentParser(description=__doc__)
parser.add_argument('inifile')
args = parser.parse_args()

INPATH = str(args.inifile)

configParser = configparser.ConfigParser(allow_no_value=True)   
configParser.read(INPATH)

RATE = float(configParser['population-model']['RATE'])
MASS_MODEL = str(configParser['population-model']['MASS_MODEL'])
MMIN = float(configParser['population-model']['MMIN'])
MMAX = float(configParser['population-model']['MMAX'])
MPARAMS = str(configParser['population-model']['MPARAMS']).split(',')
EOS = str(configParser['eos']['EOS'])
EOS_PATH = str(configParser['eos']['EOS_PATH'])
NPOP = str(configParser['observing-scenario']['NPOP'])
TIME = float(configParser['observing-scenario']['TIME'])
ZMIN = float(configParser['observing-scenario']['ZMIN'])
ZMAX = float(configParser['observing-scenario']['ZMAX'])
NETS = str(configParser['observing-scenario']['NETS']).split(',')
NOISE = int(configParser['observing-scenario']['NOISE'])
WF_MODEL = str(configParser['fisher-matrix']['WF_MODEL'])
FMIN = float(configParser['fisher-matrix']['FMIN'])
FISHER_PARAMS = str(configParser['fisher-matrix']['FISHER_PARAMS'])
CONV_COS = str(configParser['fisher-matrix']['CONV_COS']).split(',')
CONV_LOG = str(configParser['fisher-matrix']['CONV_LOG']).split(',')
USE_ROT = int(configParser['fisher-matrix']['USE_ROT'])
SEED = int(configParser['output']['SEED'])

NETS_DICT = {'XG': ['CE-40_H','CE-40_L','ET_V'], 'A+': ['A+_H','A+_L','V+_V'], 'HLV': ['aLIGO_H','aLIGO_L','aLIGO_V']} # define supported detector networks

if 'None' in MPARAMS: MPARAMS = []
MPARAMS = MPARAMS + [MMIN] # mass distribution parameters, MMAX automatically appended later

OUTDIR = './{0}_BNS_{1}_v{2}/'.format(MASS_MODEL,EOS,SEED) # output directory
if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)


# load population model and equation of state

# mass model
BNS_MASS_PDF = get_mass_model(MASS_MODEL)

# luminosity distance model
DL_PDF, DPARAMS = get_redshift_model(ZMIN,ZMAX)

# equation of state
eos_dat = np.genfromtxt(EOS_PATH,names=True,delimiter=',',dtype=None,encoding=None) # load m-Lambda table
Ms = eos_dat['M']
Lambdas = eos_dat['Lambda']

Mmax_pos = np.argmax(Ms) # truncate table at maximum NS mass
Ms = Ms[:Mmax_pos+1]
Mmax = Ms[-1]
Lambdas = Lambdas[:Mmax_pos+1]

Lambda_of_M = interp1d(Ms,Lambdas,kind='linear',bounds_error=True) # interpolate Lambda(M) function


# sample injected events from population model

np.random.seed(SEED)

# calculate size of simulated population
if NPOP == 'False':
    vt = 1e-9*cosmo.comoving_volume(ZMAX).value*TIME # sensitive volume-time product in Gpc^3*yr
    npop = math.ceil(RATE*vt)
else:
    npop = int(NPOP)

# sample from mass model
m_grid = np.arange(max(MMIN,min(Ms)),min(MMAX,max(Ms)),0.01) # regular grid in NS masses

m1m2_grid = []
for m1 in m_grid:
    for m2 in m_grid:
        m1m2_grid += [(m1,m2)] # regular 2D grid in BNS masses

m1s_grid = np.array([m1 for m1,m2 in m1m2_grid]) # m1 points from the 2D grid
m2s_grid = np.array([m2 for m1,m2 in m1m2_grid]) # m2 points from the 2D grid
grid_pts = range(len(m1m2_grid)) # label each point in 2D grid

mparams = MPARAMS + [Mmax]
m1m2_wts = BNS_MASS_PDF(m1s_grid,m2s_grid,mparams) # weight each binary mass grid point by its probability according to the chosen BNS mass distribution, and truncate at the EOS's Mmax

sample_pts = np.random.choice(grid_pts,npop,replace=True,p=m1m2_wts/np.sum(m1m2_wts)) # draw samples from BNS mass distribution
m1s = m1s_grid[sample_pts]
m2s = m2s_grid[sample_pts]

# get corresponding tidal deformability samples, given equation of state
Lambda1s = np.array([Lambda_of_M(m1) for m1 in m1s]) # use interpolated M-Lambda relation
Lambda2s = np.array([Lambda_of_M(m2) for m2 in m2s])

# sample from luminosity distance model
dl_grid = np.arange(DPARAMS[0],DPARAMS[1],10.) # regular grid in luminosity distance
dl_wts = DL_PDF(dl_grid,DPARAMS)
dls = np.random.choice(dl_grid,npop,replace=True,p=dl_wts/np.sum(dl_wts))

# sample in inclination, sky location and other nuisance parameters
thetajn_grid = np.arange(0.,2.*np.pi,0.01)
thetajn_wts = [np.abs(np.sin(thetajn)) for thetajn in thetajn_grid]
thetajns = np.random.choice(thetajn_grid,npop,replace=True,p=thetajn_wts/np.sum(thetajn_wts))

dec_grid = np.arange(0.,2.*np.pi,0.01)
dec_wts = [np.abs(np.cos(dec)) for dec in dec_grid]
decs = np.random.choice(dec_grid,npop,replace=True,p=dec_wts/np.sum(dec_wts))

ras = np.random.uniform(0.,2.*np.pi,npop)
psis = np.random.uniform(0.,np.pi,npop)
phases = np.random.uniform(0.,2.*np.pi,npop)


# save population to data frame

out_dict = {}
out_dict['m1'] = m1s
out_dict['m2'] = m2s
out_dict['mc'] = mchirp_from_mass1_mass2(m1s,m2s)
out_dict['eta'] = eta_from_mass1_mass2(m1s,m2s)
out_dict['Lambda1'] = Lambda1s
out_dict['Lambda2'] = Lambda2s
out_dict['Lambda_tilde'] = lambdatilde(Lambda1s,Lambda2s,m1s,m2s)
out_dict['deltaLambda_tilde'] = deltalambdatilde(Lambda1s,Lambda2s,m2s/m1s)
out_dict['dL'] = dls
out_dict['z'] = Distance(dls,unit=u.Mpc).compute_z(cosmology=cosmo)
out_dict['dec'] = decs
out_dict['thetajn'] = thetajns
out_dict['ra'] = ras
out_dict['psi'] = psis
out_dict['phase'] = phases
out_dat = pd.DataFrame(out_dict)


# calculate event signal to noise ratio and Fisher matrix errors based on its source parameters

for net_key in NETS:
    
    snrs, det_mcs, det_etas, det_Lts, dmcs, detas, dLts = [], [], [], [], [], [], []
    
    for i in tqdm(range(npop)):
        
        inj_params = out_dat.iloc[i]

        snr, dmc, deta, dLt = get_snrs(net_key,NETS_DICT,WF_MODEL,FMIN,FISHER_PARAMS,CONV_COS,CONV_LOG,USE_ROT,inj_params)

        snrs += [snr]

        det_mcs += [np.abs(inj_params['mc']+NOISE*np.random.normal(0.,dmc))]
        det_Lts += [np.abs(inj_params['Lambda_tilde']+NOISE*np.random.normal(0.,dLt))]
        
        eta_det = np.abs(inj_params['eta']+NOISE*np.random.normal(0.,deta))
        if eta_det >= 0.25: eta_det = 0.24999+(0.25-eta_det)
        det_etas += [eta_det]
        
        dmcs += [dmc]
        detas += [deta]
        dLts += [dLt]
    
    # save fisher matrix errors to data frame
    out_dat['snr'+'_{0}'.format(net_key)] =  snrs

    out_dat['mc'+'_{0}'.format(net_key)] =  det_mcs
    out_dat['eta'+'_{0}'.format(net_key)] =  det_etas
    out_dat['Lt'+'_{0}'.format(net_key)] =  det_Lts

    out_dat['dmc'+'_{0}'.format(net_key)] =  dmcs
    out_dat['deta'+'_{0}'.format(net_key)] =  detas
    out_dat['dLt'+'_{0}'.format(net_key)] =  dLts


# output data frame as csv

out_dat.to_csv(OUTDIR+'{0}_BNS_{1}.csv'.format(MASS_MODEL,EOS),index=False,float_format='%.4e')