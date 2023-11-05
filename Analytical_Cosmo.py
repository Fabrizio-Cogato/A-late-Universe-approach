"""
@file Analytical_Cosmo.py
@date September 6, 2023
@authors Fabrizio Cogato <fabrizio.cogato@inaf.it>
         Michele Moresco <michele.moresco@unibo.it>

Please remember to cite: https://ui.adsabs.harvard.edu/abs/2023arXiv230901375C
"""

import numpy as np
import os
import emcee
import random
from multiprocessing import Pool

from Analytical_Cosmo_Config.py import * #Configuration file


pool_obj = Pool() #Parallelization
##############################
#### Functions Definition ####
##############################
def run_MCMC(prior, cosmo, probe, dir_chain, arguments, nwalkers, initials, Nsteps):
    ############### Markov Chain Monte Carlo
    initial = initials
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]
    os.chdir(dir_chain)
    filename = "Chain_{0}Prior_{1}_{2}.h5".format(prior, cosmo, probe) #chain name
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    lnprb=globals()["lnflatprob{0}_{1}".format(probe,cosmo)] #calling prob function from Probes_Combination_Cluster_Config
    
    #with MPIPool() as pool: #Parallelize on Cluster
    with Pool() as pool: #Parallelize on single computer
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprb, args=arguments, backend=backend, pool=pool)           # Create sampler
        sampler.run_mcmc(p0, Nsteps, progress=True, store=True)                                                         # Run                                                         # Run from last sample of backend                                                                             

################################
#### Directories Definition ####
################################
dir_home = os.getcwd()
dir_data = dir_home+'/path/to/data/'
dir_chain = dir_home+'/path/to/chains/'

#############################
#### Filename Definition ####
#############################
##############
## BAO data ##
os.chdir(dir_data+'/BAO/')
# Ross et al 2015
filename = 'data_DV_ross.txt'
zR= np.genfromtxt(filename, usecols=(0), unpack=True, delimiter="\t", comments='#')
zRoss=[]
zRoss.append(zR)
zRoss=np.array(zRoss)
DVrdRoss, errDVrdRoss = np.genfromtxt(filename, usecols=(1,2), unpack=True, delimiter="\t", comments='#')
# de Mattia et al 2020
filename = 'data_DV_demattia.txt'
zD = np.genfromtxt(filename, usecols=(0), unpack=True, delimiter="\t", comments='#')
zDemattia=[]
zDemattia.append(zD)
zDemattia=np.array(zDemattia)
DVrdDemattia, errDVrdDemattia = np.genfromtxt(filename, usecols=(1,2), unpack=True, delimiter="\t", comments='#')
# du Mas des Bourboux et al 2020
filename = 'data_DM_dumas.txt'
zM = np.genfromtxt(filename, usecols=(0), unpack=True, delimiter="\t", comments='#')
zDumas=[]
zDumas.append(zM)
zDumas=np.array(zDumas)
DMrdDumas, errDMrdDumas = np.genfromtxt(filename, usecols=(1,2), unpack=True, delimiter="\t", comments='#')
filename = 'data_DH_dumas.txt'
DHrdDumas, errDHrdDumas = np.genfromtxt(filename, usecols=(1,2), unpack=True, delimiter="\t", comments='#')
#Hou et al 20xx
filename = 'data_hou.txt'
zH, dataHou = np.genfromtxt(filename, usecols=(0,1), unpack=True, delimiter="\t", comments='#')
zHou=[]
zHou.append(zH[0])
zHou=np.array(zHou)
DMrdHou=dataHou[0]
DHrdHou=dataHou[1]
filename='cov_DM_DH_hou.txt'
covHou = np.genfromtxt(filename, usecols=(0), unpack=True, comments='#')
n=len(dataHou)
covHou = np.matrix(covHou.reshape(n, n))
inv_cov_Hou = np.linalg.inv(covHou)
#Gil et al 2020
filename = 'data_gil.txt'
zG, dataGil = np.genfromtxt(filename, usecols=(0,1), unpack=True, delimiter="\t", comments='#')
zGil =[]
zGil.append(zG[0])
zGil=np.array(zGil)
DMrdGil=dataGil[0]
DHrdGil=dataGil[1]
filename='cov_DM_DH_gil.txt'
covGil = np.genfromtxt(filename, usecols=(0), unpack=True, comments='#')
n=len(dataGil)
covGil = np.matrix(covGil.reshape(n, n))
inv_cov_Gil = np.linalg.inv(covGil)
#Alam et al 2017
filename = 'data_alam.txt'
zA, dataAlam = np.genfromtxt(filename, usecols=(0,1), unpack=True, delimiter="\t", comments='#')
zAlam=np.array([zA[0],zA[2]])
DMrdAlam=[dataAlam[0],dataAlam[2]]
DHrdAlam=[dataAlam[1],dataAlam[3]]
filename='cov_DM_DH_alam.txt'
covAlam = np.genfromtxt(filename, usecols=(0), unpack=True, comments='#')
n=len(dataAlam)
covAlam = np.matrix(covAlam.reshape(n, n))
inv_cov_Alam = np.linalg.inv(covAlam)

#############
## CC data ##
os.chdir(dir_data+'/CC/')

filename = 'data_CC.txt'
zC, Hz, errHz = np.genfromtxt(filename, comments='#', usecols=(0,1,2), unpack=True, delimiter="\t")
# Cov
filename = 'cov_CC.txt' # cov matrix created with data_MM20.dat
cov = np.genfromtxt(filename, comments='#', usecols=(0), unpack=True, delimiter="\t")
cov_matC = np.matrix(cov.reshape(len(zC), len(zC)))
inv_cov_matC = np.linalg.inv(cov_matC)

##############
## SNe data ##
os.chdir(dir_data+'/SN/')

## Pantheon+
## Full Sample ##
#filename='data_SN_pantheon+.txt'
#zS, DmS, errDmS = np.genfromtxt(filename, usecols=(2,8,9), unpack=True, comments='#')
#filename='cov_SN_pantheon+.txt'
#covS_Pp = np.genfromtxt(filename, usecols=(0), unpack=True, comments='#')
#covS = np.matrix(covS_Pp.reshape(len(zS), len(zS)))
#inv_cov_matS = np.linalg.inv(covS)

## cut at z< 0.01 ##
filename='data_SN_pantheon+_cut_z0.01.txt'
zS, DmS, errDmS = np.genfromtxt(filename, usecols=(0,1,2), unpack=True, comments='#')
filename='cov_SN_pantheon+_cut_z0.01.txt'
cov = np.genfromtxt(filename, comments='#', usecols=(0), unpack=True, delimiter="\t")
cov_matS = np.matrix(cov.reshape(len(zS), len(zS)))
inv_cov_matS = np.linalg.inv(cov_matS)

os.chdir(dir_home)


######################
## Input Parameters ##
######################
# Initial parameter values (random)
in_H0= np.around(random.uniform(62, 78),1)
in_Om= np.around(random.uniform(0.1, 0.49),4)
in_OL= np.around(random.uniform(0.49, 0.89),4)
in_w0= np.around(random.uniform(-1.99, -0.51),2)
in_wa= np.around(random.uniform(-0.99, 0.99),2)
in_M= np.around(random.uniform(19, 20),1)
in_rd= np.around(random.uniform(130, 170),1)
in_a= np.around(random.uniform(0.01, 0.99),2)
in_b= np.around(random.uniform(0.99, 2.99),2)
in_intr= np.around(random.uniform(0.01, 0.99),2)


initials_FLCDM= np.array([in_H0, in_Om])

initials_OLCDM= np.array([in_H0, in_Om, in_OL])

initials_FwCDM= np.array([in_H0, in_Om, in_w0])

initials_OwCDM= np.array([in_H0, in_Om, in_OL, in_w0])

initials_Fw0waCDM= np.array([in_H0, in_Om, in_w0, in_wa])

initials_Ow0waCDM= np.array([in_H0, in_Om, in_OL, in_w0, in_wa])

#Chain properties
nwalkers=250 
Nsteps=5000

#Arguments of Probability functions
# BAO + CC + SN 
arg_BAOCCSN= (zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS)

# BAO + SN
arg_BAOSN= (zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS)
# CC + SN
arg_CCSN = (zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS)
# BAO + CC
arg_BAOCC= (zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC)


# SN
arg_SN = (zS, DmS, inv_cov_matS)
# BAO
arg_BAO = (zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)
# BAOALam
arg_BAOAlam = (zAlam, dataAlam, inv_cov_Alam)
# BAOHou
arg_BAOHou = (zHou, dataHou, inv_cov_Hou)
# BAOGil
arg_BAOGil = (zGil, dataGil, inv_cov_Gil)
# BAODumas
arg_BAODumas = (zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas)
# BAORoss
arg_BAORoss = (zRoss, DVrdRoss, errDVrdRoss)
# BAODemattia
arg_BAODemattia = (zDemattia, DVrdDemattia, errDVrdDemattia)


# CC
arg_CC= (zC, Hz, inv_cov_matC)


##############
#### Main ####
##############
## MCMC Run ##
#                                                                     example
# prior = choice of prior (flat, gauss, ..)                      ---> 'flat'
# cosmo = cosmology (flatLCDM, openw0CDM, ..)                    ---> 'FLCDM'
# probe = cosmo probe (CC, SNe, BAO, ..)                         ---> 'CC'
# dir_chain = save chain in ...                                  ---> dir_chain
# arguments = logprob arguments (z, Hz, ..)                      ---> arguments_fit=(z, Hz, errHz)
# nwalkers =  MCMC number of walkers                             ---> 250
# initials_FLCDM = parameter initial value (H0, Om)              ---> np.array([100, 0.5])
# Nsteps                                                         ---> 1000

################
####Run MCMC####
run_MCMC('Flat', 'FLCDM', 'BAOCCSNGRB', dir_chain, arg_BAOCCSN, nwalkers, initials_FLCDM, Nsteps)
run_MCMC('Flat', 'LCDM', 'BAOCCSNGRB', dir_chain, arg_BAOCCSN, nwalkers, initials_OLCDM, Nsteps)
run_MCMC('Flat', 'FwCDM', 'BAOCCSNGRB', dir_chain, arg_BAOCCSN, nwalkers, initials_FwCDM, Nsteps)
run_MCMC('Flat', 'wCDM', 'BAOCCSNGRB', dir_chain, arg_BAOCCSN, nwalkers, initials_OwCDM, Nsteps)
run_MCMC('Flat', 'Fw0waCDM', 'BAOCCSNGRB', dir_chain, arg_BAOCCSN, nwalkers, initials_Fw0waCDM, Nsteps)
run_MCMC('Flat', 'w0waCDM', 'BAOCCSNGRB', dir_chain, arg_BAOCCSN, nwalkers, initials_Ow0waCDM, Nsteps)
