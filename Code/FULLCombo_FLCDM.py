import numpy as np
import os
from astropy.cosmology import FlatLambdaCDM
import pandas as pd
import emcee
import astropy.units as u
import random
import math
##############################
#### Functions Definition ####
##############################
c = 299792.458
def run_MCMC(prior, cosmo, probe, dir_chain, arguments, nwalkers, initials, Nsteps):
    
    def mu_model_FLCDM(H0, Om, M, rd, rat, z):                                                               
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
        mu = np.array(cosmo.distmod(z)) - 25 + M
        return mu
    def E_model_FLCDM(H0, Om, M, rd, rat, z):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
        return np.array(cosmo.efunc(z))
    def Hz_model_FLCDM(H0, Om, M, rd, rat, z):
        arr = []
        for j in range(len(z)):
            arr.append(H0*E_model_FLCDM(H0, Om, M, rd, rat, z[j]))
        arr = np.array(arr)
        return arr
    def Hxratio_model_FLCDM(H0, Om, M, rd, rat, z):
        arr = []
        for j in range(len(z)):
            arr.append(H0*E_model_FLCDM(H0, Om, M, rd, rat, z[j])*rat)
        arr = np.array(arr)
        return arr
    #DH functions
    def DH_model_FLCDM(H0, Om, M, rd, rat, z):
        arr = []
        for j in range(len(z)):
            arr.append(c/(H0*E_model_FLCDM(H0, Om, M, rd, rat, z[j])))
        arr = np.array(arr)
        return arr
    def DHrd_model_FLCDM(H0, Om, M, rd, rat, z): 
        arr = []
        for j in range(len(z)):
            arr.append(c/(H0*E_model_FLCDM(H0, Om, M, rd, rat, z[j])*rd))
        arr = np.array(arr)
        return arr
    #DA 
    def DA_model_FLCDM(H0, Om, M, rd, rat, z):
        cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
        DA = np.array(cosmo.angular_diameter_distance(z))
        return DA
    #DM 
    def DM_model_FLCDM(H0, Om, M, rd, rat, z):
        arr = []
        for i in range(len(z)):
            arr.append(DA_model_FLCDM(H0, Om, M, rd, rat, z[i]) * (1+z[i]))
        arr = np.array(arr)
        arr.shape
        return arr
    def DMratio_model_FLCDM(H0, Om, M, rd, rat, z):
        arr = []
        for i in range(len(z)):
            arr.append(DA_model_FLCDM(H0, Om, M, rd, rat, z[i]) * (1+z[i]) / rat)
        arr = np.array(arr)
        arr.shape
        return arr
    def DMrd_model_FLCDM(H0, Om, M, rd, rat, z):
        dM = DM_model_FLCDM(H0, Om, M, rd, rat, z)
        arr = []
        for i in range(len(z)):
            arr.append(dM[i]/rd)
        arr = np.array(arr)
        arr.shape
        return arr
    #DV
    def rdDV_model_FLCDM(H0, Om, M, rd, rat, z):
        dh = DH_model_FLCDM(H0, Om, M, rd, rat, z)  
        dm = DM_model_FLCDM(H0, Om, M, rd, rat, z)
        arr = []
        for i in range(len(z)):
            arr.append(rd/(np.cbrt(z[i]*dh[i]*dm[i]**2)))
        arr = np.array(arr)
        arr.shape
        return arr       
    def DV_model_FLCDM(H0, Om, M, rd, rat, z):
        dh = DH_model_FLCDM(H0, Om, M, rd, rat, z)  
        dm = DM_model_FLCDM(H0, Om, M, rd, rat, z)
        arr = []
        for i in range(len(z)):
            arr.append(np.cbrt(z[i]*dh[i]*dm[i]**2))
        arr = np.array(arr)
        arr.shape
        return arr
    #A,F
    def A_model_FLCDM(H0, Om, M, rd, rat, z):
        dV = DV_model_FLCDM(H0, Om, M, rd, rat, z)  
        arr = []
        for i in range(len(z)):
            arr.append((H0*dV[i]*(Om**0.5))/(c*z[i]))
        arr = np.array(arr)
        arr.shape
        return arr        
    def F_model_FLCDM(H0, Om, M, rd, rat, z):
        dA = DA_model_FLCDM(H0, Om, M, rd, rat, z)
        H = Hz_model_FLCDM(H0, Om, M, rd, rat, z)
        arr = []
        for i in range(len(z)):
            arr.append((1+z[i])*dA[i]*H[i]*(1/c))
        arr = np.array(arr)
        arr.shape
        return arr
    #Combo - BAO analysis
    def A_F_model_FLCDM(H0, Om, M, rd, rat, z):
        dV = DV_model_FLCDM(H0, Om, M, rd, rat, z)
        dA = DA_model_FLCDM(H0, Om, M, rd, rat, z)
        H = Hz_model_FLCDM(H0, Om, M, rd, rat, z)
        A=(H0*dV*(Om**0.5))/(c*z)
        F=(1+z)*dA*H*(1/c)
        arr = []
        for i in range(len(z)):
            arr.append(A[i])
        for i in range(len(z)):
            arr.append(F[i])
        arr = np.array(arr)
        arr.shape
        return arr
    def DMrat_Hrat_model_FLCDM(H0, Om, M, rd, rat, z):
        dM = DM_model_FLCDM(H0, Om, M, rd, rat, z)
        H = Hz_model_FLCDM(H0, Om, M, rd, rat, z)
        arr = []
        for i in range(len(z)):
            arr.append(dM[i]/rat)
            arr.append(H[i]*rat)
        arr = np.array(arr)
        arr.shape
        return arr
    def DMrd_DHrd_model_FLCDM(H0, Om, M, rd, rat, z):
        dM = DM_model_FLCDM(H0, Om, M, rd, rat, z)
        dH = DH_model_FLCDM(H0, Om, M, rd, rat, z)
        arr = []
        for i in range(len(z)):
            arr.append(dM[i]/rd)
            arr.append(dH[i]/rd)
        arr = np.array(arr)
        arr.shape
        return arr
    ###############################  
    #### Likelihood Definition ####
    ###############################
    ## BAO    
    ## Alam
    def lnlikeAlam_FLCDM(theta, zAlam, dataAlam, inv_cov_alam):                 
        H0, Om, M, rd, rat=theta
        chi2=0.
        ndim=np.shape(inv_cov_alam)[0]
        residual=dataAlam-DMrat_Hrat_model_FLCDM(H0, Om, M, rd, rat, zAlam)
        for i in range(0, ndim):
                for j in range(0, ndim):
                    chi2=chi2+((residual[i])*inv_cov_alam[i,j]*(residual[j]))
        return -0.5 * chi2       
    ## Wiggle
    def lnlikeWiggle_FLCDM(theta, zWiggle, dataWiggle, inv_cov_wiggleZ):                 
        H0, Om, M, rd, rat=theta
        chi2=0.
        ndim=np.shape(inv_cov_wiggleZ)[0]
        residual=dataWiggle-A_F_model_FLCDM(H0, Om, M, rd, rat, zWiggle)
        for i in range(0, ndim):
                for j in range(0, ndim):
                    chi2=chi2+((residual[i])*inv_cov_wiggleZ[i,j]*(residual[j]))
        return -0.5 * chi2
    ## SDSS
    def lnlikeSDSS_FLCDM(theta, zSDSS, dataSDSS, inv_cov_SDSS):                 
        H0, Om, M, rd, rat=theta 
        chi2=0.
        ndim=np.shape(inv_cov_SDSS)[0]
        residual=dataSDSS-DMrd_DHrd_model_FLCDM(H0, Om, M, rd, rat, zSDSS)
        for i in range(0, ndim):
                for j in range(0, ndim):
                    chi2=chi2+((residual[i])*inv_cov_SDSS[i,j]*(residual[j]))
        return -0.5 * chi2        
    ## 6dFGS
    def lnlike6dFGS_FLCDM(theta, z6dFGS, rdDV6dFGS, err6dFGS):
        H0, Om, M, rd, rat=theta            
        sigma2 = err6dFGS ** 2
        return -0.5 * np.sum((rdDV6dFGS - rdDV_model_FLCDM(H0, Om, M, rd, rat, z6dFGS)) ** 2 / sigma2)        
    ## eBOSS
    def lnlikeeBOSS_FLCDM(theta, zeBOSS, DHrdeBOSS, erreBOSS):
        H0, Om, M, rd, rat=theta              
        sigma2 = erreBOSS ** 2
        return -0.5 * np.sum((DHrdeBOSS - DHrd_model_FLCDM(H0, Om, M, rd, rat, zeBOSS)) ** 2 / sigma2)
    ## BAO Combo
    def lnlikeBAO_FLCDM(theta, zAlam, dataAlam, inv_cov_alam, zWiggle, dataWiggle, inv_cov_wiggleZ, zSDSS, dataSDSS, inv_cov_SDSS, z6dFGS, rdDV6dFGS, err6dFGS, zeBOSS, DHrdeBOSS, erreBOSS):
        return lnlikeAlam_FLCDM(theta,zAlam, dataAlam, inv_cov_alam)+lnlikeWiggle_FLCDM(theta, zWiggle, dataWiggle, inv_cov_wiggleZ)+lnlikeSDSS_FLCDM(theta, zSDSS, dataSDSS, inv_cov_SDSS)+lnlike6dFGS_FLCDM(theta, z6dFGS, rdDV6dFGS, err6dFGS)+lnlikeeBOSS_FLCDM(theta, zeBOSS, DHrdeBOSS, erreBOSS)
    ## CC
    def lnlikeCC_FLCDM(theta, zC, Hz, inv_cov_matC):                 
        H0, Om, M, rd, rat=theta 
        chi2=0.
        residual=Hz-Hz_model_FLCDM(H0, Om, M, rd, rat, zC)
        for i in range(0, len(zC)):
                for j in range(0, len(zC)):
                    chi2=chi2+((residual[i])*inv_cov_matC[i,j]*(residual[j]))
        return -0.5 * chi2
    ## SNe
    def lnlikeSN_FLCDM(theta, zS, DmS, inv_cov_matS):                 
        H0, Om, M, rd, rat=theta 
        mu = mu_model_FLCDM(H0, Om, M, rd, rat, zS)
        residual= DmS - mu
        chi2=0.
        for i in range(0, len(zS)):
                for j in range(0, len(zS)):
                    chi2=chi2+((residual[i])*inv_cov_matS[i,j]*(residual[j]))
        return -0.5 * chi2
    ## GRB
    def lnlikeFitGRB_FLCDM(theta, z, Dm, errDm):                  
        H0, Om, M, rd, rat=theta 
        mu = mu_model_FLCDM(H0, Om, M, rd, rat, z) + deltaM_FLCDM
        return -0.5 * np.sum(((Dm - mu) / errDm)**2)

    #######################################  
    #### Probes Combination Likelihood ####
    #######################################
    ###########################
    ## BAO + CC + SN + GRB
    def lnlikeBAOCCSNGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_alam, zWiggle, dataWiggle, inv_cov_wiggleZ, zSDSS, dataSDSS, inv_cov_SDSS, z6dFGS, rdDV6dFGS, err6dFGS, zeBOSS, DHrdeBOSS, erreBOSS, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, DmG, errDmG):
        return lnlikeBAO_FLCDM(theta, zAlam, dataAlam, inv_cov_alam, zWiggle, dataWiggle, inv_cov_wiggleZ, zSDSS, dataSDSS, inv_cov_SDSS, z6dFGS, rdDV6dFGS, err6dFGS, zeBOSS, DHrdeBOSS, erreBOSS)+lnlikeCC_FLCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_FLCDM(theta, zS, DmS, inv_cov_matS) + lnlikeFitGRB_FLCDM(theta, zG, DmG, errDmG)

    ###########
    ## Prior ##
    ###########
    # Flat Prior
    def lnflatprior_FLCDM(theta):                                 
        H0, Om, M, rd, rat=theta 
        if (50.0 < H0 < 100.0 and 0.0 < Om < 1.0 and 20. < M < 30.0 and 100 < rd < 200 and 0.9 < rat < 1.1 ):
            return 0.0
        return -np.inf
    ###############
    ## Posterior ##
    ###############    
    ## BAO + CC + SN + GRB
    # Flat Prior 
    def lnflatprobBAOCCSNGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_alam, zWiggle, dataWiggle, inv_cov_wiggleZ, zSDSS, dataSDSS, inv_cov_SDSS, z6dFGS, rdDV6dFGS, err6dFGS, zeBOSS, DHrdeBOSS, erreBOSS, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, DmG, errDmG):             # Cov Probability
        lp = lnflatprior_FLCDM(theta)
        return lp + lnlikeBAOCCSNGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_alam, zWiggle, dataWiggle, inv_cov_wiggleZ, zSDSS, dataSDSS, inv_cov_SDSS, z6dFGS, rdDV6dFGS, err6dFGS, zeBOSS, DHrdeBOSS, erreBOSS, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, DmG, errDmG) if np.isfinite(lp) else -np.inf

    ##############################
    ## Markov Chain Monte Carlo ##
    ##############################
    initial = initials
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]
    os.chdir(dir_chain)
    filename = "Chain_{0}Prior_{1}_{2}.h5".format(prior, cosmo, probe)
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, locals()["lnflatprob{0}_{1}".format(probe, cosmo)], args=arguments, backend=backend)           
    sampler.run_mcmc(p0, Nsteps, progress=True, store=True)                                                                                                          

################################
#### Directories Definition ####
################################
dir_home = os.getcwd()
dir_data = dir_home+'/Data/'
dir_chain = dir_home+'/Chains/'

#############################
#### Filename Definition ####
#############################
##############
## BAO data ##
##############
os.chdir(dir_data)
filename = 'data_DM_DH_SDSSIV.dat'
z, dataSDSS = np.genfromtxt(filename, usecols=(0,1), unpack=True, delimiter="\t")
z0=z[0]
zSDSS=[]
zSDSS.append(z0)
filename='cov_DM_DH_SDSSIV.dat'
cov_SDSS = np.genfromtxt(filename, usecols=(0), unpack=True, comments='#')
cov_SDSS = np.matrix(cov_SDSS.reshape(2, 2))
inv_cov_SDSS = np.linalg.inv(cov_SDSS)

filename = 'data_A_F_wiggleZ.dat'
z, dataWiggle = np.genfromtxt(filename, usecols=(0,1), unpack=True, delimiter="\t")
z0=z[0]
z1=z[1]
z2=z[2]
zWiggle=[]
zWiggle.append(z0)
zWiggle.append(z1)
zWiggle.append(z2)
filename='1000xCov_wiggleZ.dat'
cov_wiggleZ = np.genfromtxt(filename, usecols=(0), unpack=True, comments='#')
cov_wiggleZ = cov_wiggleZ / 1000
cov_wiggleZ = np.matrix(cov_wiggleZ.reshape(6, 6))
inv_cov_wiggleZ = np.linalg.inv(cov_wiggleZ)

filename = 'data_DM_H_alam.txt'
z, dataAlam = np.genfromtxt(filename, usecols=(0,1), unpack=True, delimiter="\t")
z0=z[0]
z1=z[2]
z2=z[4]
zAlam=[]
zAlam.append(z0)
zAlam.append(z1)
zAlam.append(z2)
filename='cov_DM_H_alam.dat'
cov_alam = np.genfromtxt(filename, usecols=(0), unpack=True, comments='#')
cov_alam = np.matrix(cov_alam.reshape(6, 6))
inv_cov_alam = np.linalg.inv(cov_alam)

filename = 'data_BAO_rdDV_6dFGS.txt'
z, rdDV6dFGS, err6dFGS = np.genfromtxt(filename, comments='#', usecols=(0,1,2), unpack=True, delimiter="\t")
z6dFGS=[]
z6dFGS.append(z)

filename = 'data_DHrd_eBOSS.dat'
z, DHrdeBOSS, erreBOSS = np.genfromtxt(filename, comments='#', usecols=(0,1,2), unpack=True, delimiter="\t")
zeBOSS=[]
zeBOSS.append(z)

#############
## CC data ##
#############
filename = 'data_CC.dat'
zC, Hz, errHz = np.genfromtxt(filename, comments='#', usecols=(0,1,2), unpack=True, delimiter="\t")
# Data Sys
filename = 'data_MM20.dat'
zmod, imf, slib, sps, spsooo = np.genfromtxt(filename, comments='#', usecols=(0,1,2,3,4), unpack=True)
# Interpolation DataSys in z (data_CC)
imf_intp = np.interp(zC, zmod, imf)/100
slib_intp = np.interp(zC, zmod, slib)/100
sps_intp = np.interp(zC, zmod, sps)/100
spsooo_intp = np.interp(zC, zmod, spsooo)/100
# Systematics Matrices
cov_mat_imf = np.zeros((len(zC), len(zC)), dtype='float64')
cov_mat_slib = np.zeros((len(zC), len(zC)), dtype='float64')
cov_mat_sps = np.zeros((len(zC), len(zC)), dtype='float64')
cov_mat_spsooo = np.zeros((len(zC), len(zC)), dtype='float64')
for i in range(len(zC)):
	for j in range(len(zC)):
		cov_mat_imf[i,j] = Hz[i] * imf_intp[i] * Hz[j] * imf_intp[j]
		cov_mat_slib[i,j] = Hz[i] * slib_intp[i] * Hz[j] * slib_intp[j]
		cov_mat_sps[i,j] = Hz[i] * sps_intp[i] * Hz[j] * sps_intp[j]
		cov_mat_spsooo[i,j] = Hz[i] * spsooo_intp[i] * Hz[j] * spsooo_intp[j]
# Full Covariance Matrices w/wo systematics
cov_mat_diagC = np.zeros((len(zC), len(zC)), dtype='float64') # wo full sys
for i in range(len(zC)):
	cov_mat_diagC[i,i] = errHz[i]**2
cov_matC = cov_mat_spsooo+cov_mat_imf+cov_mat_diagC # w full sys
# Inverse Covariance
inv_cov_matC = np.linalg.inv(cov_matC)
inv_cov_mat_diagC = np.linalg.inv(cov_mat_diagC)

##############
## SNe data ##
## Pantheon ##
##############
filename='panstarrs_PS1COSMO_bin.dat'
zS_P, DmS_P, errDmS_P = np.genfromtxt(filename, usecols=(1,4,5), unpack=True, comments='#')
DmS_P=DmS_P+19.35
filename='panstarrs_PS1COSMO_bin_cov.dat'
covS_P = np.genfromtxt(filename, usecols=(0), unpack=True, comments='#')
# Full
covS_P = np.matrix(covS_P.reshape(len(zS_P), len(zS_P))) + np.diag(errDmS_P**2)
inv_cov_matS_P = np.linalg.inv(covS_P)

##############
## GRB data ##
##############
filename = 'Results_deltaM_flcdm.dat'
deltaM_FLCDM=np.genfromtxt(filename, usecols=(0), unpack=True, comments='#')

filename='data_GRB.txt'
zG, DmG, errDmG = np.genfromtxt(filename, usecols=(2,3,4), unpack=True, comments='#')

os.chdir(dir_home)
######################
## Input Parameters ##
######################

in_H0= np.around(random.uniform(51, 99),1)
in_Om= np.around(random.uniform(0.01, 0.99),4)
in_M= np.around(random.uniform(21, 27),1)
in_rd= np.around(random.uniform(110, 180),1)
in_rat= np.around(random.uniform(0.93, 1.07),4)

initials_FLCDM= np.array([in_H0, in_Om, in_M, in_rd, in_rat])

nwalkers=250
Nsteps=5000

arg_BAOCCSNGRB= (zAlam, dataAlam, inv_cov_alam, zWiggle, dataWiggle, inv_cov_wiggleZ, zSDSS, dataSDSS, inv_cov_SDSS, z6dFGS, rdDV6dFGS, err6dFGS, zeBOSS, DHrdeBOSS, erreBOSS, zC, Hz, inv_cov_matC, zS_P, DmS_P, inv_cov_matS_P, zG, DmG, errDmG)

##############
## MCMC Run ##
##############
run_MCMC('Flat', 'FLCDM', 'BAOCCSNGRB', dir_chain, arg_BAOCCSNGRB, nwalkers, initials_FLCDM, Nsteps)
