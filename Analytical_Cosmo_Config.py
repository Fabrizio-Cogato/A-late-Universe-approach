"""
@file Analytical_Cosmo_Config.py
@date September 6, 2023
@authors Fabrizio Cogato <fabrizio.cogato@inaf.it>
         Michele Moresco <michele.moresco@unibo.it>

Please remember to cite: https://ui.adsabs.harvard.edu/abs/2023arXiv230901375C
"""

import numpy as np
from astropy.cosmology import FlatLambdaCDM, FlatwCDM, Flatw0waCDM, LambdaCDM, wCDM, w0waCDM
import pandas as pd
import astropy.units as u

##############################
#### Functions Definition ####
##############################
# Selecting Column Function
def column(matrix, i):                          
    return [row[i] for row in matrix]

c = 299792.458
####------------------------------------FLCDM---------------------------------------
#SN
def mu_model_FLCDM(H0, Om, M, z):                                                               
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
    mu = np.array(cosmo.distmod(z)) - M
    return mu
# CC
def E_model_FLCDM(H0, Om, z):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
    return np.array(cosmo.efunc(z))
def Hz_model_FLCDM(H0, Om, z):
    arr = []
    for j in range(len(z)):
        arr.append(H0*E_model_FLCDM(H0, Om, z[j]))
    arr = np.array(arr)
    return arr

def dL_model_FLCDM(H0, Om, z):                                                              
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
    dl = np.array(cosmo.luminosity_distance(z))
    return dl

#GRB
def Eiso_model_FLCDM(H0,Om,Eiso,z):
    dL=dL_model_FLCDM(H0,Om,z)
    dL_cal=dL_model_FLCDM(70,0.3,z)
    E=Eiso*(dL/dL_cal)**2
    return E
def errEiso_model_FLCDM(H0,Om,errEiso,z):
    dL=dL_model_FLCDM(H0,Om,z)
    dL_cal=dL_model_FLCDM(70,0.3,z)
    err=errEiso*(dL/dL_cal)**2
    return err

#BAO
def DHrd_model_FLCDM(H0, Om, rd, z):
    arr = []
    for j in range(len(z)):
        arr.append(c/(H0*rd*E_model_FLCDM(H0, Om, z[j])))
    arr = np.array(arr)
    return arr


def DM_model_FLCDM(H0, Om, z):
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om)
    DM = np.array(cosmo.comoving_transverse_distance(z))
    return DM

 
def DMrd_model_FLCDM(H0, Om, rd, z):
    arr = []
    for i in range(len(z)):
        arr.append(DM_model_FLCDM(H0, Om, z[i]) / rd)
    arr = np.array(arr)
    arr.shape
    return arr

def DH_model_FLCDM(H0, Om, z):
    arr = []
    for j in range(len(z)):
        arr.append(c/(H0*E_model_FLCDM(H0, Om, z[j])))
    arr = np.array(arr)
    return arr

def DVrd_model_FLCDM(H0, Om, rd, z):
    dh = DH_model_FLCDM(H0, Om, z)  
    dm = DM_model_FLCDM(H0, Om, z)
    arr = []
    for i in range(len(z)):
        arr.append(np.cbrt(z[i]*dh[i]*dm[i]**2)/rd)
    arr = np.array(arr)
    arr.shape
    return arr


def DMrd_DHrd_model_FLCDM(H0, Om, rd, z):
    dMrd = DMrd_model_FLCDM(H0, Om, rd, z)
    dHrd = DHrd_model_FLCDM(H0, Om, rd, z)
    arr = []
    for i in range(len(z)):
        arr.append(dMrd[i])
        arr.append(dHrd[i])
    arr = np.array(arr)
    arr.shape
    return arr

###############################  
#### Likelihood Definition ####
###############################
## BAO    
## Cov
def lnlikeBAOCov_FLCDM(theta, zBAO, dataBAO, inv_covBAO):                 # Likelihood Cov
    H0, Om, M, a, b, intr, rd = theta
    chi2=0.
    ndim=np.shape(inv_covBAO)[0]
    residual=dataBAO-DMrd_DHrd_model_FLCDM(H0, Om, rd, zBAO)
    for i in range(0, ndim):
            for j in range(0, ndim):
                chi2=chi2+((residual[i])*inv_covBAO[i,j]*(residual[j]))
    return -0.5 * chi2

## Err
def lnlikeDMrd_FLCDM(theta, zDMrd, DMrdBAO, errDMrdBAO):
    H0, Om, M, a, b, intr, rd = theta          
    sigma2 = errDMrdBAO ** 2
    return -0.5 * np.sum((DMrdBAO - DMrd_model_FLCDM(H0, Om, rd, zDMrd)) ** 2 / sigma2)
def lnlikeDHrd_FLCDM(theta, zDHrd, DHrdBAO, errDHrdBAO):
    H0, Om, M, a, b, intr, rd = theta          
    sigma2 = errDHrdBAO ** 2
    return -0.5 * np.sum((DHrdBAO - DHrd_model_FLCDM(H0, Om, rd, zDHrd)) ** 2 / sigma2)
def lnlikeDVrd_FLCDM(theta, zDVrd, DVrdBAO, errDVrdBAO):
    H0, Om, M, a, b, intr, rd = theta          
    sigma2 = errDVrdBAO ** 2
    return -0.5 * np.sum((DVrdBAO - DVrd_model_FLCDM(H0, Om, rd, zDVrd)) ** 2 / sigma2)

def lnlikeBAO_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia):
    return lnlikeBAOCov_FLCDM(theta,zAlam, dataAlam, inv_cov_Alam)+lnlikeBAOCov_FLCDM(theta, zHou, dataHou, inv_cov_Hou)+lnlikeBAOCov_FLCDM(theta, zGil, dataGil, inv_cov_Gil)+lnlikeDMrd_FLCDM(theta, zDumas, DMrdDumas, errDMrdDumas)+lnlikeDHrd_FLCDM(theta, zDumas, DHrdDumas, errDHrdDumas)+lnlikeDVrd_FLCDM(theta, zRoss, DVrdRoss, errDVrdRoss)+lnlikeDVrd_FLCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia)

def lnlikeBAO_Alam_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam):
    return lnlikeBAOCov_FLCDM(theta,zAlam, dataAlam, inv_cov_Alam)

def lnlikeBAO_Hou_FLCDM(theta, zHou, dataHou, inv_cov_Hou):
    return lnlikeBAOCov_FLCDM(theta, zHou, dataHou, inv_cov_Hou)

def lnlikeBAO_Gil_FLCDM(theta, zGil, dataGil, inv_cov_Gil):
    return lnlikeBAOCov_FLCDM(theta, zGil, dataGil, inv_cov_Gil)

def lnlikeBAO_Dumas_FLCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas):
    return lnlikeDMrd_FLCDM(theta, zDumas, DMrdDumas, errDMrdDumas)+lnlikeDHrd_FLCDM(theta, zDumas, DHrdDumas, errDHrdDumas)

def lnlikeBAO_Ross_FLCDM(theta, zRoss, DVrdRoss, errDVrdRoss):
    return lnlikeDVrd_FLCDM(theta, zRoss, DVrdRoss, errDVrdRoss)

def lnlikeBAO_Demattia_FLCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia):
    return lnlikeDVrd_FLCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia)


## CC
def lnlikeCC_FLCDM(theta, zC, Hz, inv_cov_matC):                 # Likelihood Cov
    H0, Om, M, a, b, intr, rd=theta 
    chi2=0.
    residual=Hz-Hz_model_FLCDM(H0, Om, zC)
    for i in range(0, len(zC)):
            for j in range(0, len(zC)):
                chi2=chi2+((residual[i])*inv_cov_matC[i,j]*(residual[j]))
    return -0.5 * chi2
## SNe
def lnlikeSN_FLCDM(theta, zS, DmS, inv_cov_matS):                 # Likelihood Cov
    H0, Om, M, a, b, intr, rd=theta 
    mu = mu_model_FLCDM(H0, Om, M, zS)
    residual= DmS - mu
    chi2=0.
    for i in range(0, len(zS)):
            for j in range(0, len(zS)):
                chi2=chi2+((residual[i])*inv_cov_matS[i,j]*(residual[j]))
    return -0.5 * chi2

##GRB
def lnlikeGRB_FLCDM(theta,z,Ep,Eiso,errEp,errEiso):
    H0, Om, M, a, b, intr, rd=theta
    E_iso=Eiso_model_FLCDM(H0,Om,Eiso,z)
    err_Eiso=errEiso_model_FLCDM(H0,Om,errEiso,z) 
    logEiso=np.log10(E_iso)
    logEp=np.log10(Ep)
    errlog_iso=err_Eiso/(np.log(10)*E_iso)
    errlog_p=errEp/(np.log(10)*Ep)
    fact1=0.5*np.log((1+a**2)/(2*np.pi*(intr**2+errlog_p**2+(a*errlog_iso)**2)))
    fact2=0.5*((logEp-a*logEiso-b)**2/(intr**2+errlog_p**2+(a*errlog_iso)**2))
    lnlike=np.sum(fact1-fact2)
    return lnlike

#######################################  
#### Probes Combination Likelihood ####
#######################################
###########################
# BAO + CC + SN + GRB
def lnlikeBAOCCSNGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_FLCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_FLCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_FLCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC + GRB
def lnlikeBAOCCGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_FLCDM(theta, zC, Hz, inv_cov_matC) + lnlikeGRB_FLCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC + SN
def lnlikeBAOCCSN_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):
    return lnlikeBAO_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_FLCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_FLCDM(theta, zS, DmS, inv_cov_matS)
# BAO + SN + GRB
def lnlikeBAOSNGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS,zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeSN_FLCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_FLCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# CC + SN + GRB
def lnlikeCCSNGRB_FLCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeCC_FLCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_FLCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_FLCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC
def lnlikeBAOCC_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC):
    return lnlikeBAO_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_FLCDM(theta, zC, Hz, inv_cov_matC)
# BAO + SN
def lnlikeBAOSN_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS):
    return lnlikeBAO_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeSN_FLCDM(theta, zS, DmS, inv_cov_matS)
# BAO + GRB
def lnlikeBAOGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia) + lnlikeGRB_FLCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# SN + GRB
def lnlikeSNGRB_FLCDM(theta, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeSN_FLCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_FLCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# CC + SN
def lnlikeCCSN_FLCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):
    return lnlikeCC_FLCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_FLCDM(theta, zS, DmS, inv_cov_matS)
# CC + GRB
def lnlikeCCGRB_FLCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeCC_FLCDM(theta, zC, Hz, inv_cov_matC)+ lnlikeGRB_FLCDM(theta, zG, Ep, Eiso, errEp, errEiso)

###########
## Prior ##
###########
# Flat Prior
def lnflatprior_FLCDM(theta):                                 
    H0, Om, M, a, b, intr, rd=theta 
    if (0.0 < H0 < 100.0 and 0.0 < Om < 1.0 and 15. < M < 25.  and 0. < a < 3.0 and 0. < b < 5.0 and 0. < intr < 1.0 and 50 < rd < 250):
        return 0.0
    return -np.inf
###############
## Posterior ##
###############
## BAO + CC + SN + GRB
# Flat Prior 
def lnflatprobBAOCCSNGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAOCCSNGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + CC + GRB
# Flat Prior 
def lnflatprobBAOCCGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAOCCGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC,zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + SN + GRB
# Flat Prior 
def lnflatprobBAOSNGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAOSNGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia,zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + CC + SN
# Flat Prior
def lnflatprobBAOCCSN_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAOCCSN_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC + SN + GRB
# Flat Prior
def lnflatprobCCSNGRB_FLCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeCCSNGRB_FLCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

## BAO + CC 
# Flat Prior
def lnflatprobBAOCC_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAOCC_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC) if np.isfinite(lp) else -np.inf
## BAO + SN
# Flat Prior 
def lnflatprobBAOSN_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAOSN_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## BAO + GRB
# Flat Prior
def lnflatprobBAOGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAOGRB_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf 
## SN + GRB 
# Flat Prior
def lnflatprobSNGRB_FLCDM(theta, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeSNGRB_FLCDM(theta,zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf 
## CC + SN
# Flat Prior
def lnflatprobCCSN_FLCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeCCSN_FLCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC + GRB
# Flat Prior
def lnflatprobCCGRB_FLCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeCCGRB_FLCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

## BAO
# Flat Prior
def lnflatprobBAO_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAO_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Alam_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAO_Alam_FLCDM(theta, zAlam, dataAlam, inv_cov_Alam) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Hou_FLCDM(theta, zHou, dataHou, inv_cov_Hou):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAO_Hou_FLCDM(theta, zHou, dataHou, inv_cov_Hou) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Gil_FLCDM(theta, zGil, dataGil, inv_cov_Gil):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAO_Gil_FLCDM(theta, zGil, dataGil, inv_cov_Gil) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Dumas_FLCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAO_Dumas_FLCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Ross_FLCDM(theta, zRoss, DVrdRoss, errDVrdRoss):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAO_Ross_FLCDM(theta, zRoss, DVrdRoss, errDVrdRoss) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Demattia_FLCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeBAO_Demattia_FLCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia) if np.isfinite(lp) else -np.inf

## SN
# Flat Prior
def lnflatprobSN_FLCDM(theta, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeSN_FLCDM(theta,zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC
# Flat Prior
def lnflatprobCC_FLCDM(theta, zC, Hz, inv_cov_matC):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeCC_FLCDM(theta, zC, Hz, inv_cov_matC) if np.isfinite(lp) else -np.inf
## GRB
# Flat Prior
def lnflatprobGRB_FLCDM(theta, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FLCDM(theta)
    return lp + lnlikeGRB_FLCDM(theta, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

### --------------------------FwCDM--------------------------------------

#SN
def mu_model_FwCDM(H0, Om, w, M, z):                                                               
    cosmo = FlatwCDM(H0=H0, Om0=Om, w0=w)
    mu = np.array(cosmo.distmod(z)) - M
    return mu
# CC
def E_model_FwCDM(H0, Om, w, z):
    cosmo = FlatwCDM(H0=H0, Om0=Om, w0=w)
    return np.array(cosmo.efunc(z))
def Hz_model_FwCDM(H0, Om, w, z):
    arr = []
    for j in range(len(z)):
        arr.append(H0*E_model_FwCDM(H0, Om, w, z[j]))
    arr = np.array(arr)
    return arr
def dL_model_FwCDM(H0, Om, w, z):                                                              
    cosmo = FlatwCDM(H0=H0, Om0=Om, w0=w)
    dl = np.array(cosmo.luminosity_distance(z))
    return dl

#GRB
def Eiso_model_FwCDM(H0, Om, w, Eiso,z):
    dL=dL_model_FwCDM(H0, Om, w, z)
    dL_cal=dL_model_FLCDM(70,0.3,z)
    E=Eiso*(dL/dL_cal)**2
    return E
def errEiso_model_FwCDM(H0, Om, w, errEiso,z):
    dL=dL_model_FwCDM(H0, Om, w, z)
    dL_cal=dL_model_FLCDM(70,0.3,z)
    err=errEiso*(dL/dL_cal)**2
    return err

#BAO
def DHrd_model_FwCDM(H0, Om, w, rd, z):
    arr = []
    for j in range(len(z)):
        arr.append(c/(H0*rd*E_model_FwCDM(H0, Om, w, z[j])))
    arr = np.array(arr)
    return arr


def DA_model_FwCDM(H0, Om, w, z):
    cosmo = FlatwCDM(H0=H0, Om0=Om, w0=w)
    DA = np.array(cosmo.angular_diameter_distance(z))
    return DA

 
def DMrd_model_FwCDM(H0, Om, w, rd, z):
    arr = []
    for i in range(len(z)):
        arr.append(DA_model_FwCDM(H0, Om, w, z[i]) * (1+z[i]) / rd)
    arr = np.array(arr)
    arr.shape
    return arr

def DH_model_FwCDM(H0, Om, w, z):
    arr = []
    for j in range(len(z)):
        arr.append(c/(H0*E_model_FwCDM(H0, Om, w, z[j])))
    arr = np.array(arr)
    return arr
def DM_model_FwCDM(H0, Om, w, z):
    arr = []
    for i in range(len(z)):
        arr.append(DA_model_FwCDM(H0, Om, w, z[i]) * (1+z[i]))
    arr = np.array(arr)
    arr.shape
    return arr
def DVrd_model_FwCDM(H0, Om, w, rd, z):
    dh = DH_model_FwCDM(H0, Om, w, z)  
    dm = DM_model_FwCDM(H0, Om, w, z)
    arr = []
    for i in range(len(z)):
        arr.append(np.cbrt(z[i]*dh[i]*dm[i]**2)/rd)
    arr = np.array(arr)
    arr.shape
    return arr


def DMrd_DHrd_model_FwCDM(H0, Om, w, rd, z):
    dMrd = DMrd_model_FwCDM(H0, Om, w, rd, z)
    dHrd = DHrd_model_FwCDM(H0, Om, w, rd, z)
    arr = []
    for i in range(len(z)):
        arr.append(dMrd[i])
        arr.append(dHrd[i])
    arr = np.array(arr)
    arr.shape
    return arr

###############################  
#### Likelihood Definition ####
###############################
## BAO    
## Cov
def lnlikeBAOCov_FwCDM(theta, zBAO, dataBAO, inv_covBAO):                 # Likelihood Cov
    H0, Om, w, M, a, b, intr, rd = theta
    chi2=0.
    ndim=np.shape(inv_covBAO)[0]
    residual=dataBAO-DMrd_DHrd_model_FwCDM(H0, Om, w, rd, zBAO)
    for i in range(0, ndim):
            for j in range(0, ndim):
                chi2=chi2+((residual[i])*inv_covBAO[i,j]*(residual[j]))
    return -0.5 * chi2

## Err
def lnlikeDMrd_FwCDM(theta, zBAO, DMrdBAO, errDMrdBAO):
    H0, Om, w, M, a, b, intr, rd = theta          
    sigma2 = errDMrdBAO ** 2
    return -0.5 * np.sum((DMrdBAO - DMrd_model_FwCDM(H0, Om, w, rd, zBAO)) ** 2 / sigma2)
def lnlikeDHrd_FwCDM(theta, zBAO, DHrdBAO, errDHrdBAO):
    H0, Om, w, M, a, b, intr, rd = theta          
    sigma2 = errDHrdBAO ** 2
    return -0.5 * np.sum((DHrdBAO - DHrd_model_FwCDM(H0, Om, w, rd, zBAO)) ** 2 / sigma2)
def lnlikeDVrd_FwCDM(theta, zBAO, DVrdBAO, errDVrdBAO):
    H0, Om, w, M, a, b, intr, rd = theta          
    sigma2 = errDVrdBAO ** 2
    return -0.5 * np.sum((DVrdBAO - DVrd_model_FwCDM(H0, Om, w, rd, zBAO)) ** 2 / sigma2)

def lnlikeBAO_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia):
    return lnlikeBAOCov_FwCDM(theta,zAlam, dataAlam, inv_cov_Alam)+lnlikeBAOCov_FwCDM(theta, zHou, dataHou, inv_cov_Hou)+lnlikeBAOCov_FwCDM(theta, zGil, dataGil, inv_cov_Gil)+lnlikeDMrd_FwCDM(theta, zDumas, DMrdDumas, errDMrdDumas)+lnlikeDHrd_FwCDM(theta, zDumas, DHrdDumas, errDHrdDumas)+lnlikeDVrd_FwCDM(theta, zRoss, DVrdRoss, errDVrdRoss)+lnlikeDVrd_FwCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia)

def lnlikeBAO_Alam_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam):
    return lnlikeBAOCov_FwCDM(theta,zAlam, dataAlam, inv_cov_Alam)

def lnlikeBAO_Hou_FwCDM(theta, zHou, dataHou, inv_cov_Hou):
    return lnlikeBAOCov_FwCDM(theta, zHou, dataHou, inv_cov_Hou)

def lnlikeBAO_Gil_FwCDM(theta, zGil, dataGil, inv_cov_Gil):
    return lnlikeBAOCov_FwCDM(theta, zGil, dataGil, inv_cov_Gil)

def lnlikeBAO_Dumas_FwCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas):
    return lnlikeDMrd_FwCDM(theta, zDumas, DMrdDumas, errDMrdDumas)+lnlikeDHrd_FwCDM(theta, zDumas, DHrdDumas, errDHrdDumas)

def lnlikeBAO_Ross_FwCDM(theta, zRoss, DVrdRoss, errDVrdRoss):
    return lnlikeDVrd_FwCDM(theta, zRoss, DVrdRoss, errDVrdRoss)

def lnlikeBAO_Demattia_FwCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia):
    return lnlikeDVrd_FwCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia)

## CC
def lnlikeCC_FwCDM(theta, zC, Hz, inv_cov_matC):                 # Likelihood Cov
    H0, Om, w, M, a, b, intr, rd=theta 
    chi2=0.
    residual=Hz-Hz_model_FwCDM(H0, Om, w, zC)
    for i in range(0, len(zC)):
            for j in range(0, len(zC)):
                chi2=chi2+((residual[i])*inv_cov_matC[i,j]*(residual[j]))
    return -0.5 * chi2
## SNe
def lnlikeSN_FwCDM(theta, zS, DmS, inv_cov_matS):                 # Likelihood Cov
    H0, Om, w, M, a, b, intr, rd=theta 
    mu = mu_model_FwCDM(H0, Om, w, M, zS)
    residual= DmS - mu
    chi2=0.
    for i in range(0, len(zS)):
            for j in range(0, len(zS)):
                chi2=chi2+((residual[i])*inv_cov_matS[i,j]*(residual[j]))
    return -0.5 * chi2

##GRB
def lnlikeGRB_FwCDM(theta,z,Ep,Eiso,errEp,errEiso):
    H0, Om, w, M, a, b, intr, rd=theta
    E_iso=Eiso_model_FwCDM(H0, Om, w, Eiso,z)
    err_Eiso=errEiso_model_FwCDM(H0, Om, w, errEiso,z) 
    logEiso=np.log10(E_iso)
    logEp=np.log10(Ep)
    errlog_iso=err_Eiso/(np.log(10)*E_iso)
    errlog_p=errEp/(np.log(10)*Ep)
    fact1=0.5*np.log((1+a**2)/(2*np.pi*(intr**2+errlog_p**2+(a*errlog_iso)**2)))
    fact2=0.5*((logEp-a*logEiso-b)**2/(intr**2+errlog_p**2+(a*errlog_iso)**2))
    lnlike=np.sum(fact1-fact2)
    return lnlike

#######################################  
#### Probes Combination Likelihood ####
#######################################
###########################
# BAO + CC + SN + GRB
def lnlikeBAOCCSNGRB_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_FwCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_FwCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_FwCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC + GRB
def lnlikeBAOCCGRB_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_FwCDM(theta, zC, Hz, inv_cov_matC) + lnlikeGRB_FwCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC + SN
def lnlikeBAOCCSN_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):
    return lnlikeBAO_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_FwCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_FwCDM(theta, zS, DmS, inv_cov_matS)
# BAO + SN + GRB
def lnlikeBAOSNGRB_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS,zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeSN_FwCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_FwCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# CC + SN + GRB
def lnlikeCCSNGRB_FwCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeCC_FwCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_FwCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_FwCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC
def lnlikeBAOCC_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC):
    return lnlikeBAO_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_FwCDM(theta, zC, Hz, inv_cov_matC)
# BAO + SN
def lnlikeBAOSN_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS):
    return lnlikeBAO_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeSN_FwCDM(theta, zS, DmS, inv_cov_matS)
# BAO + GRB
def lnlikeBAOGRB_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia) + lnlikeGRB_FwCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# SN + GRB
def lnlikeSNGRB_FwCDM(theta, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeSN_FwCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_FwCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# CC + SN
def lnlikeCCSN_FwCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):
    return lnlikeCC_FwCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_FwCDM(theta, zS, DmS, inv_cov_matS)
# CC + GRB
def lnlikeCCGRB_FwCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeCC_FwCDM(theta, zC, Hz, inv_cov_matC)+ lnlikeGRB_FwCDM(theta, zG, Ep, Eiso, errEp, errEiso)

###########
## Prior ##
###########
# Flat Prior
def lnflatprior_FwCDM(theta):                                 
    H0, Om, w, M, a, b, intr, rd=theta 
    if (0.0 < H0 < 100.0 and 0.0 < Om < 1.0 and -5.0 < w < -0.3 and 15. < M < 25.  and 0. < a < 3.0 and 0. < b < 5.0 and 0. < intr < 1.0 and 50 < rd < 250):
        return 0.0
    return -np.inf
###############
## Posterior ##
###############
## BAO + CC + SN + GRB
# Flat Prior 
def lnflatprobBAOCCSNGRB_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAOCCSNGRB_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + CC + GRB
# Flat Prior 
def lnflatprobBAOCCGRB_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAOCCGRB_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC,zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + SN + GRB
# Flat Prior 
def lnflatprobBAOSNGRB_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAOSNGRB_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia,zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + CC + SN
# Flat Prior
def lnflatprobBAOCCSN_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAOCCSN_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC + SN + GRB
# Flat Prior
def lnflatprobCCSNGRB_FwCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeCCSNGRB_FwCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

## BAO + CC 
# Flat Prior
def lnflatprobBAOCC_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAOCC_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC) if np.isfinite(lp) else -np.inf
## BAO + SN
# Flat Prior 
def lnflatprobBAOSN_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAOSN_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## BAO + GRB
# Flat Prior
def lnflatprobBAOGRB_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAOGRB_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf 
## SN + GRB 
# Flat Prior
def lnflatprobSNGRB_FwCDM(theta, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeSNGRB_FwCDM(theta,zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf 
## CC + SN
# Flat Prior
def lnflatprobCCSN_FwCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeCCSN_FwCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC + GRB
# Flat Prior
def lnflatprobCCGRB_FwCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeCCGRB_FwCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

## BAO
# Flat Prior
def lnflatprobBAO_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAO_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Alam_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAO_Alam_FwCDM(theta, zAlam, dataAlam, inv_cov_Alam) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Hou_FwCDM(theta, zHou, dataHou, inv_cov_Hou):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAO_Hou_FwCDM(theta, zHou, dataHou, inv_cov_Hou) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Gil_FwCDM(theta, zGil, dataGil, inv_cov_Gil):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAO_Gil_FwCDM(theta, zGil, dataGil, inv_cov_Gil) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Dumas_FwCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAO_Dumas_FwCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Ross_FwCDM(theta, zRoss, DVrdRoss, errDVrdRoss):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAO_Ross_FwCDM(theta, zRoss, DVrdRoss, errDVrdRoss) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Demattia_FwCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeBAO_Demattia_FwCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia) if np.isfinite(lp) else -np.inf

## SN
# Flat Prior
def lnflatprobSN_FwCDM(theta, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeSN_FwCDM(theta,zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC
# Flat Prior
def lnflatprobCC_FwCDM(theta, zC, Hz, inv_cov_matC):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeCC_FwCDM(theta, zC, Hz, inv_cov_matC) if np.isfinite(lp) else -np.inf
## GRB
# Flat Prior
def lnflatprobGRB_FwCDM(theta, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_FwCDM(theta)
    return lp + lnlikeGRB_FwCDM(theta, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

####-------------------------Fw0waCDM------------------------------####

#SN
def mu_model_Fw0waCDM(H0, Om, w0, wa, M, z):                                                               
    cosmo = Flatw0waCDM(H0=H0, Om0=Om, w0=w0, wa=wa)
    mu = np.array(cosmo.distmod(z)) - M
    return mu
# CC
def E_model_Fw0waCDM(H0, Om, w0, wa, z):
    cosmo = Flatw0waCDM(H0=H0, Om0=Om, w0=w0, wa=wa)
    return np.array(cosmo.efunc(z))
def Hz_model_Fw0waCDM(H0, Om, w0, wa, z):
    arr = []
    for j in range(len(z)):
        arr.append(H0*E_model_Fw0waCDM(H0, Om, w0, wa, z[j]))
    arr = np.array(arr)
    return arr
def dL_model_Fw0waCDM(H0, Om, w0, wa, z):                                                              
    cosmo = Flatw0waCDM(H0=H0, Om0=Om, w0=w0, wa=wa)
    dl = np.array(cosmo.luminosity_distance(z))
    return dl

#GRB
def Eiso_model_Fw0waCDM(H0, Om, w0, wa, Eiso,z):
    dL=dL_model_Fw0waCDM(H0, Om, w0, wa, z)
    dL_cal=dL_model_FLCDM(70,0.3,z)
    E=Eiso*(dL/dL_cal)**2
    return E
def errEiso_model_Fw0waCDM(H0, Om, w0, wa, errEiso,z):
    dL=dL_model_Fw0waCDM(H0, Om, w0, wa, z)
    dL_cal=dL_model_FLCDM(70,0.3,z)
    err=errEiso*(dL/dL_cal)**2
    return err

#BAO
def DHrd_model_Fw0waCDM(H0, Om, w0, wa, rd, z):
    arr = []
    for j in range(len(z)):
        arr.append(c/(H0*rd*E_model_Fw0waCDM(H0, Om, w0, wa, z[j])))
    arr = np.array(arr)
    return arr


def DA_model_Fw0waCDM(H0, Om, w0, wa, z):
    cosmo = Flatw0waCDM(H0=H0, Om0=Om, w0=w0, wa=wa)
    DA = np.array(cosmo.angular_diameter_distance(z))
    return DA

 
def DMrd_model_Fw0waCDM(H0, Om, w0, wa, rd, z):
    arr = []
    for i in range(len(z)):
        arr.append(DA_model_Fw0waCDM(H0, Om, w0, wa, z[i]) * (1+z[i]) / rd)
    arr = np.array(arr)
    arr.shape
    return arr

def DH_model_Fw0waCDM(H0, Om, w0, wa, z):
    arr = []
    for j in range(len(z)):
        arr.append(c/(H0*E_model_Fw0waCDM(H0, Om, w0, wa, z[j])))
    arr = np.array(arr)
    return arr
def DM_model_Fw0waCDM(H0, Om, w0, wa, z):
    arr = []
    for i in range(len(z)):
        arr.append(DA_model_Fw0waCDM(H0, Om, w0, wa, z[i]) * (1+z[i]))
    arr = np.array(arr)
    arr.shape
    return arr
def DVrd_model_Fw0waCDM(H0, Om, w0, wa, rd, z):
    dh = DH_model_Fw0waCDM(H0, Om, w0, wa, z)  
    dm = DM_model_Fw0waCDM(H0, Om, w0, wa, z)
    arr = []
    for i in range(len(z)):
        arr.append(np.cbrt(z[i]*dh[i]*dm[i]**2)/rd)
    arr = np.array(arr)
    arr.shape
    return arr


def DMrd_DHrd_model_Fw0waCDM(H0, Om, w0, wa, rd, z):
    dMrd = DMrd_model_Fw0waCDM(H0, Om, w0, wa, rd, z)
    dHrd = DHrd_model_Fw0waCDM(H0, Om, w0, wa, rd, z)
    arr = []
    for i in range(len(z)):
        arr.append(dMrd[i])
        arr.append(dHrd[i])
    arr = np.array(arr)
    arr.shape
    return arr

###############################  
#### Likelihood Definition ####
###############################
## BAO    
## Cov
def lnlikeBAOCov_Fw0waCDM(theta, zBAO, dataBAO, inv_covBAO):                 # Likelihood Cov
    H0, Om, w0, wa, M, a, b, intr, rd = theta
    chi2=0.
    ndim=np.shape(inv_covBAO)[0]
    residual=dataBAO-DMrd_DHrd_model_Fw0waCDM(H0, Om, w0, wa, rd, zBAO)
    for i in range(0, ndim):
            for j in range(0, ndim):
                chi2=chi2+((residual[i])*inv_covBAO[i,j]*(residual[j]))
    return -0.5 * chi2

## Err
def lnlikeDMrd_Fw0waCDM(theta, zBAO, DMrdBAO, errDMrdBAO):
    H0, Om, w0, wa, M, a, b, intr, rd = theta          
    sigma2 = errDMrdBAO ** 2
    return -0.5 * np.sum((DMrdBAO - DMrd_model_Fw0waCDM(H0, Om, w0, wa, rd, zBAO)) ** 2 / sigma2)
def lnlikeDHrd_Fw0waCDM(theta, zBAO, DHrdBAO, errDHrdBAO):
    H0, Om, w0, wa, M, a, b, intr, rd = theta          
    sigma2 = errDHrdBAO ** 2
    return -0.5 * np.sum((DHrdBAO - DHrd_model_Fw0waCDM(H0, Om, w0, wa, rd, zBAO)) ** 2 / sigma2)
def lnlikeDVrd_Fw0waCDM(theta, zBAO, DVrdBAO, errDVrdBAO):
    H0, Om, w0, wa, M, a, b, intr, rd = theta          
    sigma2 = errDVrdBAO ** 2
    return -0.5 * np.sum((DVrdBAO - DVrd_model_Fw0waCDM(H0, Om, w0, wa, rd, zBAO)) ** 2 / sigma2)

def lnlikeBAO_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia):
    return lnlikeBAOCov_Fw0waCDM(theta,zAlam, dataAlam, inv_cov_Alam)+lnlikeBAOCov_Fw0waCDM(theta, zHou, dataHou, inv_cov_Hou)+lnlikeBAOCov_Fw0waCDM(theta, zGil, dataGil, inv_cov_Gil)+lnlikeDMrd_Fw0waCDM(theta, zDumas, DMrdDumas, errDMrdDumas)+lnlikeDHrd_Fw0waCDM(theta, zDumas, DHrdDumas, errDHrdDumas)+lnlikeDVrd_Fw0waCDM(theta, zRoss, DVrdRoss, errDVrdRoss)+lnlikeDVrd_Fw0waCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia)

def lnlikeBAO_Alam_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam):
    return lnlikeBAOCov_Fw0waCDM(theta,zAlam, dataAlam, inv_cov_Alam)

def lnlikeBAO_Hou_Fw0waCDM(theta, zHou, dataHou, inv_cov_Hou):
    return lnlikeBAOCov_Fw0waCDM(theta, zHou, dataHou, inv_cov_Hou)

def lnlikeBAO_Gil_Fw0waCDM(theta, zGil, dataGil, inv_cov_Gil):
    return lnlikeBAOCov_Fw0waCDM(theta, zGil, dataGil, inv_cov_Gil)

def lnlikeBAO_Dumas_Fw0waCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas):
    return lnlikeDMrd_Fw0waCDM(theta, zDumas, DMrdDumas, errDMrdDumas)+lnlikeDHrd_Fw0waCDM(theta, zDumas, DHrdDumas, errDHrdDumas)

def lnlikeBAO_Ross_Fw0waCDM(theta, zRoss, DVrdRoss, errDVrdRoss):
    return lnlikeDVrd_Fw0waCDM(theta, zRoss, DVrdRoss, errDVrdRoss)

def lnlikeBAO_Demattia_Fw0waCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia):
    return lnlikeDVrd_Fw0waCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia)

## CC
def lnlikeCC_Fw0waCDM(theta, zC, Hz, inv_cov_matC):                 # Likelihood Cov
    H0, Om, w0, wa, M, a, b, intr, rd=theta 
    chi2=0.
    residual=Hz-Hz_model_Fw0waCDM(H0, Om, w0, wa, zC)
    for i in range(0, len(zC)):
            for j in range(0, len(zC)):
                chi2=chi2+((residual[i])*inv_cov_matC[i,j]*(residual[j]))
    return -0.5 * chi2
## SNe
def lnlikeSN_Fw0waCDM(theta, zS, DmS, inv_cov_matS):                 # Likelihood Cov
    H0, Om, w0, wa, M, a, b, intr, rd=theta 
    mu = mu_model_Fw0waCDM(H0, Om, w0, wa, M, zS)
    residual= DmS - mu
    chi2=0.
    for i in range(0, len(zS)):
            for j in range(0, len(zS)):
                chi2=chi2+((residual[i])*inv_cov_matS[i,j]*(residual[j]))
    return -0.5 * chi2

##GRB
def lnlikeGRB_Fw0waCDM(theta,z,Ep,Eiso,errEp,errEiso):
    H0, Om, w0, wa, M, a, b, intr, rd=theta
    E_iso=Eiso_model_Fw0waCDM(H0, Om, w0, wa, Eiso,z)
    err_Eiso=errEiso_model_Fw0waCDM(H0, Om, w0, wa, errEiso,z) 
    logEiso=np.log10(E_iso)
    logEp=np.log10(Ep)
    errlog_iso=err_Eiso/(np.log(10)*E_iso)
    errlog_p=errEp/(np.log(10)*Ep)
    fact1=0.5*np.log((1+a**2)/(2*np.pi*(intr**2+errlog_p**2+(a*errlog_iso)**2)))
    fact2=0.5*((logEp-a*logEiso-b)**2/(intr**2+errlog_p**2+(a*errlog_iso)**2))
    lnlike=np.sum(fact1-fact2)
    return lnlike

#######################################  
#### Probes Combination Likelihood ####
#######################################
###########################
# BAO + CC + SN + GRB
def lnlikeBAOCCSNGRB_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_Fw0waCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_Fw0waCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_Fw0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC + GRB
def lnlikeBAOCCGRB_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_Fw0waCDM(theta, zC, Hz, inv_cov_matC) + lnlikeGRB_Fw0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC + SN
def lnlikeBAOCCSN_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):
    return lnlikeBAO_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_Fw0waCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_Fw0waCDM(theta, zS, DmS, inv_cov_matS)
# BAO + SN + GRB
def lnlikeBAOSNGRB_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS,zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeSN_Fw0waCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_Fw0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# CC + SN + GRB
def lnlikeCCSNGRB_Fw0waCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeCC_Fw0waCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_Fw0waCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_Fw0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC
def lnlikeBAOCC_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC):
    return lnlikeBAO_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_Fw0waCDM(theta, zC, Hz, inv_cov_matC)
# BAO + SN
def lnlikeBAOSN_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS):
    return lnlikeBAO_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeSN_Fw0waCDM(theta, zS, DmS, inv_cov_matS)
# BAO + GRB
def lnlikeBAOGRB_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia) + lnlikeGRB_Fw0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# SN + GRB
def lnlikeSNGRB_Fw0waCDM(theta, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeSN_Fw0waCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_Fw0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# CC + SN
def lnlikeCCSN_Fw0waCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):
    return lnlikeCC_Fw0waCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_Fw0waCDM(theta, zS, DmS, inv_cov_matS)
# CC + GRB
def lnlikeCCGRB_Fw0waCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeCC_Fw0waCDM(theta, zC, Hz, inv_cov_matC)+ lnlikeGRB_Fw0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)

###########
## Prior ##
###########
# Flat Prior
def lnflatprior_Fw0waCDM(theta):                                 
    H0, Om, w0, wa, M, a, b, intr, rd=theta 
    if (0.0 < H0 < 100.0 and 0.0 < Om < 1.0 and -5.0 < w0 < -0.3  and -5.0 < wa < 5 and 15. < M < 25.  and 0. < a < 3.0 and 0. < b < 5.0 and 0. < intr < 1.0 and 50 < rd < 250):
        return 0.0
    return -np.inf
###############
## Posterior ##
###############
## BAO + CC + SN + GRB
# Flat Prior 
def lnflatprobBAOCCSNGRB_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAOCCSNGRB_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + CC + GRB
# Flat Prior 
def lnflatprobBAOCCGRB_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAOCCGRB_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC,zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + SN + GRB
# Flat Prior 
def lnflatprobBAOSNGRB_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAOSNGRB_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia,zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + CC + SN
# Flat Prior
def lnflatprobBAOCCSN_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAOCCSN_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC + SN + GRB
# Flat Prior
def lnflatprobCCSNGRB_Fw0waCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeCCSNGRB_Fw0waCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

## BAO + CC 
# Flat Prior
def lnflatprobBAOCC_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAOCC_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC) if np.isfinite(lp) else -np.inf
## BAO + SN
# Flat Prior 
def lnflatprobBAOSN_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAOSN_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## BAO + GRB
# Flat Prior
def lnflatprobBAOGRB_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAOGRB_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf 
## SN + GRB 
# Flat Prior
def lnflatprobSNGRB_Fw0waCDM(theta, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeSNGRB_Fw0waCDM(theta,zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf 
## CC + SN
# Flat Prior
def lnflatprobCCSN_Fw0waCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeCCSN_Fw0waCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC + GRB
# Flat Prior
def lnflatprobCCGRB_Fw0waCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeCCGRB_Fw0waCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

## BAO
# Flat Prior
def lnflatprobBAO_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAO_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Alam_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAO_Alam_Fw0waCDM(theta, zAlam, dataAlam, inv_cov_Alam) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Hou_Fw0waCDM(theta, zHou, dataHou, inv_cov_Hou):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAO_Hou_Fw0waCDM(theta, zHou, dataHou, inv_cov_Hou) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Gil_Fw0waCDM(theta, zGil, dataGil, inv_cov_Gil):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAO_Gil_Fw0waCDM(theta, zGil, dataGil, inv_cov_Gil) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Dumas_Fw0waCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAO_Dumas_Fw0waCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Ross_Fw0waCDM(theta, zRoss, DVrdRoss, errDVrdRoss):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAO_Ross_Fw0waCDM(theta, zRoss, DVrdRoss, errDVrdRoss) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Demattia_Fw0waCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeBAO_Demattia_Fw0waCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia) if np.isfinite(lp) else -np.inf

## SN
# Flat Prior
def lnflatprobSN_Fw0waCDM(theta, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeSN_Fw0waCDM(theta,zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC
# Flat Prior
def lnflatprobCC_Fw0waCDM(theta, zC, Hz, inv_cov_matC):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeCC_Fw0waCDM(theta, zC, Hz, inv_cov_matC) if np.isfinite(lp) else -np.inf
## GRB
# Flat Prior
def lnflatprobGRB_Fw0waCDM(theta, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_Fw0waCDM(theta)
    return lp + lnlikeGRB_Fw0waCDM(theta, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf


####----------------------------------------LCDM------------------------

#SN
def mu_model_LCDM(H0, Om, OL, M, z):                                                               
    cosmo = LambdaCDM(H0=H0, Om0=Om, Ode0=OL)
    mu = np.array(cosmo.distmod(z)) - M
    return mu
# CC
def E_model_LCDM(H0, Om, OL, z):
    cosmo = LambdaCDM(H0=H0, Om0=Om, Ode0=OL)
    return np.array(cosmo.efunc(z))
def Hz_model_LCDM(H0, Om, OL, z):
    arr = []
    for j in range(len(z)):
        arr.append(H0*E_model_LCDM(H0, Om, OL, z[j]))
    arr = np.array(arr)
    return arr
def dL_model_LCDM(H0, Om, OL, z):                                                              
    cosmo = LambdaCDM(H0=H0, Om0=Om, Ode0=OL)
    dl = np.array(cosmo.luminosity_distance(z))
    return dl

#GRB
def Eiso_model_LCDM(H0,Om,OL,Eiso,z):
    dL=dL_model_LCDM(H0,Om,OL,z)
    dL_cal=dL_model_FLCDM(70,0.3,z)
    E=Eiso*(dL/dL_cal)**2
    return E
def errEiso_model_LCDM(H0,Om,OL,errEiso,z):
    dL=dL_model_LCDM(H0,Om,OL,z)
    dL_cal=dL_model_FLCDM(70,0.3,z)
    err=errEiso*(dL/dL_cal)**2
    return err

#BAO
def DHrd_model_LCDM(H0, Om, OL, rd, z):
    arr = []
    for j in range(len(z)):
        arr.append(c/(H0*rd*E_model_LCDM(H0, Om, OL, z[j])))
    arr = np.array(arr)
    return arr


def DA_model_LCDM(H0, Om, OL, z):
    cosmo = LambdaCDM(H0=H0, Om0=Om, Ode0=OL)
    DA = np.array(cosmo.angular_diameter_distance(z))
    return DA

 
def DMrd_model_LCDM(H0, Om, OL, rd, z):
    arr = []
    for i in range(len(z)):
        arr.append(DA_model_LCDM(H0, Om, OL, z[i]) * (1+z[i]) / rd)
    arr = np.array(arr)
    arr.shape
    return arr

def DH_model_LCDM(H0, Om, OL, z):
    arr = []
    for j in range(len(z)):
        arr.append(c/(H0*E_model_LCDM(H0, Om, OL, z[j])))
    arr = np.array(arr)
    return arr
def DM_model_LCDM(H0, Om, OL, z):
    arr = []
    for i in range(len(z)):
        arr.append(DA_model_LCDM(H0, Om, OL, z[i]) * (1+z[i]))
    arr = np.array(arr)
    arr.shape
    return arr
def DVrd_model_LCDM(H0, Om, OL, rd, z):
    dh = DH_model_LCDM(H0, Om, OL, z)  
    dm = DM_model_LCDM(H0, Om, OL, z)
    arr = []
    for i in range(len(z)):
        arr.append(np.cbrt(z[i]*dh[i]*dm[i]**2)/rd)
    arr = np.array(arr)
    arr.shape
    return arr


def DMrd_DHrd_model_LCDM(H0, Om, OL, rd, z):
    dMrd = DMrd_model_LCDM(H0, Om, OL, rd, z)
    dHrd = DHrd_model_LCDM(H0, Om, OL, rd, z)
    arr = []
    for i in range(len(z)):
        arr.append(dMrd[i])
        arr.append(dHrd[i])
    arr = np.array(arr)
    arr.shape
    return arr

###############################  
#### Likelihood Definition ####
###############################
## BAO    
## Cov
def lnlikeBAOCov_LCDM(theta, zBAO, dataBAO, inv_covBAO):                 # Likelihood Cov
    H0, Om, OL, M, a, b, intr, rd = theta
    chi2=0.
    ndim=np.shape(inv_covBAO)[0]
    residual=dataBAO-DMrd_DHrd_model_LCDM(H0, Om, OL, rd, zBAO)
    for i in range(0, ndim):
            for j in range(0, ndim):
                chi2=chi2+((residual[i])*inv_covBAO[i,j]*(residual[j]))
    return -0.5 * chi2

## Err
def lnlikeDMrd_LCDM(theta, zBAO, DMrdBAO, errDMrdBAO):
    H0, Om, OL, M, a, b, intr, rd = theta          
    sigma2 = errDMrdBAO ** 2
    return -0.5 * np.sum((DMrdBAO - DMrd_model_LCDM(H0, Om, OL, rd, zBAO)) ** 2 / sigma2)
def lnlikeDHrd_LCDM(theta, zBAO, DHrdBAO, errDHrdBAO):
    H0, Om, OL, M, a, b, intr, rd = theta          
    sigma2 = errDHrdBAO ** 2
    return -0.5 * np.sum((DHrdBAO - DHrd_model_LCDM(H0, Om, OL, rd, zBAO)) ** 2 / sigma2)
def lnlikeDVrd_LCDM(theta, zBAO, DVrdBAO, errDVrdBAO):
    H0, Om, OL, M, a, b, intr, rd = theta          
    sigma2 = errDVrdBAO ** 2
    return -0.5 * np.sum((DVrdBAO - DVrd_model_LCDM(H0, Om, OL, rd, zBAO)) ** 2 / sigma2)

def lnlikeBAO_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia):
    return lnlikeBAOCov_LCDM(theta,zAlam, dataAlam, inv_cov_Alam)+lnlikeBAOCov_LCDM(theta, zHou, dataHou, inv_cov_Hou)+lnlikeBAOCov_LCDM(theta, zGil, dataGil, inv_cov_Gil)+lnlikeDMrd_LCDM(theta, zDumas, DMrdDumas, errDMrdDumas)+lnlikeDHrd_LCDM(theta, zDumas, DHrdDumas, errDHrdDumas)+lnlikeDVrd_LCDM(theta, zRoss, DVrdRoss, errDVrdRoss)+lnlikeDVrd_LCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia)

def lnlikeBAO_Alam_LCDM(theta, zAlam, dataAlam, inv_cov_Alam):
    return lnlikeBAOCov_LCDM(theta,zAlam, dataAlam, inv_cov_Alam)

def lnlikeBAO_Hou_LCDM(theta, zHou, dataHou, inv_cov_Hou):
    return lnlikeBAOCov_LCDM(theta, zHou, dataHou, inv_cov_Hou)

def lnlikeBAO_Gil_LCDM(theta, zGil, dataGil, inv_cov_Gil):
    return lnlikeBAOCov_LCDM(theta, zGil, dataGil, inv_cov_Gil)

def lnlikeBAO_Dumas_LCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas):
    return lnlikeDMrd_LCDM(theta, zDumas, DMrdDumas, errDMrdDumas)+lnlikeDHrd_LCDM(theta, zDumas, DHrdDumas, errDHrdDumas)

def lnlikeBAO_Ross_LCDM(theta, zRoss, DVrdRoss, errDVrdRoss):
    return lnlikeDVrd_LCDM(theta, zRoss, DVrdRoss, errDVrdRoss)

def lnlikeBAO_Demattia_LCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia):
    return lnlikeDVrd_LCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia)
## CC
def lnlikeCC_LCDM(theta, zC, Hz, inv_cov_matC):                 # Likelihood Cov
    H0, Om, OL, M, a, b, intr, rd=theta 
    chi2=0.
    residual=Hz-Hz_model_LCDM(H0, Om, OL, zC)
    for i in range(0, len(zC)):
            for j in range(0, len(zC)):
                chi2=chi2+((residual[i])*inv_cov_matC[i,j]*(residual[j]))
    return -0.5 * chi2
## SNe
def lnlikeSN_LCDM(theta, zS, DmS, inv_cov_matS):                 # Likelihood Cov
    H0, Om, OL, M, a, b, intr, rd=theta 
    mu = mu_model_LCDM(H0, Om, OL, M, zS)
    residual= DmS - mu
    chi2=0.
    for i in range(0, len(zS)):
            for j in range(0, len(zS)):
                chi2=chi2+((residual[i])*inv_cov_matS[i,j]*(residual[j]))
    return -0.5 * chi2

##GRB
def lnlikeGRB_LCDM(theta,z,Ep,Eiso,errEp,errEiso):
    H0, Om, OL, M, a, b, intr, rd=theta
    E_iso=Eiso_model_LCDM(H0,Om,OL,Eiso,z)
    err_Eiso=errEiso_model_LCDM(H0,Om,OL,errEiso,z) 
    logEiso=np.log10(E_iso)
    logEp=np.log10(Ep)
    errlog_iso=err_Eiso/(np.log(10)*E_iso)
    errlog_p=errEp/(np.log(10)*Ep)
    fact1=0.5*np.log((1+a**2)/(2*np.pi*(intr**2+errlog_p**2+(a*errlog_iso)**2)))
    fact2=0.5*((logEp-a*logEiso-b)**2/(intr**2+errlog_p**2+(a*errlog_iso)**2))
    lnlike=np.sum(fact1-fact2)
    return lnlike

#######################################  
#### Probes Combination Likelihood ####
#######################################
###########################
# BAO + CC + SN + GRB
def lnlikeBAOCCSNGRB_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_LCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_LCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_LCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC + GRB
def lnlikeBAOCCGRB_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_LCDM(theta, zC, Hz, inv_cov_matC) + lnlikeGRB_LCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC + SN
def lnlikeBAOCCSN_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):
    return lnlikeBAO_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_LCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_LCDM(theta, zS, DmS, inv_cov_matS)
# BAO + SN + GRB
def lnlikeBAOSNGRB_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS,zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeSN_LCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_LCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# CC + SN + GRB
def lnlikeCCSNGRB_LCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeCC_LCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_LCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_LCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC
def lnlikeBAOCC_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC):
    return lnlikeBAO_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_LCDM(theta, zC, Hz, inv_cov_matC)
# BAO + SN
def lnlikeBAOSN_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS):
    return lnlikeBAO_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeSN_LCDM(theta, zS, DmS, inv_cov_matS)
# BAO + GRB
def lnlikeBAOGRB_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia) + lnlikeGRB_LCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# SN + GRB
def lnlikeSNGRB_LCDM(theta, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeSN_LCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_LCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# CC + SN
def lnlikeCCSN_LCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):
    return lnlikeCC_LCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_LCDM(theta, zS, DmS, inv_cov_matS)
# CC + GRB
def lnlikeCCGRB_LCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeCC_LCDM(theta, zC, Hz, inv_cov_matC)+ lnlikeGRB_LCDM(theta, zG, Ep, Eiso, errEp, errEiso)

###########
## Prior ##
###########
# Flat Prior
def lnflatprior_LCDM(theta):                                 
    H0, Om, OL, M, a, b, intr, rd=theta 
    if (0.0 < H0 < 100.0 and 0.0 < Om < 1.0  and 0.0 < OL < 1.0 and 15. < M < 25.  and 0. < a < 3.0 and 0. < b < 5.0 and 0. < intr < 1.0 and 50 < rd < 250):
        return 0.0
    return -np.inf
###############
## Posterior ##
###############
## BAO + CC + SN + GRB
# Flat Prior 
def lnflatprobBAOCCSNGRB_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAOCCSNGRB_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + CC + GRB
# Flat Prior 
def lnflatprobBAOCCGRB_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAOCCGRB_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC,zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + SN + GRB
# Flat Prior 
def lnflatprobBAOSNGRB_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAOSNGRB_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia,zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + CC + SN
# Flat Prior
def lnflatprobBAOCCSN_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAOCCSN_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC + SN + GRB
# Flat Prior
def lnflatprobCCSNGRB_LCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeCCSNGRB_LCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

## BAO + CC 
# Flat Prior
def lnflatprobBAOCC_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAOCC_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC) if np.isfinite(lp) else -np.inf
## BAO + SN
# Flat Prior 
def lnflatprobBAOSN_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAOSN_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## BAO + GRB
# Flat Prior
def lnflatprobBAOGRB_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAOGRB_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf 
## SN + GRB 
# Flat Prior
def lnflatprobSNGRB_LCDM(theta, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeSNGRB_LCDM(theta,zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf 
## CC + SN
# Flat Prior
def lnflatprobCCSN_LCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeCCSN_LCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC + GRB
# Flat Prior
def lnflatprobCCGRB_LCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeCCGRB_LCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

## BAO
# Flat Prior
def lnflatprobBAO_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAO_LCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Alam_LCDM(theta, zAlam, dataAlam, inv_cov_Alam):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAO_Alam_LCDM(theta, zAlam, dataAlam, inv_cov_Alam) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Hou_LCDM(theta, zHou, dataHou, inv_cov_Hou):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAO_Hou_LCDM(theta, zHou, dataHou, inv_cov_Hou) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Gil_LCDM(theta, zGil, dataGil, inv_cov_Gil):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAO_Gil_LCDM(theta, zGil, dataGil, inv_cov_Gil) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Dumas_LCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAO_Dumas_LCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Ross_LCDM(theta, zRoss, DVrdRoss, errDVrdRoss):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAO_Ross_LCDM(theta, zRoss, DVrdRoss, errDVrdRoss) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Demattia_LCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeBAO_Demattia_LCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia) if np.isfinite(lp) else -np.inf
## SN
# Flat Prior
def lnflatprobSN_LCDM(theta, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeSN_LCDM(theta,zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC
# Flat Prior
def lnflatprobCC_LCDM(theta, zC, Hz, inv_cov_matC):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeCC_LCDM(theta, zC, Hz, inv_cov_matC) if np.isfinite(lp) else -np.inf
## GRB
# Flat Prior
def lnflatprobGRB_LCDM(theta, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_LCDM(theta)
    return lp + lnlikeGRB_LCDM(theta, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

### --------------------------wCDM--------------------------------------

#SN
def mu_model_wCDM(H0, Om, OL, w, M, z):                                                               
    cosmo = wCDM(H0=H0, Om0=Om, Ode0=OL, w0=w)
    mu = np.array(cosmo.distmod(z)) - M
    return mu
# CC
def E_model_wCDM(H0, Om, OL, w, z):
    cosmo = wCDM(H0=H0, Om0=Om, Ode0=OL, w0=w)
    return np.array(cosmo.efunc(z))
def Hz_model_wCDM(H0, Om, OL, w, z):
    arr = []
    for j in range(len(z)):
        arr.append(H0*E_model_wCDM(H0, Om, OL, w, z[j]))
    arr = np.array(arr)
    return arr
def dL_model_wCDM(H0, Om, OL, w, z):                                                              
    cosmo = wCDM(H0=H0, Om0=Om, Ode0=OL, w0=w)
    dl = np.array(cosmo.luminosity_distance(z))
    return dl

#GRB
def Eiso_model_wCDM(H0, Om, OL, w, Eiso,z):
    dL=dL_model_wCDM(H0, Om, OL, w, z)
    dL_cal=dL_model_FLCDM(70,0.3,z)
    E=Eiso*(dL/dL_cal)**2
    return E
def errEiso_model_wCDM(H0, Om, OL, w, errEiso,z):
    dL=dL_model_wCDM(H0, Om, OL, w, z)
    dL_cal=dL_model_FLCDM(70,0.3,z)
    err=errEiso*(dL/dL_cal)**2
    return err

#BAO
def DHrd_model_wCDM(H0, Om, OL, w, rd, z):
    arr = []
    for j in range(len(z)):
        arr.append(c/(H0*rd*E_model_wCDM(H0, Om, OL, w, z[j])))
    arr = np.array(arr)
    return arr


def DA_model_wCDM(H0, Om, OL, w, z):
    cosmo = wCDM(H0=H0, Om0=Om, Ode0=OL, w0=w)
    DA = np.array(cosmo.angular_diameter_distance(z))
    return DA

 
def DMrd_model_wCDM(H0, Om, OL, w, rd, z):
    arr = []
    for i in range(len(z)):
        arr.append(DA_model_wCDM(H0, Om, OL, w, z[i]) * (1+z[i]) / rd)
    arr = np.array(arr)
    arr.shape
    return arr

def DH_model_wCDM(H0, Om, OL, w, z):
    arr = []
    for j in range(len(z)):
        arr.append(c/(H0*E_model_wCDM(H0, Om, OL, w, z[j])))
    arr = np.array(arr)
    return arr
def DM_model_wCDM(H0, Om, OL, w, z):
    arr = []
    for i in range(len(z)):
        arr.append(DA_model_wCDM(H0, Om, OL, w, z[i]) * (1+z[i]))
    arr = np.array(arr)
    arr.shape
    return arr
def DVrd_model_wCDM(H0, Om, OL, w, rd, z):
    dh = DH_model_wCDM(H0, Om, OL, w, z)  
    dm = DM_model_wCDM(H0, Om, OL, w, z)
    arr = []
    for i in range(len(z)):
        arr.append(np.cbrt(z[i]*dh[i]*dm[i]**2)/rd)
    arr = np.array(arr)
    arr.shape
    return arr


def DMrd_DHrd_model_wCDM(H0, Om, OL, w, rd, z):
    dMrd = DMrd_model_wCDM(H0, Om, OL, w, rd, z)
    dHrd = DHrd_model_wCDM(H0, Om, OL, w, rd, z)
    arr = []
    for i in range(len(z)):
        arr.append(dMrd[i])
        arr.append(dHrd[i])
    arr = np.array(arr)
    arr.shape
    return arr

###############################  
#### Likelihood Definition ####
###############################
## BAO    
## Cov
def lnlikeBAOCov_wCDM(theta, zBAO, dataBAO, inv_covBAO):                 # Likelihood Cov
    H0, Om, OL, w, M, a, b, intr, rd = theta
    chi2=0.
    ndim=np.shape(inv_covBAO)[0]
    residual=dataBAO-DMrd_DHrd_model_wCDM(H0, Om, OL, w, rd, zBAO)
    for i in range(0, ndim):
            for j in range(0, ndim):
                chi2=chi2+((residual[i])*inv_covBAO[i,j]*(residual[j]))
    return -0.5 * chi2

## Err
def lnlikeDMrd_wCDM(theta, zBAO, DMrdBAO, errDMrdBAO):
    H0, Om, OL, w, M, a, b, intr, rd = theta          
    sigma2 = errDMrdBAO ** 2
    return -0.5 * np.sum((DMrdBAO - DMrd_model_wCDM(H0, Om, OL, w, rd, zBAO)) ** 2 / sigma2)
def lnlikeDHrd_wCDM(theta, zBAO, DHrdBAO, errDHrdBAO):
    H0, Om, OL, w, M, a, b, intr, rd = theta          
    sigma2 = errDHrdBAO ** 2
    return -0.5 * np.sum((DHrdBAO - DHrd_model_wCDM(H0, Om, OL, w, rd, zBAO)) ** 2 / sigma2)
def lnlikeDVrd_wCDM(theta, zBAO, DVrdBAO, errDVrdBAO):
    H0, Om, OL, w, M, a, b, intr, rd = theta          
    sigma2 = errDVrdBAO ** 2
    return -0.5 * np.sum((DVrdBAO - DVrd_model_wCDM(H0, Om, OL, w, rd, zBAO)) ** 2 / sigma2)

def lnlikeBAO_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia):
    return lnlikeBAOCov_wCDM(theta,zAlam, dataAlam, inv_cov_Alam)+lnlikeBAOCov_wCDM(theta, zHou, dataHou, inv_cov_Hou)+lnlikeBAOCov_wCDM(theta, zGil, dataGil, inv_cov_Gil)+lnlikeDMrd_wCDM(theta, zDumas, DMrdDumas, errDMrdDumas)+lnlikeDHrd_wCDM(theta, zDumas, DHrdDumas, errDHrdDumas)+lnlikeDVrd_wCDM(theta, zRoss, DVrdRoss, errDVrdRoss)+lnlikeDVrd_wCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia)

def lnlikeBAO_Alam_wCDM(theta, zAlam, dataAlam, inv_cov_Alam):
    return lnlikeBAOCov_wCDM(theta,zAlam, dataAlam, inv_cov_Alam)

def lnlikeBAO_Hou_wCDM(theta, zHou, dataHou, inv_cov_Hou):
    return lnlikeBAOCov_wCDM(theta, zHou, dataHou, inv_cov_Hou)

def lnlikeBAO_Gil_wCDM(theta, zGil, dataGil, inv_cov_Gil):
    return lnlikeBAOCov_wCDM(theta, zGil, dataGil, inv_cov_Gil)

def lnlikeBAO_Dumas_wCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas):
    return lnlikeDMrd_wCDM(theta, zDumas, DMrdDumas, errDMrdDumas)+lnlikeDHrd_wCDM(theta, zDumas, DHrdDumas, errDHrdDumas)

def lnlikeBAO_Ross_wCDM(theta, zRoss, DVrdRoss, errDVrdRoss):
    return lnlikeDVrd_wCDM(theta, zRoss, DVrdRoss, errDVrdRoss)

def lnlikeBAO_Demattia_wCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia):
    return lnlikeDVrd_wCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia)

## CC
def lnlikeCC_wCDM(theta, zC, Hz, inv_cov_matC):                 # Likelihood Cov
    H0, Om, OL, w, M, a, b, intr, rd=theta 
    chi2=0.
    residual=Hz-Hz_model_wCDM(H0, Om, OL, w, zC)
    for i in range(0, len(zC)):
            for j in range(0, len(zC)):
                chi2=chi2+((residual[i])*inv_cov_matC[i,j]*(residual[j]))
    return -0.5 * chi2
## SNe
def lnlikeSN_wCDM(theta, zS, DmS, inv_cov_matS):                 # Likelihood Cov
    H0, Om, OL, w, M, a, b, intr, rd=theta 
    mu = mu_model_wCDM(H0, Om, OL, w, M, zS)
    residual= DmS - mu
    chi2=0.
    for i in range(0, len(zS)):
            for j in range(0, len(zS)):
                chi2=chi2+((residual[i])*inv_cov_matS[i,j]*(residual[j]))
    return -0.5 * chi2

##GRB
def lnlikeGRB_wCDM(theta,z,Ep,Eiso,errEp,errEiso):
    H0, Om, OL, w, M, a, b, intr, rd=theta
    E_iso=Eiso_model_wCDM(H0, Om, OL, w, Eiso,z)
    err_Eiso=errEiso_model_wCDM(H0, Om, OL, w, errEiso,z) 
    logEiso=np.log10(E_iso)
    logEp=np.log10(Ep)
    errlog_iso=err_Eiso/(np.log(10)*E_iso)
    errlog_p=errEp/(np.log(10)*Ep)
    fact1=0.5*np.log((1+a**2)/(2*np.pi*(intr**2+errlog_p**2+(a*errlog_iso)**2)))
    fact2=0.5*((logEp-a*logEiso-b)**2/(intr**2+errlog_p**2+(a*errlog_iso)**2))
    lnlike=np.sum(fact1-fact2)
    return lnlike

#######################################  
#### Probes Combination Likelihood ####
#######################################
###########################
# BAO + CC + SN + GRB
def lnlikeBAOCCSNGRB_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_wCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_wCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_wCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC + GRB
def lnlikeBAOCCGRB_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_wCDM(theta, zC, Hz, inv_cov_matC) + lnlikeGRB_wCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC + SN
def lnlikeBAOCCSN_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):
    return lnlikeBAO_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_wCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_wCDM(theta, zS, DmS, inv_cov_matS)
# BAO + SN + GRB
def lnlikeBAOSNGRB_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS,zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeSN_wCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_wCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# CC + SN + GRB
def lnlikeCCSNGRB_wCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeCC_wCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_wCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_wCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC
def lnlikeBAOCC_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC):
    return lnlikeBAO_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_wCDM(theta, zC, Hz, inv_cov_matC)
# BAO + SN
def lnlikeBAOSN_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS):
    return lnlikeBAO_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeSN_wCDM(theta, zS, DmS, inv_cov_matS)
# BAO + GRB
def lnlikeBAOGRB_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia) + lnlikeGRB_wCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# SN + GRB
def lnlikeSNGRB_wCDM(theta, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeSN_wCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_wCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# CC + SN
def lnlikeCCSN_wCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):
    return lnlikeCC_wCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_wCDM(theta, zS, DmS, inv_cov_matS)
# CC + GRB
def lnlikeCCGRB_wCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeCC_wCDM(theta, zC, Hz, inv_cov_matC)+ lnlikeGRB_wCDM(theta, zG, Ep, Eiso, errEp, errEiso)

###########
## Prior ##
###########
# Flat Prior
def lnflatprior_wCDM(theta):                                 
    H0, Om, OL, w, M, a, b, intr, rd=theta 
    if (0.0 < H0 < 100.0 and 0.0 < Om < 1.0  and 0.0 < OL < 1.0 and -5.0 < w < -0.3 and 15. < M < 25.  and 0. < a < 3.0 and 0. < b < 5.0 and 0. < intr < 1.0 and 50 < rd < 250):
        return 0.0
    return -np.inf
###############
## Posterior ##
###############
## BAO + CC + SN + GRB
# Flat Prior 
def lnflatprobBAOCCSNGRB_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAOCCSNGRB_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + CC + GRB
# Flat Prior 
def lnflatprobBAOCCGRB_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAOCCGRB_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC,zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + SN + GRB
# Flat Prior 
def lnflatprobBAOSNGRB_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAOSNGRB_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia,zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + CC + SN
# Flat Prior
def lnflatprobBAOCCSN_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAOCCSN_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC + SN + GRB
# Flat Prior
def lnflatprobCCSNGRB_wCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeCCSNGRB_wCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

## BAO + CC 
# Flat Prior
def lnflatprobBAOCC_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAOCC_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC) if np.isfinite(lp) else -np.inf
## BAO + SN
# Flat Prior 
def lnflatprobBAOSN_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAOSN_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## BAO + GRB
# Flat Prior
def lnflatprobBAOGRB_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAOGRB_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf 
## SN + GRB 
# Flat Prior
def lnflatprobSNGRB_wCDM(theta, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeSNGRB_wCDM(theta,zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf 
## CC + SN
# Flat Prior
def lnflatprobCCSN_wCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeCCSN_wCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC + GRB
# Flat Prior
def lnflatprobCCGRB_wCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeCCGRB_wCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

## BAO
# Flat Prior
def lnflatprobBAO_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAO_wCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Alam_wCDM(theta, zAlam, dataAlam, inv_cov_Alam):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAO_Alam_wCDM(theta, zAlam, dataAlam, inv_cov_Alam) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Hou_wCDM(theta, zHou, dataHou, inv_cov_Hou):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAO_Hou_wCDM(theta, zHou, dataHou, inv_cov_Hou) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Gil_wCDM(theta, zGil, dataGil, inv_cov_Gil):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAO_Gil_wCDM(theta, zGil, dataGil, inv_cov_Gil) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Dumas_wCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAO_Dumas_wCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Ross_wCDM(theta, zRoss, DVrdRoss, errDVrdRoss):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAO_Ross_wCDM(theta, zRoss, DVrdRoss, errDVrdRoss) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Demattia_wCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeBAO_Demattia_wCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia) if np.isfinite(lp) else -np.inf

## SN
# Flat Prior
def lnflatprobSN_wCDM(theta, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeSN_wCDM(theta,zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC
# Flat Prior
def lnflatprobCC_wCDM(theta, zC, Hz, inv_cov_matC):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeCC_wCDM(theta, zC, Hz, inv_cov_matC) if np.isfinite(lp) else -np.inf
## GRB
# Flat Prior
def lnflatprobGRB_wCDM(theta, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_wCDM(theta)
    return lp + lnlikeGRB_wCDM(theta, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

####-------------------------w0waCDM------------------------------####

#SN
def mu_model_w0waCDM(H0, Om, OL, w0, wa, M, z):                                                               
    cosmo = w0waCDM(H0=H0, Om0=Om, Ode0=OL, w0=w0, wa=wa)
    mu = np.array(cosmo.distmod(z)) - M
    return mu
# CC
def E_model_w0waCDM(H0, Om, OL, w0, wa, z):
    cosmo = w0waCDM(H0=H0, Om0=Om, Ode0=OL, w0=w0, wa=wa)
    return np.array(cosmo.efunc(z))
def Hz_model_w0waCDM(H0, Om, OL, w0, wa, z):
    arr = []
    for j in range(len(z)):
        arr.append(H0*E_model_w0waCDM(H0, Om, OL, w0, wa, z[j]))
    arr = np.array(arr)
    return arr
def dL_model_w0waCDM(H0, Om, OL, w0, wa, z):                                                              
    cosmo = w0waCDM(H0=H0, Om0=Om, Ode0=OL, w0=w0, wa=wa)
    dl = np.array(cosmo.luminosity_distance(z))
    return dl

#GRB
def Eiso_model_w0waCDM(H0, Om, OL, w0, wa, Eiso,z):
    dL=dL_model_w0waCDM(H0, Om, OL, w0, wa, z)
    dL_cal=dL_model_FLCDM(70,0.3,z)
    E=Eiso*(dL/dL_cal)**2
    return E
def errEiso_model_w0waCDM(H0, Om, OL, w0, wa, errEiso,z):
    dL=dL_model_w0waCDM(H0, Om, OL, w0, wa, z)
    dL_cal=dL_model_FLCDM(70,0.3,z)
    err=errEiso*(dL/dL_cal)**2
    return err

#BAO
def DHrd_model_w0waCDM(H0, Om, OL, w0, wa, rd, z):
    arr = []
    for j in range(len(z)):
        arr.append(c/(H0*rd*E_model_w0waCDM(H0, Om, OL, w0, wa, z[j])))
    arr = np.array(arr)
    return arr


def DA_model_w0waCDM(H0, Om, OL, w0, wa, z):
    cosmo = w0waCDM(H0=H0, Om0=Om, Ode0=OL, w0=w0, wa=wa)
    DA = np.array(cosmo.angular_diameter_distance(z))
    return DA

 
def DMrd_model_w0waCDM(H0, Om, OL, w0, wa, rd, z):
    arr = []
    for i in range(len(z)):
        arr.append(DA_model_w0waCDM(H0, Om, OL, w0, wa, z[i]) * (1+z[i]) / rd)
    arr = np.array(arr)
    arr.shape
    return arr

def DH_model_w0waCDM(H0, Om, OL, w0, wa, z):
    arr = []
    for j in range(len(z)):
        arr.append(c/(H0*E_model_w0waCDM(H0, Om, OL, w0, wa, z[j])))
    arr = np.array(arr)
    return arr
def DM_model_w0waCDM(H0, Om, OL, w0, wa, z):
    arr = []
    for i in range(len(z)):
        arr.append(DA_model_w0waCDM(H0, Om, OL, w0, wa, z[i]) * (1+z[i]))
    arr = np.array(arr)
    arr.shape
    return arr
def DVrd_model_w0waCDM(H0, Om, OL, w0, wa, rd, z):
    dh = DH_model_w0waCDM(H0, Om, OL, w0, wa, z)  
    dm = DM_model_w0waCDM(H0, Om, OL, w0, wa, z)
    arr = []
    for i in range(len(z)):
        arr.append(np.cbrt(z[i]*dh[i]*dm[i]**2)/rd)
    arr = np.array(arr)
    arr.shape
    return arr


def DMrd_DHrd_model_w0waCDM(H0, Om, OL, w0, wa, rd, z):
    dMrd = DMrd_model_w0waCDM(H0, Om, OL, w0, wa, rd, z)
    dHrd = DHrd_model_w0waCDM(H0, Om, OL, w0, wa, rd, z)
    arr = []
    for i in range(len(z)):
        arr.append(dMrd[i])
        arr.append(dHrd[i])
    arr = np.array(arr)
    arr.shape
    return arr

###############################  
#### Likelihood Definition ####
###############################
## BAO    
## Cov
def lnlikeBAOCov_w0waCDM(theta, zBAO, dataBAO, inv_covBAO):                 # Likelihood Cov
    H0, Om, OL, w0, wa, M, a, b, intr, rd = theta
    chi2=0.
    ndim=np.shape(inv_covBAO)[0]
    residual=dataBAO-DMrd_DHrd_model_w0waCDM(H0, Om, OL, w0, wa, rd, zBAO)
    for i in range(0, ndim):
            for j in range(0, ndim):
                chi2=chi2+((residual[i])*inv_covBAO[i,j]*(residual[j]))
    return -0.5 * chi2

## Err
def lnlikeDMrd_w0waCDM(theta, zBAO, DMrdBAO, errDMrdBAO):
    H0, Om, OL, w0, wa, M, a, b, intr, rd = theta          
    sigma2 = errDMrdBAO ** 2
    return -0.5 * np.sum((DMrdBAO - DMrd_model_w0waCDM(H0, Om, OL, w0, wa, rd, zBAO)) ** 2 / sigma2)
def lnlikeDHrd_w0waCDM(theta, zBAO, DHrdBAO, errDHrdBAO):
    H0, Om, OL, w0, wa, M, a, b, intr, rd = theta          
    sigma2 = errDHrdBAO ** 2
    return -0.5 * np.sum((DHrdBAO - DHrd_model_w0waCDM(H0, Om, OL, w0, wa, rd, zBAO)) ** 2 / sigma2)
def lnlikeDVrd_w0waCDM(theta, zBAO, DVrdBAO, errDVrdBAO):
    H0, Om, OL, w0, wa, M, a, b, intr, rd = theta          
    sigma2 = errDVrdBAO ** 2
    return -0.5 * np.sum((DVrdBAO - DVrd_model_w0waCDM(H0, Om, OL, w0, wa, rd, zBAO)) ** 2 / sigma2)

def lnlikeBAO_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia):
    return lnlikeBAOCov_w0waCDM(theta,zAlam, dataAlam, inv_cov_Alam)+lnlikeBAOCov_w0waCDM(theta, zHou, dataHou, inv_cov_Hou)+lnlikeBAOCov_w0waCDM(theta, zGil, dataGil, inv_cov_Gil)+lnlikeDMrd_w0waCDM(theta, zDumas, DMrdDumas, errDMrdDumas)+lnlikeDHrd_w0waCDM(theta, zDumas, DHrdDumas, errDHrdDumas)+lnlikeDVrd_w0waCDM(theta, zRoss, DVrdRoss, errDVrdRoss)+lnlikeDVrd_w0waCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia)

def lnlikeBAO_Alam_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam):
    return lnlikeBAOCov_w0waCDM(theta,zAlam, dataAlam, inv_cov_Alam)

def lnlikeBAO_Hou_w0waCDM(theta, zHou, dataHou, inv_cov_Hou):
    return lnlikeBAOCov_w0waCDM(theta, zHou, dataHou, inv_cov_Hou)

def lnlikeBAO_Gil_w0waCDM(theta, zGil, dataGil, inv_cov_Gil):
    return lnlikeBAOCov_w0waCDM(theta, zGil, dataGil, inv_cov_Gil)

def lnlikeBAO_Dumas_w0waCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas):
    return lnlikeDMrd_w0waCDM(theta, zDumas, DMrdDumas, errDMrdDumas)+lnlikeDHrd_w0waCDM(theta, zDumas, DHrdDumas, errDHrdDumas)

def lnlikeBAO_Ross_w0waCDM(theta, zRoss, DVrdRoss, errDVrdRoss):
    return lnlikeDVrd_w0waCDM(theta, zRoss, DVrdRoss, errDVrdRoss)

def lnlikeBAO_Demattia_w0waCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia):
    return lnlikeDVrd_w0waCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia)

## CC
def lnlikeCC_w0waCDM(theta, zC, Hz, inv_cov_matC):                 # Likelihood Cov
    H0, Om, OL, w0, wa, M, a, b, intr, rd=theta 
    chi2=0.
    residual=Hz-Hz_model_w0waCDM(H0, Om, OL, w0, wa, zC)
    for i in range(0, len(zC)):
            for j in range(0, len(zC)):
                chi2=chi2+((residual[i])*inv_cov_matC[i,j]*(residual[j]))
    return -0.5 * chi2
## SNe
def lnlikeSN_w0waCDM(theta, zS, DmS, inv_cov_matS):                 # Likelihood Cov
    H0, Om, OL, w0, wa, M, a, b, intr, rd=theta 
    mu = mu_model_w0waCDM(H0, Om, OL, w0, wa, M, zS)
    residual= DmS - mu
    chi2=0.
    for i in range(0, len(zS)):
            for j in range(0, len(zS)):
                chi2=chi2+((residual[i])*inv_cov_matS[i,j]*(residual[j]))
    return -0.5 * chi2

##GRB
def lnlikeGRB_w0waCDM(theta,z,Ep,Eiso,errEp,errEiso):
    H0, Om, OL, w0, wa, M, a, b, intr, rd=theta
    E_iso=Eiso_model_w0waCDM(H0, Om, OL, w0, wa, Eiso,z)
    err_Eiso=errEiso_model_w0waCDM(H0, Om, OL, w0, wa, errEiso,z) 
    logEiso=np.log10(E_iso)
    logEp=np.log10(Ep)
    errlog_iso=err_Eiso/(np.log(10)*E_iso)
    errlog_p=errEp/(np.log(10)*Ep)
    fact1=0.5*np.log((1+a**2)/(2*np.pi*(intr**2+errlog_p**2+(a*errlog_iso)**2)))
    fact2=0.5*((logEp-a*logEiso-b)**2/(intr**2+errlog_p**2+(a*errlog_iso)**2))
    lnlike=np.sum(fact1-fact2)
    return lnlike

#######################################  
#### Probes Combination Likelihood ####
#######################################
###########################
# BAO + CC + SN + GRB
def lnlikeBAOCCSNGRB_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_w0waCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_w0waCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_w0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC + GRB
def lnlikeBAOCCGRB_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_w0waCDM(theta, zC, Hz, inv_cov_matC) + lnlikeGRB_w0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC + SN
def lnlikeBAOCCSN_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):
    return lnlikeBAO_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_w0waCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_w0waCDM(theta, zS, DmS, inv_cov_matS)
# BAO + SN + GRB
def lnlikeBAOSNGRB_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS,zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeSN_w0waCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_w0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# CC + SN + GRB
def lnlikeCCSNGRB_w0waCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeCC_w0waCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_w0waCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_w0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# BAO + CC
def lnlikeBAOCC_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC):
    return lnlikeBAO_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeCC_w0waCDM(theta, zC, Hz, inv_cov_matC)
# BAO + SN
def lnlikeBAOSN_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS):
    return lnlikeBAO_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia)+lnlikeSN_w0waCDM(theta, zS, DmS, inv_cov_matS)
# BAO + GRB
def lnlikeBAOGRB_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeBAO_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia) + lnlikeGRB_w0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# SN + GRB
def lnlikeSNGRB_w0waCDM(theta, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeSN_w0waCDM(theta, zS, DmS, inv_cov_matS) + lnlikeGRB_w0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)
# CC + SN
def lnlikeCCSN_w0waCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):
    return lnlikeCC_w0waCDM(theta, zC, Hz, inv_cov_matC)+lnlikeSN_w0waCDM(theta, zS, DmS, inv_cov_matS)
# CC + GRB
def lnlikeCCGRB_w0waCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):
    return lnlikeCC_w0waCDM(theta, zC, Hz, inv_cov_matC)+ lnlikeGRB_w0waCDM(theta, zG, Ep, Eiso, errEp, errEiso)

###########
## Prior ##
###########
# Flat Prior
def lnflatprior_w0waCDM(theta):                                 
    H0, Om, OL, w0, wa, M, a, b, intr, rd=theta 
    if (0.0 < H0 < 100.0 and 0.0 < Om < 1.0  and 0.0 < OL < 1.0 and -5.0 < w0 < -0.3  and -5.0 < wa < 5 and 15. < M < 25.  and 0. < a < 3.0 and 0. < b < 5.0 and 0. < intr < 1.0 and 50 < rd < 250):
        return 0.0
    return -np.inf
###############
## Posterior ##
###############
## BAO + CC + SN + GRB
# Flat Prior 
def lnflatprobBAOCCSNGRB_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAOCCSNGRB_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + CC + GRB
# Flat Prior 
def lnflatprobBAOCCGRB_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAOCCGRB_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC,zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + SN + GRB
# Flat Prior 
def lnflatprobBAOSNGRB_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAOSNGRB_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia,zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf
## BAO + CC + SN
# Flat Prior
def lnflatprobBAOCCSN_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAOCCSN_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC + SN + GRB
# Flat Prior
def lnflatprobCCSNGRB_w0waCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeCCSNGRB_w0waCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

## BAO + CC 
# Flat Prior
def lnflatprobBAOCC_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAOCC_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zC, Hz, inv_cov_matC) if np.isfinite(lp) else -np.inf
## BAO + SN
# Flat Prior 
def lnflatprobBAOSN_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAOSN_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## BAO + GRB
# Flat Prior
def lnflatprobBAOGRB_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAOGRB_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf 
## SN + GRB 
# Flat Prior
def lnflatprobSNGRB_w0waCDM(theta, zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeSNGRB_w0waCDM(theta,zS, DmS, inv_cov_matS, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf 
## CC + SN
# Flat Prior
def lnflatprobCCSN_w0waCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeCCSN_w0waCDM(theta, zC, Hz, inv_cov_matC, zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC + GRB
# Flat Prior
def lnflatprobCCGRB_w0waCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeCCGRB_w0waCDM(theta, zC, Hz, inv_cov_matC, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

## BAO
# Flat Prior
def lnflatprobBAO_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAO_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam, zHou, dataHou, inv_cov_Hou, zGil, dataGil, inv_cov_Gil, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas, zRoss, DVrdRoss, errDVrdRoss, zDemattia, DVrdDemattia, errDVrdDemattia) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Alam_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAO_Alam_w0waCDM(theta, zAlam, dataAlam, inv_cov_Alam) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Hou_w0waCDM(theta, zHou, dataHou, inv_cov_Hou):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAO_Hou_w0waCDM(theta, zHou, dataHou, inv_cov_Hou) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Gil_w0waCDM(theta, zGil, dataGil, inv_cov_Gil):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAO_Gil_w0waCDM(theta, zGil, dataGil, inv_cov_Gil) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Dumas_w0waCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAO_Dumas_w0waCDM(theta, zDumas, DMrdDumas, errDMrdDumas, DHrdDumas, errDHrdDumas) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Ross_w0waCDM(theta, zRoss, DVrdRoss, errDVrdRoss):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAO_Ross_w0waCDM(theta, zRoss, DVrdRoss, errDVrdRoss) if np.isfinite(lp) else -np.inf

def lnflatprobBAO_Demattia_w0waCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeBAO_Demattia_w0waCDM(theta, zDemattia, DVrdDemattia, errDVrdDemattia) if np.isfinite(lp) else -np.inf

## SN
# Flat Prior
def lnflatprobSN_w0waCDM(theta, zS, DmS, inv_cov_matS):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeSN_w0waCDM(theta,zS, DmS, inv_cov_matS) if np.isfinite(lp) else -np.inf
## CC
# Flat Prior
def lnflatprobCC_w0waCDM(theta, zC, Hz, inv_cov_matC):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeCC_w0waCDM(theta, zC, Hz, inv_cov_matC) if np.isfinite(lp) else -np.inf
## GRB
# Flat Prior
def lnflatprobGRB_w0waCDM(theta, zG, Ep, Eiso, errEp, errEiso):             # Cov Probability
    lp = lnflatprior_w0waCDM(theta)
    return lp + lnlikeGRB_w0waCDM(theta, zG, Ep, Eiso, errEp, errEiso) if np.isfinite(lp) else -np.inf

