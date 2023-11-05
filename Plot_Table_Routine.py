"""
@file Plot_Table_Routine.py
@date September 6, 2023
@authors Fabrizio Cogato <fabrizio.cogato@inaf.it>
         Michele Moresco <michele.moresco@unibo.it>

Please remember to cite: https://ui.adsabs.harvard.edu/abs/2023arXiv230901375C
"""

###################
#### Libraries ####
###################
from operator import truth
import os
import numpy as np
import random

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
import pandas as pd
import emcee
from astropy.table import Table
from astropy.cosmology import FlatLambdaCDM, LambdaCDM, FlatwCDM, wCDM, Flatw0waCDM, w0waCDM
from scipy.stats import norm
from chainconsumer import ChainConsumer

##############################
#### Functions Definition ####
##############################
# Selecting Column Function
def column(matrix, i):                          
    return [row[i] for row in matrix]

def plot_bestfitMCMC(cosmo, probe, burnin, dir_chain, dir_plot):
    os.chdir(dir_chain+probe)
    filename = "Chain_FlatPrior_{0}_{1}.h5".format(cosmo,probe)
    chain = emcee.backends.HDFBackend(filename)
    flat_samples = chain.get_chain(discard=burnin, flat=True, thin=1)
    samples = chain.get_chain()

    if cosmo=='FLCDM':
        parstep = [samples[:,:,0], samples[:,:,1], samples[:,:,2], samples[:,:,3], samples[:,:,4], samples[:,:,5], samples[:,:,6]]
        labels = ["H0", "Om", "M", "a", "b", "int", "rd"]
        N=len(labels)
        ndim=N 
           
    if cosmo=='LCDM':
        parstep = [samples[:,:,0], samples[:,:,1], samples[:,:,2], samples[:,:,3], samples[:,:,4], samples[:,:,5], samples[:,:,6], samples[:,:,7]]
        labels = ["H0", "Om", "OL", "M", "a", 'b' , 'int', "rd"]
        N= len(labels)
        ndim=N    
    if cosmo=='FwCDM':
        parstep = [samples[:,:,0], samples[:,:,1], samples[:,:,2], samples[:,:,3], samples[:,:,4], samples[:,:,5], samples[:,:,6], samples[:,:,7]]
        labels = ["H0", "Om", "w", "M", "a", 'b' , 'int', "rd"]
        N= len(labels)
        ndim=N
    if cosmo=='wCDM':
        parstep = [samples[:,:,0], samples[:,:,1], samples[:,:,2], samples[:,:,3], samples[:,:,4], samples[:,:,5], samples[:,:,6], samples[:,:,7], samples[:,:,8]]
        labels = ["H0", "Om", "OL", "w", "M", "a", 'b' , 'int', "rd"]
        N= len(labels)
        ndim=N
    if cosmo=='Fw0waCDM' :
        parstep = [samples[:,:,0], samples[:,:,1], samples[:,:,2], samples[:,:,3], samples[:,:,4], samples[:,:,5], samples[:,:,6], samples[:,:,7], samples[:,:,8]]
        labels = ["H0", "Om", "w0", "wa", "M", "a", 'b' , 'int', "rd"]
        N= len(labels)
        ndim=N
    if cosmo=='w0waCDM':
        parstep = [samples[:,:,0], samples[:,:,1], samples[:,:,2], samples[:,:,3], samples[:,:,4], samples[:,:,5], samples[:,:,6], samples[:,:,7], samples[:,:,8], samples[:,:,9]]
        labels = ["H0", "Om", "OL", "w0$", "wa", "M", "a", 'b' , 'int', "rd"]

        N= len(labels)
        ndim=N
 
############### Create Plots
# Par vs. Step 
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    plt.suptitle(f'ParVsStep')
    for i in range(ndim):
        ax = axes[i]
        ax.plot(parstep[i], "k", alpha=0.3, zorder=1)
        ax.axvspan(0, burnin, facecolor='firebrick', alpha=0.3, zorder= 2)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i], fontsize=16)
        ax.yaxis.set_label_coords(-0.1, 0.5)
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
    axes[-1].set_xlabel("Step Number", fontsize=16)   
    os.chdir(dir_plot)
    plt.savefig(r'ParStep_FlatPrior_{0}_{1}.png'.format(cosmo, probe), dpi=350)
    os.chdir(dir_home)
    plt.clf()
    

    #Corner
    color='#b2967d'
        
    c = ChainConsumer()
    c.add_chain(flat_samples, parameters=labels, color=color, shade=True, shade_alpha=1, bar_shade=True, linewidth=1)
    c.configure(statistics='cumulative', diagonal_tick_labels=True, serif=True, usetex=False, summary=False, tick_font_size=16, label_font_size=28, max_ticks=4)
    # Save
    os.chdir(dir_plot)
    c.plotter.plot(filename=f'Corner_{cosmo}_{probe}', parameters=labels, figsize="page")
    os.chdir(dir_home)
    plt.clf()



def table_paper(probe, burnin, dir_chain, dir_out): 
    dirchain=dir_chain+probe   
    thin=1
    os.chdir(dirchain)
    FLCDM = "Chain_FlatPrior_FLCDM_{0}.h5".format(probe)
    chain = emcee.backends.HDFBackend(FLCDM)
    
    flcdm = chain.get_chain(discard=burnin[0], flat=True, thin=thin)
    prob_flcdm=chain.get_log_prob(discard=burnin[0], flat=True, thin=thin)
    
    LCDM = "Chain_FlatPrior_LCDM_{0}.h5".format(probe)
    chain = emcee.backends.HDFBackend(LCDM)
    lcdm = chain.get_chain(discard=burnin[1], flat=True, thin=thin)
    prob_lcdm=chain.get_log_prob(discard=burnin[1], flat=True, thin=thin)
    
    FWCDM = "Chain_FlatPrior_FwCDM_{0}.h5".format(probe)
    chain = emcee.backends.HDFBackend(FWCDM)
    fwcdm = chain.get_chain(discard=burnin[2], flat=True, thin=thin)
    
    WCDM = "Chain_FlatPrior_wCDM_{0}.h5".format(probe)
    chain = emcee.backends.HDFBackend(WCDM)
    wcdm = chain.get_chain(discard=burnin[3], flat=True, thin=thin)
    
    Fw0waCDM = "Chain_FlatPrior_Fw0waCDM_{0}.h5".format(probe)
    chain = emcee.backends.HDFBackend(Fw0waCDM)
    fw0wacdm = chain.get_chain(discard=burnin[4], flat=True, thin=thin)
    
    W0WaCDM = "Chain_FlatPrior_w0waCDM_{0}.h5".format(probe)
    chain = emcee.backends.HDFBackend(W0WaCDM)
    w0wacdm = chain.get_chain(discard=burnin[5], flat=True, thin=thin)


    perc = [0.15, 2.3, 15.85, 50, 84.15, 97.7, 99.85]

    sH0_flcdm, sOm_flcdm, sM_flcdm, sa_flcdm, sb_flcdm, sint_flcdm, srd_flcdm = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                            zip(*np.percentile(flcdm, perc, axis=0)))

    H0_flcdm = str(np.around(sH0_flcdm[0],1))
    H0errup_flcdm = str(np.around(sH0_flcdm[1],1))
    H0errdw_flcdm = str(np.around(sH0_flcdm[2],1))
    stringH0_flcdm = "$"+H0_flcdm+"^{+"+H0errup_flcdm+"}_{-"+H0errdw_flcdm+"}$"

    Om_flcdm = str(np.around(sOm_flcdm[0],3))
    Omerrup_flcdm = str(np.around(sOm_flcdm[1],3))
    Omerrdw_flcdm = str(np.around(sOm_flcdm[2],3))
    stringOm_flcdm = "$"+Om_flcdm+"^{+"+Omerrup_flcdm+"}_{-"+Omerrdw_flcdm+"}$"

    m=column(flcdm,2)
    M=[]
    for i in range(len(m)):
        M.append(-1*m[i])
    M_flcdm=str(np.around((np.percentile(M,50)),1))
    Merrup_flcdm = str(np.around((np.percentile(M,84.15)-np.percentile(M,50)),1))
    Merrdw_flcdm = str(np.around((np.percentile(M,50)-np.percentile(M,15.85)),1))
    stringM_flcdm = "$"+M_flcdm+"^{+"+Merrup_flcdm+"}_{-"+Merrdw_flcdm+"}$"

    a_flcdm = str(np.around(sa_flcdm[0],3))
    aerrup_flcdm = str(np.around(sa_flcdm[1],3))
    aerrdw_flcdm = str(np.around(sa_flcdm[2],3))
    stringa_flcdm = "$"+a_flcdm+"^{+"+aerrup_flcdm+"}_{-"+aerrdw_flcdm+"}$"

    b_flcdm = str(np.around(sb_flcdm[0],2))
    berrup_flcdm = str(np.around(sb_flcdm[1],2))
    berrdw_flcdm = str(np.around(sb_flcdm[2],2))
    stringb_flcdm = "$"+b_flcdm+"^{+"+berrup_flcdm+"}_{-"+berrdw_flcdm+"}$"

    int_flcdm = str(np.around(sint_flcdm[0],3))
    interrup_flcdm = str(np.around(sint_flcdm[1],3))
    interrdw_flcdm = str(np.around(sint_flcdm[2],3))
    stringint_flcdm = "$"+int_flcdm+"^{+"+interrup_flcdm+"}_{-"+interrdw_flcdm+"}$"

    rd_flcdm = str(np.around(srd_flcdm[0],1))
    rderrup_flcdm = str(np.around(srd_flcdm[1],1))
    rderrdw_flcdm = str(np.around(srd_flcdm[2],1))
    stringrd_flcdm = "$"+rd_flcdm+"^{+"+rderrup_flcdm+"}_{-"+rderrdw_flcdm+"}$"

    sH0_lcdm, sOm_lcdm, sOl_lcdm, sM_lcdm, sa_lcdm, sb_lcdm, sint_lcdm, srd_lcdm = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                            zip(*np.percentile(lcdm, perc, axis=0)))

    H0_lcdm = str(np.around(sH0_lcdm[0],1))
    H0errup_lcdm = str(np.around(sH0_lcdm[1],1))
    H0errdw_lcdm = str(np.around(sH0_lcdm[2],1))
    stringH0_lcdm = "$"+H0_lcdm+"^{+"+H0errup_lcdm+"}_{-"+H0errdw_lcdm+"}$"

    Om_lcdm = str(np.around(sOm_lcdm[0],3))
    Omerrup_lcdm = str(np.around(sOm_lcdm[1],3))
    Omerrdw_lcdm = str(np.around(sOm_lcdm[2],3))
    stringOm_lcdm = "$"+Om_lcdm+"^{+"+Omerrup_lcdm+"}_{-"+Omerrdw_lcdm+"}$"

    Ol_lcdm = str(np.around(sOl_lcdm[0],3))
    Olerrup_lcdm = str(np.around(sOl_lcdm[1],3))
    Olerrdw_lcdm = str(np.around(sOl_lcdm[2],3))
    stringsOl_lcdm = "$"+Ol_lcdm+"^{+"+Olerrup_lcdm+"}_{-"+Olerrdw_lcdm+"}$"

    Om=np.array(column(lcdm,1))
    OL=np.array(column(lcdm,2))
    Ok=[]
    for i in range(len(Om)):
        Ok.append(1-Om[i]-OL[i])
    Ok_lcdm=str(np.around((np.percentile(Ok,50)),3))
    Okerrup_lcdm = str(np.around((np.percentile(Ok,84.15)-np.percentile(Ok,50)),3))
    Okerrdw_lcdm = str(np.around((np.percentile(Ok,50)-np.percentile(Ok,15.85)),3))
    stringOk_lcdm = "$"+Ok_lcdm+"^{+"+Okerrup_lcdm+"}_{-"+Okerrdw_lcdm+"}$"

    m=column(lcdm,3)
    M=[]
    for i in range(len(m)):
        M.append(-1*m[i])
    M_lcdm=str(np.around((np.percentile(M,50)),1))
    Merrup_lcdm = str(np.around((np.percentile(M,84.15)-np.percentile(M,50)),1))
    Merrdw_lcdm = str(np.around((np.percentile(M,50)-np.percentile(M,15.85)),1))
    stringM_lcdm = "$"+M_lcdm+"^{+"+Merrup_lcdm+"}_{-"+Merrdw_lcdm+"}$"

    a_lcdm = str(np.around(sa_lcdm[0],3))
    aerrup_lcdm = str(np.around(sa_lcdm[1],3))
    aerrdw_lcdm = str(np.around(sa_lcdm[2],3))
    stringa_lcdm = "$"+a_lcdm+"^{+"+aerrup_lcdm+"}_{-"+aerrdw_lcdm+"}$"

    b_lcdm = str(np.around(sb_lcdm[0],2))
    berrup_lcdm = str(np.around(sb_lcdm[1],2))
    berrdw_lcdm = str(np.around(sb_lcdm[2],2))
    stringb_lcdm = "$"+b_lcdm+"^{+"+berrup_lcdm+"}_{-"+berrdw_lcdm+"}$"

    int_lcdm = str(np.around(sint_lcdm[0],3))
    interrup_lcdm = str(np.around(sint_lcdm[1],3))
    interrdw_lcdm = str(np.around(sint_lcdm[2],3))
    stringint_lcdm = "$"+int_lcdm+"^{+"+interrup_lcdm+"}_{-"+interrdw_lcdm+"}$"

    rd_lcdm = str(np.around(srd_lcdm[0],1))
    rderrup_lcdm = str(np.around(srd_lcdm[1],1))
    rderrdw_lcdm = str(np.around(srd_lcdm[2],1))
    stringrd_lcdm = "$"+rd_lcdm+"^{+"+rderrup_lcdm+"}_{-"+rderrdw_lcdm+"}$"

    sH0_fwcdm, sOm_fwcdm, sw_fwcdm, sM_fwcdm, sa_fwcdm, sb_fwcdm, sint_fwcdm, srd_fwcdm = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                            zip(*np.percentile(fwcdm, perc, axis=0)))

    H0_fwcdm = str(np.around(sH0_fwcdm[0],1))
    H0errup_fwcdm = str(np.around(sH0_fwcdm[1],1))
    H0errdw_fwcdm = str(np.around(sH0_fwcdm[2],1))
    stringH0_fwcdm = "$"+H0_fwcdm+"^{+"+H0errup_fwcdm+"}_{-"+H0errdw_fwcdm+"}$"

    Om_fwcdm = str(np.around(sOm_fwcdm[0],3))
    Omerrup_fwcdm = str(np.around(sOm_fwcdm[1],3))
    Omerrdw_fwcdm = str(np.around(sOm_fwcdm[2],3))
    stringOm_fwcdm = "$"+Om_fwcdm+"^{+"+Omerrup_fwcdm+"}_{-"+Omerrdw_fwcdm+"}$"

    w_fwcdm = str(np.around(sw_fwcdm[0],2))
    werrup_fwcdm = str(np.around(sw_fwcdm[1],2))
    werrdw_fwcdm = str(np.around(sw_fwcdm[2],2))
    stringw_fwcdm = "$"+w_fwcdm+"^{+"+werrup_fwcdm+"}_{-"+werrdw_fwcdm+"}$"

    m=column(fwcdm,3)
    M=[]
    for i in range(len(m)):
        M.append(-1*m[i])
    M_fwcdm=str(np.around((np.percentile(M,50)),1))
    Merrup_fwcdm = str(np.around((np.percentile(M,84.15)-np.percentile(M,50)),1))
    Merrdw_fwcdm = str(np.around((np.percentile(M,50)-np.percentile(M,15.85)),1))
    stringM_fwcdm = "$"+M_fwcdm+"^{+"+Merrup_fwcdm+"}_{-"+Merrdw_fwcdm+"}$"

    a_fwcdm = str(np.around(sa_fwcdm[0],3))
    aerrup_fwcdm = str(np.around(sa_fwcdm[1],3))
    aerrdw_fwcdm = str(np.around(sa_fwcdm[2],3))
    stringa_fwcdm = "$"+a_fwcdm+"^{+"+aerrup_fwcdm+"}_{-"+aerrdw_fwcdm+"}$"

    b_fwcdm = str(np.around(sb_fwcdm[0],2))
    berrup_fwcdm = str(np.around(sb_fwcdm[1],2))
    berrdw_fwcdm = str(np.around(sb_fwcdm[2],2))
    stringb_fwcdm = "$"+b_fwcdm+"^{+"+berrup_fwcdm+"}_{-"+berrdw_fwcdm+"}$"

    int_fwcdm = str(np.around(sint_fwcdm[0],3))
    interrup_fwcdm = str(np.around(sint_fwcdm[1],3))
    interrdw_fwcdm = str(np.around(sint_fwcdm[2],3))
    stringint_fwcdm = "$"+int_fwcdm+"^{+"+interrup_fwcdm+"}_{-"+interrdw_fwcdm+"}$"

    rd_fwcdm = str(np.around(srd_fwcdm[0],1))
    rderrup_fwcdm = str(np.around(srd_fwcdm[1],1))
    rderrdw_fwcdm = str(np.around(srd_fwcdm[2],1))
    stringrd_fwcdm = "$"+rd_fwcdm+"^{+"+rderrup_fwcdm+"}_{-"+rderrdw_fwcdm+"}$"


    sH0_wcdm, sOm_wcdm, sOl_wcdm, sw_wcdm, sM_wcdm, sa_wcdm, sb_wcdm, sint_wcdm, srd_wcdm = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                            zip(*np.percentile(wcdm, perc, axis=0)))

    H0_wcdm = str(np.around(sH0_wcdm[0],1))
    H0errup_wcdm = str(np.around(sH0_wcdm[1],1))
    H0errdw_wcdm = str(np.around(sH0_wcdm[2],1))
    stringH0_wcdm = "$"+H0_wcdm+"^{+"+H0errup_wcdm+"}_{-"+H0errdw_wcdm+"}$"

    Om_wcdm = str(np.around(sOm_wcdm[0],3))
    Omerrup_wcdm = str(np.around(sOm_wcdm[1],3))
    Omerrdw_wcdm = str(np.around(sOm_wcdm[2],3))
    stringOm_wcdm = "$"+Om_wcdm+"^{+"+Omerrup_wcdm+"}_{-"+Omerrdw_wcdm+"}$"

    Ol_wcdm = str(np.around(sOl_wcdm[0],3))
    Olerrup_wcdm = str(np.around(sOl_wcdm[1],3))
    Olerrdw_wcdm = str(np.around(sOl_wcdm[2],3))
    stringsOl_wcdm = "$"+Ol_wcdm+"^{+"+Olerrup_wcdm+"}_{-"+Olerrdw_wcdm+"}$"

    Om=np.array(column(wcdm,1))
    OL=np.array(column(wcdm,2))
    Ok=[]
    for i in range(len(Om)):
        Ok.append(1-Om[i]-OL[i])
    Ok_wcdm=str(np.around((np.percentile(Ok,50)),3))
    Okerrup_wcdm = str(np.around((np.percentile(Ok,84.15)-np.percentile(Ok,50)),3))
    Okerrdw_wcdm = str(np.around((np.percentile(Ok,50)-np.percentile(Ok,15.85)),3))
    stringOk_wcdm = "$"+Ok_wcdm+"^{+"+Okerrup_wcdm+"}_{-"+Okerrdw_wcdm+"}$"

    w_wcdm = str(np.around(sw_wcdm[0],2))
    werrup_wcdm = str(np.around(sw_wcdm[1],2))
    werrdw_wcdm = str(np.around(sw_wcdm[2],2))
    stringw_wcdm = "$"+w_wcdm+"^{+"+werrup_wcdm+"}_{-"+werrdw_wcdm+"}$"

    m=column(wcdm,4)
    M=[]
    for i in range(len(m)):
        M.append(-1*m[i])
    M_wcdm=str(np.around((np.percentile(M,50)),1))
    Merrup_wcdm = str(np.around((np.percentile(M,84.15)-np.percentile(M,50)),1))
    Merrdw_wcdm = str(np.around((np.percentile(M,50)-np.percentile(M,15.85)),1))
    stringM_wcdm = "$"+M_wcdm+"^{+"+Merrup_wcdm+"}_{-"+Merrdw_wcdm+"}$"

    a_wcdm = str(np.around(sa_wcdm[0],3))
    aerrup_wcdm = str(np.around(sa_wcdm[1],3))
    aerrdw_wcdm = str(np.around(sa_wcdm[2],3))
    stringa_wcdm = "$"+a_wcdm+"^{+"+aerrup_wcdm+"}_{-"+aerrdw_wcdm+"}$"

    b_wcdm = str(np.around(sb_wcdm[0],2))
    berrup_wcdm = str(np.around(sb_wcdm[1],2))
    berrdw_wcdm = str(np.around(sb_wcdm[2],2))
    stringb_wcdm = "$"+b_wcdm+"^{+"+berrup_wcdm+"}_{-"+berrdw_wcdm+"}$"

    int_wcdm = str(np.around(sint_wcdm[0],3))
    interrup_wcdm = str(np.around(sint_wcdm[1],3))
    interrdw_wcdm = str(np.around(sint_wcdm[2],3))
    stringint_wcdm = "$"+int_wcdm+"^{+"+interrup_wcdm+"}_{-"+interrdw_wcdm+"}$"

    rd_wcdm = str(np.around(srd_wcdm[0],1))
    rderrup_wcdm = str(np.around(srd_wcdm[1],1))
    rderrdw_wcdm = str(np.around(srd_wcdm[2],1))
    stringrd_wcdm = "$"+rd_wcdm+"^{+"+rderrup_wcdm+"}_{-"+rderrdw_wcdm+"}$"

    sH0_fw0wacdm, sOm_fw0wacdm, sw0_fw0wacdm, wsa_fw0wacdm, sM_fw0wacdm, sa_fw0wacdm, sb_fw0wacdm, sint_fw0wacdm, srd_fw0wacdm = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                            zip(*np.percentile(fw0wacdm, perc, axis=0)))
    
    H0_fw0wacdm = str(np.around(sH0_fw0wacdm[0],1))
    H0errup_fw0wacdm = str(np.around(sH0_fw0wacdm[1],1))
    H0errdwsa_fw0wacdm = str(np.around(sH0_fw0wacdm[2],1))
    stringH0_fw0wacdm = "$"+H0_fw0wacdm+"^{+"+H0errup_fw0wacdm+"}_{-"+H0errdwsa_fw0wacdm+"}$"

    Om_fw0wacdm = str(np.around(sOm_fw0wacdm[0],3))
    Omerrup_fw0wacdm = str(np.around(sOm_fw0wacdm[1],3))
    Omerrdwsa_fw0wacdm = str(np.around(sOm_fw0wacdm[2],3))
    stringOm_fw0wacdm = "$"+Om_fw0wacdm+"^{+"+Omerrup_fw0wacdm+"}_{-"+Omerrdwsa_fw0wacdm+"}$"

    w0_fw0wacdm = str(np.around(sw0_fw0wacdm[0],2))
    w0errup_fw0wacdm = str(np.around(sw0_fw0wacdm[1],2))
    w0errdw_fw0wacdm = str(np.around(sw0_fw0wacdm[2],2))
    stringw0_fw0wacdm = "$"+w0_fw0wacdm+"^{+"+w0errup_fw0wacdm+"}_{-"+w0errdw_fw0wacdm+"}$"

    wa_fw0wacdm = str(np.around(wsa_fw0wacdm[0],2))
    waerrup_fw0wacdm = str(np.around(wsa_fw0wacdm[1],2))
    waerrdw_fw0wacdm = str(np.around(wsa_fw0wacdm[2],2))
    stringwa_fw0wacdm = "$"+wa_fw0wacdm+"^{+"+waerrup_fw0wacdm+"}_{-"+waerrdw_fw0wacdm+"}$"

    m=column(fw0wacdm,4)
    M=[]
    for i in range(len(m)):
        M.append(-1*m[i])
    M_fw0wacdm=str(np.around((np.percentile(M,50)),1))
    Merrup_fw0wacdm = str(np.around((np.percentile(M,84.15)-np.percentile(M,50)),1))
    Merrdw_fw0wacdm = str(np.around((np.percentile(M,50)-np.percentile(M,15.85)),1))
    stringM_fw0wacdm = "$"+M_fw0wacdm+"^{+"+Merrup_fw0wacdm+"}_{-"+Merrdw_fw0wacdm+"}$"

    a_fw0wacdm = str(np.around(sa_fw0wacdm[0],3))
    aerrup_fw0wacdm = str(np.around(sa_fw0wacdm[1],3))
    aerrdwsa_fw0wacdm = str(np.around(sa_fw0wacdm[2],3))
    stringa_fw0wacdm = "$"+a_fw0wacdm+"^{+"+aerrup_fw0wacdm+"}_{-"+aerrdwsa_fw0wacdm+"}$"

    b_fw0wacdm = str(np.around(sb_fw0wacdm[0],2))
    berrup_fw0wacdm = str(np.around(sb_fw0wacdm[1],2))
    berrdwsa_fw0wacdm = str(np.around(sb_fw0wacdm[2],2))
    stringb_fw0wacdm = "$"+b_fw0wacdm+"^{+"+berrup_fw0wacdm+"}_{-"+berrdwsa_fw0wacdm+"}$"

    int_fw0wacdm = str(np.around(sint_fw0wacdm[0],3))
    interrup_fw0wacdm = str(np.around(sint_fw0wacdm[1],3))
    interrdwsa_fw0wacdm = str(np.around(sint_fw0wacdm[2],3))
    stringint_fw0wacdm = "$"+int_fw0wacdm+"^{+"+interrup_fw0wacdm+"}_{-"+interrdwsa_fw0wacdm+"}$"

    rd_fw0wacdm = str(np.around(srd_fw0wacdm[0],1))
    rderrup_fw0wacdm = str(np.around(srd_fw0wacdm[1],1))
    rderrdwsa_fw0wacdm = str(np.around(srd_fw0wacdm[2],1))
    stringrd_fw0wacdm = "$"+rd_fw0wacdm+"^{+"+rderrup_fw0wacdm+"}_{-"+rderrdwsa_fw0wacdm+"}$"
    
    sH0_w0wacdm, sOm_w0wacdm, sOl_w0wacdm, sw0_w0wacdm, wsa_w0wacdm, sM_w0wacdm, sa_w0wacdm, sb_w0wacdm, sint_w0wacdm, srd_w0wacdm = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                            zip(*np.percentile(w0wacdm, perc, axis=0)))
    
    H0_w0wacdm = str(np.around(sH0_w0wacdm[0],1))
    H0errup_w0wacdm = str(np.around(sH0_w0wacdm[1],1))
    H0errdwsa_w0wacdm = str(np.around(sH0_w0wacdm[2],1))
    stringH0_w0wacdm = "$"+H0_w0wacdm+"^{+"+H0errup_w0wacdm+"}_{-"+H0errdwsa_w0wacdm+"}$"

    Om_w0wacdm = str(np.around(sOm_w0wacdm[0],3))
    Omerrup_w0wacdm = str(np.around(sOm_w0wacdm[1],3))
    Omerrdwsa_w0wacdm = str(np.around(sOm_w0wacdm[2],3))
    stringOm_w0wacdm = "$"+Om_w0wacdm+"^{+"+Omerrup_w0wacdm+"}_{-"+Omerrdwsa_w0wacdm+"}$"

    Ol_w0wacdm = str(np.around(sOl_w0wacdm[0],3))
    Olerrup_w0wacdm = str(np.around(sOl_w0wacdm[1],3))
    Olerrdwsa_w0wacdm = str(np.around(sOl_w0wacdm[2],3))
    stringsOl_w0wacdm = "$"+Ol_w0wacdm+"^{+"+Olerrup_w0wacdm+"}_{-"+Olerrdwsa_w0wacdm+"}$"
    
    Om=np.array(column(w0wacdm,1))
    OL=np.array(column(w0wacdm,2))
    Ok=[]
    for i in range(len(Om)):
        Ok.append(1-Om[i]-OL[i])
    Ok_w0wacdm=str(np.around((np.percentile(Ok,50)),3))
    Okerrup_w0wacdm = str(np.around((np.percentile(Ok,84.15)-np.percentile(Ok,50)),3))
    Okerrdw_w0wacdm = str(np.around((np.percentile(Ok,50)-np.percentile(Ok,15.85)),3))
    stringOk_w0wacdm = "$"+Ok_w0wacdm+"^{+"+Okerrup_w0wacdm+"}_{-"+Okerrdw_w0wacdm+"}$"

    w0_w0wacdm = str(np.around(sw0_w0wacdm[0],2))
    w0errup_w0wacdm = str(np.around(sw0_w0wacdm[1],2))
    w0errdwsa_w0wacdm = str(np.around(sw0_w0wacdm[2],2))
    stringw0_w0wacdm = "$"+w0_w0wacdm+"^{+"+w0errup_w0wacdm+"}_{-"+w0errdwsa_w0wacdm+"}$"

    wa_w0wacdm = str(np.around(wsa_w0wacdm[0],2))
    waerrup_w0wacdm = str(np.around(wsa_w0wacdm[1],2))
    waerrdw_w0wacdm = str(np.around(wsa_w0wacdm[2],2))
    stringwa_w0wacdm = "$"+wa_w0wacdm+"^{+"+waerrup_w0wacdm+"}_{-"+waerrdw_w0wacdm+"}$"

    m=column(w0wacdm,5)
    M=[]
    for i in range(len(m)):
        M.append(-1*m[i])
    M_w0wacdm=str(np.around((np.percentile(M,50)),1))
    Merrup_w0wacdm = str(np.around((np.percentile(M,84.15)-np.percentile(M,50)),1))
    Merrdw_w0wacdm = str(np.around((np.percentile(M,50)-np.percentile(M,15.85)),1))
    stringM_w0wacdm = "$"+M_w0wacdm+"^{+"+Merrup_w0wacdm+"}_{-"+Merrdw_w0wacdm+"}$"

    a_w0wacdm = str(np.around(sa_w0wacdm[0],3))
    aerrup_w0wacdm = str(np.around(sa_w0wacdm[1],3))
    aerrdwsa_w0wacdm = str(np.around(sa_w0wacdm[2],3))
    stringa_w0wacdm = "$"+a_w0wacdm+"^{+"+aerrup_w0wacdm+"}_{-"+aerrdwsa_w0wacdm+"}$"

    b_w0wacdm = str(np.around(sb_w0wacdm[0],2))
    berrup_w0wacdm = str(np.around(sb_w0wacdm[1],2))
    berrdwsa_w0wacdm = str(np.around(sb_w0wacdm[2],2))
    stringb_w0wacdm = "$"+b_w0wacdm+"^{+"+berrup_w0wacdm+"}_{-"+berrdwsa_w0wacdm+"}$"

    int_w0wacdm = str(np.around(sint_w0wacdm[0],3))
    interrup_w0wacdm = str(np.around(sint_w0wacdm[1],3))
    interrdwsa_w0wacdm = str(np.around(sint_w0wacdm[2],3))
    stringint_w0wacdm = "$"+int_w0wacdm+"^{+"+interrup_w0wacdm+"}_{-"+interrdwsa_w0wacdm+"}$"

    rd_w0wacdm = str(np.around(srd_w0wacdm[0],1))
    rderrup_w0wacdm = str(np.around(srd_w0wacdm[1],1))
    rderrdwsa_w0wacdm = str(np.around(srd_w0wacdm[2],1))
    stringrd_w0wacdm = "$"+rd_w0wacdm+"^{+"+rderrup_w0wacdm+"}_{-"+rderrdwsa_w0wacdm+"}$"


    os.chdir(dir_out)
    name_output=r'Table_{0}_Paper.dat'.format(probe)
    fileout=open(name_output,"w")
    fileout.write("\\begin{table}[H] \n")
    fileout.write("\\renewcommand{\\arraystretch}{1.4} \n")
    fileout.write("     \\begin{center}\n")
    fileout.write("     \\begin{tabular}{ccccccc}\n")
    fileout.write("     \hline\n")
    fileout.write("     & Flat $\Lambda$CDM & $\Lambda$CDM & Flat $w$CDM & $w$CDM & Flat $w_0w_a$CDM & $w_0w_a$CDM  \\\ \n")
    fileout.write("     \hline\n")
    fileout.write("     \hline\n")
    fileout.write("     $H_0$ & "+stringH0_flcdm+" & "+stringH0_lcdm+" & "+stringH0_fwcdm+" & "+stringH0_wcdm+" & "+stringH0_fw0wacdm+" & "+stringH0_w0wacdm+" \\\ [0.3ex] \n")
    fileout.write("     \hline\n")
    fileout.write("     $\Omega_m$ & "+stringOm_flcdm+" & "+stringOm_lcdm+" & "+stringOm_fwcdm+" & "+stringOm_wcdm+" & "+stringOm_fw0wacdm+" & "+stringOm_w0wacdm+" \\\ [0.3ex] \n")
    fileout.write("     \hline\n")
    fileout.write("     $\Omega_{\Lambda}$ & -- & "+stringsOl_lcdm+" & -- & "+stringsOl_wcdm+" & -- & "+stringsOl_w0wacdm+" \\\ [0.3ex] \n")
    fileout.write("     \hline\n")
    fileout.write("     $\Omega_k$ & -- & "+stringOk_lcdm+" & -- & "+stringOk_wcdm+" & -- & "+stringOk_w0wacdm+" \\\ [0.3ex] \n")
    fileout.write("     \hline\n")
    fileout.write("     $w_0$ & -- & -- & "+stringw_fwcdm+" & "+stringw_wcdm+" & "+stringw0_fw0wacdm+" & "+stringw0_w0wacdm+" \\\ [0.3ex] \n")
    fileout.write("     \hline\n")
    fileout.write("     $w_a$ & -- & -- & -- & -- & "+stringwa_fw0wacdm+" & "+stringwa_w0wacdm+" \\\ [0.3ex] \n")
    fileout.write("     \hline\n")
    fileout.write("     M & "+stringM_flcdm+" & "+stringM_lcdm+" & "+stringM_fwcdm+" & "+stringM_wcdm+" & "+stringM_fw0wacdm+" & "+stringM_w0wacdm+" \\\ [0.3ex] \n")
    fileout.write("     \hline\n")
    fileout.write("     a & "+stringa_flcdm+" & "+stringa_lcdm+" & "+stringa_fwcdm+" & "+stringa_wcdm+" & "+stringa_fw0wacdm+" & "+stringa_w0wacdm+" \\\ [0.3ex] \n")
    fileout.write("     \hline\n")
    fileout.write("     b & "+stringb_flcdm+" & "+stringb_lcdm+" & "+stringb_fwcdm+" & "+stringb_wcdm+" & "+stringb_fw0wacdm+" & "+stringb_w0wacdm+" \\\ [0.3ex] \n")
    fileout.write("     \hline\n")
    fileout.write("     $\sigma_{int}$ & "+stringint_flcdm+" & "+stringint_lcdm+" & "+stringint_fwcdm+" & "+stringint_wcdm+" & "+stringint_fw0wacdm+" & "+stringint_w0wacdm+" \\\ [0.3ex] \n")
    fileout.write("     \hline\n")
    fileout.write("     $r_d$ & "+stringrd_flcdm+" & "+stringrd_lcdm+" & "+stringrd_fwcdm+" & "+stringrd_wcdm+" & "+stringrd_fw0wacdm+" & "+stringrd_w0wacdm+" \\\ [0.3ex] \n")
    fileout.write("     \hline\n")
    fileout.write("     \hline\n")
    fileout.write("     \end{tabular}\n")
    fileout.write("     \end{center}\n")
    fileout.write("\end{table}\n")     
    
    return fileout.close()

def Table_MCMC(prior, cosmo, cosmolatex, dir_chain, dir_out):
    burnin=4000
    thin=5
    os.chdir(dir_chain) 
    ### 4 ####
    os.chdir(dir_chain+'BAOCCSNGRB')
    filename = 'Chain_{0}Prior_{1}_BAOCCSNGRB.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    BAO_CC_SN_GRB = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(BAO_CC_SN_GRB,0))
        Om=np.array(column(BAO_CC_SN_GRB,1))
        OL=np.array(column(BAO_CC_SN_GRB,2))
        M=np.array(column(BAO_CC_SN_GRB,3))
        a=np.array(column(BAO_CC_SN_GRB,4))
        b=np.array(column(BAO_CC_SN_GRB,5))
        scat=np.array(column(BAO_CC_SN_GRB,6))
        rd=np.array(column(BAO_CC_SN_GRB,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_CC_SN_GRB = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(BAO_CC_SN_GRB,0))
        Om=np.array(column(BAO_CC_SN_GRB,1))
        OL=np.array(column(BAO_CC_SN_GRB,2))
        w=np.array(column(BAO_CC_SN_GRB,3))
        M=np.array(column(BAO_CC_SN_GRB,4))
        a=np.array(column(BAO_CC_SN_GRB,5))
        b=np.array(column(BAO_CC_SN_GRB,6))
        scat=np.array(column(BAO_CC_SN_GRB,7))
        rd=np.array(column(BAO_CC_SN_GRB,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_CC_SN_GRB = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(BAO_CC_SN_GRB,0))
        Om=np.array(column(BAO_CC_SN_GRB,1))
        OL=np.array(column(BAO_CC_SN_GRB,2))
        w0=np.array(column(BAO_CC_SN_GRB,3))
        wa=np.array(column(BAO_CC_SN_GRB,4))
        M=np.array(column(BAO_CC_SN_GRB,5))
        a=np.array(column(BAO_CC_SN_GRB,6))
        b=np.array(column(BAO_CC_SN_GRB,7))
        scat=np.array(column(BAO_CC_SN_GRB,8))
        rd=np.array(column(BAO_CC_SN_GRB,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_CC_SN_GRB = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd]))
    ### 3 ###
    os.chdir(dir_chain+'BAOCCGRB')
    filename = 'Chain_{0}Prior_{1}_BAOCCGRB.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    BAO_CC_GRB = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(BAO_CC_GRB,0))
        Om=np.array(column(BAO_CC_GRB,1))
        OL=np.array(column(BAO_CC_GRB,2))
        M=np.array(column(BAO_CC_GRB,3))
        a=np.array(column(BAO_CC_GRB,4))
        b=np.array(column(BAO_CC_GRB,5))
        scat=np.array(column(BAO_CC_GRB,6))
        rd=np.array(column(BAO_CC_GRB,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_CC_GRB = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(BAO_CC_GRB,0))
        Om=np.array(column(BAO_CC_GRB,1))
        OL=np.array(column(BAO_CC_GRB,2))
        w=np.array(column(BAO_CC_GRB,3))
        M=np.array(column(BAO_CC_GRB,4))
        a=np.array(column(BAO_CC_GRB,5))
        b=np.array(column(BAO_CC_GRB,6))
        scat=np.array(column(BAO_CC_GRB,7))
        rd=np.array(column(BAO_CC_GRB,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_CC_GRB = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(BAO_CC_GRB,0))
        Om=np.array(column(BAO_CC_GRB,1))
        OL=np.array(column(BAO_CC_GRB,2))
        w0=np.array(column(BAO_CC_GRB,3))
        wa=np.array(column(BAO_CC_GRB,4))
        M=np.array(column(BAO_CC_GRB,5))
        a=np.array(column(BAO_CC_GRB,6))
        b=np.array(column(BAO_CC_GRB,7))
        scat=np.array(column(BAO_CC_GRB,8))
        rd=np.array(column(BAO_CC_GRB,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_CC_GRB = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd]))
    os.chdir(dir_chain+'BAOSNGRB')
    filename = 'Chain_{0}Prior_{1}_BAOSNGRB.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    BAO_SN_GRB = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(BAO_SN_GRB,0))
        Om=np.array(column(BAO_SN_GRB,1))
        OL=np.array(column(BAO_SN_GRB,2))
        M=np.array(column(BAO_SN_GRB,3))
        a=np.array(column(BAO_SN_GRB,4))
        b=np.array(column(BAO_SN_GRB,5))
        scat=np.array(column(BAO_SN_GRB,6))
        rd=np.array(column(BAO_SN_GRB,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_SN_GRB = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(BAO_SN_GRB,0))
        Om=np.array(column(BAO_SN_GRB,1))
        OL=np.array(column(BAO_SN_GRB,2))
        w=np.array(column(BAO_SN_GRB,3))
        M=np.array(column(BAO_SN_GRB,4))
        a=np.array(column(BAO_SN_GRB,5))
        b=np.array(column(BAO_SN_GRB,6))
        scat=np.array(column(BAO_SN_GRB,7))
        rd=np.array(column(BAO_SN_GRB,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_SN_GRB = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(BAO_SN_GRB,0))
        Om=np.array(column(BAO_SN_GRB,1))
        OL=np.array(column(BAO_SN_GRB,2))
        w0=np.array(column(BAO_SN_GRB,3))
        wa=np.array(column(BAO_SN_GRB,4))
        M=np.array(column(BAO_SN_GRB,5))
        a=np.array(column(BAO_SN_GRB,6))
        b=np.array(column(BAO_SN_GRB,7))
        scat=np.array(column(BAO_SN_GRB,8))
        rd=np.array(column(BAO_SN_GRB,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_SN_GRB = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd]))
    os.chdir(dir_chain+'CCSNGRB')
    filename = 'Chain_{0}Prior_{1}_CCSNGRB.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    CC_SN_GRB = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(CC_SN_GRB,0))
        Om=np.array(column(CC_SN_GRB,1))
        OL=np.array(column(CC_SN_GRB,2))
        M=np.array(column(CC_SN_GRB,3))
        a=np.array(column(CC_SN_GRB,4))
        b=np.array(column(CC_SN_GRB,5))
        scat=np.array(column(CC_SN_GRB,6))
        rd=np.array(column(CC_SN_GRB,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        CC_SN_GRB = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(CC_SN_GRB,0))
        Om=np.array(column(CC_SN_GRB,1))
        OL=np.array(column(CC_SN_GRB,2))
        w=np.array(column(CC_SN_GRB,3))
        M=np.array(column(CC_SN_GRB,4))
        a=np.array(column(CC_SN_GRB,5))
        b=np.array(column(CC_SN_GRB,6))
        scat=np.array(column(CC_SN_GRB,7))
        rd=np.array(column(CC_SN_GRB,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        CC_SN_GRB = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(CC_SN_GRB,0))
        Om=np.array(column(CC_SN_GRB,1))
        OL=np.array(column(CC_SN_GRB,2))
        w0=np.array(column(CC_SN_GRB,3))
        wa=np.array(column(CC_SN_GRB,4))
        M=np.array(column(CC_SN_GRB,5))
        a=np.array(column(CC_SN_GRB,6))
        b=np.array(column(CC_SN_GRB,7))
        scat=np.array(column(CC_SN_GRB,8))
        rd=np.array(column(CC_SN_GRB,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        CC_SN_GRB = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd])) 
    os.chdir(dir_chain+'BAOCCSN')
    filename = 'Chain_{0}Prior_{1}_BAOCCSN.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    BAO_CC_SN = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(BAO_CC_SN,0))
        Om=np.array(column(BAO_CC_SN,1))
        OL=np.array(column(BAO_CC_SN,2))
        M=np.array(column(BAO_CC_SN,3))
        a=np.array(column(BAO_CC_SN,4))
        b=np.array(column(BAO_CC_SN,5))
        scat=np.array(column(BAO_CC_SN,6))
        rd=np.array(column(BAO_CC_SN,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_CC_SN = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(BAO_CC_SN,0))
        Om=np.array(column(BAO_CC_SN,1))
        OL=np.array(column(BAO_CC_SN,2))
        w=np.array(column(BAO_CC_SN,3))
        M=np.array(column(BAO_CC_SN,4))
        a=np.array(column(BAO_CC_SN,5))
        b=np.array(column(BAO_CC_SN,6))
        scat=np.array(column(BAO_CC_SN,7))
        rd=np.array(column(BAO_CC_SN,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_CC_SN = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(BAO_CC_SN,0))
        Om=np.array(column(BAO_CC_SN,1))
        OL=np.array(column(BAO_CC_SN,2))
        w0=np.array(column(BAO_CC_SN,3))
        wa=np.array(column(BAO_CC_SN,4))
        M=np.array(column(BAO_CC_SN,5))
        a=np.array(column(BAO_CC_SN,6))
        b=np.array(column(BAO_CC_SN,7))
        scat=np.array(column(BAO_CC_SN,8))
        rd=np.array(column(BAO_CC_SN,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_CC_SN = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd]))
    ### 2 ###
    os.chdir(dir_chain+'BAOCC')
    filename = 'Chain_{0}Prior_{1}_BAOCC.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    BAO_CC = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)   
    if cosmo=='LCDM':
        H0=np.array(column(BAO_CC,0))
        Om=np.array(column(BAO_CC,1))
        OL=np.array(column(BAO_CC,2))
        M=np.array(column(BAO_CC,3))
        a=np.array(column(BAO_CC,4))
        b=np.array(column(BAO_CC,5))
        scat=np.array(column(BAO_CC,6))
        rd=np.array(column(BAO_CC,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_CC = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(BAO_CC,0))
        Om=np.array(column(BAO_CC,1))
        OL=np.array(column(BAO_CC,2))
        w=np.array(column(BAO_CC,3))
        M=np.array(column(BAO_CC,4))
        a=np.array(column(BAO_CC,5))
        b=np.array(column(BAO_CC,6))
        scat=np.array(column(BAO_CC,7))
        rd=np.array(column(BAO_CC,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_CC = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(BAO_CC,0))
        Om=np.array(column(BAO_CC,1))
        OL=np.array(column(BAO_CC,2))
        w0=np.array(column(BAO_CC,3))
        wa=np.array(column(BAO_CC,4))
        M=np.array(column(BAO_CC,5))
        a=np.array(column(BAO_CC,6))
        b=np.array(column(BAO_CC,7))
        scat=np.array(column(BAO_CC,8))
        rd=np.array(column(BAO_CC,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_CC = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd]))
    os.chdir(dir_chain+'BAOSN') 
    filename = 'Chain_{0}Prior_{1}_BAOSN.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    BAO_SN = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(BAO_SN,0))
        Om=np.array(column(BAO_SN,1))
        OL=np.array(column(BAO_SN,2))
        M=np.array(column(BAO_SN,3))
        a=np.array(column(BAO_SN,4))
        b=np.array(column(BAO_SN,5))
        scat=np.array(column(BAO_SN,6))
        rd=np.array(column(BAO_SN,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_SN = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(BAO_SN,0))
        Om=np.array(column(BAO_SN,1))
        OL=np.array(column(BAO_SN,2))
        w=np.array(column(BAO_SN,3))
        M=np.array(column(BAO_SN,4))
        a=np.array(column(BAO_SN,5))
        b=np.array(column(BAO_SN,6))
        scat=np.array(column(BAO_SN,7))
        rd=np.array(column(BAO_SN,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_SN = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(BAO_SN,0))
        Om=np.array(column(BAO_SN,1))
        OL=np.array(column(BAO_SN,2))
        w0=np.array(column(BAO_SN,3))
        wa=np.array(column(BAO_SN,4))
        M=np.array(column(BAO_SN,5))
        a=np.array(column(BAO_SN,6))
        b=np.array(column(BAO_SN,7))
        scat=np.array(column(BAO_SN,8))
        rd=np.array(column(BAO_SN,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_SN = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd]))
    os.chdir(dir_chain+'CCSN')
    filename = 'Chain_{0}Prior_{1}_CCSN.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    CC_SN = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(CC_SN,0))
        Om=np.array(column(CC_SN,1))
        OL=np.array(column(CC_SN,2))
        M=np.array(column(CC_SN,3))
        a=np.array(column(CC_SN,4))
        b=np.array(column(CC_SN,5))
        scat=np.array(column(CC_SN,6))
        rd=np.array(column(CC_SN,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        CC_SN = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(CC_SN,0))
        Om=np.array(column(CC_SN,1))
        OL=np.array(column(CC_SN,2))
        w=np.array(column(CC_SN,3))
        M=np.array(column(CC_SN,4))
        a=np.array(column(CC_SN,5))
        b=np.array(column(CC_SN,6))
        scat=np.array(column(CC_SN,7))
        rd=np.array(column(CC_SN,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        CC_SN = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(CC_SN,0))
        Om=np.array(column(CC_SN,1))
        OL=np.array(column(CC_SN,2))
        w0=np.array(column(CC_SN,3))
        wa=np.array(column(CC_SN,4))
        M=np.array(column(CC_SN,5))
        a=np.array(column(CC_SN,6))
        b=np.array(column(CC_SN,7))
        scat=np.array(column(CC_SN,8))
        rd=np.array(column(CC_SN,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        CC_SN = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd]))
    os.chdir(dir_chain+'BAOGRB')
    filename = 'Chain_{0}Prior_{1}_BAOGRB.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    BAO_GRB = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(BAO_GRB,0))
        Om=np.array(column(BAO_GRB,1))
        OL=np.array(column(BAO_GRB,2))
        M=np.array(column(BAO_GRB,3))
        a=np.array(column(BAO_GRB,4))
        b=np.array(column(BAO_GRB,5))
        scat=np.array(column(BAO_GRB,6))
        rd=np.array(column(BAO_GRB,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_GRB = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(BAO_GRB,0))
        Om=np.array(column(BAO_GRB,1))
        OL=np.array(column(BAO_GRB,2))
        w=np.array(column(BAO_GRB,3))
        M=np.array(column(BAO_GRB,4))
        a=np.array(column(BAO_GRB,5))
        b=np.array(column(BAO_GRB,6))
        scat=np.array(column(BAO_GRB,7))
        rd=np.array(column(BAO_GRB,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_GRB = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(BAO_GRB,0))
        Om=np.array(column(BAO_GRB,1))
        OL=np.array(column(BAO_GRB,2))
        w0=np.array(column(BAO_GRB,3))
        wa=np.array(column(BAO_GRB,4))
        M=np.array(column(BAO_GRB,5))
        a=np.array(column(BAO_GRB,6))
        b=np.array(column(BAO_GRB,7))
        scat=np.array(column(BAO_GRB,8))
        rd=np.array(column(BAO_GRB,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO_GRB = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd]))
    os.chdir(dir_chain+'CCGRB')
    filename = 'Chain_{0}Prior_{1}_CCGRB.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    CC_GRB = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(CC_GRB,0))
        Om=np.array(column(CC_GRB,1))
        OL=np.array(column(CC_GRB,2))
        M=np.array(column(CC_GRB,3))
        a=np.array(column(CC_GRB,4))
        b=np.array(column(CC_GRB,5))
        scat=np.array(column(CC_GRB,6))
        rd=np.array(column(CC_GRB,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        CC_GRB = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(CC_GRB,0))
        Om=np.array(column(CC_GRB,1))
        OL=np.array(column(CC_GRB,2))
        w=np.array(column(CC_GRB,3))
        M=np.array(column(CC_GRB,4))
        a=np.array(column(CC_GRB,5))
        b=np.array(column(CC_GRB,6))
        scat=np.array(column(CC_GRB,7))
        rd=np.array(column(CC_GRB,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        CC_GRB = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(CC_GRB,0))
        Om=np.array(column(CC_GRB,1))
        OL=np.array(column(CC_GRB,2))
        w0=np.array(column(CC_GRB,3))
        wa=np.array(column(CC_GRB,4))
        M=np.array(column(CC_GRB,5))
        a=np.array(column(CC_GRB,6))
        b=np.array(column(CC_GRB,7))
        scat=np.array(column(CC_GRB,8))
        rd=np.array(column(CC_GRB,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        CC_GRB = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd]))
    os.chdir(dir_chain+'SNGRB')  
    filename = 'Chain_{0}Prior_{1}_SNGRB.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())  
    SN_GRB = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(SN_GRB,0))
        Om=np.array(column(SN_GRB,1))
        OL=np.array(column(SN_GRB,2))
        M=np.array(column(SN_GRB,3))
        a=np.array(column(SN_GRB,4))
        b=np.array(column(SN_GRB,5))
        scat=np.array(column(SN_GRB,6))
        rd=np.array(column(SN_GRB,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        SN_GRB = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(SN_GRB,0))
        Om=np.array(column(SN_GRB,1))
        OL=np.array(column(SN_GRB,2))
        w=np.array(column(SN_GRB,3))
        M=np.array(column(SN_GRB,4))
        a=np.array(column(SN_GRB,5))
        b=np.array(column(SN_GRB,6))
        scat=np.array(column(SN_GRB,7))
        rd=np.array(column(SN_GRB,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        SN_GRB = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(SN_GRB,0))
        Om=np.array(column(SN_GRB,1))
        OL=np.array(column(SN_GRB,2))
        w0=np.array(column(SN_GRB,3))
        wa=np.array(column(SN_GRB,4))
        M=np.array(column(SN_GRB,5))
        a=np.array(column(SN_GRB,6))
        b=np.array(column(SN_GRB,7))
        scat=np.array(column(SN_GRB,8))
        rd=np.array(column(SN_GRB,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        SN_GRB = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd]))
    ###1###
    os.chdir(dir_chain+'BAO')
    filename = 'Chain_{0}Prior_{1}_BAO.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    BAO = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(BAO,0))
        Om=np.array(column(BAO,1))
        OL=np.array(column(BAO,2))
        M=np.array(column(BAO,3))
        a=np.array(column(BAO,4))
        b=np.array(column(BAO,5))
        scat=np.array(column(BAO,6))
        rd=np.array(column(BAO,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(BAO,0))
        Om=np.array(column(BAO,1))
        OL=np.array(column(BAO,2))
        w=np.array(column(BAO,3))
        M=np.array(column(BAO,4))
        a=np.array(column(BAO,5))
        b=np.array(column(BAO,6))
        scat=np.array(column(BAO,7))
        rd=np.array(column(BAO,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(BAO,0))
        Om=np.array(column(BAO,1))
        OL=np.array(column(BAO,2))
        w0=np.array(column(BAO,3))
        wa=np.array(column(BAO,4))
        M=np.array(column(BAO,5))
        a=np.array(column(BAO,6))
        b=np.array(column(BAO,7))
        scat=np.array(column(BAO,8))
        rd=np.array(column(BAO,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        BAO = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd]))
    os.chdir(dir_chain+'CC')
    filename = 'Chain_{0}Prior_{1}_CC.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    CC = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(CC,0))
        Om=np.array(column(CC,1))
        OL=np.array(column(CC,2))
        M=np.array(column(CC,3))
        a=np.array(column(CC,4))
        b=np.array(column(CC,5))
        scat=np.array(column(CC,6))
        rd=np.array(column(CC,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        CC = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(CC,0))
        Om=np.array(column(CC,1))
        OL=np.array(column(CC,2))
        w=np.array(column(CC,3))
        M=np.array(column(CC,4))
        a=np.array(column(CC,5))
        b=np.array(column(CC,6))
        scat=np.array(column(CC,7))
        rd=np.array(column(CC,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        CC = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(CC,0))
        Om=np.array(column(CC,1))
        OL=np.array(column(CC,2))
        w0=np.array(column(CC,3))
        wa=np.array(column(CC,4))
        M=np.array(column(CC,5))
        a=np.array(column(CC,6))
        b=np.array(column(CC,7))
        scat=np.array(column(CC,8))
        rd=np.array(column(CC,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        CC = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd]))
    os.chdir(dir_chain+'SN')
    filename = 'Chain_{0}Prior_{1}_SN.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    SN = Read_Chain.get_chain(discard=2500, flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(SN,0))
        Om=np.array(column(SN,1))
        OL=np.array(column(SN,2))
        M=np.array(column(SN,3))
        a=np.array(column(SN,4))
        b=np.array(column(SN,5))
        scat=np.array(column(SN,6))
        rd=np.array(column(SN,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        SN = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(SN,0))
        Om=np.array(column(SN,1))
        OL=np.array(column(SN,2))
        w=np.array(column(SN,3))
        M=np.array(column(SN,4))
        a=np.array(column(SN,5))
        b=np.array(column(SN,6))
        scat=np.array(column(SN,7))
        rd=np.array(column(SN,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        SN = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(SN,0))
        Om=np.array(column(SN,1))
        OL=np.array(column(SN,2))
        w0=np.array(column(SN,3))
        wa=np.array(column(SN,4))
        M=np.array(column(SN,5))
        a=np.array(column(SN,6))
        b=np.array(column(SN,7))
        scat=np.array(column(SN,8))
        rd=np.array(column(SN,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        SN = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd]))
    os.chdir(dir_chain+'GRB')
    filename = 'Chain_{0}Prior_{1}_GRB.h5'.format(prior, cosmo)
    Read_Chain = emcee.backends.HDFBackend(filename)
    len_ch=len(Read_Chain.get_chain())
    GRB = Read_Chain.get_chain(discard=(len_ch-burnin), flat=True, thin=thin)
    if cosmo=='LCDM':
        H0=np.array(column(GRB,0))
        Om=np.array(column(GRB,1))
        OL=np.array(column(GRB,2))
        M=np.array(column(GRB,3))
        a=np.array(column(GRB,4))
        b=np.array(column(GRB,5))
        scat=np.array(column(GRB,6))
        rd=np.array(column(GRB,7))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        GRB = np.transpose(np.array([H0,Om,OL,Ok,M,a,b,scat,rd]))
    if cosmo=='wCDM':
        H0=np.array(column(GRB,0))
        Om=np.array(column(GRB,1))
        OL=np.array(column(GRB,2))
        w=np.array(column(GRB,3))
        M=np.array(column(GRB,4))
        a=np.array(column(GRB,5))
        b=np.array(column(GRB,6))
        scat=np.array(column(GRB,7))
        rd=np.array(column(GRB,8))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        GRB = np.transpose(np.array([H0,Om,OL,Ok,w,M,a,b,scat,rd]))
    if cosmo=='w0waCDM':
        H0=np.array(column(GRB,0))
        Om=np.array(column(GRB,1))
        OL=np.array(column(GRB,2))
        w0=np.array(column(GRB,3))
        wa=np.array(column(GRB,4))
        M=np.array(column(GRB,5))
        a=np.array(column(GRB,6))
        b=np.array(column(GRB,7))
        scat=np.array(column(GRB,8))
        rd=np.array(column(GRB,9))
        Ok=[]
        for i in range(len(Om)):
            Ok.append(1-Om[i]-OL[i])
        GRB = np.transpose(np.array([H0,Om,OL,Ok,w0,wa,M,a,b,scat,rd])) 

    perc = [0.15, 2.3, 15.85, 50, 84.15, 97.7, 99.85]
    cosmology = str(cosmolatex)
    os.chdir(dir_out)
    name_output=r'Results_{0}.dat'.format(cosmo)
    fileout=open(name_output,"w")
    fileout.write("\\begin{table}[H] \n")
    fileout.write("\\renewcommand{\\arraystretch}{1.4} \n")
    fileout.write("     \\begin{center}\n")
    fileout.write("     \\centering {MARGINALIZED 1D CONSTRAINTS \\\ "+cosmology+"} \\\ \n")

    if cosmo == 'FLCDM':
        fileout.write("     \\begin{tabular}{cccccccc}\n")
        fileout.write("     \hline\n")
        fileout.write("     Probe & $H_0$ & $\Omega_{m}$ & $M$ & $a$ & $b$ & $\sigma_{int}$ & $r_d$  \\\ \n")
        fileout.write("     - & [km/s/Mpc] & - & - & - & - & - & Mpc & - \\\ \n")
        fileout.write("     \hline\n")
        fileout.write("     \hline\n")
  
        # BAO
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO, perc, axis=0)))

        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"


        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"


        fileout.write("     BAO & "+stringH0+"& "+stringOm+" & - & - & - & - & "+stringrd+" \\\ [0.3ex] \n")
        
        # CC
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC, perc, axis=0)))

        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"
        
        fileout.write("     CC & "+stringH0+" & "+stringOm+" & - & - & - & - & - & - \\\ [0.3ex] \n")
        
        # SN
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(SN, perc, axis=0)))
        
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        
        fileout.write("     SN & "+stringH0+" & "+stringOm+" & --"+stringM+" & - & - & - & - & - \\\ [0.3ex] \n")
    
                
        # GRB
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc= map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(GRB, perc, axis=0)))
        
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        
        fileout.write("     GRB & "+stringH0+" & "+stringOm+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        
        ### 2 ###
        # BAO + CC
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"


        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        

        fileout.write("     B+C & "+stringH0+" & "+stringOm+" & - & - & - & - & "+stringrd+" \\\ [0.3ex] \n")
        
        # BAO + SN
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_SN, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+S & "+stringH0+"  & "+stringOm+" & --"+stringM+" & - & - & - & "+stringrd+" \\\ [0.3ex] \n")

        
        
        # BAO + GRB
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("    B+G & "+stringH0+"  & "+stringOm+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")
        
    

        # CC + SN
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_SN, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"
        
        fileout.write("     C+S & "+stringH0+" & "+stringOm+" & --"+stringM+" & - & - & - & - & - \\\ [0.3ex] \n")

        
        
        # CC + GRB
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_GRB, perc, axis=0)))
        
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"
        
        fileout.write("     C+G & "+stringH0+" & "+stringOm+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        


        # SN + GRB
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(SN_GRB,  perc, axis=0)))

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"
        
        fileout.write("     S+G & "+stringH0+" & "+stringOm+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
                
        ### 3 ###
        # BAO + CC + SN
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_SN,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+S & "+stringH0+" & "+stringOm+" & --"+stringM+" & - & - & - & "+stringrd+" \\\ [0.3ex] \n")

        
        
        # BAO + CC + GRB
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+G & "+stringH0+" & "+stringOm+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")

                
        # BAO + SN + GRB
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+S+G & "+stringH0+" & "+stringOm+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")

        # CC + SN + GRB
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"
        
        fileout.write("     C+S+G & "+stringH0+" & "+stringOm+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        
         ### 4 ###
        # BA0 + CC + SN + GRB
        H0_mcmc, Om_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+S+G & "+stringH0+" & "+stringOm+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")
        
        fileout.write("     \hline\n")
        fileout.write("     \hline\n")
        fileout.write("     \end{tabular}\n")
        fileout.write("     \end{center}\n")
        fileout.write("\end{table}\n")     
        return fileout.close()

    if cosmo == 'LCDM':
        fileout.write("     \\begin{tabular}{cccccccccc}\n")
        fileout.write("     \hline\n")
        fileout.write("     Probe & $H_0$ & $\Omega_{m}$ & $\Omega_{\Lambda}$ &  $\Omega_{k}$ & $M$ & $a$ & $b$ & $\sigma_{int}$ & $r_d$  \\\ \n")
        fileout.write("     - & [km/s/Mpc] & - & - & -& - & - & - & - & Mpc & - \\\ \n")
        fileout.write("     \hline\n")
        fileout.write("     \hline\n")
  
        # BAO
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO, perc, axis=0)))

        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        

        fileout.write("     BAO & "+stringH0+"& "+stringOm+" & "+stringOL+" & "+stringOk+" &  - & - & - & - & "+stringrd+" \\\ [0.3ex] \n")
        
        # CC
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC, perc, axis=0)))

        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     CC & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" &  - & - & - & - & - & - \\\ [0.3ex] \n")
        
        # SN
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(SN, perc, axis=0)))
        
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     SN & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & --"+stringM+" & - & - & - & - & - \\\ [0.3ex] \n")
    
                
        # GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc= map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(GRB, perc, axis=0)))
        
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     GRB & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" &  - & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        
        ### 2 ###
        # BAO + CC
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        

        fileout.write("     B+C & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" &  - & - & - & - & "+stringrd+" \\\ [0.3ex] \n")
        
        # BAO + SN
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_SN, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+S & "+stringH0+"  & "+stringOm+" & "+stringOL+" & "+stringOk+" & --"+stringM+" & - & - & - & "+stringrd+" \\\ [0.3ex] \n")

        
        
        # BAO + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("    B+G & "+stringH0+"  & "+stringOm+" & "+stringOL+" & "+stringOk+" &  - & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")
        
    

        # CC + SN
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_SN, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+S & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & --"+stringM+" & - & - & - & - & - \\\ [0.3ex] \n")

        
        
        # CC + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" &  - & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        

        
        # SN + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     S+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
                
        ### 3 ###
        # BAO + CC + SN
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_SN,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+S & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & --"+stringM+" & - & - & - & "+stringrd+" \\\ [0.3ex] \n")

        
        
        # BAO + CC + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" &  - & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")

                
        # BAO + SN + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+S+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")

        # CC + SN + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+S+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        
         ### 4 ###
        # BA0 + CC + SN + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+S+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")
        
        fileout.write("     \hline\n")
        fileout.write("     \hline\n")
        fileout.write("     \end{tabular}\n")
        fileout.write("     \end{center}\n")
        fileout.write("\end{table}\n")     
        return fileout.close()

    if cosmo == 'FwCDM':
        fileout.write("     \\begin{tabular}{ccccccccc}\n")
        fileout.write("     \hline\n")
        fileout.write("     Probe & $H_0$ & $\Omega_{m}$ & $w$ & $M$ & $a$ & $b$ & $\sigma_{int}$ & $r_d$  \\\ \n")
        fileout.write("     - & [km/s/Mpc] & - & - & -& - & - & - & Mpc & - \\\ \n")
        fileout.write("     \hline\n")
        fileout.write("     \hline\n")
  
        # BAO
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO, perc, axis=0)))

        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        

        fileout.write("     BAO & "+stringH0+"& "+stringOm+" & "+stringw+" & - & - & - & - & "+stringrd+" \\\ [0.3ex] \n")
        
        # CC
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC, perc, axis=0)))

        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     CC & "+stringH0+" & "+stringOm+" & "+stringw+" & - & - & - & - & - & - \\\ [0.3ex] \n")
        
        # SN
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(SN, perc, axis=0)))
        
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     SN & "+stringH0+" & "+stringOm+" & "+stringw+" & --"+stringM+" & - & - & - & - & - \\\ [0.3ex] \n")
    
                
        # GRB
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc= map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(GRB, perc, axis=0)))
        
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     GRB & "+stringH0+" & "+stringOm+" & "+stringw+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        
        ### 2 ###
        # BAO + CC
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        

        fileout.write("     B+C & "+stringH0+" & "+stringOm+" & "+stringw+" & - & - & - & - & "+stringrd+" \\\ [0.3ex] \n")
        
        # BAO + SN
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_SN, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+S & "+stringH0+"  & "+stringOm+" & "+stringw+" & --"+stringM+" & - & - & - & "+stringrd+" \\\ [0.3ex] \n")

        
        
        # BAO + GRB
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("    B+G & "+stringH0+"  & "+stringOm+" & "+stringw+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")
        
    

        # CC + SN
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_SN, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+S & "+stringH0+" & "+stringOm+" & "+stringw+" & --"+stringM+" & - & - & - & - & - \\\ [0.3ex] \n")

        
        
        # CC + GRB
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+G & "+stringH0+" & "+stringOm+" & "+stringw+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        

        
        # SN + GRB
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     S+G & "+stringH0+" & "+stringOm+" & "+stringw+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
                
        ### 3 ###
        # BAO + CC + SN
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_SN,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+S & "+stringH0+" & "+stringOm+" & "+stringw+" & --"+stringM+" & - & - & - & "+stringrd+" \\\ [0.3ex] \n")

        
        
        # BAO + CC + GRB
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+G & "+stringH0+" & "+stringOm+" & "+stringw+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")

                
        # BAO + SN + GRB
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+S+G & "+stringH0+" & "+stringOm+" & "+stringw+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")

        # CC + SN + GRB
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+S+G & "+stringH0+" & "+stringOm+" & "+stringw+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        
         ### 4 ###
        # BA0 + CC + SN + GRB
        H0_mcmc, Om_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+S+G & "+stringH0+" & "+stringOm+" & "+stringw+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")
        
        fileout.write("     \hline\n")
        fileout.write("     \hline\n")
        fileout.write("     \end{tabular}\n")
        fileout.write("     \end{center}\n")
        fileout.write("\end{table}\n")     
        return fileout.close()
        
    if cosmo == 'wCDM':
        fileout.write("     \\begin{tabular}{cccccccccccc}\n")
        fileout.write("     \hline\n")
        fileout.write("     Probe & $H_0$ & $\Omega_{m}$ & $\Omega_{\Lambda}$ &  $\Omega_{k}$ & $w$ & $M$ & $a$ & $b$ & $\sigma_{int}$ & $r_d$  \\\ \n")
        fileout.write("     - & [km/s/Mpc] & - & - & -& - & - & - & - & - & Mpc & - \\\ \n")
        fileout.write("     \hline\n")
        fileout.write("     \hline\n")
  
        # BAO
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO, perc, axis=0)))

        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        

        fileout.write("     BAO & "+stringH0+"& "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & - & - & - & - & "+stringrd+" \\\ [0.3ex] \n")
        
        # CC
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC, perc, axis=0)))

        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     CC & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & - & - & - & - & - & - \\\ [0.3ex] \n")
        
        # SN
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(SN, perc, axis=0)))
        
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     SN & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & --"+stringM+" & - & - & - & - & - \\\ [0.3ex] \n")
    
                
        # GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc= map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(GRB, perc, axis=0)))
        
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     GRB & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        
        ### 2 ###
        # BAO + CC
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        

        fileout.write("     B+C & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & - & - & - & - & "+stringrd+" \\\ [0.3ex] \n")
        
        # BAO + SN
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_SN, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+S & "+stringH0+"  & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & --"+stringM+" & - & - & - & "+stringrd+" \\\ [0.3ex] \n")

        
        
        # BAO + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("    B+G & "+stringH0+"  & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")
        
    

        # CC + SN
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_SN, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+S & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & --"+stringM+" & - & - & - & - & - \\\ [0.3ex] \n")

        
        
        # CC + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        

        
        # SN + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     S+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
                
        ### 3 ###
        # BAO + CC + SN
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_SN,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+S & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & --"+stringM+" & - & - & - & "+stringrd+" \\\ [0.3ex] \n")

        
        
        # BAO + CC + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")

                
        # BAO + SN + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+S+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")

        # CC + SN + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+S+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        
         ### 4 ###
        # BA0 + CC + SN + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w= str(np.around(w_mcmc[0],3))
        werrup = str(np.around(w_mcmc[1],3))
        werrdw = str(np.around(w_mcmc[2],3))
        stringw = "$"+w+"^{+"+werrup+"}_{-"+werrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+S+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")
        
        fileout.write("     \hline\n")
        fileout.write("     \hline\n")
        fileout.write("     \end{tabular}\n")
        fileout.write("     \end{center}\n")
        fileout.write("\end{table}\n")     
        return fileout.close()
        
    if cosmo == 'Fw0waCDM':
        fileout.write("     \\begin{tabular}{cccccccccc}\n")
        fileout.write("     \hline\n")
        fileout.write("     Probe & $H_0$ & $\Omega_{m}$ & $w_0$ & $w_a$ &$M$ & $a$ & $b$ & $\sigma_{int}$ & $r_d$  \\\ \n")
        fileout.write("     - & [km/s/Mpc] & - & - & - & - & - & - & - & Mpc & - \\\ \n")
        fileout.write("     \hline\n")
        fileout.write("     \hline\n")
  
        # BAO
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO, perc, axis=0)))

        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        

        fileout.write("     BAO & "+stringH0+"& "+stringOm+" & "+stringw0+" & "+stringwa+" & - & - & - & - & "+stringrd+" \\\ [0.3ex] \n")
        
        # CC
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC, perc, axis=0)))

        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     CC & "+stringH0+" & "+stringOm+" & "+stringw0+" & "+stringwa+" & - & - & - & - & - & - \\\ [0.3ex] \n")
        
        # SN
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(SN, perc, axis=0)))
        
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     SN & "+stringH0+" & "+stringOm+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & - & - & - & - & - \\\ [0.3ex] \n")
    
                
        # GRB
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc= map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(GRB, perc, axis=0)))
        
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     GRB & "+stringH0+" & "+stringOm+" & "+stringw0+" & "+stringwa+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        
        ### 2 ###
        # BAO + CC
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        

        fileout.write("     B+C & "+stringH0+" & "+stringOm+" & "+stringw0+" & "+stringwa+" & - & - & - & - & "+stringrd+" \\\ [0.3ex] \n")
        
        # BAO + SN
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_SN, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+S & "+stringH0+"  & "+stringOm+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & - & - & - & "+stringrd+" \\\ [0.3ex] \n")

        
        
        # BAO + GRB
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("    B+G & "+stringH0+"  & "+stringOm+" & "+stringw0+" & "+stringwa+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")
        
    

        # CC + SN
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_SN, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+S & "+stringH0+" & "+stringOm+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & - & - & - & - & - \\\ [0.3ex] \n")

        
        
        # CC + GRB
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+G & "+stringH0+" & "+stringOm+" & "+stringw0+" & "+stringwa+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        

        
        # SN + GRB
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     S+G & "+stringH0+" & "+stringOm+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
                
        ### 3 ###
        # BAO + CC + SN
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_SN,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+S & "+stringH0+" & "+stringOm+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & - & - & - & "+stringrd+" \\\ [0.3ex] \n")

        
        
        # BAO + CC + GRB
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+G & "+stringH0+" & "+stringOm+" & "+stringw0+" & "+stringwa+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")

                
        # BAO + SN + GRB
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+S+G & "+stringH0+" & "+stringOm+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")

        # CC + SN + GRB
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+S+G & "+stringH0+" & "+stringOm+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        
         ### 4 ###
        # BA0 + CC + SN + GRB
        H0_mcmc, Om_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+S+G & "+stringH0+" & "+stringOm+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")
        
        fileout.write("     \hline\n")
        fileout.write("     \hline\n")
        fileout.write("     \end{tabular}\n")
        fileout.write("     \end{center}\n")
        fileout.write("\end{table}\n")     
        return fileout.close()
        
    if cosmo == 'w0waCDM':
        fileout.write("     \\begin{tabular}{cccccccccccc}\n")
        fileout.write("     \hline\n")
        fileout.write("     Probe & $H_0$ & $\Omega_{m}$ & $\Omega_{\Lambda}$ &  $\Omega_{k}$ & $w_0$ & $w_a$ &$M$ & $a$ & $b$ & $\sigma_{int}$ & $r_d$  \\\ \n")
        fileout.write("     - & [km/s/Mpc] & - & - & - & - & - & - & Mpc & - \\\ \n")
        fileout.write("     \hline\n")
        fileout.write("     \hline\n")
  
        # BAO
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO, perc, axis=0)))

        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        

        fileout.write("     BAO & "+stringH0+"& "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & - & - & - & - & "+stringrd+" \\\ [0.3ex] \n")
        
        # CC
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC, perc, axis=0)))

        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     CC & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & - & - & - & - & - & - \\\ [0.3ex] \n")
        
        # SN
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(SN, perc, axis=0)))
        
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     SN & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & - & - & - & - & - \\\ [0.3ex] \n")
    
                
        # GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc= map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(GRB, perc, axis=0)))
        
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     GRB & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        
        ### 2 ###
        # BAO + CC
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        

        fileout.write("     B+C & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & - & - & - & - & "+stringrd+" \\\ [0.3ex] \n")
        
        # BAO + SN
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_SN, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+S & "+stringH0+"  & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & - & - & - & "+stringrd+" \\\ [0.3ex] \n")

        
        
        # BAO + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("    B+G & "+stringH0+"  & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")
        
    

        # CC + SN
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_SN, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+S & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & - & - & - & - & - \\\ [0.3ex] \n")

        
        
        # CC + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        

        
        # SN + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     S+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
                
        ### 3 ###
        # BAO + CC + SN
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_SN,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+S & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & - & - & - & "+stringrd+" \\\ [0.3ex] \n")

        
        
        # BAO + CC + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_GRB, perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & - & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")

                
        # BAO + SN + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+S+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")

        # CC + SN + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(CC_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     C+S+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & - & - \\\ [0.3ex] \n")
        
         ### 4 ###
        # BA0 + CC + SN + GRB
        H0_mcmc, Om_mcmc, OL_mcmc, Ok_mcmc, w0_mcmc, wa_mcmc, M_mcmc, a_mcmc, b_mcmc, int_mcmc, rd_mcmc = map(lambda v: (v[3], v[4]-v[3], v[3]-v[2], v[5]-v[3], v[3]-v[1], v[6]-v[3], v[3]-v[0]), 
                                zip(*np.percentile(BAO_CC_SN_GRB,  perc, axis=0)))
        H0 = str(np.around(H0_mcmc[0],2))
        H0errup = str(np.around(H0_mcmc[1],2))
        H0errdw = str(np.around(H0_mcmc[2],2))
        stringH0 = "$"+H0+"^{+"+H0errup+"}_{-"+H0errdw+"}$"

        Om= str(np.around(Om_mcmc[0],3))
        Omerrup = str(np.around(Om_mcmc[1],3))
        Omerrdw = str(np.around(Om_mcmc[2],3))
        stringOm = "$"+Om+"^{+"+Omerrup+"}_{-"+Omerrdw+"}$"

        OL= str(np.around(OL_mcmc[0],3))
        OLerrup = str(np.around(OL_mcmc[1],3))
        OLerrdw = str(np.around(OL_mcmc[2],3))
        stringOL = "$"+OL+"^{+"+OLerrup+"}_{-"+OLerrdw+"}$"

        Ok= str(np.around(Ok_mcmc[0],3))
        Okerrup = str(np.around(Ok_mcmc[1],3))
        Okerrdw = str(np.around(Ok_mcmc[2],3))
        stringOk = "$"+Ok+"^{+"+Okerrup+"}_{-"+Okerrdw+"}$"

        w0= str(np.around(w0_mcmc[0],3))
        w0errup = str(np.around(w0_mcmc[1],3))
        w0errdw = str(np.around(w0_mcmc[2],3))
        stringw0 = "$"+w0+"^{+"+w0errup+"}_{-"+w0errdw+"}$"
        
        wa= str(np.around(wa_mcmc[0],3))
        waerrup = str(np.around(wa_mcmc[1],3))
        waerrdw = str(np.around(wa_mcmc[2],3))
        stringwa = "$"+wa+"^{+"+waerrup+"}_{-"+waerrdw+"}$"

        M= str(np.around(M_mcmc[0],1))
        Merrup = str(np.around(M_mcmc[1],1))
        Merrdw = str(np.around(M_mcmc[2],1))  
        stringM = "$"+M+"^{+"+Merrup+"}_{-"+Merrdw+"}$"

        a= str(np.around(a_mcmc[0],3))
        aerrup = str(np.around(a_mcmc[1],3))
        aerrdw = str(np.around(a_mcmc[2],3))  
        stringa = "$"+a+"^{+"+aerrup+"}_{-"+aerrdw+"}$"

        b= str(np.around(b_mcmc[0],3))
        berrup = str(np.around(b_mcmc[1],3))
        berrdw = str(np.around(b_mcmc[2],3))  
        stringb = "$"+b+"^{+"+berrup+"}_{-"+berrdw+"}$"

        scat= str(np.around(int_mcmc[0],3))
        scaterrup = str(np.around(int_mcmc[1],3))
        scaterrdw = str(np.around(int_mcmc[2],3))  
        stringscat = "$"+scat+"^{+"+scaterrup+"}_{-"+scaterrdw+"}$"

        rd= str(np.around(rd_mcmc[0],1))
        rderrup = str(np.around(rd_mcmc[1],1))
        rderrdw = str(np.around(rd_mcmc[2],1))  
        stringrd = "$"+rd+"^{+"+rderrup+"}_{-"+rderrdw+"}$"

        
        
        fileout.write("     B+C+S+G & "+stringH0+" & "+stringOm+" & "+stringOL+" & "+stringOk+" & "+stringw0+" & "+stringwa+" & --"+stringM+" & "+stringa+" & "+stringb+" & "+stringscat+" & "+stringrd+" \\\ [0.3ex] \n")
        
        fileout.write("     \hline\n")
        fileout.write("     \hline\n")
        fileout.write("     \end{tabular}\n")
        fileout.write("     \end{center}\n")
        fileout.write("\end{table}\n")     
        return fileout.close()
        

################################
#### Directories Definition ####
################################
dir_home = os.getcwd()
dir_out = dir_home+'/path/to/tables/'
dir_plots = dir_home+'/path/to/plots/'
dir_chain = dir_home+'/path/to/chains/'


########################
#### Summary Tables ####
########################
Table_MCMC('Flat', 'FLCDM', r'Flat $\Lambda$CDM', dir_chain, dir_out)
Table_MCMC('Flat', 'LCDM', r'$\Lambda$CDM', dir_chain, dir_out)
Table_MCMC('Flat', 'FwCDM', r'Flat $w$CDM', dir_chain, dir_out)
Table_MCMC('Flat', 'wCDM', r'$w$CDM', dir_chain, dir_out)
Table_MCMC('Flat', 'Fw0waCDM', r'Flat $w_0w_a$CDM', dir_chain, dir_out)
Table_MCMC('Flat', 'w0waCDM', r'$w_0w_a$CDM', dir_chain, dir_out)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('BAOCCSNGRB', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'BAOCCSNGRB',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'BAOCCSNGRB',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'BAOCCSNGRB',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'BAOCCSNGRB',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'BAOCCSNGRB',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'BAOCCSNGRB',  burnin[5], dir_chain, dir_plots)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('BAO', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'BAO',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'BAO',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'BAO',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'BAO',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'BAO',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'BAO',  burnin[5], dir_chain, dir_plots)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('CC', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'CC',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'CC',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'CC',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'CC',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'CC',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'CC',  burnin[5], dir_chain, dir_plots)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('SN', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'SN',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'SN',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'SN',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'SN',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'SN',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'SN',  burnin[5], dir_chain, dir_plots)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('GRB', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'GRB',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'GRB',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'GRB',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'GRB',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'GRB',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'GRB',  burnin[5], dir_chain, dir_plots)


burnin=[4000,4000,4000,4000,4000,4000]
table_paper('BAOCCSN', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'BAOCCSN',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'BAOCCSN',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'BAOCCSN',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'BAOCCSN',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'BAOCCSN',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'BAOCCSN',  burnin[5], dir_chain, dir_plots)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('BAOCCGRB', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'BAOCCGRB',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'BAOCCGRB',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'BAOCCGRB',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'BAOCCGRB',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'BAOCCGRB',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'BAOCCGRB',  burnin[5], dir_chain, dir_plots)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('BAOSNGRB', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'BAOSNGRB',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'BAOSNGRB',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'BAOSNGRB',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'BAOSNGRB',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'BAOSNGRB',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'BAOSNGRB',  burnin[5], dir_chain, dir_plots)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('CCSNGRB', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'CCSNGRB',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'CCSNGRB',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'CCSNGRB',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'CCSNGRB',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'CCSNGRB',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'CCSNGRB',  burnin[5], dir_chain, dir_plots)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('BAOCC', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'BAOCC',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'BAOCC',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'BAOCC',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'BAOCC',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'BAOCC',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'BAOCC',  burnin[5], dir_chain, dir_plots)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('BAOSN', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'BAOSN',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'BAOSN',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'BAOSN',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'BAOSN',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'BAOSN',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'BAOSN',  burnin[5], dir_chain, dir_plots)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('BAOGRB', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'BAOGRB',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'BAOGRB',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'BAOGRB',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'BAOGRB',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'BAOGRB',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'BAOGRB',  burnin[5], dir_chain, dir_plots)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('CCSN', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'CCSN',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'CCSN',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'CCSN',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'CCSN',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'CCSN',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'CCSN',  burnin[5], dir_chain, dir_plots)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('CCGRB', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'CCGRB',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'CCGRB',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'CCGRB',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'CCGRB',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'CCGRB',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'CCGRB',  burnin[5], dir_chain, dir_plots)

burnin=[4000,4000,4000,4000,4000,4000]
table_paper('SNGRB', burnin, dir_chain, dir_out)
plot_bestfitMCMC('FLCDM', 'SNGRB',  burnin[0], dir_chain, dir_plots)
plot_bestfitMCMC('LCDM', 'SNGRB',  burnin[1], dir_chain, dir_plots)
plot_bestfitMCMC('FwCDM', 'SNGRB',  burnin[2], dir_chain, dir_plots)
plot_bestfitMCMC('wCDM', 'SNGRB',  burnin[3], dir_chain, dir_plots)
plot_bestfitMCMC('Fw0waCDM', 'SNGRB',  burnin[4], dir_chain, dir_plots)
plot_bestfitMCMC('w0waCDM', 'SNGRB',  burnin[5], dir_chain, dir_plots)
