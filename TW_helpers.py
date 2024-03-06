"""
Helper functions for working with RAM data using cmlreaders and PTSA to perform traveling wave directional analyses. 
This is geared towards helping users perform analyses of direction dynamics after calculating basic traveling wave characteristics .
"""

import numexpr
import os
import warnings
import numpy as np
import pandas as pd
import xarray as xr

from ptsa.data.filters import ButterworthFilter
from ptsa.data.filters import MorletWaveletFilter
from ptsa.data.filters import ResampleFilter
from ptsa.data.timeseries import TimeSeries
from scipy import stats, signal
from cmlreaders import CMLReader, get_data_index
from scipy.stats.mstats import zscore
from scipy.io import loadmat
from pycircstat import mean as circmean
import pycircstat
from tqdm import tqdm
from glob import glob
from sklearn import metrics
import vm_mixture
from timeout import timeout
import RAM_helpers_UM

# get the r1 dataframe on import so we don't have to keep doing it
try:
    r1_data = get_data_index("r1")
except KeyError:
    print('r1 protocol file not found')


#group subregion labels into broader regions

def subreg_to_lobe(elec_subreg):
   
    roi_dict = {'Hipp': ['Left CA1', 'Left CA2', 'Left CA3', 'Left DG', 'Left Sub', 'Right CA1', 'Right CA2',
                         'Right CA3', 'Right DG', 'Right Sub'],
                'MTL': ['Left PRC', 'Right PRC', 'Right EC', 'Right PHC', 'Left EC', 'Left PHC', 
                        'entorhinal', 'Left Amy', 'Right Amy','parahippocampal'],
#                 'Cingulate': ['posteriorcingulate','caudalanteriorcingulate', 'isthmuscingulate', 'rostralanteriorcingulate'],
                'Frontal': ['parsopercularis', 'parsorbitalis', 'parstriangularis', 'caudalmiddlefrontal', 'rostralmiddlefrontal', 'superiorfrontal', 'lateralorbitofrontal', 'medialorbitofrontal', 'frontalpole', 'precentral', 'caudalanteriorcingulate', 'rostralanteriorcingulate'],
                'Temporal': ['superiortemporal', 'middletemporal', 'inferiortemporal', 'transversetemporal', 'fusiform', 'bankssts', 'temporalpole',
                             'Left Inferior Temporal Gyrus', 'Right Inferior Temporal Gyrus', 
                             'Left Middle Temporal Gyrus', 'Right Middle Temporal Gyrus',
                             'Left Superior Temporal Gyrus', 'Right Superior Temporal Gyrus'],
                'Parietal': ['inferiorparietal', 'supramarginal', 'superiorparietal', 'precuneus','postcentral','paracentral','posteriorcingulate', 'isthmuscingulate'],
                'Occipital': ['lateraloccipital', 'lingual', 'cuneus', 'pericalcarine'],
                'Other': ['corpuscallosum','unknown']}
    
    
    



    for roi in roi_dict.keys():
        if elec_subreg in roi_dict[roi]:
            lobe = roi
    return lobe


#find lobe where majority of channels in an oscillation cluster are located


def tw_main_reg(tw_reg):    
    
    lobez = []
    for r in range(len(tw_reg)): 
        reg = subreg_to_lobe(tw_reg.iloc[r].to_frame().columns[0])
        lobez.append(reg)
    tw_reg['lobe'] = lobez
    majority_lobe = tw_reg.groupby('lobe')[tw_reg.columns[0]].sum().nlargest(1).to_frame().index[0]
    return majority_lobe

#calculate r_bar


def rbar(x):
    n=len(x)
    x1 = numexpr.evaluate('sum(cos(x) / n, axis=0)')
    x1 = numexpr.evaluate('x1 ** 2')
    x2 = numexpr.evaluate('sum(sin(x) / n, axis=0)')
    x2 = numexpr.evaluate('x2 ** 2')
    Rs = numexpr.evaluate('sqrt(x1 + x2)')
    return Rs

@timeout(300)
#calculate r_bar

# calculate number of von mises distributions needed to fit the distribution of propagation directions. Each individual fitted von Mises distribution reflects one particular direction in which the TWs on the cluster frequently propagate. Distributions fitted with more than one von Mises distribution thus showed multiple distinct propagation directions. This iterates to determine the best fitting mixture of von Mises curves, as the sum of the minimum number of von Mises curves that would fit 99% of the variance in the original distribution of propagation directions

def fit_num_peaks_vm3(tw_el_dir_dist, r2thresh, max_peaks):
    r2 = 0
    k = 0
    d = tw_el_dir_dist
    while r2<r2thresh: 
        k = k + 1
        vmf_params = vm_mixture.mixture_vonmises_pdfit(d, n=k, threshold=1e-5)
        while any(np.isnan(vmf_params[1])): 
            k = k-1
            vmf_params = vm_mixture.mixture_vonmises_pdfit(d, n=k, threshold=1e-5)
        for x in range(5):
            allvm_est = []
            for n in range(len(vmf_params[0])): 
                vm_est = stats.vonmises.rvs(vmf_params[2][n], loc = vmf_params[1][n], scale = 1, size = int(np.round(vmf_params[0][n]*len(d))))
                vm_est[vm_est < 0] = vm_est[vm_est < 0] + 2*np.pi
                vm_est[vm_est > 2*np.pi] = vm_est[vm_est > 2*np.pi] - 2*np.pi
                allvm_est.append(vm_est)
            try:
                r2 = metrics.r2_score(np.sort(d), np.sort(np.hstack(allvm_est)))
            except: 
                if len(np.hstack(allvm_est)) > len(d): 
                    r2 = metrics.r2_score(np.append(np.sort(d), 2*np.pi), np.sort(np.hstack(allvm_est)))
                else: 
                    r2 = metrics.r2_score(np.sort(d), np.append(np.sort(np.hstack(allvm_est)), 2*np.pi) )
#             print("r2: %0.4f" % r2 )
            if r2 > r2thresh: 
                break
        if k == max_peaks: 
            break
    if any(vmf_params[1]<0): 
        vmf_params[1][vmf_params[1]<0] = vmf_params[1][vmf_params[1]<0] + 2*np.pi
    return np.array(vmf_params[1]), vmf_params[0], vmf_params[2], k





def rbars_ev(trial_directions_rec, trial_directions_unrec, t, num_elecs, num_ev_rec, num_ev_unrec):

    rbarz_ev_rec = []
    ray_ev_rec = np.zeros(num_ev_rec)
    for ev in range(num_ev_rec):
        rdir_el_rec = []
        for el in (range(num_elecs)): 
            try:
                rdir_el_rec.append(np.abs(np.arctan2(-3*trial_directions_rec[el][t][2,ev],-3*trial_directions_rec[el][t][1,ev])-np.pi))
            except: 
                pass
        rbarz_ev_rec.append(rbar(rdir_el_rec))
        p, r = pycircstat.tests.rayleigh(np.hstack(rdir_el_rec))
        if p < 0.05: 
            ray_ev_rec[ev] = 1 
    ray_ev_rec = ray_ev_rec == 1

        
    rbarz_ev_unrec = []
    ray_ev_unrec = np.zeros(num_ev_unrec)

    for ev in range(num_ev_unrec):
        rdir_el_unrec = []
        for el in (range(num_elecs)): 
            try:
                rdir_el_unrec.append(np.abs(np.arctan2(-3*trial_directions_unrec[el][t][2,ev],-3*trial_directions_unrec[el][t][1,ev]-np.pi)))
            except: 
                pass
        rbarz_ev_unrec.append(rbar(rdir_el_unrec))
        p, r = pycircstat.tests.rayleigh(np.hstack(rdir_el_unrec))
        if p < 0.05: 
            ray_ev_unrec[ev] = 1 
    ray_ev_unrec = ray_ev_unrec == 1
    return rbarz_ev_rec, rbarz_ev_unrec, ray_ev_rec, ray_ev_unrec


def dir_rayleigh(recall_rad, no_recall_rad): 
    rec_include = np.zeros(len(recall_rad))
    norec_include = np.zeros(len(no_recall_rad))
    for ev in range(len(recall_rad)):
        p, r = pycircstat.tests.rayleigh(recall_rad)
        if p < 0.05: 
            rec_include[ev] = 1
    for ev in range(len(no_recall_rad)):
        p, r = pycircstat.tests.rayleigh(no_recall_rad)
        if p < 0.05: 
            norec_include[ev] = 1
    rec_include = rec_include == 1
    norec_include = norec_include == 1
    return rec_include, norec_include


# find recall rate for trials when waves are propagating in preferred encoding and preferred recall directions. Use this to calculate odds ratio
def find_pref_dir_recallrate(m_all, m_recall, prop_recall, clust_dir_t_all_ev, recall_th, pref_dir): 
    if pref_dir == None: 
        pref_dir = m_recall[np.argmax(prop_recall)]
    if len(m_all) == 2: 
        a1 = stats.circmean(m_all)
        if RAM_helpers_UM.circular_dist(m_all[0], m_all[1]) < np.pi/2:
            a1 = pref_dir + np.pi/2
    else: 
        a1 = pref_dir + np.pi/2
    if a1 > 2*np.pi: 
        a1 = a1 - 2*np.pi
    if a1>=np.pi: 
        a2 = a1 - np.pi
    if a1<np.pi:
        a2 = a1 + np.pi
    if pref_dir > a2 and pref_dir < a1: 
        a1 = a2
        a2 = a2 + np.pi
    if pref_dir > a1 and pref_dir < a2: 
        num_pref = len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])
        num_nonpref = len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])
        if len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)]) == 0: 
            pref_dir_rec = 0
            pref_odds = 0
        else: 
            pref_dir_rec = sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])/len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])
        if len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)]) == 0: 
            nonpref_dir_rec = 0
            pref_odds = (sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])*(len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])-sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)]))+.5)/(sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])*(len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])-sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)]))+.5)
        else: 
            nonpref_dir_rec = sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])/len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])
            pref_odds = (sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])*(len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])-sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])))/(sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])*(len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])-sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])))
                         
    
    if pref_dir < a1 and pref_dir < a2:  
        num_pref = len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])
        num_nonpref = len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])
        if len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])==0: 
            pref_dir_rec = 0
            pref_odds = 0
        else: 
            pref_dir_rec = sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])/len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])     
        if len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)]) == 0: 
            nonpref_dir_rec = 0
            pref_odds = (sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])*(len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])-sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)]))+.5)/(sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])*(len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])-sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)]))+.5)
        else: 
            nonpref_dir_rec = sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])/len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])
            pref_odds = (sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])*(len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])-sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])))/(sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])*(len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])-sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])))
    
    
    if pref_dir >= a1 and pref_dir >= a2:  
        num_pref = len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])
        num_nonpref = len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])
        if len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)]) == 0:
            pref_dir_rec = 0
            pref_odds = 0
        else: 
            pref_dir_rec = sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])/len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])
        if len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)]) == 0: 
            nonpref_dir_rec = 0
            pref_odds = (sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])*(len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])-sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)]))+.5)/(sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])*(len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])-sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)]))+.5)
        else: 
            nonpref_dir_rec = sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])/len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])
            pref_odds = (sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])*(len(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])-sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])))/(sum(recall_th[np.logical_and(clust_dir_t_all_ev > a1, clust_dir_t_all_ev < a2)])*(len(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])-sum(recall_th[np.logical_or(clust_dir_t_all_ev <= a1, clust_dir_t_all_ev >= a2)])))
            
    return pref_dir_rec, nonpref_dir_rec, pref_odds, num_pref, num_nonpref, a1, a2


            

    # using combination of number of von mises distributions, rayleigh stat, and kappa parameter (concentration/height) of mixture of von mises distributions
def find_distribution_modality(all_dirs): 
    p, r = pycircstat.tests.rayleigh(all_dirs)

    th = TW_helpers.find_r2thresh(all_dirs)
    m_all,prop_all, kappa_all, numpeaks_all = TW_helpers.fit_num_peaks_vm3(all_dirs, r2thresh = th-.05, max_peaks = 3)
#     print(p, m_all,prop_all, kappa_all)
    if p < 0.05: 
        if len(m_all) == 1: 
            mode = 'unimodal'
        if np.logical_and(len(m_all) == 2, np.any(kappa_all < 2)):
            mode = 'unimodal'
        if np.logical_and(len(m_all) == 2, np.all(kappa_all >= 2)):
            mode = 'bimodal'
        if len(m_all) == 3:
            mode = 'multimodal'
    if p >= 0.05: 
        if np.logical_and(len(m_all) == 1, np.any(kappa_all < 2)):
            mode = 'non-directional'
        if np.logical_and(len(m_all) == 1, np.all(kappa_all >= 2)):
            mode = 'unimodal'
        if np.logical_and(len(m_all) == 2, np.any(kappa_all < 2)):
            mode = 'unimodal'
        if np.logical_and(len(m_all) == 2, np.all(kappa_all >= 2)):
            mode = 'bimodal'
        if np.logical_and(len(m_all) == 3, np.any(kappa_all < 2)):
            mode = 'bimodal'
        if np.logical_and(len(m_all) == 3, np.all(kappa_all >= 2)):
            mode = 'multimodal'
    return mode



