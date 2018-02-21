import numpy as np
import fit
import dictionaries
import params
import sample
from scipy.stats import binned_statistic

def debias(data,params=params,dictionaries=dictionaries,question='features',
           answer='smooth',use_fit=False,use_bin=False,append_column=True,
           verbose=True):
    function_dictionary = dictionaries.function_dictionary
    questions = dictionaries.questions
    
    data_sample = sample.Sample(data,questions,params,question)
    bins = sample.Bins(data_sample,params,questions,question,answer)
    bins.voronoi_bin()
    bins.voronoi_assignment(reassign=True)
    bins.redshift_assignment()
    # bin by bin...
    fv_debiased_bin = fit.debias_by_bin(data_sample.all_data,
                                        bins.voronoi_bins,bins.z_bins,
                                        question,answer)
    
    functionfit, function_, bounds = fit.fit_bins(data_sample,bins,
                                                  function_dictionary,params,
                                                  question,answer,
                                                  verbose=verbose)
    # function fitter...
    fitted_k = fit.FitToBins(functionfit,'k',
                      params.clip_percentile).get_kc_function(verbose=verbose)
    fitted_c = fit.FitToBins(functionfit,'c',
                      params.clip_percentile).get_kc_function(verbose=verbose)
    fv, fv_debiased = fit.debias_data(data_sample.all_data,params,
                                      fitted_k,fitted_c,function_,
                                      question,answer)
    fv_debiased[np.isfinite(fv_debiased) == False] = 0

    if use_fit is True:
        fv_debiased_final = fv_debiased
    elif use_bin is True:
        fv_debiased_final = fv_debiased_bin
    else:
        # now need to check which is better -- need a volume limited sample...
        in_vl = data_sample.volume_limited_sample()
        z = data_sample.all_data[params.z_column]
        chi2_bin = get_chi2(fv_debiased_bin[in_vl],fv[in_vl],z[in_vl])
        chi2_fit = get_chi2(fv_debiased[in_vl],fv[in_vl],z[in_vl])
    
        if chi2_bin < chi2_fit:
            print('bin method preferred')
            fv_debiased_final = fv_debiased_bin
        else:
            print('fit method preferred')
            fv_debiased_final = fv_debiased
        if verbose is True:
            print('---------')
            print('chi2 (bin) = {}'.format(np.round(chi2_bin,2)))
            print('chi2 (fit) = {}'.format(np.round(chi2_fit,2)))
            print('---------')
    # give back the outputs...
    if append_column is True:
        data[question + '_' + answer + '_debiased'] = fv_debiased_final
        return data
    else:
        return fv, fv_debiased_final

def histogram_fractions(data,hist_bins):
    ''' Get raw histogram values '''
    h,bin_edges = np.histogram(data,bins=hist_bins)
    f = h/np.sum(h)
    return f


def bin_by_column(column, nbins, fixedcount=True):
    ''' Bin the data into redshift slices 
    (or by any column) '''
    
    sorted_indices = np.argsort(column)
    if fixedcount:
        bin_edges = np.linspace(0, 1, nbins + 1)
        bin_edges[-1] += 1
        values = np.empty(len(column))
        values[sorted_indices] = np.linspace(0, 1, len(column))
        bins = np.digitize(values, bins=bin_edges)
    else:
        bin_edges = np.linspace(np.min(column),np.max(column), nbins + 1)
        bin_edges[-1] += 1
        values = column
        bins = np.digitize(values, bins=bin_edges)
    x, b, n = binned_statistic(values, column, bins=bin_edges)
    return x, bins


def get_chi2(dataset,reference,redshifts):
    ''' Calculate rms of a dataset in comparison w. reference data'''
    hist_bins = np.linspace(0,1,11)
    hist_bins[0] = -1
    hist_bins[-1] = 2
    zv,zb = bin_by_column(redshifts,nbins=10)
    
    rms_values = np.zeros(len(np.unique(zb)))
    ref_fractions = histogram_fractions(reference,hist_bins)
    
    for i, z in enumerate(np.unique(zb)):
        
        sample = dataset[zb == z]
        fractions = histogram_fractions(sample,hist_bins)
        rms_values[i] = np.sum((fractions-ref_fractions)**2)
    
    return np.sum(rms_values)