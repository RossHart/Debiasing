import fit
import dictionaries
import params
import sample

def debias(data,params=params,dictionaries=dictionaries,
           question='shape',answer='smooth',append_column=True,verbose=True):
    function_dictionary = dictionaries.function_dictionary
    questions = dictionaries.questions
    
    data_sample = sample.Sample(data,questions,params,question)
    bins = sample.Bins(data_sample,params,questions,question,answer)
    bins.voronoi_bin()
    bins.voronoi_assignment(reassign=True)
    bins.redshift_assignment()
    functionfit, function_, bounds = fit.fit_bins(data_sample,bins,function_dictionary,
                                                  params,question,answer,verbose=verbose)

    fitted_k = fit.FitToBins(functionfit,'k',params.clip_percentile).get_kc_function(verbose=verbose)
    fitted_c = fit.FitToBins(functionfit,'c',params.clip_percentile).get_kc_function(verbose=verbose)
    fv, fv_debiased = fit.debias_data(data_sample.all_data,params,fitted_k,fitted_c,function_,
                                      question,answer)
    if append_column is True:
        data[question + '_' + answer + '_debiased'] = fv_debiased
        return data
    else:
        return fv, fv_debiased