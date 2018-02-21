from astropy.table import Table, column
import dictionaries
import math
import numpy as np
import params
from scipy.optimize import minimize, curve_fit

class FunctionFit():
    
    def __init__(self,sample,params,bins,question,answer):
        self.sample = sample
        self.bins = bins
        self.question = question
        self.answer = answer
        self.params = params
        self.parameter_table = Table(names=('logR50','Mr','z','k','c','chisq','success'))
        
    def chisq_fun(self,p, f, x, y):
        ''' chisquare function'''
        return ((f(x, *p) - y)**2).sum()
        
    def cumfrac_fit(self,data,function_,append=True):
        q = self.question
        a = self.answer
        fv = np.sort(data[q + '_' + a + self.params.fraction_suffix])
        fv_nonzero = fv != 0
        cf = np.linspace(0,1,len(fv))
        x, y = [np.log10(fv[fv_nonzero]),cf[fv_nonzero]]
        
        x_fit = np.log10(np.linspace(10**(self.params.log_fv_range[0]), 1, 100))
        indices = np.searchsorted(x,x_fit)
        y_fit = y[indices.clip(0, len(y)-1)]
        
        fit_setup = function_
        res =  minimize(self.chisq_fun, function_.p0,
                        args=(function_.function,x_fit,y_fit),
                        bounds=function_.bounds,method='SLSQP')
        
        Mr = np.mean(data[params.Mr_column])
        logR50 = np.mean(data['logR50'])
        z_ = np.mean(data[params.z_column])
        k, c = res.x
        if append is True:
            row = [logR50,Mr,z_,k,c,res.fun,res.success]
            self.parameter_table.add_row(row)
        return None
    

class Function():
    def __init__(self,function_dictionary,key):
        self.function = function_dictionary['func'][key]
        self.bounds = function_dictionary['bounds'][key]
        self.p0 = function_dictionary['p0'][key]
        self.inverse_function = function_dictionary['i_func'][key]
        self.label = function_dictionary['label'][key]    
        
        
class FitToBins():
    def __init__(self,functionfit,column='k',clip_percentile=5):
        self.fit_table = functionfit.parameter_table
        self.Mr = self.fit_table['Mr']
        self.R50 = self.fit_table['logR50']
        self.z = self.fit_table['z']
        self.fit_parameter = self.fit_table[column]
        self.ok_fit = self.fit_table['success'] == 1
        self.column = column
        if clip_percentile is not None:
            parameter_mean = np.mean(self.fit_parameter)
            parameter_std = np.std(self.fit_parameter)
            #self.in_sigma = np.all([self.fit_parameter <= parameter_mean + clip*parameter_std,
                                    #self.fit_parameter >= parameter_mean - clip*parameter_std],
                                   #axis=0)
            parameter_min = np.percentile(self.fit_parameter,clip_percentile)
            parameter_max = np.percentile(self.fit_parameter,100-clip_percentile)
            self.in_sigma = np.all([self.fit_parameter >= parameter_min,
                                    self.fit_parameter <= parameter_max],
                                    axis=0)
        else:
            self.in_sigma = np.full(len(parameter_table),True)
        to_fit = np.all([self.ok_fit,self.in_sigma],axis=0) 
        self.to_fit = to_fit
        # Set limits of the functions here:
        self.min_ = self.fit_parameter[to_fit].min() 
        self.max_ = self.fit_parameter[to_fit].max()
        
    def kc_function(self,M_dependence,R_dependence,z_dependence):
        def kcfunc(x,A0,AM,AR,Az):
            M_term = self.get_term(AM,x[0],M_dependence,negative=True)
            R_term = self.get_term(AR,x[1],R_dependence)
            z_term = self.get_term(Az,x[2],z_dependence)
            return A0 + M_term + R_term + z_term
        return kcfunc
        
    def get_term(self,constant,var,t='linear',negative='False'):
        if negative is True:
            var = -var
        if t == 'log':
            term = constant*np.log10(var)
        elif t == 'linear':
            term = constant*var
        elif t == 'exp':
            term = constant*(10**(var))
        return term

    def get_MRz_function(self,M_dependence,R_dependence,z_dependence):
        def kcfunc(x,A0,AM,AR,Az):
            M_term = get_term(AM,x[0],M_dependence)
            R_term = get_term(AR,x[1],R_dependence)
            z_term = get_term(Az,x[2],z_dependence)
            return A0 + M_term + R_term + z_term
    
        return kcfunc

    def normalise(self,x):
        return (x - x.mean())/x.std()
        
    def get_kc_function(self,verbose=True):
        M_dependencies = ('log','linear','exp')
        R_dependencies = ('log','linear','exp')
        z_dependencies = ('log','linear','exp')
        # Find the best functions for fitting the data:
        output_table = Table(names=('M_dependency','R_dependency',
                                    'z_dependency','p_fit','chisq'),
                             dtype=('object','object','object','object',
                                    np.float32))
    
        for M_dependency in M_dependencies:
            for R_dependency in R_dependencies:
                for z_dependency in z_dependencies:
                    kcfunc = self.kc_function(M_dependency,R_dependency,z_dependency)
                    p_fit, res, _ = self.fit_mrz(kcfunc)
                    MRz_row = [M_dependency,R_dependency,z_dependency,p_fit,res]
                    output_table.add_row(MRz_row)
        # Set the nan values to be very large (so they are automatically avoided!)
        output_table['chisq'][np.isfinite(output_table['chisq']) == False] = 10**8
    
        # Choose best functions:
        best_row = np.argmin(output_table['chisq']) 
        best_M_d = output_table['M_dependency'][best_row]
        best_R_d = output_table['M_dependency'][best_row]
        best_z_d = output_table['M_dependency'][best_row]

        kcfunc = self.kc_function(best_M_d,best_R_d,best_z_d)
        if verbose is True:
            print('--- Selected function ({}): ---'.format(self.column))
            print('{}(M),{}(R),{}(z)'.format(best_M_d,best_R_d,best_z_d))
        p_fit, _, fitted_table = self.fit_mrz(kcfunc)
     
        self.kc_function = kcfunc
        self.p_fit = p_fit
        self.output_table = output_table
        self.fitted_table = fitted_table
    
        return self

    def fit_mrz(self,kc_function,clip=None):
        ''' Fit a linear function of M, R and z to k and c '''
        x = self.Mr
        y = self.R50
        z = self.z
        xyz = np.array([x,y,z])
        fitted_table = Table(xyz.T,names=('Mr','R50','z'))
        fitted_table[self.column] = self.fit_parameter

        xyz_ok = (xyz.T[self.to_fit]).T
        fit_parameter_ok = self.fit_parameter[self.to_fit]
    
        # Fit to the data:
        p_fit, _ = curve_fit(kc_function,xyz_ok,fit_parameter_ok,maxfev=10**5)  
        res = kc_function(xyz_ok,*p_fit) - fit_parameter_ok # k residuals
        res_normalised = self.normalise(res) # normalised k residuals
        # Remove the +-2sigma fits, and then redo the fitting:
        if clip != None:
            clipped = np.absolute(res_normalised) < clip 
            p_fit, _ = curve_fit(kc_function, (xyz_ok.T[clipped]).T, 
                                 fit_parameter_ok[clipped],maxfev=10**5)
    
        fitted_table[self.column + '_fit'] = kc_function(xyz,*p_fit)
        chisq = (fitted_table[self.column + '_fit'] - fitted_table[self.column])**2
        return p_fit, chisq.sum(), fitted_table
      

def fit_bins(sample,bins,function_dictionary,params,
             question='shape',answer='smooth',verbose=True):
    
    log_function = Function(function_dictionary,0)
    exp_function = Function(function_dictionary,1)
    logfit = FunctionFit(sample,params,bins,question,answer)
    expfit = FunctionFit(sample,params,bins,question,answer)
    v_unique = np.unique(bins.voronoi_bins)
    for v in v_unique:
        in_v = bins.voronoi_bins == v
        in_vp = np.all([in_v,sample.p_mask],axis=0)
        z_unique = np.unique(bins.z_bins[in_vp])
        for z in z_unique:
            in_z = bins.z_bins == z
            in_vpz = np.all([in_vp,in_z],axis=0)
            logfit.cumfrac_fit(sample.all_data[in_vpz],log_function)
            expfit.cumfrac_fit(sample.all_data[in_vpz],exp_function)
            
    chisq_log = np.sum(logfit.parameter_table['chisq'])
    chisq_exp = np.sum(expfit.parameter_table['chisq'])
    if verbose is True:
        print('------------------')
        print('chisq (log) = {}'.format(round(chisq_log.sum(),1)))
        print('chisq (exp) = {}'.format(round(chisq_exp.sum(),1)))
        print('------------------')
    if chisq_log < chisq_exp:
        print('=> log function preferred') if verbose is True else None
        return logfit, log_function, params.logistic_bounds
    else:
        print('=> exp function preferred') if verbose is True else None
        return expfit, exp_function, params.exponential_bounds
      
      
def debias_data(data,params,fitted_k,fitted_c,function_,
                question='shape',answer='smooth'):
    
    function = function_.function
    inverse_function = function_.inverse_function
    
    fv_column = question + '_' + answer + params.fraction_suffix
    fv = data[fv_column]
    fv_debiased = np.zeros(len(fv))
    nonzero = fv > 0
    fv_nonzero = fv[nonzero]
    logfv = np.log10(fv_nonzero)
    
    x = data[params.Mr_column][nonzero]
    y = data['logR50'][nonzero]
    z = data[params.z_column][nonzero]
    xyz = np.array([x,y,z])
    low_z_limit = params.volume_redshift_limits[0]
    xyz_low_z = xyz.copy()
    xyz_low_z[-1] = np.full(len(z),low_z_limit)
   
    def fitted_parameter(xyz,f):
        p = f.kc_function(xyz,*f.p_fit)
        p[p <= f.min_] = f.min_
        p[p >= f.max_] = f.max_
        return p
    
    k = fitted_parameter(xyz,fitted_k)
    k_low_z = fitted_parameter(xyz_low_z,fitted_k)
    c = fitted_parameter(xyz,fitted_c)
    c_low_z = fitted_parameter(xyz_low_z,fitted_c)
    
    cumfrac = function(logfv, k, c)
    logfv_debiased = inverse_function(cumfrac, k_low_z, c_low_z) 
    debiased = 10**(logfv_debiased) # Get 'fv'.
    fv_debiased[nonzero] = debiased
    
    return fv, fv_debiased



###############
''' This module does the abundance matching on a bin-by-bin basis,
    depending on the bins provided from the binning.py module.'''

def find_nearest(reference,values):
    ''' given an array (reference), return the indices of the closest indices for
        for each of the numbers in the array "values" '''
    i = np.zeros(len(values))
    for m,value in enumerate(values):
        i[m] = (np.abs(reference-value)).argmin()
    return i.astype(int)


def sort_data(D):
    ''' Sort data by a given column, to get a cumulative fraction for each index
        of an array'''

    D_i = np.arange(len(D))
    order = np.argsort(D)
    D_sorted = D[order]
    D_i_sorted = D_i[order]
    cumfrac = np.linspace(0,1,len(D))
    
    D_table = Table(np.array([D_i_sorted,D_sorted,cumfrac]).T,
		    names=('index','fv','cumfrac'))
    reorder = np.argsort(D_table['index'])
    D_table = D_table[reorder]
    
    for f in np.unique(D_table['fv']):
        f_select = D_table['fv'] == f
        D_table['cumfrac'][f_select] = np.mean(D_table['cumfrac'][f_select])
    
    return D_table


def debias_by_bin(full_data,vbins,zbins,question,answer):
    ''' Debias the data in a bin-by-bin basis'''
    
    # Get the raw and debiased fractions:
    fraction_column = question + '_' + answer + params.fraction_suffix
    data_column = full_data[fraction_column]
    debiased_column = np.zeros(len(data_column))

    for v in np.unique(vbins):
        select_v = vbins == v
        zbins_v = zbins[select_v] # redshift bins for this voronoi bin.
        
        data_v0 = data_column[(select_v) & (zbins == 1)]
        v0_table = sort_data(data_v0) # Reference array (ie. the low-z sample 
        # for each voronoi bin).

        for z in np.unique(zbins_v): # Now go through each bin in turn:
            select_z = zbins == z
            
            data_vz = data_column[(select_v) & (select_z)] 
            vz_table = sort_data(data_vz)
            
            # Now match to the low redshft sample:
            debiased_i = find_nearest(v0_table['cumfrac'],vz_table['cumfrac'])
            debiased_fractions = v0_table['fv'][debiased_i]
            debiased_column[(select_v) & (select_z)] = debiased_fractions
    
    debiased_column[data_column == 0] = 0 # Don't 'debias up' 0s.
    debiased_column[data_column == 1] = 1 # Don't 'debias down' the 1s.
    
    return debiased_column