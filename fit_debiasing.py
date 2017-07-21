import numpy as np
import time
from scipy.optimize import minimize,curve_fit
from astropy.table import Table
import matplotlib.pyplot as plt

''' Now have a code for doing debiasing on a fit-basis to each of the bins: '''

def make_fit_setup(function_dictionary,key):
    ''' For a given "key", get the fit setup (function, inverse, bounds etc.)'''
    fit_setup = {}
    fit_setup['func'] = function_dictionary['func'][key]
    fit_setup['bounds'] = function_dictionary['bounds'][key]
    fit_setup['p0'] = function_dictionary['p0'][key]
    fit_setup['inverse'] = function_dictionary['i_func'][key]
    return fit_setup
  
  
def get_fit_setup(fit_setup):
    ''' Get a fit setup, given a dictionary outputted form the make_fit_setup
        function '''
    func = fit_setup['func']
    p0 = fit_setup['p0']
    bounds = fit_setup['bounds']
    
    return func, p0, bounds
  
  
def chisq_fun(p, f, x, y):
    ''' chisquare function'''
    return ((f(x, *p) - y)**2).sum()


def get_best_function(data,vbins,zbins,function_dictionary
                      ,question,answer,min_log_fv):
    ''' Choose the function that captures the overall best fit using "coarse" 
        bins'''

    fv_all = np.sort(data[question + '_' + answer + '_fraction'])
    fv_nonzero = fv_all != 0
    cf = np.linspace(0,1,len(fv_all))
    x,y = [np.log10(fv_all[fv_nonzero]),cf[fv_nonzero]] # x and y values are 
    # log(fv) vs. cumulative fraction.
    
    x_fit = np.log10(np.linspace(10**(min_log_fv), 1, 100)) # Equally space
    # in log space.
    indices = np.searchsorted(x,x_fit)
    y_fit = y[indices.clip(0, len(y)-1)]
    
    # Save the output k, c and overall chisquare values in the following
    # lists: 
    chisq_tot = np.zeros(len(function_dictionary['func'].keys()))
    k_tot = np.zeros(len(function_dictionary['func'].keys()))
    c_tot = np.zeros(len(function_dictionary['func'].keys()))
    
    for n,key in enumerate(function_dictionary['func'].keys()):
        # Overall data fitting for each of the functions in the dictionary:
        fit_setup = make_fit_setup(function_dictionary,key)
        func = fit_setup['func']
        p0 = fit_setup['p0']
        bounds = fit_setup['bounds']
        
        res =  minimize(chisq_fun, p0,
                        args=(func,x_fit,y_fit),
                        bounds=bounds,method='SLSQP')
        function_dictionary['p0'][key] = x # Best fit value
        
        if res.success == False:
            # Try again if no fit was found:
            print('Failed to minimise total dataset')
            popt,pcov = curve_fit(func,x_fit,y_fit,maxfev=10**5) 
            # Try to use scipy.optimize.curve_fit to at least get some 'starting'
            # values.
            res =  minimize(chisq_fun, popt,
                        args=(func,x_fit,y_fit),
                        bounds=bounds,method='SLSQP')
            if res.success == False:
                print('Still failed to minimise!')    
        
        fit_vbin_results = fit_vbin_function(data,vbins,zbins,fit_setup,
                                             question,answer,min_log_fv,
                                             clip=None)

        finite_chisq = np.isfinite(fit_vbin_results['chi2nu'])
        
        # Deal with chisq nans here:
        chisq = np.sum((fit_vbin_results['chi2nu'][finite_chisq])
                       /(np.sum(finite_chisq))) # Mean chisquare value.
        k = np.mean(fit_vbin_results['k'][finite_chisq])
        c = np.mean(fit_vbin_results['c'][finite_chisq])
        chisq_tot[n] = chisq
        k_tot[n] = k
        c_tot[n] = c
        print('chisq({}) = {}'.format(function_dictionary['label'][key],chisq))
    
    # Finally, compare the chisq values and choose the best function to proceed:
    n = np.argmin(chisq_tot)
    keys = [key for key in function_dictionary['func'].keys()]
    key = keys[n]
    fit_setup = make_fit_setup(function_dictionary,key) # Choose function with
    # the lowest chisquare value fit to the coarse bins.
       
    return fit_setup 


def fit_vbin_function(data, vbins, zbins, fit_setup,
                      question,answer,min_log_fv,
                      kc_fit_results=None,
                      even_sampling=True,clip=2):
    ''' Fit a function to each of the voronoi and zbins (given by vbins and 
        and zbins).'''
    
    start_time = time.time() # Check how long the fitting takes:
    
    min_fv = 10**(min_log_fv)
    
    redshift = data['Z_TONRY'] # redshifts
    fv = question + '_' + answer + '_fraction'# raw column name.
    
    if kc_fit_results is not None:
        kcfunc, kparams, cparams, lparams,kclabel = kc_fit_results
    
    # Set up the list to write the parameters in to:
    param_data = []
    
    # Get parameters from the given fit_setup:
    bounds = fit_setup['bounds']
    p0 = fit_setup['p0']
    func = fit_setup['func']
    
    vbins_unique = []
    
    for v in np.unique(vbins):
        if np.sum(vbins == v) < 50:
            print('vbin {} has too low signal!'.format(v))
        else:
            vbins_unique.append(v)
    
    for v in vbins_unique:
        # Get the data for a given voronoi bin:
        vselect = vbins == v
        data_v = data[vselect]
        zbins_v = zbins[vselect]
        zbins_unique = np.unique(zbins_v)
        for z in zbins_unique:
            data_z = data_v[zbins_v == z]
            n = len(data_z)
            #print(D[fv])
            D = data_z[[fv]]
            D.sort(fv)
            D['cumfrac'] = np.linspace(0, 1, n)
            if (D[fv] > 0).sum() > 0:
                D = D[D[fv] > min_fv]
                D['log10fv'] = np.log10(D[fv])
            
                if even_sampling:
            # Evenly sample in log(fv):
                    D_fit_log10fv = np.log10(np.linspace(10**(min_log_fv), 1, 100))
                    D = D[(D['log10fv'] > min_log_fv)]
                    indices = np.searchsorted(D['log10fv'], D_fit_log10fv)
                    D_fit = D[indices.clip(0, len(D)-1)]
                else:
                    D_fit = D[D['log10fv'] > min_log_fv]

                res = minimize(chisq_fun, p0,
                           args=(func,
                                 D_fit['log10fv'].astype(np.float64),
                                 D_fit['cumfrac'].astype(np.float64)),
                                 bounds=bounds, method='SLSQP')
            
                p = res.x # Best fit
                chi2nu = res.fun / (n - len(p)) 
            
            #if (v == 1) & (z == 2):
                #plt.plot(D_fit['log10fv'].astype(np.float64),
                         #D_fit['cumfrac'].astype(np.float64),lw=3,color='k')
                #x_guide = np.linspace(-2,0,100)
                #plt.plot(x_guide,func(x_guide,*p),color='g',lw=2)
            
                if res.success == False:
                    print('Fit not found for z={},v={}'.format(z,v))
                
                means = [data_z['absmag_r_stars'].mean(),
                         np.log10(data_z['GALRE_r_kpc']).mean(),
                         data_z['Z_TONRY'].mean()] # Mean values for each bin.

            #if len(p) < 2:
                #p = np.array([p[0], 10])

                param_data.append([v,z] + means + p.tolist() + [chi2nu]) # Make
            # final table.                         
            
    fit_vbin_results = Table(rows=param_data,
                             names=('vbin','zbin', 'Mr','R50', 
                                    'redshift', 'k', 'c', 'chi2nu'))
                                    
    print('All bins fitted! {}s in total'.format(time.time()-start_time))
    
    # Include a column of 'outlier' fittings:
    if clip != None:
        k_values = fit_vbin_results['k']
        k_mean = np.mean(k_values)
        k_std = np.std(k_values)
        k_range = [k_mean-clip*k_std,k_mean+clip*k_std]
        
        c_values = fit_vbin_results['c']
        c_mean = np.mean(c_values)
        c_std = np.std(c_values)
        c_range = [c_mean-clip*c_std,c_mean+clip*c_std]
        
        select = ((k_values > k_range[0]) & (k_values < k_range[1]) 
                  & (c_values > c_range[0]) & (c_values < c_range[1]))

        fit_vbin_results['in_2sigma'] = select
    else:
        fit_vbin_results['in_2sigma'] = np.ones(len(fit_vbin_results))
    
    return fit_vbin_results


def normalise(x):
    return (x - x.mean())/x.std()
  
  
def normalise_tot(x,mean,std):
    return (x - mean)/std


def fit_mrz(d, f_k, f_c, clip=None):
    ''' Fit a linear function of M, R and z to k and c '''
    
    dout = d.copy()
    
    kparams = []
    cparams = []
    dout['kf'] = np.zeros(len(d))
    dout['cf'] = np.zeros(len(d))

    x = np.array([d[c] for c in ['Mr', 'R50', 'redshift']], np.float64)
    good = dout['in_2sigma'] == 1 # Only use the 'good' fits (w/o outliers).
    x_good = ((x.T)[good]).T 
    k = d['k'].astype(np.float64)[good]
    c = d['c'].astype(np.float64)[good]
    
    # Set limits of the functions here:
    kmin = d['k'][good].min() 
    kmax = d['k'][good].max() 
    cmin = d['c'][good].min()
    cmax = d['c'][good].max()
    
    # Fit to the data:
    kp, kc = curve_fit(f_k, x_good, k, maxfev=100000)
    cp, cc = curve_fit(f_c, x_good, c, maxfev=100000)  
    kres = f_k(x_good, *kp) - k # k residuals
    knormres = normalise(kres) # normalised k residuals
    cres = f_c(x_good, *cp) - c # c residuals
    cnormres = normalise(cres) # normalised c residuals
    
    bins = np.linspace(-3,3,15) 
    
    # Remove the +-2sigma fits, and then redo the fitting:
    if clip != None:
        clipped = ((np.absolute(knormres) < clip) & (np.absolute(cnormres) < clip))# 'clip' sigma clipping
        kp, kc = curve_fit(f_k, ((x_good.T)[clipped]).T, k[clipped], maxfev=100000)
        cp, cc = curve_fit(f_c, ((x_good.T)[clipped]).T, c[clipped], maxfev=100000)
        
    dout['kf'] = f_k(x, *kp) # Continuously fitted k values
    dout['cf'] = f_c(x, *cp) # Continuously fitted c values
        
    kparams.append(kp) # Best fits to the data.
    cparams.append(cp)

    return kparams, cparams, dout, kmin, kmax, cmin, cmax


def get_term(constant,var,t='linear',negative='False'):
    ''' Get a term for a part of an equation. If -ve is True, then we 
        take the -ve value (eg. Mr is usually -ve, so wouldn't have a 
        log solution '''
    if negative == True:
        var = -var
    
    if t == 'log':
        term = constant*np.log10(var)
    elif t == 'linear':
        term = constant*var
    elif t == 'exp':
        term = constant*(10**(var))
    
    return term
    

def get_func(M_dependence,R_dependence,z_dependence):
    ''' Get a function (log/linear/exp) '''
    def kcfunc(x,A0,AM,AR,Az):
        M_term = get_term(AM,x[0],M_dependence,negative='True')
        R_term = get_term(AR,x[1],R_dependence)
        z_term = get_term(Az,x[2],z_dependence)
        return A0 + M_term + R_term + z_term
    
    return kcfunc
  
  
def get_kc_functions(fit_vbin_results):
    ''' Cycle through M, R50 and z dependences, to find the best overall
        function '''  
  
    # Loop through M,R and z functional forms:
    M_ds = ['log','linear','exp']
    R_ds = ['log','linear','exp']
    z_ds = ['log','linear','exp']
    
    c_residuals = np.zeros(len(M_ds)*len(R_ds)*len(z_ds))
    k_residuals = np.zeros(len(M_ds)*len(R_ds)*len(z_ds))
    i = 0
    M_dependences = []
    R_dependences = []
    z_dependences = []
    # Only keep correctly fitted values.
    finite_select = ((np.isfinite(fit_vbin_results['k'])) & 
                     (np.isfinite(fit_vbin_results['c'])))
    fit_vbin_results_finite = fit_vbin_results[finite_select] 
    fit_vbin_results.write('vbin_results.fits',overwrite=True)
    # Find the best functions for fitting the data:
    for M_dependence in M_ds:
        for R_dependence in R_ds:
            for z_dependence in z_ds:

                kcfunc = get_func(M_dependence,R_dependence,z_dependence)
                (kparams, cparams, dout, 
                 kmin, kmax, cmin, cmax) = fit_mrz(fit_vbin_results_finite, 
                                                   kcfunc, kcfunc,clip=None)
               
                k_fit_residuals = (dout['kf']-dout['k'])**2
                k_fit_residuals = k_fit_residuals[np.isfinite(k_fit_residuals)]
                c_fit_residuals = (dout['cf']-dout['c'])**2
                c_fit_residuals = c_fit_residuals[np.isfinite(c_fit_residuals)]
                k_residuals[i] = np.mean(k_fit_residuals)
                c_residuals[i] = np.mean(c_fit_residuals)
                i = i+1
            
                M_dependences.append(M_dependence)
                R_dependences.append(R_dependence)
                z_dependences.append(z_dependence)
          
    k_residuals[np.isfinite(k_residuals) == False] = 10**8
    c_residuals[np.isfinite(c_residuals) == False] = 10**8
    
    # Choose best functions:
    best_k = np.argmin(k_residuals) 
    best_c = np.argmin(c_residuals)
    best_M_k = M_dependences[best_k]
    best_R_k = R_dependences[best_k]
    best_z_k = z_dependences[best_k]
    best_M_c = M_dependences[best_c]
    best_R_c = R_dependences[best_c]
    best_z_c = z_dependences[best_c]

    k_func = get_func(best_M_k,best_R_k,best_z_k)
    c_func = get_func(best_M_c,best_R_c,best_z_c)
    print('Selected functions:------')
    print('k: {}(M),{}(R),{}(z)'.format(best_M_k,best_R_k,best_z_k))
    print('c: {}(M),{}(R),{}(z)'.format(best_M_c,best_R_c,best_z_c))
    
    return k_func,c_func
  
  
def function_inversion(value,func,k,kb,c,cb):
    ''' Function for use when function has no mathematical inverse'''
    xg = np.log10(np.linspace(0.01,1,100))
    low_z_values = func(xg,kb,cb,lb)
    high_z_value = func(value,k,c,l)
    i = (np.abs(low_z_values-high_z_value)).argmin()
    x = xg[i]
    return x
  
  
def debias(data, z_base, k_func,c_func, kparams, cparams,
           question,answer,kmin,kmax,cmin,cmax,fit_setup):
    ''' Given a functional form, now debias all of the data'''
    
    fv_col = question + '_' + answer + '_fraction'
    fv = data[fv_col]
    debiased = np.zeros(len(fv))
    fv_nonzero = fv > 0
    log10fv = np.log10(np.asarray(fv[fv_nonzero]))
    func, _, _ = get_fit_setup(fit_setup)
    i_func = fit_setup['inverse']
    bounds = fit_setup['bounds']

    d  = data[fv_nonzero] # Only keep the non-zero data   
    x = np.array([d['absmag_r_stars'],
                 np.log10(d['GALRE_r_kpc']),
                 d['Z_TONRY']], np.float64) # Parameter array.
    xb  = x.copy()
    xb[-1] = z_base # Low redshift equivalent of the parameter array.
        
    k = k_func(x, *kparams[0])
    c = c_func(x, *cparams[0])
    k[k < kmin] = kmin
    k[k > kmax] = kmax
    c[c < cmin] = cmin
    c[c > cmax] = cmax

    #create version of x with all redshifts at z_base
    kb = k_func(xb, *kparams[0])
    cb = c_func(xb, *cparams[0])
    kb[kb < kmin] = kmin
    kb[kb > kmax] = kmax
    cb[cb < cmin] = cmin
    cb[cb > cmax] = cmax
        
    cumfrac = func(log10fv, k, c) # Get the fitted 'y' value.
    log10fv_debiased = i_func(cumfrac, kb, cb) # Find corresponding 'log(fv)'.    
    fv_debiased = 10**(log10fv_debiased) # Get 'fv'.
    debiased[fv_nonzero] = fv_debiased

    return debiased


def debias_by_fit(data,full_data,vbins,zbins,zbins_coarse,question,
                  answer,function_dictionary,min_log_fv,coarse=False):
    ''' Find the best function, fit to the data, and then return the debiased 
        values '''
        
    low_z = 0.03 # debias down to this redshift.
    #low_z_select = fit_vbin_results['zbin'] == 1
    #low_z = np.mean(fit_vbin_results['redshift'][low_z_select])

    if coarse == True: # can choose whether to coarsely bin here.
        zbins = zbins_coarse.copy()
    
    # Firstly, choose the best function (logistic, exp.power etc.):
    fit_setup = get_best_function(data,vbins,zbins_coarse,function_dictionary,
                                  question,answer,min_log_fv)
    
    # Fit to each of the bins in turn (w. a 2sigma clipping):
    fit_vbin_results = fit_vbin_function(data, vbins, zbins, fit_setup,
                                         question,answer,min_log_fv,clip=2)
    
    # Get the best k or c functional form:
    k_func,c_func = get_kc_functions(fit_vbin_results) 
    
    # Now get the the k and c functional values (w. a 2sigma clipping):
    kparams, cparams,dout,kmin, kmax, cmin, cmax = fit_mrz(fit_vbin_results,
                                                           k_func,c_func,
                                                           clip=2)
    
    # Use the functions calculated above to debias all of the data:
    debiased_fit = debias(full_data,low_z, k_func,c_func, kparams, cparams,
                          question,answer,kmin,kmax,cmin,cmax,fit_setup)
    
    return debiased_fit,dout,fit_setup,zbins,fit_vbin_results