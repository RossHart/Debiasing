import numpy as np
from astropy.table import Table

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


def debias(data,full_data,vbins,zbins,vbins_all,zbins_all,question,answer):
    ''' Debias the data in a bin-by-bin basis'''
    
    # Get the raw and debiased fractions:
    fraction_column = question + '_' + answer + '_weighted_fraction'
    data_column = data[fraction_column]
    all_data_column = full_data[fraction_column]
    debiased_column = np.zeros(len(all_data_column))

    for v in np.unique(vbins):
        select_v = vbins == v
        select_v_all = vbins_all == v
        zbins_v = zbins[select_v] # redshift bins for this voronoi bin.
        
        data_v0 = data_column[(select_v) & (zbins == 1)]
        v0_table = sort_data(data_v0) # Reference array (ie. the low-z sample 
        # for each voronoi bin).

        for z in np.unique(zbins_v): # Now go through each bin in turn:
            select_z = zbins == z
            select_z_all = zbins_all == z
    
            data_vz = data_column[(select_v) & (select_z)]
            vz_table = sort_data(data_vz)
            
            all_data_vz = all_data_column[(select_v_all) & (select_z_all)] 
            all_vz_table = sort_data(all_data_vz)
            
            # Now find the nearest value to each of the galaxies in the voronoi
            # bin:
            fv_i = find_nearest(vz_table['fv'],all_vz_table['fv'])
            all_vz_table['cumfrac'] = vz_table['cumfrac'][fv_i]
            
            # Now match to the low redshft sample:
            debiased_i = find_nearest(v0_table['cumfrac'],all_vz_table['cumfrac'])
            debiased_fractions = v0_table['fv'][debiased_i]
            debiased_column[(select_v_all) & (select_z_all)] = debiased_fractions
    
    debiased_column[data_column == 0] = 0 # Don't 'debias up' 0s.
    debiased_column[data_column == 1] = 1 # Don't 'debias down' the 1s.
    
    return debiased_column