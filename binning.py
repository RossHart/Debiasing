import numpy as np
import math
from voronoi_2d_binning import voronoi_2d_binning
from sklearn.neighbors import NearestNeighbors
from astropy.table import Table
import matplotlib.pyplot as plt

''' Code for binning the data in terms of Mr,R50 and z '''

def voronoi_binning(R50, Mr,n_rect_bins=500, n_per_voronoi_bin=5000,save=False):
    ''' Voronoi bin in terms of R50 and Mr'''
    
    rect_bin_val, R50_bin_edges, Mr_bin_edges = np.histogram2d(R50, Mr, n_rect_bins)

    rect_bins_table = Table(data=[R50_bin_edges, Mr_bin_edges],
                            names=['R50_bin_edges', 'Mr_bin_edges'])
    rect_bins_table.meta['nrectbin'] = n_rect_bins # add value for number of 
    # bins to the table. 
    
    # Get bin centres + number of bins:
    R50_bin_centres = 0.5*(R50_bin_edges[:-1] + R50_bin_edges[1:])
    Mr_bin_centres = 0.5*(Mr_bin_edges[:-1] + Mr_bin_edges[1:]) 
    n_R50_bins = len(R50_bin_centres)
    n_Mr_bins = len(Mr_bin_centres)

    # Get ranges:
    R50_bins_min, Mr_bins_min = map(np.min, (R50_bin_centres, Mr_bin_centres))
    R50_bins_max, Mr_bins_max = map(np.max, (R50_bin_centres, Mr_bin_centres))
    R50_bins_range = R50_bins_max - R50_bins_min
    Mr_bins_range = Mr_bins_max - Mr_bins_min
    
    # 'Ravel' out the coordinate bins (length of this array = n_bin*n_bin)
    R50_bin_coords = R50_bin_centres.repeat(n_rect_bins).reshape(n_rect_bins, n_rect_bins).ravel()
    Mr_bin_coords = Mr_bin_centres.repeat(n_rect_bins).reshape(n_rect_bins, n_rect_bins).T.ravel()

    # Only keep bins that contain a galaxy:
    signal = rect_bin_val.ravel() # signal=number of gals.
    ok_bin = (signal > 0).nonzero()[0]
    signal = signal[ok_bin]

    # Normalise x + y to be between 0 and 1:
    x = (R50_bin_coords[ok_bin] - R50_bins_min) / R50_bins_range
    y = (Mr_bin_coords[ok_bin] - Mr_bins_min) / Mr_bins_range

    # Voronoi_2d_binning aims for a target S/N
    noise = np.sqrt(signal)
    targetSN = np.sqrt(n_per_voronoi_bin)

    output = voronoi_2d_binning(x, y, signal, noise, targetSN, plot=0,
				quiet=1, wvt=True)
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = output

    vbin = np.unique(binNum)
    count = (sn**2).astype(np.int)
    
    count = count
    R50_vbin_mean = (xBar * R50_bins_range + R50_bins_min)
    Mr_vbin_mean = (yBar * Mr_bins_range + Mr_bins_min)
    nPixels = nPixels
    
    vbins_table = Table(data=[vbin, R50_vbin_mean, Mr_vbin_mean,
                              count, nPixels],
                        names=['vbin', 'R50', 'Mr', 
                               'count_gals', 'count_rect_bins'])
    vbins_table.meta['nrectbin'] = n_rect_bins
    vbins_table.meta['nperbin'] = n_per_voronoi_bin
    
    # Populate elements of the rectangular grid with
    # the voronoi bin indices and counts
    rect_bin_voronoi_bin = (np.zeros(np.product(rect_bin_val.shape), np.int)
			    - 1)
    rect_bin_voronoi_bin[ok_bin] = binNum
    rect_bin_count = np.zeros_like(rect_bin_voronoi_bin)
    rect_bin_count[ok_bin] = count
    
    rect_vbins_table = Table(data=[R50_bin_coords, Mr_bin_coords,
                             rect_bin_voronoi_bin],
                             names=['R50', 'Mr', 'vbin'])
    
    rect_bins_table.meta['nrectbin'] = n_rect_bins
    rect_bins_table.meta['nperbin'] = n_per_voronoi_bin
    
    if save == True:
        rect_bins_table.write(save_directory + 'rect_bins_table.fits',
                              overwrite=True)
        vbins_table.write(save_directory + 'vbins_table.fits',
                          overwrite=True)
        rect_vbins_table.write(save_directory + 'rect_vbins_table.fits',
                               overwrite=True)

    return (rect_bins_table, vbins_table, rect_vbins_table, Mr_bins_min,
            Mr_bins_range, R50_bins_min, R50_bins_range)
  
  
def redshift_binning(data,voronoi_bins,min_gals=100,coarse=False):
    ''' Bin each of the voronoi bins in terms in to bins of equal sample sizes, 
        each with >=min_gals galaxies'''
    
    redshift = data['REDSHIFT_1']
    z_bins = []

    for N in np.unique(voronoi_bins):
        inbin = voronoi_bins == N
        n_with_morph = np.sum(inbin)
        if coarse == True:
            n_zbins = 4 # Split into 4 bins per voronoi bin if 'coarse'
        else:
            n_zbins = n_with_morph/min_gals
        #n_zbins = 5
        z = redshift[inbin]
        z = np.sort(z)
        bin_edges = np.linspace(0, len(z)-1, n_zbins+1, dtype=np.int)
        z_edges = z[bin_edges]
        z_edges[0] = 0
        z_edges[-1] = 1
        
        z_bins.append(z_edges)
        
    return z_bins
  
  
def voronoi_assignment(data, rect_bins_table, rect_vbins_table,
                       Mr_bins_min, Mr_bins_range, R50_bins_min, R50_bins_range,
                       reassign=False):
    ''' Assign each of the galaxies a voronoi bin. If reassign is True, then
        even the galaxies not in the sample are given a voronoi bin'''
  
    # Load outputs from the 'voronoi_binning' module:
    R50_bin_edges = rect_bins_table['R50_bin_edges']
    Mr_bin_edges = rect_bins_table['Mr_bin_edges']
    n_R50_bins = len(R50_bin_edges) - 1
    n_Mr_bins = len(Mr_bin_edges) - 1
    
    # Load R50, Mr data for each galaxy:
    R50 = np.log10(data['PETROR50_R_KPC'])
    Mr = data['PETROMAG_MR']
    
    # Get the R50 and Mr bin for each galaxy in the sample
    R50_bins = np.digitize(R50, bins=R50_bin_edges).clip(1, n_R50_bins)
    Mr_bins = np.digitize(Mr, bins=Mr_bin_edges).clip(1, n_Mr_bins)

    # convert R50 and Mr bin indices to indices of bins
    # in the combined rectangular grid
    rect_bins = (Mr_bins - 1) + n_Mr_bins * (R50_bins - 1)

    # get the voronoi bin for each galaxy in the sample
    rect_bin_vbins = rect_vbins_table['vbin']
    voronoi_bins = rect_bin_vbins[rect_bins]
    
    if reassign is True: # Find nearest bin if none are available: 
        rect_bins_assigned = rect_vbins_table[rect_vbins_table['vbin'] != -1]
        R50_bin = rect_bins_assigned['R50']
        Mr_bin = rect_bins_assigned['Mr']
        
        x = (R50_bin - R50_bins_min) / R50_bins_range
        y = (Mr_bin - Mr_bins_min) / Mr_bins_range
        
        unassigned = voronoi_bins == -1
        R50u = (R50[unassigned] - R50_bins_min) / R50_bins_range
        Mru = (Mr[unassigned] - Mr_bins_min) / Mr_bins_range
        
        xy = (np.array([R50u,Mru])).T
        xy_ref = (np.array([x,y])).T
        
        nbrs = NearestNeighbors(n_neighbors=1,algorithm='ball_tree').fit(xy_ref,xy)
        d,i = nbrs.kneighbors(xy)
        i = i.squeeze()
        vbins_reassigned = rect_bins_assigned['vbin'][i]
        voronoi_bins[voronoi_bins == -1] = vbins_reassigned
    
    return voronoi_bins
  
  
def redshift_assignment(data,vbins,zbin_ranges):
    ''' Assign a redshift bin to each galaxies (using the ranges from the 
        redshift_binning function'''
    
    zbins = np.zeros(len(data))
    
    for v in (np.unique(vbins)):
        z_range = zbin_ranges[v]
        v_data = data[vbins == v]['REDSHIFT_1']
        z_bin = np.digitize(v_data,bins=z_range)
        zbins[vbins == v] = z_bin
        
    return zbins
  
  
def bin_data(data,full_data,question,answer,n_vbins=30,signal=50):
    ''' This function applies all of the binning (voronoi and then redshift):'''
  
    raw_column = data[question + '_' + answer]
    fv_nonzero = raw_column > 0 # Select only the non-zero data to add to the 
    # 'signal' for each bin.
    R50 = data['PETROR50_R_KPC'][fv_nonzero]
    Mr = data['PETROMAG_MR'][fv_nonzero]
    
    npv = np.sum(fv_nonzero)/(n_vbins) # Number of galaxies in each voronoi bin.
    nrb = math.floor(np.sqrt(np.sum(fv_nonzero))) # If we have too many
    # rectangular bins, the code doesn't work correctly?
    
    # Get voronoi bin values in Mr,R50 space:
    (rect_bins_table,vbins_table,rect_vbins_table,
     Mr_bins_min,Mr_bins_range,R50_bins_min,
     R50_bins_range) = voronoi_binning(np.log10(R50),
                                       Mr,
                                       n_rect_bins=nrb,
                                       n_per_voronoi_bin=npv)
                                       
    
    # Now assign each of the galaxies in the sample to a voronoi bin:
    vbins = voronoi_assignment(data[fv_nonzero],rect_bins_table,
                               rect_vbins_table,Mr_bins_min,
                               Mr_bins_range, R50_bins_min, R50_bins_range)
    
    # Get zbin ranges:
    zbin_ranges = redshift_binning(data[fv_nonzero],vbins,min_gals=signal)
    zbin_ranges_coarse = redshift_binning(data[fv_nonzero],vbins,min_gals=None,
					  coarse=True)
    
    # redo the voronoi binning, applying it to all of the data, not just those
    # that contribute to the 'signal':
    vbins = voronoi_assignment(data, rect_bins_table, rect_vbins_table,
                               Mr_bins_min, Mr_bins_range, R50_bins_min, 
                               R50_bins_range, reassign=True)
    zbins = redshift_assignment(data,vbins,zbin_ranges)
    zbins_coarse = redshift_assignment(data,vbins,zbin_ranges_coarse)
    
    # Assign each of the galaxies in the FULL sample a voronoi and redshift bin:
    vbins_all = voronoi_assignment(full_data, rect_bins_table, rect_vbins_table,
                                   Mr_bins_min, Mr_bins_range, R50_bins_min, 
                                   R50_bins_range, reassign=True)
    zbins_all = redshift_assignment(full_data,vbins,zbin_ranges)
    zbins_coarse_all = redshift_assignment(full_data,vbins,zbin_ranges_coarse)
    
    # Print the number of voronoi bins, and the mean number of z-bins in each:
    N_v = np.unique(vbins)
    N_z = []
    for v in N_v:
        zbins_v = zbins[vbins == v]
        N_z.append(np.max(zbins_v))    
    print('{} voronoi bins'.format(len(N_v)))
    print('{} redshift bins per voronoi bin'.format(np.mean(N_z)))
    # Try to have a 'caveat' if there aren't enough bins???? -->
    if np.mean(N_z) < 2:
        zbins = zbins_coarse.copy()
        zbins_all = zbins_coarse_all.copy()
        print('Using fixed width bins')

    return vbins,zbins,zbins_coarse,vbins_all,zbins_all,zbins_coarse_all,vbins_table
