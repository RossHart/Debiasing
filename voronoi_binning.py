import numpy as np
from astropy.io import fits
from astropy.table import Table
from voronoi_2d_binning import voronoi_2d_binning
import os
import params

import matplotlib as mpl
from matplotlib import pyplot as plt
# better-looking plots
#from prefig import Prefig
#Prefig()
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.figsize'] = (10.0, 8)
plt.rcParams['font.size'] = 18
mpl.ticker.AutoLocator.default_params['nbins'] = 5
mpl.ticker.AutoLocator.default_params['prune'] = 'both'

source_directory = params.source_directory
save_directory = params.numpy_save_directory
full_sample = params.full_sample

os.mkdir(save_directory) if os.path.isdir(save_directory) is False else None

# This function should select the galaxies with >=5 arm number answers and p>0.5. 
# (ie. a 'robust' selection)
def select_data_arm_number(data,N_cut=5,p_cut=0.5
                           ,p_questions=['t01_smooth_or_features','t02_edgeon'
                                         ,'t04_spiral']
                           ,p_answers=['a02_features_or_disk','a05_no',
                                       'a08_spiral']): 
    
    strings = [p_questions[s] + '_' + p_answers[s] for s in range(len(p_answers))]
    p_strings = [s + '_debiased' for s in strings]
    N_string = strings[-1] + '_count'

    p_values = np.ones(len(data))
    N_values = np.ones(len(data))

    for Q in p_strings:
        p_values = p_values*data[Q]
    N_values = N_values*data[N_string]

    select = (p_values > p_cut) & (N_values >= N_cut)
    data = data[select]
    
    return data
 

def voronoi_binning(R50, Mr, n_rect_bins=500, n_per_voronoi_bin=5000,save=True):
    
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
    
    # 'Ravel' out the coordinate bins (.'. length=n_bin*n_bin)
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

    output = voronoi_2d_binning(x, y, signal, noise, targetSN, plot=0, quiet=1, wvt=True)
    binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale = output

    vbin = np.unique(binNum)
    count = (sn**2).astype(np.int) # N_gals for each voronoi bin.
    R50_vbin_mean = xBar * R50_bins_range + R50_bins_min
    Mr_vbin_mean = yBar * Mr_bins_range + Mr_bins_min
    
    vbins_table = Table(data=[vbin, R50_vbin_mean, Mr_vbin_mean,
                              count, nPixels],
                        names=['vbin', 'R50', 'Mr', 
                               'count_gals', 'count_rect_bins'])
    vbins_table.meta['nrectbin'] = n_rect_bins
    vbins_table.meta['nperbin'] = n_per_voronoi_bin

    # Populate elements of the rectangular grid with
    # the voronoi bin indices and counts
    rect_bin_voronoi_bin = np.zeros(np.product(rect_bin_val.shape), np.int) - 1
    rect_bin_voronoi_bin[ok_bin] = binNum
    rect_bin_count = np.zeros_like(rect_bin_voronoi_bin)
    rect_bin_count[ok_bin] = count
    
    rect_vbins_table = Table(data=[R50_bin_coords, Mr_bin_coords,
                             rect_bin_voronoi_bin],
                             names=['R50', 'Mr', 'vbin'])
    rect_bins_table.meta['nrectbin'] = n_rect_bins
    rect_bins_table.meta['nperbin'] = n_per_voronoi_bin
    
    if save == True:
        rect_bins_table.write(save_directory + 'rect_bins_table.fits', overwrite=True)
        vbins_table.write(save_directory + 'vbins_table.fits', overwrite=True)
        rect_vbins_table.write(save_directory + 'rect_vbins_table.fits', overwrite=True)
    
    # rect_bins_table: contains all of the bin edges (len=N_bins)
    # rect_vbins_table: has bin centre values + assigned v-bin (len=N_bins**2)
    # vbins_table: for each bin, contains the number of gals, Mr+R50 mean
    # values + the number of rectangular bins it is made up of (len=N_v-bins)
    return rect_bins_table, vbins_table, rect_vbins_table

 
def voronoi_assignment(data, rect_bins_table, rect_vbins_table):
    R50_bin_edges = rect_bins_table['R50_bin_edges']
    Mr_bin_edges = rect_bins_table['Mr_bin_edges']
    n_R50_bins = len(R50_bin_edges) - 1
    n_Mr_bins = len(Mr_bin_edges) - 1
    
    R50 = np.log10(data['PETROR50_R_KPC'])
    Mr = data['PETROMAG_MR']
    
    # get the R50 and Mr bin for each galaxy in the sample
    R50_bins = np.digitize(R50, bins=R50_bin_edges).clip(1, n_R50_bins)
    Mr_bins = np.digitize(Mr, bins=Mr_bin_edges).clip(1, n_Mr_bins)

    # convert R50 and Mr bin indices to indices of bins
    # in the combined rectangular grid
    rect_bins = (Mr_bins - 1) + n_Mr_bins * (R50_bins - 1)

    # get the voronoi bin for each galaxy in the sample
    rect_bin_vbins = rect_vbins_table['vbin']
    voronoi_bins = rect_bin_vbins[rect_bins]
    
    return voronoi_bins # Gives each galaxy a voronoi bin.

# Load the required data:
all_data = fits.getdata(source_directory+full_sample,1)
all_data = Table(all_data)
#spiral_data = select_data_arm_number(all_data) # keep spirals for binning.

R50, Mr = [all_data[c] for c in ['PETROR50_R_KPC','PETROMAG_MR']]
R50 = np.log10(R50) # binning performed in log10(R50)

rect_bins_table, vbins_table, rect_vbins_table = voronoi_binning(R50, Mr)
voronoi_bins = voronoi_assignment(all_data, rect_bins_table, rect_vbins_table)

# Now make some plots to show the voronoi bins:

# First check if the directory exists:
os.mkdir('figures/voronoi_binning/') if os.path.isdir('figures/voronoi_binning/') is False else None
  
# Bin count histogram:
count = vbins_table['count_gals']
count = count[count > 0]
plt.hist(count)
_ = plt.vlines([vbins_table.meta['nperbin']], *plt.axis()[2:])
plt.xlabel('$N_{gal}$')
plt.ylabel('$N_{bins}$')

plt.savefig('figures/voronoi_binning/voronoi_histogram.pdf',dpi=100)

# Map of voronoi bins:
vbin = rect_vbins_table['vbin']
R50_bin_coords = rect_vbins_table['R50']
Mr_bin_coords = rect_vbins_table['Mr']

# Sort bins by relative position to avoid neighbouring bins having the same colour.
relative_r = [((T-T.min())/(T.max()-T.min())) for T in [np.log10(vbins_table['R50']),-vbins_table['Mr']]]
relative_r = relative_r[0] + relative_r[1]
r_sort = np.argsort(relative_r)

for N in np.unique(vbin[vbin >= 0])[r_sort]:
    inbin = vbin == N
    plt.plot(10**R50_bin_coords[inbin], Mr_bin_coords[inbin], '.')
plt.ylabel(r"$M_r$")
plt.xlabel(r"$R_{50}$ (kpc)")
plt.xscale('log')
_ = plt.axis((0.5, 60, -16, -25))

plt.savefig('figures/voronoi_binning/voronoi_bins.pdf',dpi=100)

# Finally save in FITS format for easy future access:
bin_table = Table()
bin_table['voronoi_bin'] = voronoi_bins
bin_table.write(save_directory + 'bins.fits',overwrite=True)

print('Done! Voronoi bin values saved.')