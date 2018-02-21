data_file = 'GAMA_fits/gama_to_debias.fits'
output_directory = 'output_files/'

logistic_bounds = ((0.5,10), (-10, 10))
exponential_bounds = ((10**(-5),10),(10**(-5),10)) 

log_fv_range = (-1.5,0.01)
count_suffix = '_total_raw'
fraction_suffix = '_frac'

Mr_column = 'absmag_r'
R50_column = 'GALRE_r_kpc'
z_column = 'Z_TONRY'

n_voronoi = 30
n_per_z = 50
low_signal_limit = 100
clip_percentile = 5 

volume_redshift_limits = (0.03,0.10)
survey_mag_limit = 19.8