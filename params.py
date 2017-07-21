data_file = 'GAMA_fits/gama09_less.fits'
output_directory = 'output_files/'

#question_dictionary = 'questions.pickle'

logistic_bounds = ((0.5,10), (-10, 10))
exponential_bounds = ((10**(-5),10),(10**(-5),10)) 

log_fv_range = (-1.5,0.01)
count_suffix = ''
fraction_suffix = '_fraction'

Mr_column = 'absmag_r'
R50_column = 'GALRE_r_kpc'
z_column = 'Z_TONRY'

n_voronoi = 10
n_per_z = 10
low_signal_limit = 40
clip_percentile = 1

volume_redshift_limits = (0.02,0.10)
survey_mag_limit = 19.4