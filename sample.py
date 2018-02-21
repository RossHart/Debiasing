from astropy.table import Table, column
from luminosities_magnitudes_and_distances import mag_to_Mag
import dictionaries
import math
import numpy as np
import os
import params
from sklearn.neighbors import NearestNeighbors
from voronoi_2d_binning import voronoi_2d_binning

class Sample():
    
    def __init__(self,full_data,questions,params,
                 question='shape',p_cut=0.5,N_cut=5,use_normalised=False):
        
        full_data['logR50'] = np.log10(full_data[params.R50_column])
    
        if use_normalised == True:
            suffix = '_debiased_normalised'
        else:
            suffix = '_debiased'
        previous_q = questions[question]['pre_questions']
        previous_a = questions[question]['pre_answers']
    
        if previous_q is not None:
            p_col = np.ones(len(full_data))
            for q, a in zip(previous_q, previous_a):
                p_col = p_col*(full_data[q + '_' + a + suffix])
            #N_col = (full_data[q + '_' + a + params.count_suffix])
            N_col = full_data[question + params.count_suffix]
            select = (p_col > p_cut) & (N_col >= N_cut)
            less_data = full_data[select]
            print('{}/{} galaxies with p>{} and N>={}.'.format(len(less_data),
                                                               len(full_data),
                                                                 p_cut,N_cut))
        else:
            less_data = full_data.copy()
            select = np.ones(len(full_data),dtype=np.bool)
            print('Primary question, so all {} galaxies used.'.format(
            	                                              len(less_data)))
    
        self.all_data = full_data
        self.less_data = less_data
        self.params = params
        self.p_mask = select
        return None

    def volume_limited_sample(self):
        lower_z_limit, upper_z_limit = self.params.volume_redshift_limits
        mag_limit = self.params.survey_mag_limit
        Mag_limit = mag_to_Mag(mag_limit,upper_z_limit)
        in_vl = np.all([self.all_data[self.params.z_column] >= lower_z_limit,
                        self.all_data[self.params.z_column] <= upper_z_limit,
                        self.all_data[self.params.Mr_column] <= Mag_limit],
                        axis=0)
        
        self.in_volume_limit = in_vl
        self.Mag_limit = Mag_limit
        return in_vl

    def z_slices(self):
    	in_vl = self.volume_limited_sample()
    	lower_z_limit, upper_z_limit = self.params.volume_redshift_limits
    	
    	low_z_limits = [lower_z_limit,
    	                lower_z_limit+(upper_z_limit-lower_z_limit)/10]

    	upper_z_limits =[upper_z_limit-(upper_z_limit-lower_z_limit)/10,
    	                 upper_z_limit]

    	in_low_z = np.all([self.all_data[self.params.z_column] 
    		                                >= low_z_limits[0],
    		               self.all_data[self.params.z_column] 
    		                                <= low_z_limits[1]],axis=0)

    	in_high_z = np.all([self.all_data[self.params.z_column] 
    		                               >= upper_z_limits[0],
    		                self.all_data[self.params.z_column] 
    		                               <= upper_z_limits[1]],axis=0)

    	return (in_vl*in_low_z).astype(bool), (in_vl*in_high_z).astype(bool)

class Bins():
    def __init__(self,sample,params,questions,
                 question='shape',answer='smooth'):
        self.n_voronoi = params.n_voronoi
        self.n_per_z = params.n_per_z
        self.Mr = sample.less_data[params.Mr_column]
        self.R50 = sample.less_data['logR50']
        self.z = sample.less_data[params.z_column]
        self.Mr_all = sample.all_data[params.Mr_column]
        self.R50_all = sample.all_data['logR50']
        self.z_all = sample.all_data[params.z_column]
        self.n_min = params.low_signal_limit
        
        self.params = params
        self.sample = sample
        self.question = question
        self.answer = answer
        
        save_directory = params.output_directory + question + answer
        if os.path.isdir(params.output_directory + '/' + question) is False:
            os.mkdir(params.output_directory + '/' + question)
        if os.path.isdir(save_directory) is False:
            os.mkdir(save_directory)
        self.save_directory = save_directory
        return None
    
    def voronoi_bin(self,save=False):
        
        Mr = self.Mr
        R50 = self.R50
        n_gal = len(Mr)
        n_rect_bins = (math.sqrt(n_gal))/2
        n_rect_bins = int(math.floor(n_rect_bins))
        n_per_voronoi_bin = n_gal/self.params.n_voronoi
        
        rect_bin_val, R50_bin_edges, Mr_bin_edges = np.histogram2d(R50, Mr, 
        	                                                      n_rect_bins)

        rect_bins_table = Table(data=[R50_bin_edges, Mr_bin_edges],
                                names=['R50_bin_edges', 'Mr_bin_edges'])
    
        # Get bin centres + number of bins:
        R50_bin_centres = 0.5*(R50_bin_edges[:-1] + R50_bin_edges[1:])
        Mr_bin_centres = 0.5*(Mr_bin_edges[:-1] + Mr_bin_edges[1:]) 
        n_R50_bins = len(R50_bin_centres)
        n_Mr_bins = len(Mr_bin_centres)

        # Get ranges:
        R50_bins_min, Mr_bins_min = map(np.min, (R50_bin_centres, 
        	                                     Mr_bin_centres))
        R50_bins_max, Mr_bins_max = map(np.max, (R50_bin_centres, 
        	                                     Mr_bin_centres))
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
                                    quiet=True, wvt=True)
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
        rect_bin_voronoi_bin = (np.zeros(np.product(rect_bin_val.shape), np.int)- 1)
        rect_bin_voronoi_bin[ok_bin] = binNum
        #rect_bin_count = np.zeros_like(rect_bin_voronoi_bin)
        #rect_bin_count[ok_bin] = count
    
        rect_vbins_table = Table(data=[R50_bin_coords, Mr_bin_coords,
                                 rect_bin_voronoi_bin],
                                 names=['R50', 'Mr', 'vbin'])
    
        rect_bins_table.meta['nrectbin'] = n_rect_bins
        rect_bins_table.meta['nperbin'] = n_per_voronoi_bin
    
        if save == True:
            rect_bins_table.write(self.save_directory + '/rect_bins_table.fits',
                                  overwrite=True)
            vbins_table.write(self.save_directory + '/vbins_table.fits',
                              overwrite=True)
            rect_vbins_table.write(self.save_directory + '/rect_vbins_table.fits',
                                   overwrite=True)
            
        rect_vbins_table['R50_norm'] = (rect_vbins_table['R50'] - R50_bins_min) / R50_bins_range
        rect_vbins_table['Mr_norm'] = (rect_vbins_table['Mr'] - Mr_bins_min) / R50_bins_range
    
        self.rect_bins_table = rect_bins_table
        self.vbins_table = vbins_table
        self.rect_vbins_table = rect_vbins_table
        self.Mr_bins_min = Mr_bins_min
        self.Mr_bins_range = Mr_bins_range
        self.R50_bins_min = R50_bins_min
        self.R50_bins_range = R50_bins_range
    
        return None
    
        
    def voronoi_assignment(self,reassign=True):
        ''' Assign each of the galaxies a voronoi bin. If reassign is True, then
            even the galaxies not in the sample are given a voronoi bin'''
    
        Mr_all = self.Mr_all
        R50_all = self.R50_all
        z_all = self.z_all
        rect_bins_table = self.rect_bins_table 
        rect_vbins_table = self.rect_vbins_table
        vbins_table = self.vbins_table
        Mr_bins_min = self.Mr_bins_min
        Mr_bins_range = self.Mr_bins_range
        R50_bins_min = self.R50_bins_min
        R50_bins_range = self.R50_bins_range
        
        
        # Load outputs from the 'voronoi_binning' module:
        R50_bin_edges = rect_bins_table['R50_bin_edges']
        Mr_bin_edges = rect_bins_table['Mr_bin_edges']
        n_R50_bins = len(R50_bin_edges) - 1
        n_Mr_bins = len(Mr_bin_edges) - 1
    
        # Get the R50 and Mr bin for each galaxy in the sample
        R50_bins = np.digitize(R50_all, bins=R50_bin_edges).clip(1, n_R50_bins)
        Mr_bins = np.digitize(Mr_all, bins=Mr_bin_edges).clip(1, n_Mr_bins)
        
        self.rect_vbins_table = rect_vbins_table

        # convert R50 and Mr bin indices to indices of bins
        # in the combined rectangular grid
        rect_bins = (Mr_bins - 1) + n_Mr_bins * (R50_bins - 1)
        
        # get the voronoi bin for each galaxy in the sample
        rect_bin_vbins = rect_vbins_table['vbin']
        voronoi_bins = rect_bin_vbins[rect_bins]
        in_p = self.sample.p_mask
        if reassign is True:
            for v in np.unique(voronoi_bins[in_p]):
                in_v = voronoi_bins[in_p] == v
                if in_v.sum() <= self.n_min:
                    voronoi_bins[voronoi_bins == v] = -1
            assigned = voronoi_bins >= 0
            if (assigned == False).sum() != 0:
                xy = np.array([(R50_all - R50_bins_min) / R50_bins_range,
                              (Mr_all - Mr_bins_min) / Mr_bins_range]).T
                nbrs = NearestNeighbors(n_neighbors=1).fit(xy[assigned])
                d, i = nbrs.kneighbors(xy[assigned == False])
                voronoi_bins[assigned == False] = voronoi_bins[assigned][i.squeeze()]
        
        self.voronoi_bins = voronoi_bins
        return None
    
    def redshift_assignment(self):
        ''' Bin each of the voronoi bins in terms in to bins of equal sample sizes, 
            each with >=min_gals galaxies'''
        n_per_bin = self.params.n_per_z
        
        q = self.question
        a = self.answer
        
        z_bins = np.zeros(len(self.z_all),dtype=np.int)

        for v in np.unique(self.voronoi_bins):
            in_v = self.voronoi_bins == v
            in_p = self.sample.p_mask
            in_q = self.sample.all_data[q + '_' + a + self.params.fraction_suffix] > 0
            in_vpq = np.all([in_v,in_p,in_q],axis=0)
            z_v = self.z_all[in_v]
            z_vpq = self.z_all[in_vpq]
            
            n_zbins = np.max([5,int(math.floor(in_vpq.sum()/self.n_per_z))])
            z_vpq = np.sort(z_vpq)
            bin_edges = np.linspace(0, len(z_vpq)-1, n_zbins+1, dtype=np.int)
            z_edges = z_vpq[bin_edges]
            z_edges[0] -= 1
            z_edges[-1] += 1
            z_bins_v = np.digitize(z_v,z_edges)
            z_bins[in_v] = z_bins_v
        
        self.z_bins = z_bins
        return None