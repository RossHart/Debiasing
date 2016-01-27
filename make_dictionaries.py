import params
import numpy as np
import math

#-------------------------------------------------------------------------------
'''List the functions (and their respective inverses)'''

def f_logistic(x, k, c):
    # Function to fit the data bin output from the raw plot function
    L = (1 + np.exp(c))
    r = L / (1.0 + np.exp(-k * x + c))
    return r


def f_exp_pow(x, k, c):
    # Function to fit the data bin output from the raw plot function
    r = np.exp(-k * (-x) ** c)
    return r


def i_f_logistic(y, k, c):
    # inverse of f_logistic
    L = (1 + np.exp(c))
    x = -(np.log(L / y - 1) - c) / k
    return x


def i_f_exp_pow(y, k, c):
    # inverse of f_exp_pow
    ok = k > 0
    x = np.zeros_like(y) - np.inf
    x[ok] = -(-np.log(y[ok]) /k[ok] )**(1.0/c[ok])
    return x
#-------------------------------------------------------------------------------
'''This dictionary lists all of the functions and bounds to be used in the
to fit the data'''

function_dictionary = {}
function_dictionary['func'] = {0: f_logistic,
                               1: f_exp_pow,
                               #2: f_inv
                               }

function_dictionary['bounds'] = {0: params.logistic_bounds,
                                 1: params.exponential_bounds
                                 #2: params.inverse_bounds,
                                 }

function_dictionary['p0'] = {0: [3,-3],
                             1: [2,1],
                             #2: [1,1]
                             }

function_dictionary['i_func'] = {0: i_f_logistic,
                                 1: i_f_exp_pow
                                 #2:None
                                 }

function_dictionary['label'] = {0: 'logistic',
                                1: 'exp. power'
                                #2:'inverse'
                                 }
#-------------------------------------------------------------------------------
'''Make a dictionary of questions, answers, and which questions precede others
'''

# List of questions in order:
q = ['t01_smooth_or_features'
     ,'t02_edgeon'
     ,'t03_bar'
     ,'t04_spiral'
     ,'t05_bulge_prominence'
     ,'t06_odd'
     ,'t07_rounded'
     ,'t08_odd_feature'
     ,'t09_bulge_shape'
     ,'t10_arms_winding'
     ,'t11_arms_number']

# Labels for each of the questions (for plotting):
label_q = ['Smooth or features'
     ,'Edge on'
     ,'Bar'
     ,'Spiral'
     ,'Bulge prominence'
     ,'Anything odd'
     ,'Roundedness'
     ,'Odd features'
     ,'Bulge shape'
     ,'Arm winding'
     ,'Arm number']

# Answers for each of the questions in turn:
a = [['a01_smooth','a02_features_or_disk','a03_star_or_artifact']
     ,['a04_yes','a05_no']
     ,['a06_bar','a07_no_bar']
     ,['a08_spiral','a09_no_spiral']
     ,['a10_no_bulge','a11_just_noticeable','a12_obvious','a13_dominant']
     ,['a14_yes','a15_no']
     ,['a16_completely_round','a17_in_between','a18_cigar_shaped']
     ,['a19_ring','a20_lens_or_arc','a21_disturbed','a22_irregular','a23_other','a24_merger','a38_dust_lane']
     ,['a25_rounded','a26_boxy','a27_no_bulge']
     ,['a28_tight','a29_medium','a30_loose']
     ,['a31_1','a32_2','a33_3','a34_4','a36_more_than_4','a37_cant_tell']]

# Answer labels (for plotting):
label_a = [['Smooth','Features','Artifact']
     ,['Yes','No']
     ,['Yes','No']
     ,['Yes','No']
     ,['None','Noticeable','Obvious','Dominant']
     ,['Yes','No']
     ,['Round','In between','Cigar shaped']
     ,['Ring','Lens/Arc','Disturbed','Irregular','Other','Merger','Dust lane']
     ,['Rounded','Boxy','None']
     ,['Tight','Medium','Loose']
     ,['1','2','3','4','5+','??']]

# 'Previously answered questions' for each question in turn:
pre_q = [None
         ,[0]
         ,[0,1]
         ,[0,1]
         ,[0,1]
         ,None
         ,[0]
         ,[5]
         ,[0,1]
         ,[0,1,3]
         ,[0,1,3]]

# Required answers for each previously answered question:
pre_a = [None
         ,[1]
         ,[1,1]
         ,[1,1]
         ,[1,1]
         ,None
         ,[0]
         ,[0]
         ,[1,1]
         ,[1,1,0]
         ,[1,1,0]]

#-------------------------------------------------------------------------------
'''Put all of this together in a single dictionary called "questions" '''

questions = {}

for s in range(len(q)):
    
    if pre_q[s] is not None:
        pq = [q[v] for v in pre_q[s]] 
    else:
        pq = None
    
    questions[q[s]] = {'answers': a[s]
                       ,'answerlabels': label_a[s]
                       ,'questionlabel': label_q[s]
                       ,'pre_questions': pq}
    
    if pre_a[s] is not None:
        pa_array = [questions[q[v]]['answers'] 
		    for v in pre_q[s]]
        answer_arrays = [pa_array[v] 
			 for v in range(len(pre_a[s]))]
        answer_indices = [pre_a[s][v] 
			  for v in range(len(pre_a[s]))]
        pa = [answer_arrays[v2][answer_indices[v2]] 
	      for v2 in range(len(answer_indices))]
 
    else:
        pa = None # if there are no previous questions
    
    questions[q[s]].update({'pre_answers': pa})
#-------------------------------------------------------------------------------