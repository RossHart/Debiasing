# Debiasing
Final version of the debiasing method git repo:

You need the file 'full sample.fits' from my dropbox (you may have to email me so I can share it with you), saved in an external directory one folder 'up'(ie ../fits from this directory). 

1. Can use volume_limiting.ipynb to add a boolean column to the full sample, with 1= inside luminosity limited sample. However, the full_sample.fits already has this, so this stage is not necessary.

2. Then run debiasing_final.ipynb to generate the debiased values for each of the questions in turn.

3. Use 'Recover_full_data.ipynb' to add the debiased values to the end of the full_sample.fits

*Plotting codes (a little sporadic at the moment) are all in the 'Plotting_codes' folder.
