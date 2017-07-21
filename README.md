# Debiasing

A code for removing any redshift dependent biases in GZ data.

All of the debiasing takes place in the 'debias.ipynb'. The relevant files to adjust are:

params.py: input file, containing information such as where to find the data on the system. Can also adjust the input parameters to the fitting here.
dictionaries.py: here, you can adjust what the questions are to be debiased, and their order. *Note that the order does matter, you need to debias the questions further up the tree than the question you are currently debiasing.*


