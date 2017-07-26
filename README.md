# FLISP
Computational Cognitive Model of Idiom Processing

This collection of files represents an ongoing research project on the cognitive processing of idioms (sequences like “kick the bucket” that have both a literal and figurative interpretation). I have developed a computational model that aims to describe the competition between literal and figurative interpretations of idioms in the mind of a human as they are hearing sentences that contain such idioms. This model is called FLISP (Figurative and Literal Interpretations in Sentence Processing). It calculates, at each time step, a “strength” value for the two competing interpretations of a given idiom. There are 2 types of times steps: word steps and intra-word steps. Word steps represent the comprehension of each successive word in a sentence as it is perceived by the human. Intra-word steps represent processing cycles that occur between successive words (number of intra-word steps is a parameter of the model). Strength values are a mixture of n-gram probabilities (the probability of an interpretation given the words seen so far), spreading activation from semantically related concept nodes (summed & normalized using a logistic function), and strength at previous time steps.

The implementation of the model is spread across 3 Python module files:

1) FLISP_model.py : contains the classes that make up the model itself.

2) FLISP_model_FITTER.py : contains an implementation of a grid search procedure that evaluates the fit of the model to empirical data, given different parameter settings and a set of target idioms. Fit is evaluated using r-squared, and output is stored in a fitting_results subfolder. Note that when run from the command line, it can be run in a parallelized fashion. For a large grid of parameter settings, this really is necessary due to time and memory considerations.

3) fitting_results_processing.py : contains a class for sorting through the fitting results, to find the best set of parameters for model-to-data fit.


In addition, the model requires a number of spreadsheets that serve as input to the model. These are housed in Input_dfs. The model code expects that these particular files occur in this particular directory, so names/locations shouldn’t be modified. The current empirical data contained in this folder did not yield strong empirical findings—thus, there are not particularly good empirical results to currently fit! Nonetheless, using this empirical data can be fruitful for debugging and to examine behavior of the model across different regions of the parameter space.
