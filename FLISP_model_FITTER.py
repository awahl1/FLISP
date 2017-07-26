from __future__ import division
import random

import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr
from multiprocessing import Pool
from multiprocessing import Process
import json
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import sys
import traceback

from FLISP_model import *


"""
This module is used to fit the model in the FLISP_model.py module. It can be run from the command line
(see if __name__=="__main__" at bottom) or from within the python shell (by instantiating the Fitter class).
"""

def AutoRun(parameter_collection, indices):
    
    fitter = Fitter()
    fitter.set_training(indices)
    fitter.set_model()
    #fitter.iteratively_fit(parameter_collection)
    
    try:
        fitter.iteratively_fit(parameter_collection)
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_fn = "".join(("./",str(os.getpid()),"_CRASH_ERROR"))
        with open(error_fn,"w") as f:
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                      limit=30, file=f)
        lp_fn = "".join(("./",str(os.getpid()),"_LAST_PARAMS"))
        lp = ",".join((str(fitter.lam), str(fitter.cycles), str(fitter.memory_mixing), str(fitter.assoc_mixing), str(fitter.K)))
        with open(lp_fn,"w") as f:
            f.write(lp)
    


class Fitter(object):
    
    """To assess model fit for a collection of sets of parameters values from within the shell, instantiate this class.
    Next, call "set_training" followed by "set_model" followed by "iteratively_fit".
    See below for further details.
    """
    
    def __init__(self, empirical_data=None):
        
        """
        Default location for empirical_data.csv is Input_dfs subfolder, but can be overridden.
        """
        
        #Each idiom should be on a line! That means at least 3 reaction times per line...
        self.empirical_data = pd.read_csv("./Input_dfs/empirical_data.csv", index_col=0)  
            
    def set_training(self, indices=None, n=27, save_training=True):
        
        """
        n is number of idioms in training set. save_training will save list of those idioms. indices is passed from
        above.
        """
        
        self.n = n
        if indices:
            self.rand_indices = indices
        else:
            self.rand_indices = random.sample(self.empirical_data.index, self.n)
        if save_training:
            with open("training_items.pickle","w") as f:
                pickle.dump(self.rand_indices,f)
        self.empirical_training = self.empirical_data.loc[self.rand_indices]
        #print self.empirical_training
    
    def set_model(self):
        
        self.idiom_and_sentence = []
        self.reaction_times = []
        for idiom,row in self.empirical_training.iterrows():
            self.idiom = idiom
            self.row = row
            self.idiom_and_sentence.append((self.idiom,self.row["Model Sentence"]))
            self.reaction_times.append(self.empirical_training["Figurative RT"].loc[self.idiom])
            self.reaction_times.append(self.empirical_training["Literal RT"].loc[self.idiom])
            self.reaction_times.append(self.empirical_training["Unrelated RT"].loc[self.idiom])
        self.condition = [0,1,2]*27
            
            
    def iteratively_fit(self, parameter_collection=None, save=True):
        
        """
        parameter_collection should be a list of 5-d tuples that enumerate the current parameter values to be
        tested. The order is lam,cycles,memory_mixing,assoc_mixing,K.
        """

        self.parameter_collection = parameter_collection
        self.save = save
        self.fit_results = {}
        for lam,cycles,memory_mixing,assoc_mixing,K in self.parameter_collection:
            self.lam = lam
            self.cycles = cycles
            self.memory_mixing = memory_mixing
            self.assoc_mixing = assoc_mixing
            self.K = K
            self._calculate_fit((lam,cycles,memory_mixing,assoc_mixing,K,))
        if self.save:
            with open("./fitting_results/"+str(os.getpid())+".pickle","w") as f:
                pickle.dump(self.fit_results,f)
            
    def _calculate_fit(self,(lam,cycles,memory_mixing,assoc_mixing,K,)):
        
        self.lam = lam
        self.cycles = cycles
        self.memory_mixing = memory_mixing
        self.assoc_mixing = assoc_mixing
        self.K = K
        self.message_1 = "Current Parameter Settings:\n"\
        "Lambda: " + str(self.lam) + "\n"\
        "Cycles: " + str(self.cycles) + "\n"\
        "Memory Mixing: " + str(self.memory_mixing) + "\n"\
        "Association Mixing: " + str(self.assoc_mixing) + "\n"\
        "K: " + str(self.K) + "\n"\
        "\n"\
        "Current Process: " + str(os.getpid()) + "\n"
        print self.message_1
        self.activations = []
        self.idiom_posteriors = []
        self.idiom_activations = []
        
        self.fig_activations = []
        self.fig_rts = []
        self.lit_activations = []
        self.lit_rts = []
        for idiom,sentence in self.idiom_and_sentence:
            self.idiom = idiom
            self.sentence = sentence
            self.model_runner = ModelRunner(idiom=self.idiom, sentence=self.sentence)
            self.model_runner.run(self.lam, self.cycles, self.memory_mixing, self.assoc_mixing, self.K)
            self.fig_tar = self.empirical_training["Figurative Target"].loc[self.idiom]
            self.fig_tar_index = self.model_runner.df_concept_counts["Form"]==self.fig_tar
            self.fig_tar_concept = self.model_runner.df_concept_counts.index[self.fig_tar_index][0]
            self.activations.append(self.model_runner.semantic_manager.activations.get_final_context(self.fig_tar_concept))

            self.lit_tar = self.empirical_training["Literal Target"].loc[self.idiom]
            self.lit_tar_index = self.model_runner.df_concept_counts["Form"]==self.lit_tar
            self.lit_tar_concept = self.model_runner.df_concept_counts.index[self.lit_tar_index][0]
            self.activations.append(self.model_runner.semantic_manager.activations.get_final_context(self.lit_tar_concept))
            
            self.unrel_tar = self.empirical_training["Unrelated Target"].loc[self.idiom]
            self.unrel_tar_index = self.model_runner.df_concept_counts["Form"]==self.unrel_tar
            self.unrel_tar_concept = self.model_runner.df_concept_counts.index[self.unrel_tar_index][0]
            self.activations.append(self.model_runner.semantic_manager.activations.get_final_context(self.unrel_tar_concept))
        
        self.condition_cleaned = []
        self.reaction_times_cleaned = []
        self.activations_cleaned = []
        for index,rt in enumerate(self.reaction_times):
            if not np.isnan(rt):
                self.condition_cleaned.append(self.condition[index])
                self.reaction_times_cleaned.append(rt)
                self.activations_cleaned.append(self.activations[index])
        self.predictors = pd.DataFrame({"Condition":self.condition_cleaned,"RT":self.reaction_times_cleaned})
        self.interaction_matrix = PolynomialFeatures(2,interaction_only=True)
        self.predictors_interactions = self.interaction_matrix.fit_transform(self.predictors)
        self.model = LinearRegression()
        self.model.fit(self.predictors_interactions,self.activations_cleaned)
        self.r2 = self.model.score(self.predictors_interactions,self.activations_cleaned)
        
        self.condition_cleaned_df = pd.DataFrame({"Condition":self.condition_cleaned})
        self.model2 = LinearRegression()
        self.model2.fit(self.condition_cleaned_df,self.activations_cleaned)
        self.r2_monofactorial = self.model2.score(self.condition_cleaned_df,self.activations_cleaned)
        
        self.fit_results[(self.lam,self.cycles,self.memory_mixing,self.assoc_mixing,self.K)] = (self.model.intercept_,
                                                                                                self.model.coef_[1:],
                                                                                                self.r2,
                                                                                                self.model2.intercept_,
                                                                                                self.model2.coef_,
                                                                                                self.r2_monofactorial)
        self.message_2 = "\n"\
        "R2: " + str(self.r2) + "\n"\
        "\n"\
        "------------------------------\n"\
        "\n"
        print self.message_2
            
            
    def set_lowest_error(self):
        
        self.lowest_error_parameters,self.lowest_error = sorted(self.fit_results.items(), key=lambda tup: tup[1])[0]
        self.lam,self.cycles,self.memory_mixing,self.assoc_mixing,self.K = self.lowest_error_parameters
        
    def increase_parameter_resolution(self):
        
        self.parameter_manager.increase_parameter_resolution(self.lam,"lam")
        self.parameter_manager.increase_parameter_resolution(self.memory_mixing,"memory_mixing")
        self.parameter_manager.increase_parameter_resolution(self.assoc_mixing,"assoc_mixing")
        self.parameter_manager.increase_parameter_resolution(self.K,"K")
        
    def calculate_average_fit(self, parameters=None, iterations=20):
        
        self.parameters = parameters
        self.iterations = iterations
        self.all_fits = []
        for i in range(self.iterations):
            self.set_training(save_training=False)
            self.set_model()
            self.fit_results = {}
            self._calculate_fit(self.parameters)
            self.all_fits.append(self.fit_results.values()[0])
        self.average_fit = sum(self.all_fits)/len(self.all_fits)
            
        
        

        



class ParameterManager(object):
    
    def __init__(self):
        
        self.parameter_fits = {}
    
    def set_parameters(self, **kwargs):
        
        for attname,(minimum,maximum,increment) in kwargs.iteritems():
            self.set_parameter_attributes(attname,minimum,maximum,increment)
    
    def set_parameter_attributes(self,attname,minimum,maximum,increment):
        
        setattr(self,attname+"_incr",increment)
        setattr(self,attname+"_min",minimum)
        setattr(self,attname+"_max",maximum)
        setattr(self,attname+"_range",np.arange(minimum,maximum+increment,increment))
    
    def __iter__(self):
        
        for lam in self.lam_range:
            for cycles in self.cycles_range:
                for memory_mixing in self.memory_mixing_range:
                    for assoc_mixing in self.assoc_mixing_range:
                        for K in self.K_range:
                            yield lam,cycles,memory_mixing,assoc_mixing,K
                            
    def increase_parameter_resolution(self,value,attname,incr_count=10):
        
        self._set_hyper_max(value,attname)
        self._set_hyper_min(value,attname)
        self.curr_hyper_incr = (self.curr_hyper_max-self.curr_hyper_min)/incr_count
        self.set_parameter_attributes(attname,self.curr_hyper_min,self.curr_hyper_max,self.curr_hyper_incr)
        
    def _set_hyper_max(self, value, attname):
        
        self.curr_max = getattr(self,attname+"_max")
        self.curr_incr = getattr(self,attname,"_incr")
        if value+self.curr_incr<=self.curr_max:
            self.curr_hyper_max = value+self.curr_incr
        else:
            self.curr_hyper_max = value
        
    def _set_hyper_min(self, value, attname):
        
        self.curr_min = getattr(self,attname+"_min")
        self.curr_incr = getattr(self,attname,"_incr")
        if value-self.curr_incr>=self.curr_min:
            self.curr_hyper_min = value-self.curr_incr
        else:
            self.curr_hyper_min = value
        




if __name__=="__main__":
    parameter_manager = ParameterManager()
    parameter_manager.set_parameters(lam=(0.1, 0.9, 0.1),
                                     cycles=(1, 6, 1),
                                     memory_mixing=(0, 0.9, 0.1),
                                     assoc_mixing=(0, 0.9, 0.1),
                                     K=(0.25, 3.0, 0.25))
    

    all_params = [params for params in parameter_manager]
    
    #Integer should be number of processes that you want to launch (e.g., number of cores); I use just 4 on my Macbook. I use 30 on my University's server cluster.
    size_of_group = int(round(len(all_params)/4))
    
    #The integers in the 2 following lines should be number of processes - 1
    all_groups = [all_params[s*size_of_group:(s+1)*size_of_group] for s in range(3)]
    all_groups.append(all_params[3*size_of_group:])
    
    empirical_data = pd.read_csv("./Input_dfs/empirical_data.csv", index_col=0)
    indices = random.sample(empirical_data.index, 27)
    processes = []
    for group in all_groups:
        curr_process = Process(target=AutoRun, args=(group,indices))
        curr_process.start()
        processes.append(curr_process)
    for p in processes:
        p.join()
            
    
        