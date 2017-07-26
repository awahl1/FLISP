



from __future__ import division
import os
from collections import defaultdict
from collections import namedtuple
import itertools
import re



import pandas as pd
import numpy as np




class CorpusManager(object):
    
    """
    Tool for finding word and word sequence co-occurrences in a corpus, and for finding turns/sentences that contain a
    particular sequence
    """
    
    def __init__(self, source_dir="/Users/alexanderwahl/Documents/Corpora/BNC_cleaned/", corpusfile_ext=".txt"):
        
        """
        Pass the directory where the plain text corpus files are listed
        """
        
        self.filepaths = ["".join((source_dir,filename)) for filename in os.listdir(source_dir) if filename.endswith(corpusfile_ext)]
        
    def get_cooccurrence(self,context,tail):
        
        """
        The "context" is the first sequence of words to be searched. The "tail" is the second. This routine will find all
        turns in the corpus containing both sequences, in the designated order, with any size discontinuity between them.
        Results are assigned to a self.cooccurrences attribute.
        """
        
        self.context = context.split()
        self.context_len = len(self.context)
        self.first_context_word = self.context[0]
        self.tail = tail.split()
        self.tail_len = len(self.tail)
        self.first_tail_word = self.tail[0]
        self.cooccurrences = []
        
        for lineindex,line in self._line_getter():
            self.lineindex = lineindex
            self.line = line
            self.tokenized_line = self.line.split()
            self._capture_cooccurrence_tokens()
            
        self.cooccurrences = set(self.cooccurrences)
            
    def _capture_cooccurrence_tokens(self):
        
        for wordindex,word in enumerate(self.tokenized_line):
            if word==self.first_context_word:
                try:
                    curr_context = self.tokenized_line[wordindex:wordindex+self.context_len]
                    if self.context==curr_context:
                        for secondindex in range(wordindex+self.context_len,len(self.tokenized_line)):
                            self.secondword = self.tokenized_line[secondindex]
                            if self.secondword==self.first_tail_word:
                                try:
                                    curr_tail = self.tokenized_line[secondindex:secondindex+self.tail_len]
                                    if self.tail==curr_tail:
                                        self.cooccurrences.append(self.line)
                                except:
                                    pass
                except:
                    pass
            
        
    def get_containing_turns(self,sequence):
        
        """
        This routine will find all turns in the corpus containing the passed sequence.
        Hits are assigned to self.hits attribute, and misses to self.misses attribute.
        """
        
        self.corpus_size = 0
        self.hits = []
        self.misses = []
        self.str_sequence = sequence
        self.sequence = sequence.split()
        self.seq_len = len(self.sequence)
        self.first_word = self.sequence[0]
        
        for lineindex,line in self._line_getter():
            self.lineindex = lineindex
            self.line = line
            self.tokenized_line = line.split()
            self._capture_seq_tokens()
        
        self.misses = set(self.misses)
        self.hits = set(self.hits)
        self.hit_counter = len(self.hits)
        
    def _line_getter(self):
        
        for filepath in self.filepaths:
            self.filepath = filepath
            with open(self.filepath, "r") as f:
                for lineindex,line in enumerate(f):
                    yield lineindex,line
                    
    def _capture_seq_tokens(self):
        
        for wordindex,word in enumerate(self.tokenized_line):
            self.corpus_size += 1
            if word==self.first_word:
                try:
                    curr_seq = self.tokenized_line[wordindex:wordindex+self.seq_len]
                    if self.sequence==curr_seq:
                        self.hits.append(self.line)
                    else:
                        self.misses.append(self.line)
                except:
                    self.misses.append(self.line)
                    
    def save_hits_and_misses(self, directory="/Users/alexanderwahl/Documents/FLISP/"):
        
        """
        If you have called self.get_containing_turns, use this method to store results to the path that you pass
        """
        
        self.directory = directory
        with open("".join((self.directory,"BNChits_",self.str_sequence)),"w") as f:
            f.write("\n".join(self.hits))
        with open("".join((self.directory,"BNCmisses_",self.str_sequence)),"w") as f:
            f.write("\n".join(self.misses))
    
    def get_word_freqs(self,words):
        
        """
        Gets the frequencies of the passed words (pass as a list) in the corpus.
        Returns a dictionary with word:count pairs
        """
        
        self.words = set(words)
        self.word_freqs = defaultdict(int)
        self.corpus_size = 0
        for filepath in self.filepaths:
            self.filepath = filepath
            with open(self.filepath, "r") as f:
                for lineindex,line in enumerate(f):
                    self.tokenized_line = line.split()
                    for word in self.tokenized_line:
                        self.corpus_size += 1
                        if word in self.words:
                            self.word_freqs[word]+=1
        return self.word_freqs


        
        
        
class ModelRunner(object):
    
    """
    This class is a top level interface for running the computational model of idiom processing.
    You must pass an idiom and its containing sentence when you instantiate. The model will be run for this combination only.
    DataTableLoader is a dependency of this class, so you must change the paths in the __init__ of that class to where
    you have the input spreadsheets stored.
    """
    
    def __init__(self, idiom=None, sentence=None):
        
        self.idiom = idiom
        self.words_in_idiom = idiom.split()
        self.sentence = sentence
        self.words_in_sentence = sentence.split()
        self._set_local_refs_to_data()
        self._instantiate_managers()
        
    def _set_local_refs_to_data(self):
        
        self.data_table_loader = DataTableLoader(self.idiom)
        self.df_concepts_in_contexts = self.data_table_loader.df_concepts_in_contexts
        self.df_context_counts = self.data_table_loader.df_context_counts
        self.df_concept_counts = self.data_table_loader.df_concept_counts
        self.df_sem_connect = self.data_table_loader.df_sem_connect
        
    def _instantiate_managers(self):
        
        self.probability_manager = ProbabilityManager(self.df_concepts_in_contexts,
                                                      self.df_context_counts,
                                                      self.df_concept_counts)
        self.posterior_manager = ConceptValueHolder(self.df_concept_counts.index)
        self.semantic_manager = SemanticManager(self.df_sem_connect)
                
    def run(self, lam=0.8, cycles=1, memory_mixing=0, assoc_mixing=0.5, K=1):
        
        """
        Call this method to run the model. Free parameters of the model are passed.
        Output is a dict whose keys are the current context (1st through nth word of sentence).
        Vals are Pandas Series whose indexes are each concept and whose values are the activation of that concept
        at that context position.
        """
        
        self.lam = lam
        self.cycles = cycles
        self.memory_mixing = memory_mixing
        self.assoc_mixing = assoc_mixing
        self.K = K
        self.all_posteriors = {}
        
        self.probability_manager.set_parameter(lam=self.lam)
        self.semantic_manager.set_parameters(cycles=self.cycles, K=self.K, memory_mixing=self.memory_mixing, assoc_mixing=self.assoc_mixing)
        for current_position in range(1,len(self.words_in_sentence)+1):
            self.current_context = self.words_in_sentence[0:current_position]
            self.probability_manager.update_concept_probabilities(self.current_context, self.posterior_manager)
            self.posterior_manager = self.probability_manager.posterior_manager
            self.semantic_manager.conduct_spread(self.current_context, self.posterior_manager)
            self.current_context_str = " ".join(self.current_context)
            self.all_posteriors[self.current_context_str] = pd.Series(index=self.df_concept_counts.index)
            for concept,posterior in self.posterior_manager:
                self.all_posteriors[self.current_context_str].loc[concept] = posterior
            

class DataTableLoader(object):
    
    def __init__(self,
                 idiom,
                 concept_counts_path = "./Input_dfs/concept_counts.csv",
                 sem_connect_path = "./Input_dfs/sem_connect.csv"):
        
        if idiom=='\xc3\xa9\xc3\xa9n lijn trekken':
            self.idiom_fn_prefix = 'e\xdd\x81e\xdd\x81N_LIJN_TREKKEN'
        else:
            self.idiom_fn_prefix = re.sub(" ", "_", idiom).upper()
        self.concepts_in_contexts_fn = self.idiom_fn_prefix + "_concepts_in_contexts.csv"
        self.concepts_in_contexts_path = "./Input_dfs/" + self.concepts_in_contexts_fn
        self.context_counts_fn = self.idiom_fn_prefix + "_context_counts.csv"
        self.context_counts_path = "./Input_dfs/" + self.context_counts_fn 
        self.concept_counts_path = concept_counts_path
        self.sem_connect_path = sem_connect_path
        self.load_data()
        
    def load_data(self):
        
        self.df_concepts_in_contexts = pd.read_csv(self.concepts_in_contexts_path,index_col=0)
        self.df_context_counts = pd.read_csv(self.context_counts_path,index_col=0)
        self.df_concept_counts = pd.read_csv(self.concept_counts_path,index_col=0)
        self.df_sem_connect = pd.read_csv(self.sem_connect_path) #Note no index column!


class ProbabilityManager(object):
    
    def __init__(self, concepts_in_contexts, context_counts, concept_counts):
        
        self.df_concepts_in_contexts = concepts_in_contexts
        self.df_context_counts = context_counts
        self.df_concept_counts = concept_counts
    
    def set_parameter(self, lam=None):
        
        self.lam = lam
    
    def update_concept_probabilities(self, current_context, posterior_manager):
        
        self.current_context = current_context
        self.current_context_str = " ".join(self.current_context)
        self.curr_cntxt_plus_cncpt_str = "_".join((self.current_context_str,"CNCPT"))
        self.curr_cntxt_plus_cncptfrms_str = "_".join((self.current_context_str,"CNCPT_FRMS"))
        self.posterior_manager = posterior_manager
        self.run_update_loop()
    
    def run_update_loop(self):
        
        for concept in self.df_concepts_in_contexts.index:                
            self.curr_concept = concept
            self.calculate_obs_forms_given_concept()
            self.calculate_obs_forms_given_concept_forms()
            self.calculate_concept_given_concept_forms()
            self.calculate_concept_forms_given_obs_forms()
            self.calculate_smoothing_term()
            self.calculate_product()
    
    def calculate_obs_forms_given_concept(self):
        
        numer = self.df_concepts_in_contexts[self.curr_cntxt_plus_cncpt_str].loc[self.curr_concept]
        denom = self.df_concept_counts["Concept Count"].loc[self.curr_concept]
        self.obs_forms_given_concept = numer/denom
        
    def calculate_obs_forms_given_concept_forms(self):
        
        numer = self.df_concepts_in_contexts[self.curr_cntxt_plus_cncptfrms_str].loc[self.curr_concept]
        denom = self.df_concept_counts["Form Count"].loc[self.curr_concept]
        self.obs_forms_given_concept_forms = numer/denom
        
    def calculate_concept_given_concept_forms(self):
        
        numer = self.df_concept_counts["Concept Count"].loc[self.curr_concept]
        denom = self.df_concept_counts["Form Count"].loc[self.curr_concept]
        self.concept_given_concept_forms = numer/denom
        
    def calculate_concept_forms_given_obs_forms(self):
        
        numer = self.df_concepts_in_contexts[self.curr_cntxt_plus_cncptfrms_str].loc[self.curr_concept]
        denom = self.df_context_counts["Adj Freq"].loc[self.current_context_str]
        self.concept_forms_given_obs_forms = numer/denom
    
    def calculate_smoothing_term(self):
        
        if np.isnan(self.df_concept_counts["Smoothed Concept Given Concept Forms"].loc[self.curr_concept]):
            self.smoothing_term = 0
        else:
            self.smoothing_term = self.df_concept_counts["Smoothed Concept Given Concept Forms"].loc[self.curr_concept] * self.concept_forms_given_obs_forms
            
    def calculate_product(self):
            
        if self.obs_forms_given_concept_forms>0:
            #self.product = (self.lam*self.obs_forms_given_concept/self.obs_forms_given_concept_forms + (1-self.lam)) * self.smoothing_term*self.concept_forms_given_obs_forms
            self.main_term = self.obs_forms_given_concept*self.concept_given_concept_forms*self.concept_forms_given_obs_forms/self.obs_forms_given_concept_forms
        else:
            self.main_term = 0
        if self.smoothing_term>0:
            self.product = self.lam*self.main_term + (1-self.lam)*self.concept_forms_given_obs_forms*self.smoothing_term
        else:
            self.product = self.main_term
        self.posterior_manager.set_concept_val_pair(self.curr_concept, self.product)
        

    

    
class SemanticManager(object):
    
    def __init__(self, df_sem_connect):

        self.df_sem_connect = df_sem_connect
        self.activations = ActivationHolder()
        
    def set_parameters(self, cycles=None, K=None, memory_mixing=None, assoc_mixing=None):
        
        self.cycles = cycles
        self.K = K
        self.memory_mixing = memory_mixing
        self.assoc_mixing = assoc_mixing
                    
    def conduct_spread(self, current_context, posterior_manager):
        
        self.posterior_manager = posterior_manager
        self.current_context = current_context
        self.global_strengths = ConceptValueHolder(self.posterior_manager.get_concepts())
        for curr_cycle in range(self.cycles):
            self.inbound_holder = defaultdict(list)
            #on the first cycle after a new word
            if curr_cycle==0:
                #update global strengths as mixture of final global strengths at last step and new posteriors
                #using these updated globals, calculate the outbound/inbound spread
                for concept,posterior in self.posterior_manager:
                    self.curr_starting = self.memory_mixing*self.activations.get_final_context(concept) + (1-self.memory_mixing)*posterior
                    self.calculate_outbounds(concept,self.curr_starting)
                    self.global_strengths.set_concept_val_pair(concept,self.curr_starting)
                #update global strengths (again) as mixture of current global strength (after above mixing) and inbound activation
                for concept,raw_inputs in self.inbound_holder.iteritems():
                    curr_global = self.global_strengths.get_value(concept)
                    self.update_globals(concept,curr_global,raw_inputs)
            #on the later cycles
            else:
                #calculate outbound/inbound spread
                for concept,strength in self.global_strengths:
                    self.calculate_outbounds(concept,strength)
                #update global strengths as mixture of current global strength (after above mixing) and inbound activation
                for concept,raw_inputs in self.inbound_holder.iteritems():
                    curr_strength = self.global_strengths.get_value(concept)
                    self.update_globals(concept,curr_strength,raw_inputs)
        self.activations.add(self.current_context, self.global_strengths)
    
    def calculate_outbounds(self, concept, value):
        
        self.curr_concept = concept
        self.curr_value = value
        self.curr_concept_row_indices = np.where(self.df_sem_connect["Origins"]==self.curr_concept)[0]
        for index in self.curr_concept_row_indices:
            self.curr_index = index
            self.curr_destination = self.df_sem_connect["Destinations"].loc[self.curr_index]
            self.curr_assoc_score = self.df_sem_connect["Similarity"].loc[self.curr_index]
            self.inbound_holder[self.curr_destination].append(self.curr_value*self.curr_assoc_score)
            
    def update_globals(self,concept,first_term,raw_inputs):
            
        self.concept = concept
        self.first_term = first_term
        self.raw_inputs = raw_inputs
        self.exponent = -1 * self.K * sum(self.raw_inputs)
        self.normalized_inbound_activation = 2/(1 + np.power(np.e,self.exponent)) - 1
        self.curr_global_strength = (1-self.assoc_mixing) * self.first_term + self.assoc_mixing * self.normalized_inbound_activation
        self.global_strengths.set_concept_val_pair(self.concept,self.curr_global_strength)      
            
            

class ActivationHolder(object):
    
    def __init__(self):
        
        self.data = {}
        
    def add(self, context, concept_values):
        
        self.data[tuple(context)] = concept_values
        
    def get_final_context(self, concept):
        
        if self.data:
            self.final_context = sorted(self.data.keys(),reverse=True)[0]
            return self.data[self.final_context].get_value(concept)
        else:
            return 0
        
    def set_df(self):
        
        self.data_ordered = {}
        for context,concept_values in self.data.iteritems():
            concepts,values = zip(*concept_values.data.items())
            self.data_ordered[" ".join(context)] = pd.Series(values,index=concepts)
        self.data_df = pd.DataFrame(self.data_ordered)
        

class ConceptValueHolder(object):
    
    def __init__(self, concepts):
        
        self.data = dict.fromkeys(concepts,0)
        
    def set_concept_val_pair(self, concept, val):
        
        self.data[concept] = val
        
    def get_value(self, concept):
        
        return self.data[concept]
    
    def __iter__(self):
        
        for concept,posterior in self.data.iteritems():
            yield concept,posterior
            
    def get_concepts(self):
        
        return self.data.keys()



