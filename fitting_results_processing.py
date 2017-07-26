from __future__ import division
import os
from collections import defaultdict

import pickle


class Results(object):
    
    def __init__(self):
        
        self.filenames = os.listdir("./fitting_results/")
        self.data = {}
        for filename in self.filenames:
            with open("./fitting_results/"+filename, "r") as f:
                self.data.update(pickle.load(f))
                
    def rank_by_parameter(self):
        
        self.by_lam = defaultdict(list)
        self.by_cycles = defaultdict(list)
        self.by_memory = defaultdict(list)
        self.by_assoc = defaultdict(list)
        self.by_K = defaultdict(list)
        for params,correlation in self.data.iteritems():
            lam,cycles,memory_mixing,assoc_mixing,K = params
            self.by_lam[lam].append((params,correlation))
            self.by_cycles[cycles].append((params,correlation))
            self.by_memory[memory_mixing].append((params,correlation))
            self.by_assoc[assoc_mixing].append((params,correlation))
            self.by_K[K].append((params,correlation))
        self.by_lam = self._val_sorter(self.by_lam)
        self.by_cycles = self._val_sorter(self.by_cycles)
        self.by_memory = self._val_sorter(self.by_memory)
        self.by_assoc = self._val_sorter(self.by_assoc)
        self.by_K = self._val_sorter(self.by_K)
    
    def _val_sorter(self, dictionary):
        
        for value in dictionary.itervalues():
            value.sort(key=lambda tup: tup[1])
        return dictionary

