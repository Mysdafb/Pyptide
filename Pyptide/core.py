#!/usr/bin/env python
# _*_coding:utf-8_*_

import numpy as np
import pandas as pd
import os
import random

from .utils import *

class Sequences(object):
    """
    Main class to load sequences and their information
    Load sequences from fasta files, pure str objects, python built-in lists of str,
    and from a csv file saved with the save method.
    """
    def __init__(self, seqs):
        if type(seqs) == str and seqs.endswith('.fasta'):
            self.names, self.seqs, self.targets = read_fasta(seqs)
            self.descriptors = np.array([[0] for i in range(len(self.seqs))])
            self.features_names = ['None']
        elif type(seqs) == str and IS_A_SEQ([seqs.strip().upper()]):
            self.seqs = [seqs.strip().upper()]
            self.names = ['Seq1']
            self.descriptors = np.array([[0]])
            self.targets = np.zeros(1, dtype=int)
            self.features_names = ['None']
        elif type(seqs) == list:
            self.seqs = []
            for s in seqs:
                if IS_A_SEQ([s.strip().upper()]):
                    self.seqs.append(s.strip().upper())
                else:
                    raise Exception('Sorry, some sequence was invalid!')
            self.names = ['Seq'+str(i) for i in range(len(seqs))]
            self.descriptors = np.array([[0] for i in range(len(seqs))])
            self.targets = np.zeros(len(seqs), dtype=int)
            self.features_names = ['None']
        elif type(seqs) == str and seqs.endswith('.csv'):
            df = pd.read_csv(seqs)
            self.names = df.IDs.tolist()
            self.seqs = df.Seqs.tolist()
            self.targets = df.Target.values
            self.descriptors = df[df.columns.tolist()[3:-1]].values
            self.features_names = df.columns[3:-1].tolist()
        else:
            print('Invalid format of inputs!')

    def get_descriptors(self):
        return self.descriptors

    def get_feature_names(self):
        return self.features_names

    def get_lens(self):
        return np.array([len(s) for s in self.seqs]) 

    def get_names(self):
        return self.names
    
    def get_number_of_seqs(self):
        return len(self.seqs)

    def get_sequences(self):
        return self.seqs

    def get_targets(self):
        return self.targets

    def set_descriptors(self, descriptors, headers=False):
        if type(descriptors) == list:
            if headers:
                self.descriptors = np.asarray(descriptors[1:])
                self.features_names = descriptors[0]
            else:
                self.descriptors = np.asarray(descriptors)
        elif type(descriptors) == str and descriptors.endswith('.csv'):
            df = pd.read_csv(descriptors)
            self.features_names = df.columns.tolist()
            self.descriptors = df.values
        else:
            print('Descriptors must be a list or a csv file')

    def set_names(self, names):
        if type(names) == list:
            self.names = names
        else:
            print('Input must be a list of names')

    def set_targets(self, targets, headers=False):
        if type(targets) == list:
            self.targets = np.array(targets)
        elif type(targets) == str and (targets.endswith('.csv') or targets.endswith('.txt')):
            t = []
            with open(targets, 'r') as handle:
                for line in handle:
                    if headers:
                        headers = False
                        continue
                    t.append(int(line.strip()))
            self.targets = np.array(t)
        else:
            print('Targets must be a list, a csv or txt file of integer values')

    def set_features_names(self, features):
        if type(features) == list:
            self.features_names = features
        else:
            print('Features names must be a list of string values')

    def save(self, filename):
        save_csv(filename, self.names, self.seqs, self.get_lens(), self.descriptors, self.targets, self.features_names)

    def save_seqs_to_fasta(self, filename):
        save_fasta(filename, self.seqs, self.names, self.targets)
