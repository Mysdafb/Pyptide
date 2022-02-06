#!/usr/bin/env python
# _*_coding:utf-8_*_
import random

import numpy as np
import pandas as pd

from .MLModeler import Modeler
from .utils import generate_superseq, get_decoys, save_fasta, read_fasta

class Preprocessor(Modeler):
    """
    Main class to prepare data for modeling
    """
    def clean(self):
        self.filter_by_length()
        self.filter_unnatural_aa()

    def filter_by_length(self, min_len=5, max_len=100):
        seqs = []
        desc = []
        names = []
        target = []
        for i, s in enumerate(self.seqs):
            if len(s) >= min_len and len(s) <= max_len:
                seqs.append(s.upper())
                if hasattr(self, 'descriptors') and self.descriptors.size:
                    desc.append(self.descriptors[i])
                if hasattr(self, 'names') and self.names:
                    names.append(self.names[i])
                if hasattr(self, 'targets') and self.targets.size:
                    target.append(self.targets[i])
        self.seqs = seqs
        self.names = names
        self.descriptors = np.array(desc)
        self.targets = np.array(target, dtype='int')

    def filter_unnatural_aa(self, return_=False):
        natural_aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        seqs = []
        desc = []
        names = []
        target = []
        unnatural = []
        for i, s in enumerate(self.seqs):
            seq = list(s.upper())
            if all(c in natural_aa for c in seq):
                seqs.append(s.upper())
                if hasattr(self, 'descriptors') and self.descriptors.size:
                    desc.append(self.descriptors[i])
                if hasattr(self, 'names') and self.names:
                    names.append(self.names[i])
                if hasattr(self, 'targets') and self.targets.size:
                    target.append(self.targets[i])
            else:
                unnatural.append(s)
        self.seqs = seqs
        self.names = names
        self.descriptors = np.array(desc)
        self.targets = np.array(target, dtype='int')
        if return_:
            return unnatural

    def filter_duplicates(self):
        df = pd.DataFrame(list(zip(self.seqs, self.names, self.descriptors.tolist(), self.targets.tolist())),
                          columns=['Seqs', 'Names', 'Descriptors', 'Targets'])
        df = df.drop_duplicates('Seqs')  
        self.seqs = df['Seqs'].tolist()
        self.names = df['Names'].tolist()
        self.descriptors = df['Descriptors'].to_numpy()
        self.targets = df['Targets'].to_numpy()

    def get_decoys(self, candidates, max_len=100, ratio=1, seed=1700):
        superseq = generate_superseq(candidates)
        decoys = get_decoys(self.seqs, superseq, max_len, ratio, seed)
        names = ['decoy'+str(i) for i in range(len(decoys))]
        targets = [0] * len(decoys)
        save_fasta(candidates[:-6]+'_decoys.fasta', decoys, names, targets)

    def identical_overlap(self, another_file, remove_from_seqs=False):
        n, another, t = read_fasta(another_file)
        if remove_from_seqs:
            seqs = []
            desc = []
            names = []
            target = []
            identical_overlap = list(set(self.seqs).intersection(set(another)))
            for i, s in enumerate(self.seqs):
                if s not in identical_overlap:
                    seqs.append(s)
                    if hasattr(self, 'descriptors') and self.descriptors.size:
                        desc.append(self.descriptors[i])
                    if hasattr(self, 'names') and self.names:
                        names.append(self.names[i])
                    if hasattr(self, 'targets') and self.targets.size:
                        target.append(self.targets[i])
            self.seqs = seqs
            self.names = names
            self.descriptors = np.array(desc)
            self.targets = np.array(target, dtype='int')
        else:
            return list(set(self.seqs).intersection(set(another)))

    def insilico_digestion(self, window=10):
        if len(self.seqs) == 1:
            digest = []
            names = []
            targets = []
            for i in range(0, len(self.seqs[0]) - window + 1):
                digest.append(self.seqs[0][i:i+window])
                names.append('seq_' + str(i))
                targets.append(1)
            save_fasta('./' + self.names[0]+'.fasta', digest, names, targets)
        elif len(self.seqs) >= 1:
            for i in range(len(self.seqs)):
                digest = []
                names = []
                targets = []
                for j in range(0, len(self.seqs[i]) - window + 1):
                    digest.append(self.seqs[i][j:j+window])
                    names.append('seq_' + str(j))
                    targets.append(1)
                save_fasta('./' + self.names[i]+'.fasta', digest, names, targets)
        else:
            print('No sequences to digest!')

    def select_seqs_at_rand(self, n, only_return=True):
        if only_return:
            selected_seqs = np.random.choice(self.seqs, n , replace=False)
            return list(selected_seqs)
        else:
            sel = np.random.choice(len(self.seqs), size=n, replace=False)
            self.seqs = np.array(self.seqs)[sel].tolist()
            if hasattr(self, 'descriptors') and self.descriptors.size:
                self.descriptors = self.descriptors[sel]
            if hasattr(self, 'names') and self.names:
                self.names = np.array(self.names)[sel].tolist()
            if hasattr(self, 'targets') and self.targets.size:
                self.targets = self.targets[sel]

    def split_train_test_random(self, p, only_return=True):
        train = []
        test = []
        for i in set(self.targets):
            size = round(len([self.seqs[j] for j in range(len(self.seqs)) if self.targets[j] == i]) * p)
            train.extend(list(np.random.choice([self.seqs[j] for j in range(len(self.seqs)) if self.targets[j] == i], size, replace=False)))
            test.extend(list(set([self.seqs[j] for j in range(len(self.seqs)) if self.targets[j] == i])-set(train)))
        print('{} sequences in training and {} sequences in testing.'.format(len(train), len(test)))
        if only_return:
            return train, test
        else:
            seqs = []
            desc = []
            names = []
            target = []
            for i, s in enumerate(self.seqs):
                if s in train:
                    seqs.append(s)
                    if hasattr(self, 'descriptors') and self.descriptors.size:
                        desc.append(self.descriptors[i])
                    if hasattr(self, 'names') and self.names:
                        names.append(self.names[i])
                    if hasattr(self, 'targets') and self.targets.size:
                        target.append(self.targets[i])
            self.seqs = seqs
            self.names = names
            self.descriptors = np.array(desc)
            self.targets = np.array(target, dtype='int')
            return test

    

    