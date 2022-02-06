#!/usr/bin/env python
# _*_coding:utf-8_*_

import numpy as np
import os

from collections import Counter
from .core import Sequences
from .globalVariables import *
from .utils import *

class MDC(Sequences):
    def aminoacid_composition(self, append=False):
        AA = 'ACDEFGHIKLMNPQRSTVWY'
        desc = []
        for s in self.seqs:
            aa_count = Counter(s)
            for key in aa_count:
                aa_count[key] = aa_count[key] / len(s)
            code = []
            for aa in AA:
                code.append(aa_count[aa])
            desc.append(code)
        if append:
            self.descriptors = np.hstack((self.descriptors, np.array(desc)))
            self.features_names.extend([aa for aa in AA])
        else:
            self.descriptors = np.asarray(desc)
            self.features_names = [aa for aa in AA]

    def aliphatic_index(self, append=False):
        desc = []
        aa_dict = aa_weights()
        for seq in self.seqs:
            d = {aa: seq.count(aa) for aa in aa_dict.keys()}
            d = {k: (float(d[k]) / len(seq)) * 100 for k in d.keys()}
            desc.append(d['A'] + 2.9 * d['V'] + 3.9 * (d['I'] + d['L']))
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptors = np.hstack((self.descriptors, np.array(desc)))
            self.features_names.append('AliphaticI')
        else:
            self.descriptors = np.array(desc)
            self.features_names = ['AliphaticI']

    def aromaticity(self, append=False):
        desc = []
        for seq in self.seqs:
            f = seq.count('F')
            w = seq.count('W')
            y = seq.count('Y')
            desc.append(float(f + w + y) / len(seq))
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptors = np.hstack((self.descriptors, np.array(desc)))
            self.features_names.append('Aromaticity')
        else:
            self.descriptors = np.array(desc)
            self.features_names = ['Aromaticity']

    def calculate_charge(self, ph=7.0, amide=False, append=False):
        desc = []
        for seq in self.seqs:
            desc.append(charge(seq, ph, amide))
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptors = np.hstack((self.descriptors, np.array(desc)))
            self.features_names.append('Charge')
        else:
            self.descriptors = np.array(desc)
            self.features_names = ['Charge']

    def composition(self, append=False):
        desc = []
        headers = []
        for p in property:
            for g in range(1, len(groups) + 1):
                headers.append(p + '.C' + str(g))
        for seq in self.seqs:
            code = []
            for p in property:
                c1 = count_C(group1[p], seq) / len(seq)
                c2 = count_C(group2[p], seq) / len(seq)
                c3 = 1 - c1 - c2
                code = code + [c1, c2, c3]
            desc.append(code)
        if append:
            self.descriptors = np.hstack((self.descriptors, np.array(desc)))
            self.features_names.extend(headers)
        else:
            self.descriptors = np.array(desc)
            self.features_names = headers

    def distribution(self, append=False):
        desc = []
        headers = []
        for p in property:
            for g in range(len(groups)):
                for d in ['0', '25', '50', '75', '100']:
                    headers.append(p + '.' + str(g+1) + '.residue' + d)
        for seq in self.seqs:
            code = []
            for p in property:
                code = code + count_D(group1[p], seq) + count_D(group2[p], seq) + count_D(group3[p], seq)
            desc.append(code)
        if append:
            self.descriptors = np.hstack((self.descriptors, np.array(desc)))
            self.features_names.extend(headers)
        else:
            self.descriptors = np.array(desc)
            self.features_names = headers

    def hydrophobic_ratio(self, append=False):
        desc = []
        aa_dict = aa_weights()
        for seq in self.seqs:
            pa = {aa: seq.count(aa) for aa in aa_dict.keys()}
            desc.append((pa['A'] + pa['C'] + pa['F'] + pa['I'] + pa['L'] + pa['M'] + pa['V']) / float(len(seq)))
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptors = np.hstack((self.descriptors, np.array(desc)))
            self.features_names.append('HydrophRatio')
        else:
            self.descriptors = np.array(desc)
            self.features_names = ['HydrophRatio']

    def isoelectric_point(self, amide=False, append=False):
        ph, ph1, ph2 = float(), float(), float()
        desc = []
        for seq in self.seqs:
            ph = 7.0
            charge_ = charge(seq, ph, amide)
            if charge_ > 0.0:
                ph1 = ph
                charge1 = charge_
                while charge1 > 0.0:
                    ph = ph1 + 1.0
                    charge_ = charge(seq, ph, amide)
                    if charge_ > 0.0:
                        ph1 = ph
                        charge1 = charge_
                    else:
                        ph2 = ph
                        break
            else:
                ph2 = ph
                charge2 = charge_
                while charge2 < 0.0:
                    ph = ph2 - 1.0
                    charge_ = charge(seq, ph, amide)
                    if charge_ < 0.0:
                        ph2 = ph
                        charge2 = charge_
                    else:
                        ph1 = ph
                        break
            while ph2 - ph1 > 0.0001 and charge_ != 0.0:
                ph = (ph1 + ph2) / 2.0
                charge_ = charge(seq, ph, amide)
                if charge_ > 0.0:
                    ph1 = ph
                else:
                    ph2 = ph
            desc.append(ph)
        desc = np.asarray(desc).reshape(len(desc), 1)
        if append:
            self.descriptors = np.hstack((self.descriptors, np.array(desc)))
            self.features_names.append('pI')
        else:
            self.descriptors = np.array(desc)
            self.features_names = ['pI']

    def molecular_weight(self, amide=False, append=False):
        desc = []
        weights = aa_weights()
        for seq in self.seqs:
            mw = []
            for aa in seq:
                mw.append(weights[aa])
            desc.append(round(sum(mw) - 18.015 * (len(seq) - 1), 2))
        desc = np.asarray(desc).reshape(len(desc), 1)
        if amide:
            desc = [d - 0.98 for d in desc]
        if append:
            self.descriptors = np.hstack((self.descriptors, np.array(desc)))
            self.features_names.append('MW')
        else:
            self.descriptors = np.array(desc)
            self.features_names = ['MW']

    def pseudo_aminoacid_composition(self, lambda_value=2, w=0.05, append=False):
        if self.get_lens().min() < lambda_value + 1:
            return 0
        dataFile = os.path.split(os.path.realpath('file'))[0] + '\Pyptide\data\PAAC.txt'
        with open(dataFile, 'r') as f:
            records = f.readlines()
        AA = ''.join(records[0].rstrip().split()[1:])
        AADict = {}
        for i in range(len(AA)):
            AADict[AA[i]] = i
        AAProperty = []
        AAPropertyNames = []
        for i in range(1, len(records)):
            array = records[i].rstrip().split() if records[i].rstrip() != '' else None
            AAProperty.append([float(j) for j in array[1:]])
            AAPropertyNames.append(array[0])
        AAProperty1 = []
        for i in AAProperty:
            meanI = sum(i) / 20
            fenmu = math.sqrt(sum([(j - meanI) ** 2 for j in i]) / 20)
            AAProperty1.append([(j - meanI) / fenmu for j in i])
        desc = []
        headers = []
        for aa in AA:
            headers.append('PAAC.' + aa)
        for n in range(1, lambda_value + 1):
            headers.append('PAAC.lambda' + str(n))
        for seq in self.seqs:
            code = []
            theta = []
            for n in range(1, lambda_value + 1):
                theta.append(
                    sum([rValue(seq[j], seq[j + n], AADict, AAProperty1) for j in range(len(seq) - n)]) / (
                        len(seq) - n))
            myDict = {}
            for aa in AA:
                myDict[aa] = seq.count(aa)
            code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
            code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
            desc.append(code)
        if append:
            self.descriptors = np.hstack((self.descriptors, np.array(desc)))
            self.features_names.extend(headers)
        else:
            self.descriptors = np.array(desc)
            self.features_names = headers
        
    def transition(self, append=False):
        desc = []
        headers = []
        for p in property:
            for tr in ('Tr1221', 'Tr1331', 'Tr2332'):
                headers.append(p + '.' + tr)
        for seq in self.seqs:
            code = []
            aaPair = [seq[j:j + 2] for j in range(len(seq) - 1)]
            for p in property:
                c1221, c1331, c2332 = 0, 0, 0
                for pair in aaPair:
                    if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                        c1221 = c1221 + 1
                        continue
                    if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                        c1331 = c1331 + 1
                        continue
                    if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                        c2332 = c2332 + 1
                code = code + [c1221/len(aaPair), c1331/len(aaPair), c2332/len(aaPair)]
            desc.append(code)
        if append:
            self.descriptors = np.hstack((self.descriptors, np.array(desc)))
            self.features_names.extend(headers)
        else:
            self.descriptors = np.array(desc)
            self.features_names = headers

    def calculate_all(self):
        self.aminoacid_composition()
        self.aliphatic_index(append=True)
        self.aromaticity(append=True)
        self.calculate_charge(append=True)
        self.composition(append=True)
        self.distribution(append=True)
        self.hydrophobic_ratio(append=True)
        self.isoelectric_point(append=True)
        self.molecular_weight(append=True)
        self.pseudo_aminoacid_composition(append=True)
        self.transition(append=True)
        


    