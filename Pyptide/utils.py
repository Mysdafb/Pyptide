#!/usr/bin/env python
# _*_coding:utf-8_*_

import numpy as np

import csv
import math
import random
import re 

from array import array
from Bio import SeqIO
from collections import Counter

def aa_weights():
    weights = {'A': 89.093, 'C': 121.158, 'D': 133.103, 'E': 147.129, 'F': 165.189, 'G': 75.067,
               'H': 155.155, 'I': 131.173, 'K': 146.188, 'L': 131.173, 'M': 149.211, 'N': 132.118,
               'P': 115.131, 'Q': 146.145, 'R': 174.20, 'S': 105.093, 'T': 119.119, 'V': 117.146,
               'W': 204.225, 'Y': 181.189}
    return weights

def charge(seq, ph=7.0, amide=False):   
    pos_pks = {'Nterm': 9.38, 'K': 10.67, 'R': 12.10, 'H': 6.04}
    if amide:
        neg_pks = {'Cterm': 15., 'D': 3.71, 'E': 4.15, 'C': 8.14, 'Y': 10.10}
    else:
        neg_pks = {'Cterm': 2.15, 'D': 3.71, 'E': 4.15, 'C': 8.14, 'Y': 10.10}
    aa_content = Counter(seq)
    aa_content['Nterm'] = 1.0
    aa_content['Cterm'] = 1.0
    pos_charge = 0.0
    for aa, pK in pos_pks.items():
        c_r = 10 ** (pK - ph)
        partial_charge = c_r / (c_r + 1.0)
        pos_charge += aa_content[aa] * partial_charge
    neg_charge = 0.0
    for aa, pK in neg_pks.items():
        c_r = 10 ** (ph - pK)
        partial_charge = c_r / (c_r + 1.0)
        neg_charge += aa_content[aa] * partial_charge
    return round(pos_charge - neg_charge, 3)

def count_C(seq1, seq2):
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum

def count_D(aaSet, sequence):
    number = 0
    for aa in sequence:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >=1 else 1 for i in cutoffNums]
    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(sequence)):
            if sequence[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(sequence) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code

def generate_superseq(filename):
    n, seqs, t = read_fasta(filename)
    ss = ''
    for s in seqs:
        ss += s
    return ss

def get_decoys(template, superseq, max_len=100, ratio=1, seed=1700):
    random.seed(seed)
    decoys = []
    for s in template:
        for r in range(0,ratio):
            i = random.randint(0,(len(superseq)- max_len))
            d = superseq[i:i+len(s)]
            decoys.append(d)
    return decoys

def IS_A_SEQ(seq):
    OK = True
    for s in seq:
        if not s.isalpha():
            OK = False
    return OK

def levenshtein(seq1, seq2, max_dist=-1):	
	if seq1 == seq2:
		return 0
	len1, len2 = len(seq1), len(seq2)
	if max_dist >= 0 and abs(len1 - len2) > max_dist:
		return -1
	if len1 == 0:
		return len2
	if len2 == 0:
		return len1
	if len1 < len2:
		len1, len2 = len2, len1
		seq1, seq2 = seq2, seq1
	column = array('L', range(len2 + 1))
	for x in range(1, len1 + 1):
		column[0] = x
		last = x - 1
		for y in range(1, len2 + 1):
			old = column[y]
			cost = int(seq1[x - 1] != seq2[y - 1])
			column[y] = min(column[y] + 1, column[y - 1] + 1, last + cost)
			last = old
		if max_dist >= 0 and min(column) > max_dist:
			return -1
	if max_dist >= 0 and column[len2] > max_dist:
		return -1
	return column[len2]

def read_fasta(seqs):
        names = []
        targets = []
        sequences = []           
        for s in SeqIO.parse(seqs, 'fasta'):
            sequences.append(str(s.seq).upper())
            if len(str(s.id).split('|')) == 2 and str(s.id).split('|')[-1].isdigit():
                name, target = str(s.id).split('|')
                names.append(name)
                targets.append(int(target))
            else:
                names.append(str(s.id))
                targets.append(0)
        return names, sequences, np.asarray(targets)

def rValue(aa1, aa2, AADict, Matrix):
    return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

def save_csv(filename, names, seqs, lens, descriptors, targets, features):
    if len(names) < len(seqs):
            names = ['Seq_'+str(i) for i in range(len(seqs))]
    if len(targets) < len(seqs):
        targets = [0 for i in range(len(seqs))]
    if descriptors.shape[0] < len(seqs):
        descriptors = np.asarray([[0] for i in range(len(seqs))])
        features = ['None']
    with open(filename, 'w', newline='') as output:
        text = csv.writer(output, delimiter=",")
        text.writerow(['IDs', 'Seqs', 'Length']+features+['Target'])
        for i in range(len(seqs)):
            text.writerow([names[i]] + [seqs[i]] + [lens[i]] + descriptors[i].tolist() + [targets[i]])

def save_fasta(filename, seqs, names, targets):
    if len(names) < len(seqs):
        names = ['Seq_'+str(i) for i in range(len(seqs))]
    if len(targets) < len(seqs):
        targets = [0 for i in range(len(seqs))]
    with open(filename, 'w') as output:
        for i in range(len(seqs)):
            output.write('>'+ names[i] + '|' + str(targets[i]) + '\n' + seqs[i] + '\n')

