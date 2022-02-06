#!/usr/bin/env python
# _*_coding:utf-8_*_

import csv
import math
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from collections import Counter
from joblib import dump, load
from .MDCalculator import MDC
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score, accuracy_score, cohen_kappa_score, confusion_matrix, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from .utils import *


class Modeler(MDC):
    def init(self, seqs):
        MDC.init(self, seqs)
        self.numerator_ = np.array([])
        self.denominator_ = np.array([])
        self.model = None

    def calculate_metrics(self, y_trues, y_preds):
        tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
        sn = (tp / (tp + fn))
        sp = (tn / (fp + tn))
        accuracy = accuracy_score(y_trues, y_preds)
        mcc = matthews_corrcoef(y_trues, y_preds)
        rocAuc = roc_auc_score(y_trues, y_preds)
        k = cohen_kappa_score(y_trues, y_preds)
        return sn, sp, accuracy, mcc, rocAuc, k

    def find_best_model(self, algorithm='RF', search_space=None, cv=5, scaler=None):
        clf = self.select_model(algorithm)
        if not search_space:
            if algorithm.upper() == 'RF':
                search_space = {'n_estimators':[100, 200, 300, 400, 500], 'criterion': ['gini', 'entropy'], 'max_depth': list(range(4,17,2)), 'max_features': ['auto', 'sqrt', 'log2']}
            elif algorithm.upper() == 'SVM':
                search_space = [{'kernel': ['rbf'], 'gamma': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5],'C': [0.0001, 0.001, 0.01, 0.1, 1, 10]},{'kernel': ['linear'], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10]}]
            elif algorithm.upper() == 'KNN':
                search_space = {'n_neighbors':[3, 5, 7, 9], 'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']}
            else:
                print('The algorithm selected is not supported!')
                return 0
        if scaler:
            self.scaling(scaler)
        gridSearch = GridSearchCV(estimator=clf, param_grid=search_space, scoring=make_scorer(matthews_corrcoef), cv=cv, n_jobs=-1)
        gridSearch.fit(self.descriptors, self.targets)
        print(gridSearch.best_params_)
        self.model = gridSearch.best_estimator_

    def k_fold_cross_validation(self, folds, scaler=None, history=False):
        if scaler:
            self.scaling(scaler)
        kf = StratifiedKFold(n_splits=folds, shuffle=True)
        snList = []
        spList = []
        accuracyList = []
        mccList = []
        rocAucList = []
        kList = []
        if self.model:
            for train, test in kf.split(self.descriptors, self.targets):
                x_tr = self.descriptors[train]
                y_tr = self.targets[train]
                x_te = self.descriptors[test]
                y_te = self.targets[test]
                self.model.fit(x_tr,y_tr)
                preds = self.model.predict(x_te)
                sn, sp, accuracy, mcc, rocAuc, k = self.calculate_metrics(y_te, preds)
                snList.append(sn)
                spList.append(sp)
                accuracyList.append(accuracy)
                mccList.append(mcc)
                rocAucList.append(rocAuc)
                kList.append(k)
            if history:
                return {'SN': snList, 'SP': spList, 'ACC': accuracyList, 'MCC': mccList, 'ROC-AUC': rocAucList, 'Kappa': kList}
            return np.mean(snList), np.mean(spList), np.mean(accuracyList), np.mean(mccList), np.mean(rocAucList), np.mean(kList)
        else:
            print('You need to define a model!')
            return 0

    def load_model_from_file(self, dirname):
        self.numerator_ = np.load(dirname + 'numerator.npy')
        self.denominator_ = np.load(dirname + 'denominator.npy')
        self.model = load(dirname + 'model.joblib')

    def plot_aa_composition(self, targets_names, save=False):
        plt.rcParams.update({'font.size': 15})
        aafreq = np.zeros((len(set(self.targets)), 20), dtype='float64')
        AAs = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        colors = ['#FA6900', '#69D2E7', '#542437', '#53777A', '#CCFC8E', '#9CC4E4']
        for c in sorted(set(self.targets)):
            concatseq = ''.join([self.seqs[i] for i in range(len(self.seqs)) if self.targets[i] == c])
            d_aa = Counter(concatseq)
            for key in d_aa:
                d_aa[key] = d_aa[key] / len(concatseq)
            aafreq[c] = [d_aa[a] for a in AAs]
        fig, ax = plt.subplots(1, 1, figsize=(10.,7.))
        hands = [mpatches.Patch(label=targets_names[i], facecolor=colors[i], alpha=0.8) for i in range(len(targets_names))]
        w = .9 / len(targets_names)
        offsets = np.arange(start=-w, step=w, stop=len(targets_names) * w)
        for i, l in enumerate(aafreq):
            for a in range(20):
                ax.bar(a - offsets[i], l[a], w, color=colors[i], alpha=0.8)
        ax.set_xlim([-1., 20.])
        ax.set_ylim([0, 1.05 * np.max(aafreq)])
        ax.set_xticks(range(20))
        ax.set_xticklabels(sorted(d_aa.keys()), fontweight='bold')
        ax.set_ylabel('Fraction', fontweight='bold', fontsize=14.)
        ax.set_xlabel('Amino Acids', fontweight='bold', fontsize=14.)
        ax.legend(handles=hands, labels=targets_names)
        if save:
            plt.savefig('./aa_composition.png', dpi=300)

    def plot_frequency_histogram(self, targets_names, feature='lens', save=False):
        plt.rcParams.update({'font.size': 15})
        fig = plt.figure()
        fig.set_size_inches(10.,7.)
        if feature == 'lens':
            f = self.get_lens()
            plt.title('Frequency historgram of sequence length')
        else:
            if feature in self.features_names:
                f = self.descriptors[:,self.features_names.index(feature)]
                plt.title('Frequency historgram of ' + feature)
            else:
                print(f'The feature {feature} does not exist!')
                return 0
        for c in sorted(set(self.targets)):
            data_to_plot = [f[i] for i in range(len(f)) if self.targets[i] == c]
            plt.hist(data_to_plot, bins=20, alpha=0.5, label=targets_names[c])
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()       
        if save:
            plt.savefig('./'+ feature + '.png', dpi=300)

    def plot_one_feature(self, targets_names, feature='lens', save=False):
        plt.rcParams.update({'font.size': 15})
        fig = plt.figure()
        fig.set_size_inches(10.,7.)
        if feature == 'lens':
            f = self.get_lens()
            plt.ylabel('Sequence length')
        else:
            if feature in self.features_names:
                f = self.descriptors[:,self.features_names.index(feature)]
                plt.ylabel(feature)
            else:
                print(f'The feature {feature} does not exist!')
                return 0
        data_to_plot = []
        for c in sorted(set(self.targets)):
            data_to_plot.append([f[i] for i in range(len(f)) if self.targets[i] == c])
        plt.boxplot(data_to_plot)
        plt.xticks([i for i in range(1, len(set(self.targets)) + 1)], targets_names)
        plt.show()       
        if save:
            plt.savefig('./'+ feature + '.png', dpi=300)

    def plot_3D(self, targets_names, save=False):
        plt.rcParams.update({'font.size': 15})
        fig = plt.figure(1, figsize=(8., 6.))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
        plt.cla()
        pca = PCA(n_components=3)
        pca.fit(self.descriptors)
        X = pca.transform(self.descriptors)
        y = self.targets
        for label, name in enumerate(targets_names):
            ax.text3D(X[y == label, 0].mean() + label,
                    X[y == label, 1].mean() + label,
                    X[y == label, 2].mean(), name,
                    horizontalalignment='center',
                    bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap=plt.cm.Set3,
                edgecolor='k')
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_title("First three PCA directions")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        if save:
            plt.savefig('./Three-Dimensions-graph.png', dpi=300)

    def predict(self, test_file, do_scaling=False):
        df = pd.read_csv(test_file, index_col='Sequence')
        if self.model:
            if do_scaling:
                X_test = (df.values - self.numerator_) / self.denominator_
                preds = self.model.predict(X_test)
            else:
                preds = self.model.predict(df.values)
            with open('./predictions.csv', 'w', newline='') as outfile:
                text = csv.writer(outfile, delimiter=",")
                text.writerow(['Sequence', 'Prediction'])
                for i in range(len(preds)):
                    text.writerow([df.index.tolist()[i]] + [preds[i]])
        else:
            print('You need to define a model!')
            return 0

    def random_feature_validation(self):
        X_randomized = np.zeros(self.descriptors.shape)
        for i in range(self.descriptors.shape[0]):
            for j in range(self.descriptors.shape[1]):
                X_randomized[i, j] = random.uniform(min(self.descriptors[:,j]), max(self.descriptors[:,j]))
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        snList = []
        spList = []
        accuracyList = []
        mccList = []
        rocAucList = []
        kList = []
        if self.model:
            print('Starting 10-fold cross-validation process')
            for train, test in kf.split(X_randomized, self.targets):
                x_tr = X_randomized[train]
                y_tr = self.targets[train]
                x_te = X_randomized[test]
                y_te = self.targets[test]
                self.model.fit(x_tr,y_tr)
                preds = self.model.predict(x_te)
                sn, sp, accuracy, mcc, rocAuc, k = self.calculate_metrics(y_te, preds)
                snList.append(sn)
                spList.append(sp)
                accuracyList.append(accuracy)
                mccList.append(mcc)
                rocAucList.append(rocAuc)
                kList.append(k)
            return np.mean(snList), np.mean(spList), np.mean(accuracyList), np.mean(mccList), np.mean(rocAucList), np.mean(kList)
        else:
            print('You need to define a model!')
            return 0

    def repeated_holdout_validation(self, nrepetitions, p=0.1,  scaler=None, history=False):
        if scaler:
            self.scaling(scaler)
        snList = []
        spList = []
        accuracyList = []
        mccList = []
        rocAucList = []
        kList = []
        if self.model:
            for i in range(nrepetitions):
                x_tr, x_te, y_tr, y_te = train_test_split(self.descriptors, self.targets, test_size=p)
                self.model.fit(x_tr,y_tr)
                preds = self.model.predict(x_te)
                sn, sp, accuracy, mcc, rocAuc, k = self.calculate_metrics(y_te, preds)
                snList.append(sn)
                spList.append(sp)
                accuracyList.append(accuracy)
                mccList.append(mcc)
                rocAucList.append(rocAuc)
                kList.append(k)
            if history:
                return {'SN': snList, 'SP': spList, 'ACC': accuracyList, 'MCC': mccList, 'ROC-AUC': rocAucList, 'Kappa': kList}
            return np.mean(snList), np.mean(spList), np.mean(accuracyList), np.mean(mccList), np.mean(rocAucList), np.mean(kList)
        else:
            print('You need to define a model!')
            return 0

    def save_model(self, dirname):
        np.save(dirname + 'numerator.npy', self.numerator_)
        np.save(dirname + 'denominator.npy', self.denominator_)
        dump(self.model, dirname + 'model.joblib')

    def scaling(self, scaler='mean-std'):
        if scaler == 'mean-std':
            self.numerator_ = self.descriptors.mean(axis=0)
            self.denominator_ = self.descriptors.std(axis=0)
            self.descriptors = (self.descriptors - self.numerator_) / self.denominator_
        if scaler == 'range': 
            self.numerator_ = np.zeros(self.descriptors.shape[1])
            self.denominator_ = self.descriptors.max(axis=0) - self.descriptors.min(axis=0)
            self.descriptors = (self.descriptors - self.numerator_) / self.denominator_
        if scaler == 'mean-range':
            self.numerator_ = self.descriptors.mean(axis=0)
            self.denominator_ = self.descriptors.max(axis=0) - self.descriptors.min(axis=0)
            self.descriptors = (self.descriptors - self.numerator_) / self.denominator_
        if scaler == 'min-max':
            self.numerator_ = self.descriptors.min(axis=0)
            self.denominator_ = self.descriptors.max(axis=0)
            self.descriptors = (self.descriptors - self.numerator_) / self.denominator_

    def select_model(self, algorithm='RF'):
        if algorithm.upper() == 'RF':
            clf = RandomForestClassifier(n_jobs=-1)
        elif algorithm.upper() == 'SVM':
            clf = SVC()
        elif algorithm.upper() == 'KNN':
            clf = KNeighborsClassifier(n_jobs=-1)
        else:
            print('The algorithm selected is not supported!')
            return 0
        return clf

    def sequence_based_clustering(self):
        seqs = np.asarray(self.seqs)
        similarity = np.array([[levenshtein(s1,s2) for s1 in seqs] for s2 in seqs])
        sc = SpectralClustering(n_clusters=len(set(self.targets)), affinity="precomputed", n_jobs=-1)
        sc.fit(similarity)
        sn, sp, accuracy, mcc, rocAuc, k = self.calculate_metrics(self.targets, sc.labels_)
        return sn, sp, accuracy, mcc, rocAuc, k

    def train(self, scaler=None):
        if scaler:
            self.scaling(scaler)
        if self.model:
            self.model.fit(self.descriptors, self.targets)
        else:   
            print('You need to define a model')
            return 0

    def y_randomization_validation(self, p=0.1):
        y_randomized = np.copy(self.targets)
        for label in set(self.targets):
            indexes = np.random.choice(np.where(self.targets == label)[0], int(len(np.where(self.targets == label)[0]) * p), replace=False)
            y_randomized[indexes] = np.random.choice(list(set(self.targets)- set([label])),1)
        kf = StratifiedKFold(n_splits=10, shuffle=True)
        snList = []
        spList = []
        accuracyList = []
        mccList = []
        rocAucList = []
        kList = []
        if self.model:
            print('Starting 10-fold cross-validation process')
            for train, test in kf.split(self.descriptors, y_randomized):
                x_tr = self.descriptors[train]
                y_tr = y_randomized[train]
                x_te = self.descriptors[test]
                y_te = y_randomized[test]
                self.model.fit(x_tr,y_tr)
                preds = self.model.predict(x_te)
                sn, sp, accuracy, mcc, rocAuc, k = self.calculate_metrics(y_te, preds)
                snList.append(sn)
                spList.append(sp)
                accuracyList.append(accuracy)
                mccList.append(mcc)
                rocAucList.append(rocAuc)
                kList.append(k)
            return np.mean(snList), np.mean(spList), np.mean(accuracyList), np.mean(mccList), np.mean(rocAucList), np.mean(kList)
        else:
            print('You need to define a model!')
            return 0

    