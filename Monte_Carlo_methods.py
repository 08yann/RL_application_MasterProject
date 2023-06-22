import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import keras
import json
import tensorflow as tf
import scipy
import scipy.stats
import math
from discrete_methods import *


class Monte_Carlo:
    # Monte-Carlo method for the case when training data corresponds to two uncorrelated Markov chains.
    def __init__(self,training_data):
        self.training_data = np.array(copy.deepcopy(training_data))
        self.nb_states1=len(np.unique(self.training_data[0,:]))
        self.nb_states2=len(np.unique(self.training_data[1,:]))
        self.delta1=np.round(np.max(np.unique(self.training_data[0,:]))-np.min(np.unique(self.training_data[0,:])),5)/(self.nb_states1-1)
        self.delta2=np.round(np.max(np.unique(self.training_data[1, :])) - np.min(np.unique(self.training_data[1, :])),5) / (
                    self.nb_states2 - 1)
        self.trans1=np.zeros((self.nb_states1,self.nb_states1))
        self.trans2=np.zeros((self.nb_states2,self.nb_states2))
        self.nb_visits1 = np.zeros((self.nb_states1, self.nb_states1))
        self.nb_visits2 = np.zeros((self.nb_states2, self.nb_states2))
        self.q_val={}

    def get_number_state1(self, state):
        min_val = int(np.ceil(self.nb_states1 / 2)) - 1

        return int((state-1)/self.delta1 + min_val)

    def get_number_state2(self, state):
        min_val = int(np.ceil(self.nb_states2 / 2)) - 1

        return int((state - 1) / self.delta2 + min_val)

    def state_as_list(self, state):
        list_state = pd.Series(state[0, :]).to_list()
        list_state.extend(pd.Series(state[1, :]).to_list())

        return list_state

    def train(self,true_qval=None, timestep_error=10000):
        # Straightforward, simply derive empirical transition matrix for each process in the first or second dimension.
        max_error=list()
        nb_wrong_action = list()
        for i in range(self.training_data.shape[1]-1):
            s1 = self.get_number_state1(self.training_data[0,i])
            s2 = self.get_number_state2(self.training_data[1,i])
            s1_next = self.get_number_state1(self.training_data[0,i+1])
            s2_next = self.get_number_state2(self.training_data[1,i+1])

            self.nb_visits1[s1,s1_next] += 1
            self.nb_visits2[s2,s2_next] += 1

            if (i%timestep_error==0) & (i!=0):
                if true_qval is not None:
                    self.trans1 = self.nb_visits1 / i
                    self.trans2 = self.nb_visits2 / i

                    # Create Markov chain processes using the empirical and use its method to derive the optimal q-value
                    # table through fixed-point iteration.
                    MC_estimated = MarkovChain_2D(delta1=self.delta1, p1=self.trans1, p2=self.trans2,
                                                  delta2=self.delta2, gamma=0.98)
                    estimated_qval = MC_estimated.true_qval(1e-8)
                    err_dict=conv_qtables(true_qval,estimated_qval)

                    max_error.extend([err_dict['max']])
                    nb_wrong_action.extend([err_dict['nb_wrong']])

        self.trans1 = self.nb_visits1/self.training_data.shape[1]
        self.trans2 = self.nb_visits2/self.training_data.shape[1]

        MC_estimated = MarkovChain_2D(delta1=self.delta1, p1=self.trans1,p2=self.trans2,delta2=self.delta2,gamma=0.98)

        self.q_val = MC_estimated.true_qval(1e-8)

        if true_qval is not None:
            err_dict = conv_qtables(true_qval, self.q_val)
            max_error.extend([err_dict['max']])
            nb_wrong_action.extend([err_dict['nb_wrong']])

            plt.plot(timestep_error * np.arange(1, len(max_error) + 1), np.array(max_error))
            plt.ticklabel_format(useOffset=False)
            plt.title(
                'Evolution of maximal absolute error compared as a function of number of training steps')
            plt.xlabel('Size of training data')
            plt.ylabel('Max. absolute error')
            plt.show()

            plt.plot(timestep_error * np.arange(1, len(max_error) + 1), np.array(nb_wrong_action))
            plt.title(
                'Evolution of number of wrong action taken by the learned policy as a function of training steps')
            plt.xlabel('Size of training data')
            plt.ylabel('Nb. of wrong actions')
            plt.show()

        return [max_error,nb_wrong_action]
