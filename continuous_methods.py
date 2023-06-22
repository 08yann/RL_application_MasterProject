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


# AR and GARCH classes used to simulate data
class ARProcess:

    def __init__(self, coeffs, phi=1,mean=0.):
        self.coeffs = np.array(coeffs)
        self.num_coeffs = len(coeffs)
        self.mean=mean
        self.phi=phi

    def simulate(self, num_samples):
        samples = np.zeros(num_samples + self.num_coeffs - 1)
        samples[:self.num_coeffs] = np.random.randn(self.num_coeffs) + self.mean
        for i in range(self.num_coeffs, num_samples + self.num_coeffs - 1):
            samples[i] = np.dot(self.phi*samples[i-self.num_coeffs:i], self.coeffs[::-1]) + np.random.randn()
        return samples[self.num_coeffs:]


class ARCHProcess:
    def __init__(self, coeffs,mean=0):
        self.coeffs = coeffs
        self.mean = mean
    def simulate(self, nb_samples):
        samples = np.zeros(nb_samples + 100)
        z = np.random.randn(nb_samples + 100 + 1)

        samples[0] = self.mean + np.sqrt(self.coeffs[0] / (1 - self.coeffs[1])) * np.random.randn()
        for i in range(1, nb_samples + 100):
            samples[i] = self.mean + z[i] * np.sqrt(self.coeffs[0] + self.coeffs[1] * (samples[i - 1]-self.mean) ** 2)

        return samples[100:]


class GARCHProcess:
    def __init__(self, coeffs,distr='normal'):

        if distr=='normal':
            self.distr=distr
        else:
            self.distr='student'

        if coeffs[1]+coeffs[2] < 1:
            self.coeffs = coeffs
        else:
            coeffs_corrected=coeffs
            coeffs_corrected[2]=1-coeffs[1]
            print('Error : coefficient alpha and beta must add to less than one !!')

    def simulate(self, nb_samples):
        samples = np.zeros(nb_samples + 100)
        if self.distr=='normal':
            z = np.random.randn(nb_samples + 100)
            previous_sig=self.coeffs[0] / (1 - self.coeffs[1]-self.coeffs[2])
            samples[0] = np.sqrt(previous_sig) * np.random.randn()
        else:
            z = np.random.standard_t(1,nb_samples + 100 + 1)
            previous_sig=self.coeffs[0] / (1 - self.coeffs[1]-self.coeffs[2])
            samples[0] = np.sqrt(previous_sig) * np.random.standard_t(3)

        for i in range(1, nb_samples + 100):
            previous_sig=self.coeffs[0] + self.coeffs[1] * samples[i - 1] ** 2 + self.coeffs[2] * previous_sig ** 2
            samples[i] = z[i] * np.sqrt(previous_sig)

        return samples[100:]

class eGARCHProcess:
    def __init__(self,omega,alpha,beta,leverage):
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.leverage = leverage

    def simulate(self, nb_samples):
        samples = np.zeros(nb_samples + 100)
        z = np.random.standard_t(2, nb_samples + 100)

        previous_sig = self.omega / (1 - self.alpha - self.beta)
        samples[0] = previous_sig * np.random.randn()

        for i in range(1, nb_samples + 100):
            previous_sig = np.exp(self.omega + self.alpha * (np.abs(z[i - 1]) - np.sqrt(2 / math.pi)) + self.beta * np.log(previous_sig) + self.leverage * z[i - 1])
            samples[i] = z[i] * np.sqrt(previous_sig)
        return samples[100:]

class corr_GARCH2d:
    def __init__(self,omega,alpha, beta,corr=0,mean=[0,0]):
        self.omega=omega
        self.alpha=alpha
        self.beta = beta
        self.corr=corr
        self.mean=np.array(mean).reshape(-1,1)

    def simulate(self,nb_samples):
        samples = np.zeros((2, nb_samples + 100))
        z = np.random.randn(2, nb_samples + 100)

        previous_sig1 = self.omega[0] / (1 - self.alpha[0] - self.beta[0])
        previous_sig2 = self.omega[1] / (1 - self.alpha[1] - self.beta[1])
        samples[:, 0] = np.squeeze(self.mean + np.matmul(np.matmul(np.matmul(np.diag([previous_sig1,previous_sig2]), np.array([[1,self.corr],[self.corr,1]])),np.diag([previous_sig1,previous_sig2])),np.random.randn(2,1)))
        for i in range(1, nb_samples + 100):
            previous_sig1 = self.omega[0] + self.alpha[0]*(samples[0,i-1]-self.mean[0,0])**2 + self.beta[0]*previous_sig1**2
            previous_sig2 = self.omega[1] + self.alpha[1]*(samples[1,i-1]-self.mean[1,0])**2 + self.beta[1]*previous_sig2**2

            cov=np.matmul(np.matmul(np.diag([np.sqrt(previous_sig1),np.sqrt(previous_sig2)]), np.array([[1,self.corr],[self.corr,1]])),np.diag([np.sqrt(previous_sig1),np.sqrt(previous_sig2)]))
            samples[:, i] = np.squeeze(self.mean + np.matmul(scipy.linalg.cholesky(cov), z[:, i].reshape(-1,1)))

        return samples[:, 100:]


class multi_GARCH:
    def __init__(self,omega,alpha, beta):
        self.omega=omega
        self.alpha=alpha
        self.beta = beta

    def simulate(self,nb_samples):
        samples = np.zeros((self.omega.shape[0], nb_samples + 100))
        z = np.random.randn(self.omega.shape[0], nb_samples + 100)

        previous_sig = np.array([[self.omega[0,0]/(1-self.alpha[0,0]-self.beta[0,0]),0],[0,self.omega[1,1]/(1-self.alpha[1,1]-self.beta[1,1])]])


        samples[:, 0] = np.squeeze(np.matmul(previous_sig,np.random.randn(self.omega.shape[0],1)))
        for i in range(1, nb_samples + 100):
            previous_sig = self.omega + self.alpha * np.matmul(samples[:,i-1].reshape(-1,1),samples[:,i-1].reshape(-1,1).transpose()) + self.beta * previous_sig

            samples[:, i] = np.squeeze(np.matmul(scipy.linalg.cholesky(previous_sig), z[:, i].reshape(-1,1)))

        return samples[:, 100:]

class diag_BEKK_GARCH:
    def __init__(self,G,E, F):
        self.G=G
        self.E=np.diag(E)
        self.F = np.diag(F)

    def simulate(self,nb_samples):
        samples = np.zeros((self.G.shape[0], nb_samples + 100))
        z = np.random.randn(self.G.shape[0], nb_samples + 100)

        previous_sig = np.matmul(self.G.transpose(),self.G) * np.array([[1/(1-self.E[0,0]**2-self.F[0,0]**2),1],[1,1/(1-self.E[1,1]**2-self.F[1,1]**2)]])

        samples[:, 0] = np.squeeze(np.matmul(previous_sig,np.random.randn(self.G.shape[0],1)))
        for i in range(1, nb_samples + 100):
            previous_sig = np.matmul(self.G.transpose(),self.G) + np.matmul(np.matmul(self.E.transpose(), np.matmul(samples[:,i-1].reshape(-1,1),samples[:,i-1].reshape(-1,1).transpose())),self.E) + np.matmul(np.matmul(self.F.transpose(), previous_sig),self.F)

            samples[:, i] = np.squeeze(np.matmul(scipy.linalg.cholesky(previous_sig), z[:, i].reshape(-1,1)))

        return samples[:, 100:]


class GARCHProcess_vech:
    def __init__(self,omega,A,B):
        self.omega=omega
        self.A=A
        self.B=B

    def simulate(self, nb_samples):
        size = 2

        samples = np.zeros((size,nb_samples + 100))
        z = np.random.randn(size,nb_samples + 100)

        previous_cov=np.eye(size)
        samples[:,0] = np.random.randn(size)
        for i in range(1, nb_samples + 100):
            cov_vech = vech(self.omega) + np.matmul(self.A,vech(np.matmul(z[:,i-1].reshape(-1,1),z[:,i-1].reshape(-1,1).transpose())))+ np.matmul(self.B,vech(previous_cov))
            previous_cov=matrix_from_vech(cov_vech)
            samples[:,i] =  np.matmul(scipy.linalg.sqrtm(previous_cov),z[:,i])

        return samples[:,100:]


class ARProcess2d:
    # Two-dimensional AR processes with some correlation for the white noise
    def __init__(self, coeffs1, mean=np.zeros((2,1)), coeffs2=[], variance=np.eye(2), corr=0.0, gamma=0.98):
        self.coeffs1 = np.array(coeffs1)
        self.order1 = len(coeffs1)-1
        if coeffs2 == []:
            self.coeffs2 = np.array(coeffs1)
            self.order2 = len(coeffs1)-1
        else:
            self.coeffs2 = np.array(coeffs2)
            self.order2 = len(coeffs2)-1

        self.order = max(self.order1, self.order2)
        self.min_order = min(self.order1, self.order2)

        self.mean = np.array(mean)
        self.variance = variance
        self.corr = corr
        self.gamma=gamma

        # Create a correlation matrix with the given correlation coefficient
        self.corr_matrix = np.array([[1.0, corr], [corr, 1.0]])

    def simulate(self, num_samples):
        noise = np.random.multivariate_normal(np.zeros(2), np.dot(self.variance, self.corr_matrix),
                                               num_samples + self.order+100).transpose()
        samples = np.zeros((2,num_samples+self.order+100))
        for i in range(self.order, num_samples + self.min_order+100):

            samples[0, i] = np.dot(samples[0, i - self.order1:i][::-1], self.coeffs1[1:]) + self.coeffs1[0]*noise[0,i] + self.mean[0]
            samples[1, i] = np.dot(samples[1, i - self.order2:i][::-1], self.coeffs2[1:]) + self.coeffs2[0]*noise[1, i] + self.mean[1]

        # Don't consider the first 100 samples to erase the bias induced by setting the first values at zero.
        return pd.DataFrame(samples[:, 100+self.order:],index=['S1','S2'])

    def state_as_list(self,state):
        list_state = pd.Series(state[0, :]).to_list()
        list_state.extend(pd.Series(state[1, :]).to_list())

        return list_state

    def discrete_true_qval(self,delta=1e-5,min_val=0.96,max_val=1.04,rounding=0.01, Q0=0):
        # Compute the true optimal q-value function for the discretized AR process.
        # Straightforward by defining the transition matrix over discretized states and performing fixed-point iteration
        # on such transitions.
        nb_state = int((max_val-min_val)/rounding)+1
        error = np.inf
        true_qval = dict()

        if self.order == 1:
            for s1 in [min_val + x * rounding for x in range(nb_state)]:
                for s2 in [min_val + x * rounding for x in range(nb_state)]:
                    true_qval[str(self.state_as_list(np.array([[s1],[s2]])))]=Q0*np.ones(11)
            time = 1
            trans_proba = self.transition_matrix(min_val,max_val,rounding)
            while error > delta:
                temp_error = 0
                previous_qval = copy.deepcopy(true_qval)
                for s1 in [min_val + x * rounding for x in range(nb_state)]:
                    for s2 in [min_val + x * rounding for x in range(nb_state)]:
                        current_state=str(self.state_as_list(np.array([[s1],[s2]])))
                        true_qval[current_state]=np.zeros(11)
                        for s1_next in [min_val + x * rounding for x in range(nb_state)]:
                            for s2_next in [min_val + x * rounding for x in range(nb_state)]:
                                next_state=str(self.state_as_list(np.array([[s1_next],[s2_next]])))
                                state_trans= current_state+ '+'+ next_state
                                temp_proba = trans_proba[state_trans]
                                true_qval[current_state] += temp_proba * (np.array([reward(action, np.array([[s1_next],[s2_next]]))[0, 0] for action in range(11)]) + self.gamma * np.max(previous_qval[next_state]))

                        if temp_error < np.max(np.abs(previous_qval[current_state]-true_qval[current_state])):
                            temp_error = np.max(np.abs(previous_qval[current_state]-true_qval[current_state]))
                error=copy.deepcopy(temp_error)
                time += 1
            print("Time needed to converge with delta = ", delta, " : ", time)

        elif self.order == 2:
            for s1_1 in [min_val + x * rounding for x in range(nb_state)]:
                for s2_1 in [min_val + x * rounding for x in range(nb_state)]:
                    for s1_2 in [min_val + x * rounding for x in range(nb_state)]:
                        for s2_2 in [min_val + x * rounding for x in range(nb_state)]:
                            true_qval[str(self.state_as_list(np.array([[s1_1,s1_2], [s2_1,s2_2]])))] = Q0 * np.ones(11)
            time = 1
            trans_proba = self.transition_matrix(min_val, max_val, rounding)
            while error > delta:
                temp_error=0
                previous_qval = copy.deepcopy(true_qval)
                for s1_1 in [min_val + x * rounding for x in range(nb_state)]:
                    for s2_1 in [min_val + x * rounding for x in range(nb_state)]:
                        for s1_2 in [min_val + x * rounding for x in range(nb_state)]:
                            for s2_2 in [min_val + x * rounding for x in range(nb_state)]:
                                current_state = str(self.state_as_list(np.array([[s1_1,s1_2], [s2_1,s2_2]])))
                                true_qval[current_state] = np.zeros(11)
                                for s1_next in [min_val + x * rounding for x in range(nb_state)]:
                                    for s2_next in [min_val + x * rounding for x in range(nb_state)]:
                                        next_state=str(self.state_as_list(np.array([[s1_2,s1_next],[s2_2,s2_next]])))
                                        state_trans = current_state + '+' + next_state
                                        temp_proba = trans_proba[state_trans]
                                        true_qval[current_state] += temp_proba * (np.array(
                                            [reward(action, np.array([[s1_next], [s2_next]]))[0, 0] for action in range(11)]) + self.gamma * np.max(
                                            previous_qval[next_state]))
                                if temp_error < np.max(np.abs(previous_qval[current_state] - true_qval[current_state])):
                                    temp_error = np.max(np.abs(previous_qval[current_state] - true_qval[current_state]))
                error = copy.deepcopy(temp_error)
                time += 1
            print("Time needed to converge with delta = ", delta, " : ", time)

        return true_qval

    def transition_matrix(self,min_val=0.96,max_val=1.04,rounding=0.01):
        # Function returning the transition for discretized state. It exploits the definition of the AR processes and
        # the distribution of the white.
        nb_state=int((max_val-min_val)/rounding)+1
        trans=dict()
        if self.order ==1 :
            for s1 in range(nb_state):
                state1=min_val + s1 * rounding
                for s2 in range(nb_state):
                    state2 = min_val + s2 * rounding

                    distr=scipy.stats.multivariate_normal(np.zeros(2),np.dot(self.variance, self.corr_matrix))
                    for s1_next in range(nb_state):
                        next_state1=min_val + s1_next*rounding
                        for s2_next in range(nb_state):
                            next_state2 = min_val + s2_next * rounding
                            if (self.order1==1) & (self.order2==1):
                                up_bound1=((np.log(next_state1+rounding/2)-self.mean[0]-self.coeffs1[1]*np.log(state1))/self.coeffs1[0])[0]
                                up_bound2=((np.log(next_state2+rounding/2)-self.mean[1]-self.coeffs2[1]*np.log(state2))/self.coeffs2[0])[0]

                                low_bound1 = ((np.log(next_state1 - rounding / 2) - self.mean[0] - self.coeffs1[1] * (
                                    np.log(state1))) / self.coeffs1[0])[0]
                                low_bound2 = ((np.log(next_state2 - rounding / 2) - self.mean[1] - self.coeffs2[1] * (
                                    np.log(state2))) / self.coeffs2[0])[0]

                                trans[str(self.state_as_list(np.array([[state1],[state2]])))+'+'+str(self.state_as_list(np.array([[next_state1],[next_state2]])))] = distr.cdf(
                                    np.array([up_bound1, up_bound2])) - distr.cdf(
                                    np.array([up_bound1, low_bound2])) - distr.cdf(
                                    np.array([low_bound1, up_bound2]))+ distr.cdf(
                                    np.array([low_bound1 , low_bound2]))
        else:
            for s1_1 in range(nb_state):
                state1_1=min_val + s1_1 * rounding
                for s2_1 in range(nb_state):
                    state2_1 = min_val + s2_1 * rounding
                    for s1_2 in range(nb_state):
                        state1_2 = min_val + s1_2 * rounding
                        for s2_2 in range(nb_state):
                            state2_2 = min_val + s2_2 * rounding

                            distr=scipy.stats.multivariate_normal(np.zeros(2),np.dot(self.variance, self.corr_matrix))
                            for s1_next in range(nb_state):
                                next_state1=min_val + s1_next*rounding
                                for s2_next in range(nb_state):
                                    next_state2 = min_val + s2_next * rounding
                                    if self.order1 == 1:
                                        up_bound1 = ((np.log(next_state1+rounding/2)-self.mean[0]-self.coeffs1[1]*np.log(state1_2))/self.coeffs1[0])[0]
                                        low_bound1 = ((np.log(next_state1 - rounding / 2) - self.mean[0] - self.coeffs1[1] * (np.log(state1_2))) / self.coeffs1[0])[0]
                                    else:
                                        up_bound1 = ((np.log(next_state1 + rounding / 2) - self.mean[0] - self.coeffs1[1] * np.log(state1_2)-self.coeffs1[2]* np.log(state1_1)) / self.coeffs1[0])[0]
                                        low_bound1 = ((np.log(next_state1 - rounding / 2) - self.mean[0] - self.coeffs1[1] * np.log(state1_2) - self.coeffs1[2]* np.log(state1_1)) / self.coeffs1[0])[0]

                                    if self.order2 == 1:
                                        up_bound2 = ((np.log(next_state2+rounding/2)-self.mean[1]-self.coeffs2[1]*np.log(state2_2))/self.coeffs2[0])[0]
                                        low_bound2 = ((np.log(next_state2 - rounding / 2) - self.mean[1] - self.coeffs2[1] * np.log(state2_2)) / self.coeffs2[0])[0]
                                    else:
                                        up_bound2=((np.log(next_state2+rounding/2)-self.mean[1]-self.coeffs2[1]*np.log(state2_2)-self.coeffs2[2]*np.log(state2_1))/self.coeffs2[0])[0]
                                        low_bound2 =((np.log(next_state2 - rounding / 2) - self.mean[1] - self.coeffs2[1] * np.log(state2_2) - self.coeffs2[2] * np.log(state2_1)) / self.coeffs2[0])[0]


                                    trans[str(self.state_as_list(np.array([[state1_1,state1_2],[state2_1,state2_2]])))+'+'+str(self.state_as_list(np.array([[state1_2,next_state1],[state2_2,next_state2]])))] = distr.cdf(np.array([up_bound1, up_bound2])) - distr.cdf(np.array([up_bound1, low_bound2])) - distr.cdf(np.array([low_bound1, up_bound2]))+ distr.cdf(np.array([low_bound1 , low_bound2]))

        return trans


class MAProcess:
    def __init__(self, coeffs, mean=0.0, variance=1.0):
        self.coeffs = np.array(coeffs)
        self.order = len(coeffs)
        self.mean = mean
        self.variance = variance

    def simulate(self, num_samples):
        noise = np.random.randn(num_samples + self.order) * np.sqrt(self.variance) + self.mean
        samples = np.zeros(num_samples)
        for i in range(self.order, num_samples + self.order):
            samples[i - self.order] = np.dot(noise[i-self.order:i], self.coeffs)
        return samples




class MAProcess2d:
    def __init__(self, coeffs1, mean=np.zeros((2,1)), coeffs2=[],variance=np.eye(2), corr=0.0):
        self.coeffs1 = np.array(coeffs1)
        self.order1 = len(coeffs1)
        if coeffs2==[]:
            self.coeffs2 = np.array(coeffs1)
            self.order2 = len(coeffs1)
        else:
            self.coeffs2=np.array(coeffs2)
            self.order2=len(coeffs2)

        self.order=max(self.order1,self.order2)
        self.min_order=min(self.order1,self.order2)

        self.mean = np.array(mean)
        self.variance = variance
        self.corr = corr

        # Create a correlation matrix with the given correlation coefficient
        self.corr_matrix = np.array([[1.0, corr], [corr, 1.0]])

    def simulate(self, num_samples):
        noise = np.random.multivariate_normal(np.zeros(2), np.dot(self.variance, self.corr_matrix),
                                               num_samples + self.order).transpose()
        samples = np.zeros((2, num_samples))
        for i in range(self.order, num_samples + self.min_order):
            samples[0,i - self.order1] = self.mean[0] + np.dot(noise[0,i-self.order1:i], self.coeffs1)
            samples[1,i - self.order2] = self.mean[1] + np.dot(noise[1,i-self.order2:i], self.coeffs2)

        return pd.DataFrame(samples,index=['S1','S2'])

    def state_as_list(self,state):
        list_state = pd.Series(state[0, :]).to_list()
        list_state.extend(pd.Series(state[1, :]).to_list())

        return list_state
    def discrete_true_qval(self,delta=1e-5,min_val=0.96,max_val=1.04,rounding=0.01,Q0=0):
        nb_state=int((max_val-min_val)/rounding)+1
        error=np.inf
        true_qval=dict()
        for s1 in [min_val + x * rounding for x in range(nb_state)]:
            for s2 in [min_val + x * rounding for x in range(nb_state)]:
                true_qval[str(self.state_as_list(np.array([[s1],[s2]])))]=Q0*np.ones(11)
        time=1
        while error>delta:
            temp_error=0
            previous_qval = copy.deepcopy(true_qval)
            distr=scipy.stats.multivariate_normal(np.zeros(2),np.dot(self.variance, self.corr_matrix))
            for s1 in [min_val + x * rounding for x in range(nb_state)]:
                for s2 in [min_val + x * rounding for x in range(nb_state)]:
                    current_state=str(self.state_as_list(np.array([[s1],[s2]])))
                    true_qval[current_state]=0
                    for s1_next in [min_val + x * rounding for x in range(nb_state)]:
                        for s2_next in [min_val + x * rounding for x in range(nb_state)]:

                            temp_proba=distr.cdf(np.array())
                            next_state=str(self.state_as_list(self.get_true_state(np.array([[s1_next],[s2_next]]))))
                            true_qval[current_state] += temp_proba * (np.array([reward(action, self.get_true_state(np.array([[s1_next],[s2_next]])))[0, 0] for action in range(11)]) + self.gamma * np.max(previous_qval[next_state]))
                    if temp_error < np.max(np.abs(previous_qval[current_state]-true_qval[current_state])):
                        temp_error = np.max(np.abs(previous_qval[current_state]-true_qval[current_state]))
            error=copy.deepcopy(temp_error)
            time += 1
        print("Time needed to converge with delta = ", delta, " : ", time)

        return true_qval


class DQN2:
    # Deep Q-network model
    def __init__(self, train_data, window=1, nb_asset=2, nb_action=11, gamma=0.98, batch_size=32, lr=0.1, units1=64,
                 units2=64, last_activation='relu', nb_epochs=1,adaptive_lr=False,tau=1):
        self.window = window
        self.lr = lr
        self.batch_size = batch_size
        self.last_activation = last_activation
        self.units1 = units1
        self.units2 = units2
        self.nb_epochs = nb_epochs
        self.nb_asset = train_data.shape[0]
        self.nb_action = nb_action
        self.model = self.build_model()
        self.model.summary()
        self.train_data = train_data
        self.gamma = gamma
        self.target_model = self.build_model()
        self.update_target_model()
        self.adaptive_lr=adaptive_lr
        self.tau = tau

    def build_model(self):
        # Model definition
        inp = tf.keras.layers.Input(shape=(self.nb_asset * self.window,), name="Prev_ret")
        lay = tf.keras.layers.Dense(units=self.units1, activation='relu', name='Dense_1')(inp)
        lay = tf.keras.layers.Dense(units=self.units2, activation='relu', name='Dense_2')(lay)
        q_val = tf.keras.layers.Dense(units=self.nb_action, activation=self.last_activation, name='linear',
                                      kernel_initializer=keras.initializers.Ones(),
                                      bias_initializer=keras.initializers.Zeros())(lay)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

        model = tf.keras.models.Model(inp, q_val)
        model.compile(loss='mse', optimizer=optimizer)
        return model

    def update_target_model(self):
        # Update weights of target model.
        self.target_model.set_weights(self.model.get_weights())

    def chooseW(self, state):
        q_val = self.model.predict(state,verbose=0)[0]
        print(q_val)
        w = np.argmax(q_val) / (self.nb_action - 1)
        return w

    def train(self):
        # Learning for DQN by performing the gradient descent for each batch_size time-steps
        # and, then, update target network.
        if self.window == 2:
            nb_episode = int(np.floor(self.train_data.shape[1]) / (1 + self.batch_size))
        else:
            nb_episode = int(np.floor(self.train_data.shape[1]) / self.batch_size)

        print('NB. episodes : ', nb_episode)

        for e in range(nb_episode - 1):
            if self.window == 2:
                update_input = self.get_state(np.array(
                    self.train_data.iloc[:, e * (1 + self.batch_size):(e + 1) * (1 + self.batch_size)])).transpose()
                update_nextstate = self.get_state(np.array(self.train_data.iloc[:,
                                                           e * (1 + self.batch_size) + 1:(e + 1) * (
                                                                       1 + self.batch_size) + 1])).transpose()
            else:
                update_input = self.get_state(
                    np.array(self.train_data.iloc[:, e * self.batch_size:(e + 1) * self.batch_size])).transpose()
                update_nextstate = self.get_state(np.array(
                    self.train_data.iloc[:, e * self.batch_size + 1:(e + 1) * self.batch_size + 1])).transpose()

            # q_val = self.model.predict(update_input)

            q_val_next = self.target_model.predict(update_nextstate,verbose=0)
            # q_val_next = self.model.predict(update_nextstate,verbose=0)
            q_val = np.zeros((self.batch_size, self.nb_action))
            for i in range(self.batch_size):
                for a in range(self.nb_action):
                    rew_temp = np.dot(np.array([[a / (self.nb_action - 1)], [1 - a / (self.nb_action - 1)]]).transpose(),
                           update_nextstate[i, (self.window - 1) * self.nb_asset:])[0]
                    q_val[i, a] = np.log(rew_temp) + self.gamma * np.max(q_val_next[i, :])

            # Change learning rate if we utilize the adaptive version.
            if self.adaptive_lr:
                keras.backend.set_value(self.model.optimizer.learning_rate, self.lr/(1+e)**self.tau)

            self.model.fit(update_input, q_val, batch_size=self.batch_size, epochs=self.nb_epochs, verbose=0)
            self.update_target_model()

    def exploit(self, test_data, test=True, init_inv=100):
        # Exploit learned policy
        w = np.zeros((2, test_data.shape[1] - self.window))
        w[0, :] = np.argmax(self.model.predict(self.get_state(np.array(test_data.iloc[:, :-1])).transpose(),verbose=0),
                            axis=1) / (self.nb_action - 1)
        w[1, :] = 1 - w[0, :]

        # ret = np.sum(w * test_data.iloc[:, 1:])
        # plt.plot(np.cumprod(ret))
        # plt.plot(np.cumprod(test_data.iloc[0,:]))
        # plt.plot(np.cumprod(test_data.iloc[1,:]))
        # plt.legend(['Trained policy','S1','S2'])
        # plt.show()

        return w

    def compare_DQN2_discrete(self, true_qval, plot=True,print_wrong_action=False):
        # Compare the learned policy from DQN to the discretized optimal q-value function.
        states = np.zeros((self.nb_asset * self.window, len(true_qval.keys())))
        for i in range(len(true_qval.keys())):
            states[:, i] = np.array(json.loads(list(true_qval.keys())[i]))

        if self.window == 2:
            temp = copy.deepcopy(states)
            states[1, :] = temp[2, :]
            states[2, :] = temp[1, :]

        pred = self.model.predict(states.transpose(),verbose=0)

        estimated_qval = dict()
        for i in range(len(true_qval.keys())):
            estimated_qval[list(true_qval.keys())[i]] = pred[i, :]

        no_wrong_action= True
        string_param=' Parameters are : lr = '+str(self.lr) + ', window = '+str(self.window)+', batch size = '+ str(self.batch_size)+', nb. epochs = '+str(self.nb_epochs)
        if self.units1 !=64:
            string_param += ', units1 = '+str(self.units1)
        if self.units2 !=64:
            string_param += ', units1 = '+str(self.units2)

        if self.last_activation!='relu':
            string_param += ', last activ. = '+self.last_activation

        if plot:
            print(string_param,flush=True)
            print('\n','Wrong actions : ',flush=True)
            nb_wrong_action=0
            for k in true_qval.keys():
                if np.argmax(true_qval[k]) != np.argmax(estimated_qval[k]):
                    nb_wrong_action+=1

            if nb_wrong_action ==0:
                print('Optimal policy derived !!', flush=True)
            elif nb_wrong_action <=10:
                for k in true_qval.keys():
                    if np.argmax(true_qval[k]) != np.argmax(estimated_qval[k]):
                        print(k, '; Action from discrete policy : ', np.argmax(true_qval[k]), '; Action from DQN2 algorithm : ',np.argmax(estimated_qval[k]),flush=True)
            elif print_wrong_action:
                for k in true_qval.keys():
                    if np.argmax(true_qval[k]) != np.argmax(estimated_qval[k]):
                        print(k, '; Action from discrete policy : ', np.argmax(true_qval[k]),
                              '; Action from DQN2 algorithm : ', np.argmax(estimated_qval[k]), flush=True)

        # print(estimated_qval)

        return [conv_qtables(true_qval, estimated_qval),estimated_qval, string_param]

    def get_state(self, data):
        state = copy.deepcopy(data)

        if self.window == 2:
            state = np.vstack((state[:, :-1], state[:, 1:]))

        return state


def discretize(returns, min_val=0.96,max_val=1.04,rounding=0.01):
    # Discretization routine
    disc=np.round(returns, decimals=int(np.log10(1/rounding)))
    disc[disc>max_val]=max_val
    disc[disc<min_val]=min_val
    return disc


def vech(array):
    v = array[np.tril_indices(array.shape[0], k = 0)]
    return v

def matrix_from_vech(v):
    size=int((-1+np.sqrt(1+8*v.shape[0]))/2)
    arr=np.zeros((size,size))
    arr[np.tril_indices(arr.shape[0], k = 0)]=v
    arr = arr + arr.T - np.diag(np.diag(arr))
    return arr


def delay_simulation(data,delay=1):
    # Delay the second variable from a 2-dimensional process.
    sim_data=np.zeros((2,data.shape[1]-delay))
    sim_data[0,:]= data[0,:-delay]
    sim_data[1,:]=data[1,delay:]

    return sim_data


def create_param_DQN(train_data,test_data, window=1, batch_size=32, lr=0.1, units1=64,
                 units2=64, last_activation='relu', nb_epochs=100,true_qval=None,adaptive_lr=False,tau=1):
    # As for Q-Learning we introduce functions to perform multiprocessing on the deep Q-network to speed up computations
    # for grid search on the parameters of the Q-networks.

    param = dict()
    param['test_data']=test_data
    param['train_data']=train_data
    param['w']=window
    param['batch_size']=batch_size
    param['lr']=lr
    param['units1']=units1
    param['units2']=units2
    param['last_act']=last_activation
    param['nb_epochs']=nb_epochs

    param['true_qval']=true_qval

    param['adaptive_lr'] = adaptive_lr

    param['tau'] = tau

    return param


def multi_train(params):
    # DQN for a dictionary input
    train_data=params['train_data']
    window = params['w']
    nb_asset = 2
    nb_action = 11
    gamma = 0.98
    batch_size = params['batch_size']
    lr = params['lr']
    units1 = params['units1']
    units2 = params['units2']
    last_activation = params['last_act']
    nb_epochs = params['nb_epochs']
    adapt_lr=params['adaptive_lr']
    tau = params['tau']

    pol=DQN2(train_data, window , nb_asset , nb_action , gamma, batch_size, lr , units1, units2, last_activation , nb_epochs,adapt_lr,tau)
    pol.train()

    res = analysis_result(pol, np.array(params['test_data']))

    result=dict()

    result['exploit_test']=res
    result['weights']=pol.model.get_weights()
    if params['true_qval'] is not None:
        if (len(list(params['true_qval'].keys())[0])<=13) & (params['w']==2):
            err = pol.compare_DQN2_discrete(extend_qtable_discrete(params['true_qval']),plot=False)
        else:
            err = pol.compare_DQN2_discrete(params['true_qval'],plot=False)

        result['error']=err
    return result


def plot_res_from_multi(result, true_qval, print_wrong_actions=False):
    # Plot essential metrics on the performance of the DQN
    print('\n \n')
    string_param = result['error'][2]
    print(string_param)

    nb_wrong_action = 0

    if (len(list(true_qval.keys())[0]) <= 13) & (len(list(result['error'][1].keys())[0]) > 13):
        q_val = extend_qtable_discrete(true_qval)
    else:
        q_val = true_qval
    print('\n', 'Wrong actions : ')
    for k in q_val.keys():
        if np.argmax(q_val[k]) != np.argmax(result['error'][1][k]):
            nb_wrong_action += 1

    if nb_wrong_action == 0:
        print('Optimal policy derived !!')

    elif nb_wrong_action <= 10:
        for k in q_val.keys():
            if np.argmax(q_val[k]) != np.argmax(result['error'][1][k]):
                print(k, '; Action from discrete policy : ', np.argmax(q_val[k]), '; Action from DQN2 algorithm : ',
                      np.argmax(result['error'][1][k]))

    else:
        if print_wrong_actions:
            for k in q_val.keys():
                if np.argmax(q_val[k]) != np.argmax(result['error'][1][k]):
                    print(k, '; Action from discrete policy : ', np.argmax(q_val[k]), '; Action from DQN2 algorithm : ',
                          np.argmax(result['error'][1][k]))
        else:
            print('There are a lot of wrong actions : ', result['error'][0]['nb_wrong'])

    plot_error_per_state(result['error'][0])
