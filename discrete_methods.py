# Implementation of different discrete classes such as Markov chains, Binomial processes for simulation
# As well as Q-Learning algorithm as a class + functions to analyze such policies

import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import keras
import tensorflow as tf
import itertools
import json
import gc


class MarkovChain:
    # one-dimensional Markov chain for some transition: p,
    # and some delta representing the difference in value between state.
    def __init__(self, delta, p,gamma=0.98):
        self.nb_states = p.shape[0]
        self.delta = delta
        self.p = p
        self.gamma=gamma

    def get_true_state(self, state):
        states = 1 + self.delta * (np.arange(self.nb_states)[state] - int(np.ceil(self.nb_states / 2)) + 1)
        return states

    def sample(self, N):
        # Sample N data points from the transition matrix
        list_states = []
        list_states.append(np.random.choice(np.arange(self.nb_states)))
        for t in range(1, N):
            list_states.append(np.random.choice(np.arange(self.nb_states), p=self.p[list_states[-1]]))

        return list_states

    def reward(self, action, next_state):
        return action * np.log(self.get_true_state(next_state))

    def true_qval(self, delta=1e-8, Q0=1):
        # Computation of the true q-value function using fixed-point iteration.
        # Straightforward using the specification of p.

        error = np.inf
        true_qval = Q0 * np.ones((self.nb_states, 2))
        time = 1

        # Iteration will end when the change in value is smaller than delta.
        while error > delta:
            previous_qval = copy.deepcopy(true_qval)
            true_qval = np.zeros((self.nb_states, 2))
            for s1 in range(self.nb_states):
                for s1_next in range(self.nb_states):
                    temp_proba = self.p[s1, s1_next]
                    for a in range(2):
                        true_qval[s1, a] += temp_proba * self.reward(a,s1_next)

                    true_qval[s1, :] += temp_proba * self.gamma * np.max(previous_qval[s1_next, :])

            temp_error = np.max(np.abs(previous_qval - true_qval))
            error = temp_error
            time += 1
        print("Time needed to converge with delta = ", delta, " : ", time)
        return true_qval


class Policy_MC1d:
    # Q-Learning for a simple one-dimensional Markov chain with actions corresponding either to 0 or 1.
    def __init__(self, MC, exploration=0.2, lr=0.2):
        self.MC = MC
        self.gamma = MC.gamma
        self.expl = exploration
        self.q_val = np.zeros((self.MC.nb_states, 2))

    def chooseAction(self, state):
        # epsilon-greedy policy
        action = 0
        if np.random.uniform() < self.expl:
            action = np.random.randint(low=0, high=2)
        else:
            action = np.argmax(self.q_val[state])

        return action

    def update_value(self, state, action, next_state, alpha):
        # Update from Q-Learning
        self.q_val[state, action] += alpha * (self.MC.reward(action, next_state) + self.gamma * np.max(self.q_val[next_state,:]) - self.q_val[state, action])

    def train(self, train_states, true_qval=None, timestep_error=10000,decay_rate=.1):
        # Q-Learning over the train data, train_states.
        # Possibility to compare estimated q-value function at each 10'000 time-steps of the training data to compare
        # with the true q-value function as insights in the convergence of this method.

        max_error = list()
        nb_wrong_action = list()
        for t in range(len(train_states) - 1):
            current_state = train_states[t]
            next_state = train_states[t + 1]
            for i in range(2):
                current_action = i
                self.update_value(state=current_state, action=current_action, next_state=next_state,alpha=1/(1+decay_rate*t))
            if true_qval is not None:
                if t % timestep_error == 0:
                    err_dict = dict()
                    err_dict['max'] = -np.inf
                    err_dict['nb_wrong'] = 0
                    for k in range(self.q_val.shape[0]):
                        if np.argmax(true_qval[k,:]) != np.argmax(self.q_val[k,:]):
                            err_dict['nb_wrong'] += 1
                        if np.max(np.abs(true_qval[k,:]-self.q_val[k,:])) > err_dict['max']:
                            err_dict['max'] = np.max(np.abs(true_qval[k,:]-self.q_val[k,:]))

                    max_error.extend([err_dict['max']])
                    nb_wrong_action.extend([err_dict['nb_wrong']])

        # Plot of maximal absolute error and number of wrong actions as a function of the number of time-steps
        # used in training.
        if true_qval is not None:
            plt.plot(timestep_error * np.arange(1, len(max_error) + 1), np.array(max_error))
            plt.title('Evolution of maximal absolute error compared as a function of number of training steps')
            plt.xlabel('Size of training data')
            plt.ylabel('Max. absolute error')
            plt.show()

            plt.plot(timestep_error * np.arange(1, len(max_error) + 1), np.array(nb_wrong_action))
            plt.title(
                'Evolution of number of wrong action taken by the learned policy as a function of training steps')
            plt.xlabel('Size of training data')
            plt.ylabel('Nb. of wrong actions')
            plt.show()
        return max_error[-1]

    def exploit(self, states):
        # Greedy policy
        time = len(states) - 1
        port_weight = np.zeros((time,))
        for t in range(time):
            current_state = states[t]
            act = np.argmax(self.q_val[current_state])
            port_weight[t] = act

        return port_weight

    def reset_qval(self):
        self.q_val = np.zeros((self.MC.nb_states, 2))


class MarkovChain_2D():
    # Class to simulate a two-dimensional process with each dimension corresponding to some Markov chain specified
    # respectively with transition p1 and p2.
    def __init__(self, delta1, p1,p2=0,delta2=0,gamma=0.98):
        self.nb_states1 = p1.shape[0]
        self.p1 = p1
        self.delta1 = delta1
        self.gamma = gamma
        if delta2 == 0:
            self.delta2 = self.delta1
        else:
            self.delta2 = delta2

        if type(p2) == int:
            self.p2=self.p1
            self.nb_states2=self.nb_states1
        else:
            self.nb_states2 = p2.shape[0]
            self.p2=p2

    def get_true_state(self, state):
        states=np.zeros(state.shape)
        for i in range(state.shape[1]):
            states[:,i] = np.array([1 + self.delta1 * (np.arange(self.nb_states1)[state[0,i]] - int(np.ceil(self.nb_states1 / 2)) + 1),1 + self.delta2 * (np.arange(self.nb_states2)[state[1,i]] - int(np.ceil(self.nb_states2 / 2)) + 1)])
        return states

    def simulate(self, N):
        # Straightforward simulation using probabilities from p1 and p2
        states = np.zeros((2,N)).astype(int)
        states[:,0]=np.array([np.random.choice(np.arange(self.nb_states1)),np.random.choice(np.arange(self.nb_states1))])

        for t in range(1, N):
            states[:,t]=np.array([np.random.choice(np.arange(self.nb_states1), p=self.p1[states[0,t-1],:]),np.random.choice(np.arange(self.nb_states2), p=self.p2[states[1,t-1],:])])

        states = self.get_true_state(states)
        return states

    def state_as_list(self, state):
        list_state = pd.Series(state[0,:]).to_list()
        list_state.extend(pd.Series(state[1,:]).to_list())

        return list_state

    def true_qval(self, delta=1e-5,Q0=0):
        # Fixed-point iteration to derive true optimal policy. This iteration goes until the update is less than delta.
        error = np.inf
        true_qval = dict()
        for s1 in range(self.nb_states1):
            for s2 in range(self.nb_states2):
                true_qval[str(self.state_as_list(self.get_true_state(np.array([[s1],[s2]]))))]=Q0*np.ones(11)
        time = 1
        while error > delta:
            temp_error = 0
            previous_qval = copy.deepcopy(true_qval)
            for s1 in range(self.nb_states1):
                for s2 in range(self.nb_states2):
                    current_state = str(self.state_as_list(self.get_true_state(np.array([[s1],[s2]]))))
                    true_qval[current_state] = 0
                    for s1_next in range(self.nb_states1):
                        for s2_next in range(self.nb_states2):
                            temp_proba = self.p1[s1,s1_next] * self.p2[s2,s2_next]
                            next_state = str(self.state_as_list(self.get_true_state(np.array([[s1_next],[s2_next]]))))
                            true_qval[current_state] += temp_proba * (np.array([reward(action, self.get_true_state(np.array([[s1_next],[s2_next]])))[0, 0] for action in range(11)]) + self.gamma * np.max(previous_qval[next_state]))
                    if temp_error < np.max(np.abs(previous_qval[current_state]-true_qval[current_state])):
                        temp_error = np.max(np.abs(previous_qval[current_state]-true_qval[current_state]))
            error=copy.deepcopy(temp_error)
            time += 1
        # print("Time needed to converge with delta = ", delta, " : ", time)

        return true_qval
class MarkovChain_delayed():
    # Same as previously but both processes are the same Markov chain with the second one corresponding to the values
    # of the first delayed by one or two time-steps.
    def __init__(self, delta1, p1,delay=1,gamma=0.98):
        self.nb_states1 = p1.shape[0]
        self.p1 = p1
        self.delta1 = delta1
        self.gamma=gamma
        self.delta2=self.delta1
        self.p2=self.p1
        self.nb_states2=self.nb_states1
        self.delay = delay
        if delay > 2:
            print("Maximal delay is two.")
            self.delay = 2

    def get_true_state(self, state):
        states=np.zeros(state.shape)
        for i in range(state.shape[1]):
            states[:,i] = np.array([1 + self.delta1 * (np.arange(self.nb_states1)[state[0,i]] - int(np.ceil(self.nb_states1 / 2)) + 1),1 + self.delta2 * (np.arange(self.nb_states2)[state[1,i]] - int(np.ceil(self.nb_states2 / 2)) + 1)])
        return states

    def simulate(self, N):
        states = np.zeros((1,N+self.delay)).astype(int)
        states[0,0]=np.random.choice(np.arange(self.nb_states1))

        for t in range(1, N+self.delay):
            states[0,t]=np.random.choice(np.arange(self.nb_states1), p=self.p1[states[0,t-1],:])

        sim_states = np.zeros((2,N)).astype(int)
        sim_states[0,:] = states[0,:-self.delay]
        sim_states[1,:] = states[0,self.delay:]

        sim_states = self.get_true_state(sim_states)

        return sim_states

    def state_as_list(self, state):
        list_state = pd.Series(state[0,:]).to_list()
        list_state.extend(pd.Series(state[1,:]).to_list())

        return list_state

    def true_qval(self, delta=1e-5,Q0=0):
        error=np.inf
        true_qval=dict()

        time = 1
        if self.delay==1:
            for s1 in range(self.nb_states1):
                for s2 in range(self.nb_states2):
                    true_qval[str(self.state_as_list(self.get_true_state(np.array([[s1], [s2]]))))] = Q0 * np.ones(11)

            while error>delta:

                temp_error=0
                previous_qval = copy.deepcopy(true_qval)
                for s2 in range(self.nb_states2):
                    for s1 in range(self.nb_states1):
                        current_state=str(self.state_as_list(self.get_true_state(np.array([[s1],[s2]]))))
                        true_qval[current_state]=0
                        for s2_next in range(self.nb_states2):
                            temp_proba=self.p2[s2,s2_next]
                            next_state=str(self.state_as_list(self.get_true_state(np.array([[s2],[s2_next]]))))
                            true_qval[current_state] += temp_proba * (np.array([reward(action, self.get_true_state(np.array([[s2],[s2_next]])))[0, 0] for action in range(11)]) + self.gamma * np.max(previous_qval[next_state]))
                        if np.max(np.abs(previous_qval[current_state]-true_qval[current_state])) > temp_error:
                            temp_error = np.max(np.abs(previous_qval[current_state]-true_qval[current_state]))
                error=copy.deepcopy(temp_error)
                time += 1

        elif self.delay == 2:
            for s1_1,s1_2 in itertools.product(range(self.nb_states1),range(self.nb_states1)):
                for s2_1, s2_2 in itertools.product(range(self.nb_states2), range(self.nb_states2)):
                    true_qval[str(self.state_as_list(self.get_true_state(np.array([[s1_1,s1_2], [s2_1,s2_2]]))))] = Q0 * np.ones(11)
            while error > delta:
                temp_error = 0
                previous_qval = copy.deepcopy(true_qval)

                for s2_1,s2_2 in itertools.product(range(self.nb_states2),range(self.nb_states2)):
                    for s1_1,s1_2 in itertools.product(range(self.nb_states1),range(self.nb_states1)):
                        current_state = str(self.state_as_list(self.get_true_state(np.array([[s1_1,s1_2], [s2_1,s2_2]]))))
                        true_qval[current_state] = 0

                        for s2_next in range(self.nb_states2):
                            temp_proba = self.p2[s2_2, s2_next]
                            next_state = str(self.state_as_list(self.get_true_state(np.array([[s1_2,s2_1], [s2_2,s2_next]]))))
                            true_qval[current_state] += temp_proba * (np.array(
                                [reward(action, self.get_true_state(np.array([[s2_1], [s2_next]])))[0, 0] for action in
                                 range(11)]) + self.gamma * np.max(previous_qval[next_state]))

                        if np.max(np.abs(previous_qval[current_state] - true_qval[current_state])) > temp_error:
                            temp_error = np.max(np.abs(previous_qval[current_state] - true_qval[current_state]))
                time += 1
                error=copy.deepcopy(temp_error)
        print("Time needed to converge with delta = ", delta, " : ", time)

        return true_qval


def get_random_transition(size,dec=3):
    # Function to derive a random transition matrix of shape (size,size)
    temp=np.exp(np.random.uniform(size=size))
    for i in range(temp.shape[0]):
        temp[i] = temp[i]/np.sum(temp[i])
    return temp


def bernoulli_sample(p=0.5, q=0.5, n=100, rho=0):
    # Create 2-dimensional Bernoulli process with correlation, rho, utilized for simulating Binomial processes.
    p1 = rho * np.sqrt(p * q * (1 - p) * (1 - q)) + (1 - p) * (1 - q)
    p2 = 1 - p - p1
    p3 = 1 - q - p1
    p4 = p1 + p + q - 1

    rv = np.zeros((2, n))
    for i in range(n):
        samples = np.random.choice([0, 1, 2, 3], size=1, replace=True, p=[p1, p2, p3, p4])
        samples = list(map(lambda x: np.array(tuple(np.binary_repr(x, width=2))).astype(int), samples))
        rv[:, i] = samples[0]
    return rv


class SimpleBinomial:
    # Class used to simulate Binomial processes with possibility to include delay and correlation.
    def __init__(self, u1, d1, p1, u2=0, d2=0, p2=-1, corr=0,gamma=0.98,delay=0):
        self.p1 = p1
        self.u1 = u1
        self.d1 = d1
        if u2 == 0:
            self.u2 = self.u1
            self.d2 = self.d1
        elif d2==0:
            self.u2 = u2
            self.d2 = self.d1
        else:
            self.u2 = u2
            self.d2 = d2

        self.corr = corr
        if 0 <= p2 <= 1:
            self.p2 = p2
        else:
            self.p2 = self.p1
        self.gamma=gamma
        self.delay=delay

    def simulate(self, N):
        S = np.zeros((2, N))

        # Call to the Bernoulli function to get process of zero and ones.
        z = bernoulli_sample(self.p1, self.p2, N+self.delay, self.corr)

        if self.delay >0:
            bern = np.zeros((2, N))
            if self.corr == 0:
                bern[0,:]=z[0,:-self.delay]
                bern[1,:]=z[0,self.delay:]
            else:
                bern[0,:]=z[0,:-self.delay]
                bern[1,:]=z[1,self.delay:]

            z = copy.deepcopy(bern)

        # Simulate the stock return for each time step
        for i in range(N):
            S[0, i] = (1 - z[0, i]) * self.d1 + z[0, i] * self.u1
            S[1, i] = (1 - z[1, i]) * self.d2 + z[1, i] * self.u2

        return S

    def true_qval(self, delta=1e-5,Q0=0):
        # Fixed-point iteration to derive true q-value function.
        # Quite straightforward when looking at the different delay length.
        # If delay = 2, then optimal q-value function has to be defined using a window size of two.

        error=np.inf
        true_qval=dict()
        time=1
        if self.delay<2:
            for s1 in [self.u1,self.d1]:
                for s2 in [self.u2,self.d2]:
                    true_qval[str([s1,s2])]=Q0*np.ones(11)

            while error>delta:
                temp_error=0
                previous_qval = copy.deepcopy(true_qval)
                for s1 in [self.u1,self.d1]:
                    for s2 in [self.u2,self.d2]:
                        current_state=str([s1,s2])
                        true_qval[current_state]=0
                        if self.delay==0:
                            for s1_next in [self.u1,self.d1]:
                                for s2_next in [self.u2,self.d2]:
                                    temp_proba=0
                                    if s1_next==self.u1:
                                        if s2_next == self.u2:
                                            temp_proba=self.p1*self.p2
                                        else:
                                            temp_proba=self.p1*(1-self.p2)
                                    else:
                                        if s2_next == self.u2:
                                            temp_proba = (1-self.p1) * self.p2
                                        else:
                                            temp_proba = (1-self.p1 )* (1 - self.p2)

                                    next_state=str([s1_next,s2_next])
                                    true_qval[current_state] += temp_proba * (np.array([reward(action, np.array([[s1_next],[s2_next]]))[0,0] for action in range(11)]) + self.gamma * np.max(previous_qval[next_state]))
                            temp_error = np.max(np.abs(previous_qval[current_state]-true_qval[current_state]))
                        else:
                            if self.corr == 0:
                                for s2_next in [self.u2, self.d2]:
                                    temp_proba=0
                                    if s2_next == self.u2:
                                        temp_proba = self.p2
                                    else:
                                        temp_proba = 1-self.p2

                                    next_state = str([s2, s2_next])
                                    true_qval[current_state] += temp_proba * (np.array(
                                        [reward(action, np.array([[s2], [s2_next]]))[0, 0] for action in
                                         range(11)]) + self.gamma * np.max(previous_qval[next_state]))
                                temp_error += np.max(np.abs(previous_qval[current_state] - true_qval[current_state]))

                            else:
                                mu = self.corr * np.sqrt(self.p1 * self.p2 * (1 - self.p1) * (1 - self.p2)) + (
                                        1 - self.p1) * (1 - self.p2)
                                for s1_next in [self.u1, self.d1]:
                                    for s2_next in [self.u2, self.d2]:
                                        temp_proba = 0
                                        if s2_next == self.u2:
                                            if s1_next == self.u1:
                                                if s2 == self.u2:
                                                    temp_proba = self.p2 * (mu + self.p1 + self.p2 - 1)
                                                else:
                                                    temp_proba = self.p2 * (1 - self.p2 - mu)
                                            else:
                                                if s2 == self.u2:
                                                    temp_proba = self.p2 * (1 - self.p1 - mu)
                                                else:
                                                    temp_proba = self.p2 * mu
                                        else:
                                            temp_proba=0
                                            if s1_next == self.u1:
                                                if s2 == self.u2:
                                                    temp_proba = (1 - self.p2) * (mu + self.p1 + self.p2 - 1)
                                                else:
                                                    temp_proba = (1 - self.p2) * (1 - self.p2 - mu)
                                            else:
                                                if s2 == self.u2:
                                                    temp_proba = (1 - self.p2) * (1 - self.p1 - mu)
                                                else:
                                                    temp_proba = (1 - self.p2) * mu
                                        next_state = str([s1_next, s2_next])
                                        true_qval[current_state] += temp_proba * (np.array(
                                            [reward(action, np.array([[s1_next], [s2_next]]))[0, 0] for action in
                                             range(11)]) + self.gamma * np.max(previous_qval[next_state]))
                                temp_error += np.max(np.abs(previous_qval[current_state] - true_qval[current_state]))

                error=copy.deepcopy(temp_error)
                time += 1

        elif self.delay==2:
            for s1_1 in [self.u1,self.d1]:
                for s1_2 in [self.u1, self.d1]:
                    for s2_1 in [self.u2, self.d2]:
                        for s2_2 in [self.u2,self.d2]:
                            true_qval[str([s1_1,s1_2,s2_1,s2_2])]=Q0*np.ones(11)

            while error > delta:
                temp_error = 0
                previous_qval = copy.deepcopy(true_qval)
                for s1_1 in [self.u1, self.d1]:
                    for s1_2 in [self.u1, self.d1]:
                        for s2_1 in [self.u2, self.d2]:
                            for s2_2 in [self.u2, self.d2]:
                                current_state = str([s1_1,s1_2,s2_1, s2_2])
                                true_qval[current_state] = 0
                                if self.corr==0:
                                    for s2_next in [self.u2, self.d2]:
                                        temp_proba = 0
                                        if s2_next == self.u1:
                                            temp_proba = self.p2
                                        else:
                                            temp_proba =  (1 - self.p2)

                                        next_state = str([s1_2,s2_1,s2_2, s2_next])
                                        true_qval[current_state] += temp_proba * (np.array(
                                            [reward(action, np.array([[s2_1], [s2_next]]))[0, 0] for action in
                                             range(11)]) + self.gamma * np.max(previous_qval[next_state]))
                                    if temp_error < np.max(np.abs(previous_qval[current_state] - true_qval[current_state])):
                                        temp_error = np.max(np.abs(previous_qval[current_state] - true_qval[current_state]))

                                else:
                                    mu = self.corr * np.sqrt(self.p1 * self.p2 * (1 - self.p1) * (1 - self.p2)) + (
                                                1 - self.p1) * (1 - self.p2)
                                    for s1_next in [self.u1, self.d1]:
                                        for s2_next in [self.u2, self.d2]:
                                            temp_proba = 0
                                            if s2_next == self.u2:
                                                if s1_next == self.u1:
                                                    if s2_1 == self.u2:
                                                        temp_proba = self.p2 *  (mu + self.p1+self.p2 -1)
                                                    else:
                                                        temp_proba = self.p2 * (1-self.p2-mu)
                                                else:
                                                    if s2_1 == self.u2:
                                                        temp_proba = self.p2 * (1-self.p1 - mu)
                                                    else:
                                                        temp_proba = self.p2 * mu
                                            else:
                                                if s1_next == self.u1:
                                                    if s2_1 == self.u2:
                                                        temp_proba = (1-self.p2) * (mu + self.p1 + self.p2 - 1)
                                                    else:
                                                        temp_proba = (1 - self.p2) * (1 - self.p2 - mu)
                                                else:
                                                    if s2_1 == self.u2:
                                                        temp_proba = (1-self.p2) * (1 - self.p1 - mu)
                                                    else:
                                                        temp_proba = (1 - self.p2) * mu

                                            next_state = str([s1_2,s1_next, s2_2,s2_next])
                                            true_qval[current_state] += temp_proba * (np.array(
                                                [reward(action, np.array([[s1_next], [s2_next]]))[0, 0] for action in
                                                 range(11)]) + self.gamma * np.max(previous_qval[next_state]))
                                    if temp_error < np.max(
                                            np.abs(previous_qval[current_state] - true_qval[current_state])):
                                        temp_error = np.max(
                                            np.abs(previous_qval[current_state] - true_qval[current_state]))
                error = copy.deepcopy(temp_error)
                time += 1

        else:
            true_qval=dict()
            time=0

        print("Time needed to converge with delta = ", delta, " : ", time)

        return true_qval


class Policy:
    # Define policy as a class for discrete RL. Q value function defined as dictionary.
    def __init__(self, window, gamma, exploration=0.1):
        self.expl = exploration
        self.q_val = {}
        self.window = window
        self.gamma=gamma
        print(window)

    def chooseAction(self, state):
        # Espsilon-greedy policy
        w = 0

        if np.random.uniform() < self.expl:
            w = np.random.randint(low=0, high=11)
            if self.q_val.get(str(state)) is None:
                self.q_val[str(state)] = np.zeros(11)
        else:
            temp_val = -np.inf
            for action in range(11):
                if self.q_val.get(str(state)) is None:
                    self.q_val[str(state)] = np.zeros(11)
                else:
                    if self.q_val[str(state)][action] >= temp_val:
                        w = action
                        temp_val = self.q_val[str(state)][action]
        return w

    def update_value(self, state, action, next_state, alpha):
        # Update the q-value table, if no entry yet for the current state, we initialize it at zero.
        next_ret = np.array([next_state[self.window - 1::self.window]]).reshape(-1, 1)
        if self.q_val.get(str(state)) is None:
            self.q_val[str(state)] = np.zeros(11)
        if self.q_val.get(str(next_state)) is None:
            self.q_val[str(next_state)] = np.zeros(11)
            self.q_val[str(state)][action] += alpha * (
                        reward(action, next_ret)[0, 0] + self.gamma*np.max(self.q_val[str(next_state)]) - self.q_val[str(state)][
                    action])

        else:
            self.q_val[str(state)][action] += alpha * (
                        reward(action, next_ret)[0, 0] + self.gamma*np.max(self.q_val[str(next_state)]) - self.q_val[str(state)][
                    action])

    def train(self, returns,true_qval=None, timestep_error=10000,decay_rate=0.1,beta=0,tau=1,plots=True,adaptive_lr=False):
        # Training from Q-Learning while comparing with the true q-value function at each 10'000 time steps in training.
        max_error=list()
        nb_wrong_action=list()
        nb_visits = dict()

        for t in range(self.window, returns.shape[1] - 1):
            current_state = get_state(returns, t, self.window)
            next_state = get_state(returns, t + 1, self.window)

            if str(current_state) in nb_visits.keys():
                nb_visits[str(current_state)] += 1
            else:
                nb_visits[str(current_state)] = 1

            # Specification of the learning rate, either adaptive or simple
            if adaptive_lr:
                alpha = beta + 1 / (1 + (decay_rate * nb_visits[str(current_state)]) ** tau)
            else:
                alpha = beta + 1 / (1 + (decay_rate * t) ** tau)

            # Update the q-values for all actions at each time-steps
            for a in range(11):
                current_action = a
                self.update_value(state=current_state, action=current_action, next_state=next_state,alpha=alpha)
            if true_qval is not None:
                if t % timestep_error == 0:
                    err_dict = conv_qtables(true_qval, self.q_val)
                    max_error.extend([err_dict['max']])
                    nb_wrong_action.extend([err_dict['nb_wrong']])

        if true_qval is not None:
            err_dict = conv_qtables(true_qval, self.q_val)
            max_error.extend([err_dict['max']])
            nb_wrong_action.extend([err_dict['nb_wrong']])
            if plots:
                plt.plot(timestep_error*np.arange(1, len(max_error)+1),np.array(max_error))
                plt.title('Evolution of maximal absolute error compared as a function of number of training steps')
                plt.xlabel('Size of training data')
                plt.ylabel('Max. absolute error')
                plt.show()

                plt.plot(timestep_error * np.arange(1, len(max_error) + 1), np.array(nb_wrong_action))
                plt.title('Evolution of number of wrong action taken by the learned policy as a function of training steps')
                plt.xlabel('Size of training data')
                plt.ylabel('Nb. of wrong actions')
                plt.show()
        return [max_error,nb_wrong_action]

    def exploit(self, returns,test=False):
        # Greedy policy
        time = returns.shape[1] - self.window
        port_weight = np.zeros((returns.shape[0], time))
        for t in range(time):
            current_state = get_state(returns, self.window + t, self.window)
            act = np.argmax(self.q_val.get(str(current_state)))
            port_weight[:, t] = np.array([act / 10., 1 - act / 10.])

        return port_weight

    def reset_qval(self):
        # Reset q-values table to re-learn with different parameters from zero.
        for k in self.q_val.keys():
            self.q_val[k] = np.zeros(11)


class Policy_init_val:
    # Same policy class as before, but here possibility to set a different initialization of the Q-values table.
    def __init__(self, window, gamma, exploration=0.1,Q0=0):
        self.expl = exploration
        self.q_val = {}
        self.window = window
        self.gamma=gamma
        self.Q0=Q0
        print(window)

    def chooseAction(self, state):
        # write state as a list
        w = 0

        if np.random.uniform() < self.expl:
            w = np.random.randint(low=0, high=11)
            if self.q_val.get(str(state)) is None:
                self.q_val[str(state)] = self.Q0*np.ones(11)
        else:
            temp_val = -np.inf
            for action in range(11):
                if self.q_val.get(str(state)) is None:
                    self.q_val[str(state)] = self.Q0*np.ones(11)
                else:
                    if self.q_val[str(state)][action] >= temp_val:
                        w = action
                        temp_val = self.q_val[str(state)][action]
        return w

    def update_value(self, state, action, next_state, alpha):
        next_ret = np.array([next_state[self.window - 1::self.window]]).reshape(-1, 1)
        if self.q_val.get(str(state)) is None:
            self.q_val[str(state)] = self.Q0*np.ones(11)
        if self.q_val.get(str(next_state)) is None:
            self.q_val[str(next_state)] = self.Q0*np.ones(11)
            self.q_val[str(state)][action] += alpha * (
                        reward(action, next_ret)[0, 0] + self.gamma*np.max(self.q_val[str(next_state)]) - self.q_val[str(state)][
                    action])

        else:
            self.q_val[str(state)][action] += alpha * (
                        reward(action, next_ret)[0, 0] + self.gamma*np.max(self.q_val[str(next_state)]) - self.q_val[str(state)][
                    action])

    def train(self, returns,true_qval=None, timestep_error=10000,decay_rate=0.1,beta=0,tau=1,plots=True,adaptive_lr=False):
        max_error=list()
        nb_wrong_action=list()
        nb_visits = dict()

        for t in range(self.window, returns.shape[1] - 1):
            current_state = get_state(returns, t, self.window)
            next_state = get_state(returns, t + 1, self.window)

            if str(current_state) in nb_visits.keys():
                nb_visits[str(current_state)] += 1
            else:
                nb_visits[str(current_state)] = 1

            if adaptive_lr:
                alpha = beta + 1 / (1 + (decay_rate * nb_visits[str(current_state)]) ** tau)
            else:
                alpha = beta + 1 / (1 + (decay_rate * t) ** tau)
            for a in range(11):
                current_action = a
                self.update_value(state=current_state, action=current_action, next_state=next_state,alpha=alpha)
            if true_qval is not None:
                if t % timestep_error==0:
                    err_dict = conv_qtables(true_qval, self.q_val)
                    max_error.extend([err_dict['max']])
                    nb_wrong_action.extend([err_dict['nb_wrong']])

        if true_qval is not None:
            err_dict = conv_qtables(true_qval, self.q_val)
            max_error.extend([err_dict['max']])
            nb_wrong_action.extend([err_dict['nb_wrong']])
            if plots:
                plt.plot(timestep_error*np.arange(1,len(max_error)+1),np.array(max_error))
                plt.title('Evolution of maximal absolute error compared as a function of number of training steps')
                plt.xlabel('Size of training data')
                plt.ylabel('Max. absolute error')
                plt.show()

                plt.plot(timestep_error * np.arange(1, len(max_error) + 1), np.array(nb_wrong_action))
                plt.title('Evolution of number of wrong action taken by the learned policy as a function of training steps')
                plt.xlabel('Size of training data')
                plt.ylabel('Nb. of wrong actions')
                plt.show()
        return [max_error,nb_wrong_action]

    def exploit(self, returns,test=False):
        time = returns.shape[1] - self.window
        port_weight = np.zeros((returns.shape[0], time))
        for t in range(time):
            current_state = get_state(returns, self.window + t, self.window)
            act = np.argmax(self.q_val.get(str(current_state)))
            port_weight[:, t] = np.array([act / 10., 1 - act / 10.])

        return port_weight

    def reset_qval(self):
        for k in self.q_val.keys():
            self.q_val[k]=self.Q0*np.ones(11)


def analysis_result(policy, test_data):
    # INPUTS:
    # policy: class of the policy with build-in methods to exploit test data
    # test_data: data on which we want the policy to be applied

    # OUTPUT:
    # dictionary containing the weights, returns and also test_data

    pandas_data = pd.DataFrame(test_data, index=['S1', 'S2'])

    res = dict()

    weight = policy.exploit(pandas_data,test=True)
    res['w'] = weight

    ret = np.sum(weight * test_data[:, policy.window:], axis=0)
    res['ret'] = ret

    res['data'] = test_data

    return res


def plot_result(res, S0=1,w_comparison=None,window=1):
    # Given dictionary from function analysis_result(), we derive performance plot of the learned policy on the testing
    # data as well as histogram of weights for the first stock.

    plt.plot(np.cumprod(res['ret']))
    plt.plot(np.cumprod(res['data'][0, :]))
    plt.plot(np.cumprod(res['data'][1, :]))
    if w_comparison is None:
        plt.plot(np.cumprod(0.5 * res['data'][1, :] + 0.5 * res['data'][0, :]), color='red')
        plt.legend(['Trained policy', 'S1', 'S2', 'Eq. weight'])

    else:
        plt.plot(np.cumprod(np.sum(w_comparison*res['data'][:,window:],axis=0)), color='red')
        plt.legend(['Trained policy', 'S1', 'S2', 'Benchmark policy'])

    plt.title('Portfolio evolutions starting at 1$')
    plt.xlabel('Days')
    plt.ylabel('Value')
    plt.show()

    plt.hist(res['w'][0, :], bins=np.arange(0, 12) / 10)
    plt.title('Histogram of weights in stock 1')
    plt.xlabel('Action')
    plt.show()


def discretize(returns, min_val=0.96,max_val=1.04,rounding=0.01):
    # Discretization routine given minimal and maximal values.
    disc=np.round(returns, decimals=int(np.log10(1/rounding)))
    disc[disc>max_val]=max_val
    disc[disc<min_val]=min_val
    return disc


def get_state(ret, t, window=2):
    ret_window=ret.iloc[:,t-window:t]
    list_state=ret_window.loc["S1"].to_list()
    list_state.extend(ret_window.loc["S2"].to_list())

    return list_state


def reward(action, new_ret):
    # Reward function as log-return of the weights applied to the next returns.
    w = np.array([action/10.,1-action/10.]).reshape(-1,1)
    return np.log(np.dot(w.transpose(),new_ret))


def conv_qtables(true, estimate):
    # Compare the true optimal q-values to the estimated q-values
    error=dict()
    error['max'] =-np.inf
    error['nb_wrong']= 0
    error['Frobenius']=0
    error['state_max']=dict()
    for k in estimate.keys():
        error['state_max'][k]= np.max(np.abs(true[k]-estimate[k]))
        error['Frobenius'] += np.sum(np.abs(true[k]-estimate[k])**2)
        if np.argmax(true[k])!=np.argmax(estimate[k]):
            error['nb_wrong'] += 1
        if error['state_max'][k]> error['max']:
            error['max']=error['state_max'][k]

    error['Frobenius']=np.sqrt(error['Frobenius'])
    return error


def plot_error_per_state(error_dict):
    # Benchmark plot, absolute errors for each state between true optimal q-values and learned policy.
    err=np.zeros(len(error_dict['state_max'].keys()))
    n=0
    for k in error_dict['state_max'].keys():
        err[n] = error_dict['state_max'][k]
        n += 1

    plt.plot(err)
    plt.title('Error of estimated Q-values for all possible states')
    plt.xlabel('State')
    plt.ylabel('Max. absolute error')
    plt.show()


def extend_qtableMC(MC,q_val):
    # Method to expand a q-values table from a window's specification w=1 to a w=2 for Markov chains.
    extend_qval=dict()
    for k in q_val.keys():
        for s1 in range(MC.nb_states1):
            for s2 in range(MC.nb_states2):
                states=MC.get_true_state(np.array([[s1],[s2]]))
                state_list=json.loads(k)
                state_list.insert(0,states[0,0])
                state_list.insert(2,states[1,0])
                extend_qval[str(state_list)]=q_val[k]
    return extend_qval


def extend_qtablebin(bin,q_val):
    # Method to expand a q-values table from a window's specification w=1 to a w=2 for Binomial processes.
    extend_qval=dict()
    for k in q_val.keys():
        for s1 in [bin.u1,bin.d1]:
            for s2 in [bin.u2,bin.d2]:
                state_list=json.loads(k)
                state_list.insert(0,s1)
                state_list.insert(2,s2)
                extend_qval[str(state_list)]=q_val[k]
    return extend_qval


def extend_qtable_discrete(q_val,min_val=0.96,max_val=1.04,rounding=0.01):
    # Method to expand a q-values table from a window's specification w=1 to a w=2 for discrete q-values.
    extend_qval=dict()
    nb_state = int((max_val-min_val)/rounding)+1

    for k in q_val.keys():
        for s1 in [min_val + x * rounding for x in range(nb_state)]:
            for s2 in [min_val + x * rounding for x in range(nb_state)]:
                state_list=json.loads(k)
                state_list.insert(0,s1)
                state_list.insert(2,s2)
                extend_qval[str(state_list)]=q_val[k]
    return extend_qval


def exploit_optimal(true_qval, returns,window=1):
    # Function to exploit a true optimal policy on some test data, used as benchmark.
    time = returns.shape[1] - window
    port_weight = np.zeros((returns.shape[0], time))
    for t in range(time):
        current_state = get_state(returns, window + t, window)
        act = np.argmax(true_qval.get(str(current_state)))
        port_weight[:, t] = np.array([act / 10., 1 - act / 10.])

    return port_weight


def create_param_QLearning(train_data,test_data, window=1,decay_rate=1,tau=1,beta=0 ,true_qval=None,adaptive_lr=False):
    # Write parameters for Q-Learning algorithm in a dictionary essential to the implementation of multiprocessing.
    param=dict()

    param['test_data']=test_data
    param['train_data']=train_data
    param['w']=window
    param['decay_rate']=decay_rate
    param['tau']=tau
    param['beta']=beta

    param['true_qval']=true_qval
    param['adaptive_lr']=adaptive_lr

    return param


def multi_train_QLearning(params):
    # Rewrite Q-Learning algorithm taking as input the dictionary of parameters and return all important variables.
    train_data=params['train_data']
    window = params['w']
    gamma = 0.98

    q_val=copy.deepcopy(params['true_qval'])

    if q_val is not None:
        if (len(list(params['true_qval'].keys())[0])<=13) & (params['w']==2):
            q_val = None
        elif (len(list(params['true_qval'].keys())[0])>13) & (params['w']==1):
            q_val=None

    pol=Policy(window,gamma)
    errors=pol.train(train_data,q_val, timestep_error=10000,decay_rate=params['decay_rate'],beta=params['beta'],tau=params['tau'],plots=False,adaptive_lr=params['adaptive_lr'])

    res = analysis_result(pol, np.array(params['test_data']))
    # plot_result(res)
    if params['adaptive_lr']:
        string_param = 'Learning rate dependent on states: '
    else:
        string_param = 'Simple learning rate: '

    string_param+='Decay rate : '+str(params['decay_rate'])+', Beta : '+str(params['beta'])+', Tau : '+str(params['tau'])

    result=dict()
    result['max_error']=errors[0]
    result['nb_wrong']=errors[1]
    result['exploit_test']=res
    result['q_val']=pol.q_val
    result['string_param']=string_param

    return result


def plot_res_multi_QLearning(result,true_qval):
    # Present results from the ouput of multi_train_QLearning
    print('\n \n')
    print(result['string_param'])

    if (len(list(true_qval.keys())[0]) <= 13) & (len(list(result['q_val'].keys())[0]) > 13):
        q_val = None
    elif (len(list(true_qval.keys())[0]) > 13) & (len(list(result['q_val'].keys())[0]) <= 13):
        q_val = None
    else:
        q_val=copy.deepcopy(true_qval)

    if q_val is not None:
        plt.plot(10000 * np.arange(1, len(result['max_error']) + 1), np.array(result['max_error']))
        plt.title('Evolution of maximal absolute error compared as a function of number of training steps')
        plt.xlabel('Size of training data')
        plt.ylabel('Max. absolute error')
        plt.show()

        plt.plot(10000 * np.arange(1, len(result['max_error']) + 1), np.array(result['nb_wrong']))
        plt.title('Evolution of number of wrong action taken by the learned policy as a function of training steps')
        plt.xlabel('Size of training data')
        plt.ylabel('Nb. of wrong actions')
        plt.show()

        nb_wrong_action=0
        print('\n', 'Wrong actions : ')
        for k in result['q_val'].keys():
            if np.argmax(q_val[k]) != np.argmax(result['q_val'][k]):
                nb_wrong_action += 1

        if nb_wrong_action == 0:
            print('Optimal policy derived !!')

        elif nb_wrong_action <= 10:
            for k in result['q_val'].keys():
                if np.argmax(q_val[k]) != np.argmax(result['q_val'][k]):
                    print(k, '; Action from discrete policy : ', np.argmax(q_val[k]), '; Action from Q-Learning algorithm : ',
                          np.argmax(result['q_val'][k]))

        else:
            print('There are a lot of wrong actions : ', nb_wrong_action)

        conv = conv_qtables(q_val,result['q_val'])
        plot_error_per_state(conv)