# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:00:37 2021

@author: qtckp
"""
import numpy as np


import random

class Observation:
    def __init__(self, last_act = None, reward = 0, step = 0):
        self.lastActions = last_act
        self.reward = reward
        self.step = step

class Configuration:
    def __init__(self, banditCount):
        self.banditCount = banditCount
conf = Configuration(100)
        

def sample():
    """Obtain a value between 0 and sampleResolution to check against a bandit threshold."""
    return random.randint(0, 100)


def run_env(agent1, agent2):
    
    thresholds = [sample() for _ in range(conf.banditCount)]
    
    reward1 = 0
    reward2 = 0
    lastact = None
    
    for step in range(2000):
        
        act1 = agent1(Observation(lastact, reward1, step), conf)
        act2 = agent2(Observation(lastact, reward2, step), conf)
        
        reward1 += 1 if sample() < thresholds[act1] else 0
        reward2 += 1 if sample() < thresholds[act2] else 0
        
        thresholds[act1] *= 0.97
        thresholds[act2] *= 0.97
        
        lastact = (act1, act2)
        

    return (reward1, reward2)

# https://stackoverflow.com/questions/18622781/why-is-numpy-random-choice-so-slow
def fast_choice(options, probs):
    x = random.random()#np.random.rand()
    cum = 0
    for i, p in enumerate(probs):
        cum += p
        if x < cum:
            return options[i]
    return options[-1]


def probsnorm(x):
    return x/x.sum()

def softmax(x, tau):
    x2 = x/tau
    e = np.exp(x2 - x2.max())
    return e/e.sum()



def creator(EXPLORE_STEPS, FIRST_SELECTION, START_TAU, TAU_MULT):

    ROUNDS = 2000


    c_arr = np.empty(ROUNDS) # array of coefs 1, 0.97, 0.97^2, ...
    c_arr[0] = 1
    for i in range(1, c_arr.size):
        c_arr[i] = c_arr[i-1]*0.97

    x_arr = np.linspace(0, 100, 101) # net of predicted thresholds
    tau = START_TAU
 
    #@profile
    def get_sample_probs(array, probs, best_of):

        p = probsnorm(probs)# to probability form

        args = np.argsort(p)[-best_of:] # select best_of values with biggest probs

        # return array[np.random.choice(args, 1, p = softmax(p[args]))[0]]
        return array[fast_choice(args, probsnorm(p[args]))]
    
    def get_sample_softmax(array, probs):
        nonlocal tau
        tau *= TAU_MULT
        
        p = softmax(probs, tau)# to probability form

        # return array[np.random.choice(args, 1, p = softmax(p[args]))[0]]
        return fast_choice(array, p)
    

    cached_x = {}
    def get_floor_x(c):
        nonlocal cached_x
        if c in cached_x:
            return cached_x[c]
        
        arr = np.floor(x_arr * c_arr[c])
        cached_x[c] = arr
        return arr



    BANDITS = 100 # count of bandits

    bandits_counts = np.zeros(BANDITS, dtype = np.int16) # choices count for each bandit

    probs = np.ones((BANDITS, x_arr.size)) # matrix bandit*threshold probs

    bandits_indexes = np.arange(BANDITS)

    start_bandits = np.random.choice(bandits_indexes, int(BANDITS*EXPLORE_STEPS/3), replace = True) # just start random sequence of bandits selection before start of main algorithm



    my_last_action = 0
    #@profile
    def update_counts(act1, act2, my_reward):
        nonlocal bandits_counts, probs
        opp = [act != my_last_action for act in (act1, act2)]
        opp = (act1, act2)[opp[0]] if len(opp) > 0 else my_last_action

        mlt = get_floor_x(bandits_counts[my_last_action])/100

        if my_reward == 1:
            probs[my_last_action, :] *= mlt
        else:
            probs[my_last_action, :] *= 1 - mlt

        bandits_counts[my_last_action] += 1
        bandits_counts[opp] += 1
    #@profile
    def get_best_action():

        #inds = np.unravel_index(probs.argmax(), probs.shape)

        #return inds[0] # select best bandit


        #likeh = np.array([np.argmax(probs[i, :]) for i in range(BANDITS)])

        #likeh = np.array([x_arr[ind]*c_arr[b]*probs[bandit, ind]/probs[bandit, :].sum() for bandit, (ind, b) in enumerate(zip(likeh, bandit_counts))])

        likeh = np.array([get_sample_probs(get_floor_x(b), probs[bandit, :], FIRST_SELECTION) for bandit, b in enumerate(bandits_counts)])

        return get_sample_softmax(bandits_indexes, likeh)# if random.random() < PROB else random.randrange(BANDITS)    



    last_reward = 0
    #@profile
    def pasa_agent(observation, configuration):

        nonlocal BANDITS, start_bandits, bandits_counts, probs, last_reward, bandits_indexes, my_last_action

        if observation.step == 0:

            BANDITS = configuration.banditCount
            #print(f"there are {BANDITS} bandits")

            bandits_indexes = np.arange(BANDITS, dtype = np.int16)   

            start_bandits = np.random.choice(bandits_indexes, int(BANDITS*EXPLORE_STEPS/3), replace = True)

            bandits_counts = np.zeros(BANDITS, dtype = np.int16)

            probs = np.ones((BANDITS, x_arr.size))


            my_last_action = start_bandits[0]

        elif observation.step < start_bandits.size:

            update_counts(int(observation.lastActions[0]), int(observation.lastActions[1]), observation.reward - last_reward)

            my_last_action = start_bandits[observation.step]

        else:

            update_counts(int(observation.lastActions[0]), int(observation.lastActions[1]), observation.reward - last_reward)

            my_last_action = get_best_action()


        last_reward = observation.reward 
        my_last_action = int(my_last_action)

        return my_last_action
    
    return pasa_agent







def random_agent(observation, configuration):
    return random.randrange(configuration.banditCount)

for i in range(10): 
    print(f"i = {i+1}")
    print(run_env(creator(5, 10, i+1, 0.95), random_agent))
    print()








