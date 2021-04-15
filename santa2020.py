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

def err(x, y):
    #return np.sum(np.abs(x-y)*( 1000 + x*x/(y+1)))
    return np.sum(np.abs(x-y)/(x+1))

def run_env(agent1, agent2):
    
    thresholds = [sample() for _ in range(conf.banditCount)]
    
    start_thresh = np.array(thresholds)
    
    reward1 = 0
    reward2 = 0
    lastact = None
    
    for step in range(2000):
        
        act1 = agent1[0](Observation(lastact, reward1, step), conf)
        act2 = agent2(Observation(lastact, reward2, step), conf)
        
        reward1 += 1 if sample() < thresholds[act1] else 0
        reward2 += 1 if sample() < thresholds[act2] else 0
        
        thresholds[act1] *= 0.97
        thresholds[act2] *= 0.97
        
        lastact = (act1, act2)
        
        #print(np.sum(np.abs(agent1[1]() - start_thresh)))
    
    mat = np.array([start_thresh, agent1[1]()]).T
    mat2 = np.array([np.floor(thresholds), agent1[2]()]).T
    #raise Exception()
    #print(np.hstack((mat, mat2)))
    print(err(start_thresh, agent1[1]()))
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

def softmax(x, tau_param):
    x2 = x/tau_param
    e = np.exp(x2 - x2.max())
    return e/e.sum()






def creator(EXPLORE_STEPS, START_TAU, TAU_MULT):

    ROUNDS = 4000


    c_arr = np.empty(ROUNDS) # array of coefs 1, 0.97, 0.97^2, ...
    c_arr[0] = 1
    for i in range(1, c_arr.size):
        c_arr[i] = c_arr[i-1]*0.97

    x_arr = np.linspace(0, 100, 101) # net of predicted thresholds
    tau = START_TAU
 
    #@profile
    def get_sample_probs(array, probs):

        #p = probsnorm(probs)# to probability form

        #return fast_choice(array, p)
        return array[np.argmax(probs)]
    
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

    start_bandits = np.random.choice(bandits_indexes, int(BANDITS*EXPLORE_STEPS), replace = True) # just start random sequence of bandits selection before start of main algorithm



    my_last_action = 0
    #@profile
    def update_counts(act1, act2, my_reward):
        nonlocal bandits_counts, probs
        opp = [i for i, act in enumerate((act1, act2)) if act != my_last_action]
        opp = (act1, act2)[opp[0]] if len(opp) > 0 else my_last_action

        mlt = get_floor_x(bandits_counts[my_last_action])/100

        if my_reward == 1:
            probs[my_last_action, :] *= mlt
        else:
            probs[my_last_action, :] *= 1 - mlt

        bandits_counts[my_last_action] += 1
        bandits_counts[opp] += 1
    
    def get_bound():
        
        return np.array([x_arr[np.argmax(arr)] for arr in probs])    
    
    def get_bound2():
        return np.array([get_sample_probs(get_floor_x(b), probs[bandit, :]) for bandit, b in enumerate(bandits_counts)])
    
    #@profile
    def get_best_action(acts):
        #opp = [act != my_last_action for act in acts]
        #opp = acts[opp[0]] if len(opp) > 0 else my_last_action
        #inds = np.unravel_index(probs.argmax(), probs.shape)

        #return inds[0] # select best bandit


        #likeh = np.array([np.argmax(probs[i, :]) for i in range(BANDITS)])

        #likeh = np.array([x_arr[ind]*c_arr[b]*probs[bandit, ind]/probs[bandit, :].sum() for bandit, (ind, b) in enumerate(zip(likeh, bandit_counts))])

        likeh = np.array([get_sample_probs(get_floor_x(b), probs[bandit, :]) for bandit, b in enumerate(bandits_counts)])
        #likeh[opp] += 15
        return get_sample_softmax(bandits_indexes, likeh)# if random.random() < PROB else random.randrange(BANDITS)
    
    
    

    
    
    current_bandit = 0
    steps = 0
    fall = False
    


    last_reward = 0
    #@profile
    def pasa_agent(observation, configuration):

        nonlocal BANDITS, start_bandits, bandits_counts, probs, last_reward, bandits_indexes, my_last_action, steps, fall, current_bandit

        if observation.step == 0:

            BANDITS = configuration.banditCount
            #print(f"there are {BANDITS} bandits")

            bandits_indexes = np.arange(BANDITS, dtype = np.int16)   

            start_bandits = np.random.choice(bandits_indexes, int(BANDITS*EXPLORE_STEPS), replace = True)

            bandits_counts = np.zeros(BANDITS, dtype = np.int16)

            probs = np.ones((BANDITS, x_arr.size))
            
            probs = np.array([x_arr/100]*BANDITS)
     
            
            my_last_action = current_bandit #start_bandits[0]
            steps += 1
        
        #elif current_bandit != BANDITS-1 and fall:
        #    update_counts(int(observation.lastActions[0]), int(observation.lastActions[1]), observation.reward - last_reward)
        #    
        #    if observation.reward - last_reward == 0:
        #        fall = True
            

        #    if steps >= EXPLORE_STEPS and fall:
        #        steps = 0
        #        fall = False
        #        current_bandit += 1
            
        #    my_last_action = min(current_bandit, BANDITS - 1)
        #    steps += 1            
                
            
        
        elif observation.step < start_bandits.size:

            update_counts(int(observation.lastActions[0]), int(observation.lastActions[1]), observation.reward - last_reward)

            my_last_action = start_bandits[observation.step]

        else:

            update_counts(int(observation.lastActions[0]), int(observation.lastActions[1]), observation.reward - last_reward)

            my_last_action = get_best_action(observation.lastActions)


        last_reward = observation.reward 
        my_last_action = int(my_last_action)
        
        #print(get_bound())
        
        return my_last_action
    
    
    return pasa_agent, get_bound, get_bound2




def multbeta():

    bandit_state = None
    total_reward = 0
    last_step = None
    
    def multi_armed_bandit_agent(observation, configuration):
        
        nonlocal bandit_state, total_reward, last_step
    
        step = 1.5 #you can regulate exploration / exploitation balacne using this param
        
        decay_rate = 0.97 # how much do we decay the win count after each call
        
            
        if observation.step == 0:
            # initial bandit state
            bandit_state = [[1,1] for i in range(configuration.banditCount)]
        else:       
            # updating bandit_state using the result of the previous step
            last_reward = observation.reward - total_reward
            total_reward = observation.reward
            
            # we need to understand who we are Player 1 or 2
            player = int(last_step == observation.lastActions[1])
            
            if last_reward > 0:
                bandit_state[observation.lastActions[player]][0] += step
            else:
                bandit_state[observation.lastActions[player]][1] += step
            
            bandit_state[observation.lastActions[0]][0] = (bandit_state[observation.lastActions[0]][0] - 1) * decay_rate + 1
            bandit_state[observation.lastActions[1]][0] = (bandit_state[observation.lastActions[1]][0] - 1) * decay_rate + 1
    
    #     generate random number from Beta distribution for each agent and select the most lucky one
        best_proba = -1
        best_agent = None
        for k in range(configuration.banditCount):
            proba = np.random.beta(bandit_state[k][0],bandit_state[k][1])
            if proba > best_proba:
                best_proba = proba
                best_agent = k
            
        last_step = best_agent
        return best_agent
    
    return multi_armed_bandit_agent





def combine_pasa(EXPLORE_STEPS, START_TAU, TAU_MULT, STEP, PROB):

    ROUNDS = 4000


    c_arr = np.empty(ROUNDS) # array of coefs 1, 0.97, 0.97^2, ...
    c_arr[0] = 1
    for i in range(1, c_arr.size):
        c_arr[i] = c_arr[i-1]*0.97

    x_arr = np.linspace(0, 100, 101) # net of predicted thresholds
    tau = START_TAU
 
    #@profile
    def get_sample_probs(array, probs):

        #p = probsnorm(probs)# to probability form

        #return fast_choice(array, p)
        return array[np.argmax(probs)]
    
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


    bandit_state = None
    total_reward = 0
    last_step = None
    step = STEP #you can regulate exploration / exploitation balacne using this param 
    decay_rate = 0.97 # how much do we decay the win count after each call


    start_bandits = np.random.choice(bandits_indexes, int(BANDITS*EXPLORE_STEPS), replace = True) # just start random sequence of bandits selection before start of main algorithm



    my_last_action = 0
    #@profile
    def update_counts(act1, act2, my_reward):
        nonlocal bandits_counts, probs
        opp = [i for i, act in enumerate((act1, act2)) if act != my_last_action]
        opp = (act1, act2)[opp[0]] if len(opp) > 0 else my_last_action

        mlt = get_floor_x(bandits_counts[my_last_action])/100

        if my_reward == 1:
            probs[my_last_action, :] *= mlt
        else:
            probs[my_last_action, :] *= 1 - mlt

        bandits_counts[my_last_action] += 1
        bandits_counts[opp] += 1
    
    def update_state(observation):
            
            nonlocal bandit_state, total_reward

            last_reward = observation.reward - total_reward
            total_reward = observation.reward
            
            # we need to understand who we are Player 1 or 2
            player = int(last_step == observation.lastActions[1])
            
            if last_reward > 0:
                bandit_state[observation.lastActions[player]][0] += step
            else:
                bandit_state[observation.lastActions[player]][1] += step
            
            bandit_state[observation.lastActions[0]][0] = (bandit_state[observation.lastActions[0]][0] - 1) * decay_rate + 1
            bandit_state[observation.lastActions[1]][0] = (bandit_state[observation.lastActions[1]][0] - 1) * decay_rate + 1




    def get_bound():
        
        return np.array([x_arr[np.argmax(arr)] for arr in probs])    
    
    def get_bound2():
        return np.array([get_sample_probs(get_floor_x(b), probs[bandit, :]) for bandit, b in enumerate(bandits_counts)])
    
    #@profile
    def get_best_action(acts):
        #opp = [act != my_last_action for act in acts]
        #opp = acts[opp[0]] if len(opp) > 0 else my_last_action
        #inds = np.unravel_index(probs.argmax(), probs.shape)

        #return inds[0] # select best bandit


        #likeh = np.array([np.argmax(probs[i, :]) for i in range(BANDITS)])

        #likeh = np.array([x_arr[ind]*c_arr[b]*probs[bandit, ind]/probs[bandit, :].sum() for bandit, (ind, b) in enumerate(zip(likeh, bandit_counts))])

        likeh = np.array([get_sample_probs(get_floor_x(b), probs[bandit, :]) for bandit, b in enumerate(bandits_counts)])
        #likeh[opp] += 15
        return get_sample_softmax(bandits_indexes, likeh)# if random.random() < PROB else random.randrange(BANDITS)
    
    def get_best_answer():
        nonlocal last_step

        best_proba = -1
        best_agent = None
        for k in range(BANDITS):
            proba = np.random.beta(bandit_state[k][0],bandit_state[k][1])
            if proba > best_proba:
                best_proba = proba
                best_agent = k
            
        last_step = best_agent
        return best_agent
    

    
    
    current_bandit = 0
    steps = 0
    fall = False
    


    last_reward = 0
    #@profile
    def pasa_agent(observation, configuration):

        nonlocal BANDITS, start_bandits, bandits_counts, probs, last_reward, bandits_indexes, my_last_action, steps, fall, current_bandit, bandit_state

        if observation.step == 0:

            BANDITS = configuration.banditCount

            bandits_indexes = np.arange(BANDITS, dtype = np.int16)   

            start_bandits = np.random.choice(bandits_indexes, int(BANDITS*EXPLORE_STEPS), replace = True)

            bandits_counts = np.zeros(BANDITS, dtype = np.int16)

            probs = np.ones((BANDITS, x_arr.size))
            
            #probs = np.array([x_arr/100]*BANDITS)
     
            bandit_state = [[1,1] for i in range(BANDITS)]
            
            my_last_action = get_best_answer()        
                
        
        elif observation.step < start_bandits.size:

            update_counts(int(observation.lastActions[0]), int(observation.lastActions[1]), observation.reward - last_reward)
            update_state(observation)

            my_last_action = get_best_answer()

        else:

            update_counts(int(observation.lastActions[0]), int(observation.lastActions[1]), observation.reward - last_reward)
            update_state(observation)

            my_last_action = get_best_action(observation.lastActions) if random.random() < PROB else get_best_answer()


        last_reward = observation.reward 
        my_last_action = int(my_last_action)
        
        #print(get_bound())
        
        return my_last_action
    
    
    return pasa_agent, get_bound, get_bound2






def setseed(seed = 2):
    np.random.seed(seed)
    random.seed(seed)



def random_agent(observation, configuration):
    return random.randrange(configuration.banditCount)

for i in range(1): 
    print(f"i = {i+1}")
    setseed()
    print(run_env(creator(5, i+1, 0.95), multbeta()))
    setseed()
    print(run_env(combine_pasa(5, 30, 0.98, 3, 0.3), multbeta()))
    #print(run_env(creator(5, i+1, 0.97), creator(5, i+1, 0.97)))
    print()








