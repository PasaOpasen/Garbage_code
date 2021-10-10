# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:07:54 2020

@author: qtckp
"""


import itertools
import random
import subprocess

import numpy as np

from geneticalgorithm2 import geneticalgorithm2 as ga


from app.recomendation.config import DEVELOPMENT


def print_data(start_scores, climbing_score = 'NaN', GA_score = 'NaN'):
    
    mn, mx, avg, med, sd = start_scores.min(), start_scores.max(), start_scores.mean(), np.median(start_scores), np.std(start_scores)
    
    q10, q25, q75, q90 = [np.quantile(start_scores, q/100) for q in (10,25,75,90)]

    string = f"{mn},{mx},{avg},{med},{sd},{q10},{q25},{q75},{q90},{climbing_score},{GA_score}"
    # subprocess.check_output(f"echo {string} >> scores.csv",shell=True)


def get_best_res(dim, function_discrete, rows = None, SEARCH_UP_BORDER = 300, start_rd = None, info = None):
    
    
    #position, value = get_best_res_naive_discrete(rows, function_discrete)
    
    hash_val = random.randint(0, 100)

    (position, value, positions, values), cp = get_best_res_greedy(rows, function_discrete, SEARCH_UP_BORDER = SEARCH_UP_BORDER, hash_val = hash_val, start_rd = start_rd)
    
    start_values = cp
    #position, value = get_best_res_greedy_mutation(rows, function_discrete)
    
    #position, value = get_best_res_greedy_multiple(rows, function_discrete)
    
    #position, value = get_best_res_naive(dim, function_real)
    
    #position, value = get_best_res_beehive(dim, function_real)
    
    #position, value = get_best_res_genetic(rows, function_discrete)
    
    #raise Exception()
    if info['check_quality'] and value > min(100, SEARCH_UP_BORDER): # SEARCH_UP_BORDER
        subprocess.check_output(f"echo {value} >> {info['filename']}",shell=True)
        return None
    before_value = value

    ######plot_pop(values, hash_val, 'after Hill Climbing optimization')
    
    position, value, positions, values = get_best_res_genetic2(rows, function_discrete, positions, values)
    
    ######plot_pop(values, hash_val, 'after GA optimization')

    #print_data(start_values, before_value, value)

    # if GA got progress, try to coord opt best solution (more water maybe or smth else)
    if value < before_value:
        #before_value = value
        #print()
        position, value, positions, values = local_descent(position, value, 0, positions, values, rows, function_discrete, min_improvement = 0, max_fails_count = 2)
        #print()
        #if value < before_value:
        #    raise Exception()
    # subprocess.check_output(f"echo {value} >> {info['filename']}",shell=True)
    return position



def get_best_res_simpler(start_position, function_discrete, rows):
    """
    то же самое, что и без simpler, но исходим из того, что входные данные уже ок и не надо делать некоторые проверки на качество
    """

    value = function_discrete(start_position)

    positions, values = random_start_population(rows, function_discrete, size = 450)

    positions[1] = start_position
    values[1] = value

    position, value, positions, values = local_descent(start_position, value, 0, positions, values, rows, function_discrete, min_improvement = 0, max_fails_count = 2)

    before_value = value

    #position, value, positions, values = get_best_res_genetic3(rows, function_discrete, positions, values)
    
    #if value < before_value:
    #    print()
    #    position, value, positions, values = local_descent(position, value, 0, positions, values, rows, function_discrete, min_improvement = 0, max_fails_count = 2)
    #    print()
    #else:
    #    print("No progress from GA((")

    return position, value



















def random_start_population(rows, function, size = 100):

    ln = len(rows)
    count = size
        
    arr = np.empty((count, ln), dtype = np.float)
    
    for col in range(ln):
        arr[:, col] = np.random.choice(rows[col], count, replace = True)

    for col in range(ln):
        arr[0, col] = rows[col][0]
        arr[-1, col] = rows[col][-1]


    vals = np.array([function(val) for val in arr])

    return arr, vals    





def get_best_res_naive(dim, function):
        
        rd = np.random.random(dim)
        f = function(rd)
        
        for _ in range(100):
            tmp = np.random.random(dim)
            tf = function(tmp)
            if tf < f:
                f = tf
                rd = tmp
        
        return rd, f

def get_best_res_naive_discrete(rows, function, start_rd = None):
    
    stard_rd_flag = start_rd is None 
    
    arr, vals = random_start_population(rows, function, size = 100 if stard_rd_flag else 500)

    if not stard_rd_flag and len(rows) == start_rd.size:
        mask = np.array([v in row for v, row in zip(start_rd, rows)])

        ln = 1 if mask.sum() == mask.size else 50

        arr[1:(ln+1),mask] = start_rd[mask]
        
        for i in range(1,ln+1): vals[i] = function(arr[i])
        #arr[1] = np.array([(v if v in row else row[-1])  for v, row in zip(start_rd, rows)]) # редактирую, чтоб не превышал границы, если что
        #vals[1] = function(arr[1])

        #arr[2] = np.array([(v if v in row else row[int(len(row) / 2)]) for v, row in zip(start_rd, rows)])
        #vals[2] = function(arr[2])

        #print(arr[1:3])
        #print(f"\n\nstart rd values = {vals[1:(ln+1)]}\n\n")
    
    
    if not stard_rd_flag:
        arg = np.argsort(vals)[:100]
        vals = vals[arg]
        arr = arr[arg]
        
    
    ind = vals.argmin()    
    
    return arr[ind, :], vals[ind], ind, arr, vals


def local_descent(pos, val, index, poses, vals, rows, function, min_improvement = 5, max_fails_count = 2):
    print(f'start value = {val}')
    
    cols = np.arange(len(rows))
    pos_local = pos.copy()
    fails = 0
    val_local = val
    
    while True:
        
        np.random.shuffle(cols)
        failures = 0
        
        for col in cols:
            
            tmp = []
            for value in rows[col]:           
                pos_local[col] = value
                tmp.append(function(pos_local))
            
            best_index = np.argmin(np.array(tmp))
            pos_local[col] = rows[col][best_index]
            
            if tmp[best_index] == val_local:
                failures += 1
            else:
                val_local = tmp[best_index]
        
        if failures == cols.size:
            poses[index, :] = pos_local
            vals[index] = val_local
            return pos_local, val_local, poses, vals
        
        else:
            
            if val - val_local < min_improvement:
                fails += 1
            else:
                fails = 0
            
            val = val_local
            
            print(f'better value {val} and {failures} failures')
            
            if fails == max_fails_count:
                # print(f'Stopped cuz of low improvement (<{min_improvement}) during {fails} iterations')
                poses[index, :] = pos_local
                vals[index] = val_local
                return pos_local, val_local, poses, vals


def get_best_res_greedy(rows, function, min_improvement = 5, max_fails_count = 2, SEARCH_UP_BORDER = 300, hash_val = 0, start_rd = None):
    
    pos, val, index, poses, vals = get_best_res_naive_discrete(rows, function, start_rd)
    cp = vals.copy()
    
    #import pandas as pd
    #ss = pd.Series(vals)
    #ss.plot.hist(grid=True, bins=20, rwidth=0.9)    
    ######plot_pop(vals, hash_val, 'Random population')

    if val > SEARCH_UP_BORDER:
        print(f'bad val {val}')
        return (pos, val, poses, vals), cp
    #else:
    #    poses, vals = positions, values
    #    index = 0
    #    pos = best_position
    #    val = vals[0]

    return local_descent(pos, val, index, poses, vals, rows, function, min_improvement, max_fails_count), cp
            
    
    
def get_best_res_greedy2(rows, function, pairs_count = 10, SEARCH_UP_BORDER = 300):
    
    pos, val = get_best_res_naive_discrete(rows, function)
    
    if val > SEARCH_UP_BORDER:
        print(f'bad val {val}')
        return pos, val
    
    print(f'start value = {val}')
    
    
    
    cols = np.arange(len(rows))
    pos_local = pos.copy()
    
    while True:
        
        np.random.shuffle(cols)
        failures = 0
        
        for col in cols:
            
            tmp = []
            for value in rows[col]:           
                pos_local[col] = value
                tmp.append(function(pos_local))
            
            best_index = np.argmin(np.array([tmp]))
            pos_local[col] = rows[col][best_index]
            
            if tmp[best_index] == val:
                failures += 1
            else:
                val = tmp[best_index]
        
        if failures == cols.size:
            #return pos_local, val
            break
        else:
            
            print(f'better value {val} and {failures} failures')
    
    
    for a, b in random.sample(list(itertools.combinations(cols, 2)), pairs_count):
        arows = rows[a]
        brows = rows[b]
        
        tmp = np.empty((arows.size, brows.size))
        
        for i, r1 in enumerate(arows):
            pos_local[a] = r1
            for j, r2 in enumerate(brows):
                pos_local[b] = r2
                tmp[i,j] = function(pos_local)
        
        best_args = np.unravel_index(np.argmin(tmp, axis=None), tmp.shape)
        
        
        pos_local[a] = arows[best_args[0]]
        pos_local[b] = brows[best_args[1]]
        
        val = tmp.min()
        print(f'new minimum {val} after 2step')
    
    return pos_local, val

    
def get_best_res_greedy_mutation(rows, function, samples = 100, SEARCH_UP_BORDER = 300):
    
    pos, val = get_best_res_naive_discrete(rows, function)
    
    if val > SEARCH_UP_BORDER:
        print(f'bad val {val}')
        return pos, val
    
    print(f'start value = {val}')
    
    
    
    cols = np.arange(len(rows))
    pos_local = pos.copy()
    
    while True:
        
        np.random.shuffle(cols)
        failures = 0
        
        for col in cols:
            
            tmp = []
            for value in rows[col]:           
                pos_local[col] = value
                tmp.append(function(pos_local))
            
            best_index = np.argmin(np.array([tmp]))
            pos_local[col] = rows[col][best_index]
            
            if tmp[best_index] == val:
                failures += 1
            else:
                val = tmp[best_index]
        
        if failures == cols.size:
            #return pos_local, val
            break
        else:
            
            print(f'better value {val} and {failures} failures')
    
    
    mutants = np.tile(pos_local, (samples, 1))
    for mutant in mutants:
        for col in np.random.choice(cols, 3, replace = False):
            mutants[col] = np.random.choice(rows[col], 1)[0]
    
    vals = np.array([function(arr) for arr in mutants])
    
    arg = vals.argmin()
    
    if val > vals[arg]:
        print(f'better mutant with val = {vals[arg]}')
        return mutants[arg], vals[arg]
    return pos_local, val

    
     
def get_best_res_greedy_multiple(rows, function, samples = 20, SEARCH_UP_BORDER = 300):
    
    pos, val = get_best_res_naive_discrete(rows, function)
    
    if val > SEARCH_UP_BORDER:
        print(f'bad val {val}')
        return pos, val
    
    print(f'start value = {val}')
    
    
    def go_down(position, val_):
        print()
        cols = np.arange(len(rows))
        pos_local = position.copy()
        
        while True:
            
            np.random.shuffle(cols)
            failures = 0
            
            for col in cols:
                
                tmp = []
                for value in rows[col]:           
                    pos_local[col] = value
                    tmp.append(function(pos_local))
                
                best_index = np.argmin(np.array([tmp]))
                pos_local[col] = rows[col][best_index]
                
                if tmp[best_index] == val_:
                    failures += 1
                else:
                    val_ = tmp[best_index]
            
            if failures == cols.size:
                return pos_local, val_
            else:
                
                print(f'better value {val_} and {failures} failures')
    
    
    examples = np.tile(pos, (samples, 1))
    ps = []
    vl = []
    for example in examples:
        p, v = go_down(example, val)
        ps.append(p)
        vl.append(v)
    
    ps = np.array(ps)
    vals = np.array(vl)
    
    arg = vals.argmin()
    
    print(f'best results with val = {vals[arg]}, mean result with val {vals.mean()}')
    return examples[arg], vals[arg]
    
    

# def get_best_res_beehive(dim, function, SEARCH_UP_BORDER = 300):
    
#         from BeeHiveOptimization import Bees, Hive, BeeHive, TestFunctions, RandomPuts
        
        
#         arr = np.random.random((100, dim))
#         bees = Bees(arr, width = 0.5)
        
#         hive = Hive(bees, 
#             function, 
#             parallel = False, # use parallel evaluating of functions values for each bee? (recommented for heavy functions, f. e. integtals) 
#             verbose = True) # show info about hive 
        
#         if hive.best_val > SEARCH_UP_BORDER:
#             return hive.best_pos, hive.best_val


#         best_result, best_position = hive.get_result(max_step_count = 30, # maximun count of iteraions
#                       max_fall_count = 10, # maximum count of continious iterations without better result
#                       w = 0.6, fp = 2, fg = 8, # parameters of algorithm
#                       latency = 1e-9, # if new_result/old_result > 1-latency then it was the iteration without better result
#                       verbose = True # show the progress
#                       )
        
#         return best_position, best_result



def get_best_res_genetic(rows, function):
    
    from geneticalgorithm2 import geneticalgorithm2 as ga
    
    def f(X):
        return function(np.array([rows[i][int(j)] for i, j in enumerate(X)]))
    
    
    varbound = np.array([[0, arr.size - 1] for arr in rows])
    
    param = {
        'max_num_iteration': 500, 
        'population_size': 100, 
        'mutation_probability': 0.1, 
        'elit_ratio': 0.05, 
        'crossover_probability': 0.5, 
        'parents_portion': 0.3, 
        'crossover_type': 'two_point', 
        'max_iteration_without_improv': 10
        }
    
    model = ga(function=f, dimension = len(rows), variable_type='int', variable_boundaries = varbound, algorithm_parameters = param)

    model.run()
    
    res = model.output_dict
    
    return np.array([rows[i][int(j)] for i, j in enumerate(res['variable'])]), res['function']

#@profile        
def get_best_res_genetic2(rows, function, samples, samples_scores):
    
    from geneticalgorithm2 import np_lru_cache
    @np_lru_cache(maxsize = 10000)
    def f(X):
        return function(np.array([rows[i][int(j)] for i, j in enumerate(X)]))
    
    #
    # decode between values 10, 20, 30
    # to indexes 0, 1, 2
    #
    
    converter = [{value: key for key, value in enumerate(row)} for row in rows]
    for column in range(samples.shape[1]):
        for row in range(samples.shape[0]):
            samples[row, column] = converter[column][samples[row, column]]
    
    
    
    varbound = np.array([[0, arr.size - 1] for arr in rows])
    
    
    param = {
            'max_num_iteration': 200, 
            'population_size': 100, 
            'mutation_probability': 0.1, 
            'elit_ratio': 0.05, 
            'crossover_probability': 0.5, 
            'parents_portion': 0.2, 
            'crossover_type': 'uniform', # best by tests 
            'selection_type': 'sigma_scaling',# best by tests
            'max_iteration_without_improv': 40
            }
    
    model = ga(function=f, dimension = len(rows), 
               variable_type='int', 
               variable_boundaries = varbound, 
               algorithm_parameters = param)

    #from geneticalgorithm2 import Actions, ActionConditions, MiddleCallbacks
    model.run(no_plot = True, 
              start_generation={'variables':samples, 'scores': samples_scores},
              disable_progress_bar = not DEVELOPMENT,
              disable_printing = not DEVELOPMENT,

              remove_duplicates_generation_step=5#,

            #   middle_callbacks=[
            #       MiddleCallbacks.UniversalCallback(
            #       Actions.PlotPopulationScores(
            #           lambda data: f"Gen {data['current_generation']}", 
            #           None# lambda data: f"{data['current_generation']}_{data['last_generation']['scores'].min()}.png"
            #           ),
            #       ActionConditions.Always()
            #       )
            #       ]
              )
    
    res = model.output_dict
    #raise Exception()
    f.cache_clear()
    
    #model.plot_results(main_color = 'green', title = 'GA process', save_as = f"GA_{random.randint(0, 100)}.png")
    
    return np.array([rows[i][int(j)] for i, j in enumerate(res['variable'])]), res['function'], model.output_dict['last_generation']['variables'], model.output_dict['last_generation']['scores']



#@profile        
def get_best_res_genetic3(rows, function, samples, samples_scores):
    

    def f(X):
        return function(np.array([rows[i][int(j)] for i, j in enumerate(X)]))
    
    #
    # decode between values 10, 20, 30
    # to indexes 0, 1, 2
    #
    
    converter = [{value: key for key, value in enumerate(row)} for row in rows]
    for column in range(samples.shape[1]):
        for row in range(samples.shape[0]):
            samples[row, column] = converter[column][samples[row, column]]
    
    
    
    varbound = np.array([[0, arr.size - 1] for arr in rows])
    
    param = {
            'max_num_iteration': 4000,#200, 
            'mutation_probability': 0.5, 
            'elit_ratio': 0.01, 
            'crossover_probability': 0.5, 
            'parents_portion': 0.1, 
            'crossover_type': 'two_point',
            'selection_type': 'linear_ranking',
            'max_iteration_without_improv': 500#30
            }
    
    model = ga(function=f, dimension = len(rows), 
               variable_type='int', 
               variable_boundaries = varbound, 
               algorithm_parameters = param)

    from geneticalgorithm2 import Population_initializer
    model.run(no_plot = True, 
              start_generation={'variables':samples, 'scores': samples_scores},
              studEA=True,
              population_initializer = Population_initializer(select_best_of = 3, local_optimization_step = 'never', local_optimizer = None),
              disable_progress_bar = not DEVELOPMENT)
    
    res = model.output_dict
    
    #model.plot_results(main_color = 'green', title = 'GA process', save_as = f"GA_{random.randint(0, 100)}.png")
    
    return np.array([rows[i][int(j)] for i, j in enumerate(res['variable'])]), res['function'], model.output_dict['last_generation']['variables'], model.output_dict['last_generation']['scores']






def get_best_res_genetic2_test2(rows, function, samples, samples_scores):
    """
    функция для  сравнения двух разных конфигураций гиперпараметров
    как правило, используемой сейчас и потенциально лучшей
    """

    from geneticalgorithm2 import geneticalgorithm2 as ga
    import pandas as pd
    
    def f(X):
        return function(np.array([rows[i][int(j)] for i, j in enumerate(X)]))
    
    #
    # decode between values 10, 20, 30
    # to indexes 0, 1, 2
    #
    
    converter = [{value: key for key, value in enumerate(row)} for row in rows]
    for column in range(samples.shape[1]):
        for row in range(samples.shape[0]):
            samples[row, column] = converter[column][samples[row, column]]
    
    
    
    varbound = np.array([[0, arr.size - 1] for arr in rows])
    

    count = 200    
    
    table = pd.DataFrame(np.zeros((count, 2)), columns = ['current', 'new'])
    
    best_before = np.min(samples_scores)
    
    br = False
    for i in range(count):
        if br: break

        np.random.seed(i)
        param = {
            'max_num_iteration': 200, 
            'population_size': 100, 
            'mutation_probability': 0.1, 
            'elit_ratio': 0.05, 
            'crossover_probability': 0.5, 
            'parents_portion': 0.3, 
            'crossover_type': 'two_point', 
            'max_iteration_without_improv': 40
            }
        
        model = ga(function=f, dimension = len(rows), 
                variable_type='int', 
                variable_boundaries = varbound, 
                algorithm_parameters = param)

        model.run(no_plot = True, 
                start_generation={'variables':samples, 'scores': samples_scores},
                disable_progress_bar = not DEVELOPMENT)
        
        r1 = model.output_dict['function']
        
        np.random.seed(i)
        param = {
                'max_num_iteration': 200, 
                'population_size': 100, 
                'mutation_probability': 0.1, 
                'elit_ratio': 0.05, 
                'crossover_probability': 0.5, 
                'parents_portion': 0.2, 
                'crossover_type': 'uniform', # best by tests 
                'selection_type': 'sigma_scaling',# best by tests
                'max_iteration_without_improv': 40
                }
        
        model = ga(function=f, dimension = len(rows), 
                variable_type='int', 
                variable_boundaries = varbound, 
                algorithm_parameters = param)

        model.run(no_plot = True, 
                start_generation={'variables':samples, 'scores': samples_scores},
                disable_progress_bar = not DEVELOPMENT)
        
        r2 = model.output_dict['function']
        
        print(f"old = {r1}  new = {r2}")

        table.iloc[i, 0] = r1/best_before
        table.iloc[i, 1] = r2/best_before
        
        if i == 10:
            br = (table.values[:,0] != table.values[:,1]).sum() == 0

                
    if not br:
        table.to_csv(f'{round(best_before)} {random.randint(0, 100)} stats.csv', index = False)
    
    res = model.output_dict
    
    return np.array([rows[i][int(j)] for i, j in enumerate(res['variable'])]), res['function'], model.output_dict['last_generation']['variables'], model.output_dict['last_generation']['scores']





Table = None
def get_best_res_genetic2_test(rows, function, samples, samples_scores):
    """
    функция для сравнения эффективность разных значений какого-то гиперпараметра
    """
    
    global Table
    
    from geneticalgorithm2 import geneticalgorithm2 as ga
    import pandas as pd
    
    def f(X):
        return function(np.array([rows[i][int(j)] for i, j in enumerate(X)]))
    
    #
    # decode between values 10, 20, 30
    # to indexes 0, 1, 2
    #
    
    converter = [{value: key for key, value in enumerate(row)} for row in rows]
    for column in range(samples.shape[1]):
        for row in range(samples.shape[0]):
            samples[row, column] = converter[column][samples[row, column]]
    
    
    
    varbound = np.array([[0, arr.size - 1] for arr in rows])
    
    #p = [0.05, 0.1, 0.2, 0.25, 0.3] elit
    # p = [0.1, 0.2, 0.3, 0.5] parents
    # p = [0.1, 0.3, 0.5, 0.7] crossprob
    #p = [0.1, 0.2, 0.3] mutprob
    #cross_type = ['one_point', 'two_point', 'uniform', 'segment', 'shuffle']
    #p = ['fully_random', 'roulette', 'stochastic', 'sigma_scaling', 'ranking', 'linear_ranking', 'tournament']
    

    count = 150    
    
    elit = [0.001, 0.05, 0.1, 0.2]
    columnes = [f'elit_{el}'.replace('.','_') for el in elit]
    
    table = pd.DataFrame(np.zeros((count, len(elit))), columns = columnes)
    
    best_before = np.min(samples_scores)
    
    br = False
    for i in range(count):
        if br: break
        for j, el in zip(table.columns, elit):
            
            np.random.seed(i)
            print(j)
            param = {
                'max_num_iteration': 200, 
                'population_size': 100, 
                'mutation_probability': 0.1, 
                'elit_ratio': el, 
                'crossover_probability': 0.5, 
                'parents_portion': 0.2, 
                'crossover_type': 'uniform', # best by tests 
                'selection_type': 'sigma_scaling',# best by tests
                'max_iteration_without_improv': 40
                }
            
            model = ga(function=f, dimension = len(rows), 
                       variable_type='int', 
                       variable_boundaries = varbound, 
                       algorithm_parameters = param)
            
            
            
            model.run(no_plot = True, start_generation={'variables':samples, 'scores': samples_scores})
            
            table.loc[i, j] = model.output_dict['function']/best_before
            
            if i == 10:
                sds = np.std(table.values, axis = 1)
                br = (sds > 0).sum() == 0
                
    if not br:
        table.to_csv(f'{round(best_before)} {random.randint(0, 100)} stats.csv', index = False)

    res = model.output_dict

    return np.array([rows[i][int(j)] for i, j in enumerate(res['variable'])]), res['function'], model.output_dict['last_generation']['variables'], model.output_dict['last_generation']['scores']











#
#
# PLOTTING
#
#

def plot_pop(scores, hash_val, title):

    
    import matplotlib.pyplot as plt 
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    from matplotlib.ticker import NullLocator
    
    sc = sorted(scores)[::-1]

    fig, ax = plt.subplots( figsize=(7, 5))
    ax.xaxis.set_major_locator(NullLocator())

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects[-1:]:
            height = round(rect.get_height(), 2)
            ax.annotate('{}'.format(height),
                        xy = (rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=14, fontweight='bold')
    


    cols = np.zeros(len(sc))
    cols[-1] = 1

    x_coord = np.arange(len(sc))
    my_norm = Normalize(vmin=0, vmax=1)
    
    rc = ax.bar(x_coord, sc,  width = 0.7, color = cm.get_cmap('Set2')(my_norm(cols)))
    
    autolabel(rc)
    #raise Exception()
    #ax.set_xticks(x_coord)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Population objects')
    ax.set_ylabel('Cost value')
    #ax.set_ylim([0, max(subdict.values())*1.2])
    #fig.suptitle(title, fontsize=15, fontweight='bold')



    
    fig.tight_layout()
    
    plt.savefig(f'results/{hash_val} {title}.png', dpi = 200)
    
    plt.show()






