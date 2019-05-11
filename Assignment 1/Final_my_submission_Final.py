'''

2018 Assigment One : Differential Evolution
    
Scafolding code

Complete the missing code at the locations marked 
with 'INSERT MISSING CODE HERE'

To run task_2 you will need to download an unzip the file dataset.zip

If you have questions, drop by in one of the pracs on Wednesday 
     11am-1pm in S503 or 3pm-5pm in S517
You can also send questions via email to f.maire@qut.edu.au


'''

'''
Project Team:
    n10143416 Dmitrii Menshikov
    n10030581 Chen-Yen Chou
    n10080236 Victor Manuel Villamil
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn import model_selection

# ----------------------------------------------------------------------------

def differential_evolution(fobj, 
                           bounds, 
                           mut=2, 
                           crossp=0.7, 
                           popsize=20, 
                           maxiter=100,
                           verbose = True):
    '''
    This generator function yields the best solution x found so far and 
    its corresponding value of fobj(x) at each iteration. In order to obtain 
    the last solution,  we only need to consume the iterator, or convert it 
    to a list and obtain the last value with list(differential_evolution(...))[-1]    
    
    
    @params
        fobj: function to minimize. Can be a function defined with a def 
            or a lambda expression.
        bounds: a list of pairs (lower_bound, upper_bound) for each 
                dimension of the input space of fobj.
        mut: mutation factor
        crossp: crossover probability
        popsize: population size
        maxiter: maximum number of iterations
        verbose: display information if True    
    '''
    def affine(w,bounds):
        '''
        Denormalization function
        maps each element of w into bounds. 
        len(w)==len(bounds)
        
        @params
            w: numpy array. Each element belongs to interval [0,1]
            bounds: a list of pairs (lower_bound, upper_bound)
        '''
        lower_bound =np.array(bounds)[:,0]
        upper_bound = np.array(bounds)[:,1]      
        w_new=w*(upper_bound-lower_bound)+lower_bound
        return w_new
#-------end of affine function-------------------------------------------------    
    
    n_dimensions = len(bounds) # dimension of the input space of 'fobj'
    #This generates our initial population of 10 random normilized vectors.
    w=np.random.random((popsize,n_dimensions)) 
    #    Each component x[i] is normalized between [0, 1]. 
    #    We will use the bounds to denormalize each component only for 
    #    evaluating them with fobj.
    
    'INSERT MISSING CODE HERE'
    #calculating costs for each vector in initial population
    cost=np.array([fobj(affine(individual,bounds)) for individual in w])
    #handling nan's
    cost[np.where(np.isnan(cost))]=np.inf
    best_idx = cost.argmin()
    
    if verbose:
        print(
        '** Lowest cost in initial population = {} '
        .format(abs(cost[best_idx]))  )      
    for i in range(maxiter):
        if verbose:
            print('** Starting generation {}, '.format(i), end="")        
  
        for j, wi in enumerate(w): # for each element in population
            #MUTATION
            #identifying random distinct vectors a,b and c
            abc_indexes = [n for n in range(len(w))]
            abc_indexes.remove(j)
            abc_indexes=np.random.permutation(abc_indexes)[:3]
            a,b,c=w[abc_indexes]

            #calculating mutant vector
            mutant=a+mut*(b-c)
            #clip mutant vector to the bounds [0,1]
            mutant=np.clip(mutant,0,1)
            #RECOMBINATION
            #creating trial vector
            trial_mutation = np.random.binomial(1,crossp,n_dimensions)==True
            trial=np.where(trial_mutation,mutant,wi)
            
            #REPLACEMENT
            #calculating cost for trial vector
            trial_cost = fobj(affine(trial,bounds))
            if np.isnan(trial_cost): trial_cost=np.inf #handling nan's
            wi_cost = cost[j]
            if trial_cost<=wi_cost:
                w[j]=trial
                cost[j]=trial_cost
    
        best_idx=cost.argmin() #minimum error
        best=affine(w[best_idx],bounds)

        yield best, cost[best_idx]

# ----------------------------------------------------------------------------

def task_1():
    '''
    Our goal is to fit a curve (defined by a polynomial) to the set of points 
    that we generate. 
    '''

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
    def fmodel(x, w):
        '''
        Compute and return the value y of the polynomial with coefficient 
        vector w at x.  
        For example, if w is of length 5, this function should return
        w[0] + w[1]*x + w[2] * x**2 + w[3] * x**3 + w[4] * x**4 
        The argument x can be a scalar or a numpy array.
        The shape and type of x and y are the same (scalar or ndarray).
        '''
        if isinstance(x, float) or isinstance(x, int):
            y = 0
        else:
            assert type(x) is np.ndarray
            y = np.zeros_like(x)
            for i, wi in enumerate(w):
                y=y+wi*x**i
            
        'INSERT MISSING CODE HERE'
        return y

    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
        
    def rmse(w):
        '''
        Compute and return the root mean squared error (RMSE) of the 
        polynomial defined by the weight vector w. 
        The RMSE is is evaluated on the training set (X,Y) where X and Y
        are the numpy arrays defined in the context of function 'task_1'.        
        '''
        Y_pred = fmodel(X, w)
        return np.sqrt(((Y - Y_pred)**2).mean()) #INSERT MISSING CODE HERE'


    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    
    # Create the training set
    X = np.linspace(-5, 5, 500)
    Y = np.cos(X) + np.random.normal(0, 0.2, len(X))
    
    # Create the DE generator
    de_gen = differential_evolution(rmse, [(-5, 5)] * 6, mut=1, maxiter=2000)
    
    # We'll stop the search as soon as we found a solution with a smaller
    # cost than the target cost
    target_cost = 0.5
    
    # Loop on the DE generator
    for i , p in enumerate(de_gen):
        w, c_w = p
        print("error= " + str(round(c_w,3)))
        # w : best solution so far
        # c_w : cost of w        
        # Stop when solution cost is less than the target cost
        if c_w<target_cost:# 'INSERT MISSING CODE HERE':
            break
        
    # Print the search result
    print('Stopped search after {} generation. Best cost found is {}'.format(i,c_w))
    #    result = list(differential_evolution(rmse, [(-5, 5)] * 6, maxiter=1000))    
    #    w = result[-1][0]
        
    # Plot the approximating polynomial
    plt.scatter(X, Y, s=2)
    plt.plot(X, np.cos(X), 'r-',label='cos(x)')
    plt.plot(X, fmodel(X, w), 'g-',label='model')
    plt.legend()
    plt.title('Polynomial fit using DE')
    plt.show()    
    

# ----------------------------------------------------------------------------

def task_2():
    '''
    Goal : find hyperparameters for a MLP
    
       w = [nh1, nh2, alpha, learning_rate_init]
    '''
    
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  
    def eval_hyper(w):
        '''
        Return the negative of the accuracy of a MLP, trained 
        with the hyperparameters from vector w
        
        alpha : float, optional, default 0.0001
                L2 penalty (regularization term) parameter.
        '''
        
        nh1, nh2, alpha, learning_rate_init  = (
                int(1+w[0]), # nh1
                int(1+w[1]), # nh2
                10**w[2], # alpha on a log scale
                10**w[3]  # learning_rate_init  on a log scale
                )


        clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                            max_iter=100, 
                            alpha=alpha, #1e-4
                            learning_rate_init=learning_rate_init, #.001
                            solver='sgd', verbose=False , tol=1e-4, random_state=1,
                            )
        
        clf.fit(X_train_transformed, y_train)
        # compute the accurary on the test set
        mean_accuracy =  clf.score(X_test_transformed,y_test)
 
        return -mean_accuracy
    
    # . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .  

    # Load the dataset
    X_all = np.loadtxt('dataset_inputs.txt', dtype=np.uint8)[:1000]
    y_all = np.loadtxt('dataset_targets.txt',dtype=np.uint8)[:1000]    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                X_all, y_all, test_size=0.4, random_state=42)
       
    # Preprocess the inputs with 'preprocessing.StandardScaler'
    
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_test_transformed =  scaler.transform(X_test) #'INSERT MISSING CODE HERE'


    
    bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters
    
    de_gen = differential_evolution(
            eval_hyper, 
            bounds, 
            mut = 1,
            popsize=10, 
            maxiter=20,
            verbose=True)
    print('Starting DE. Population size: {}. Maximum Iterations: {}'.format("10","20"))
    for i, p in enumerate(de_gen):
        w, c_w =p#    'INSERT MISSING CODE HERE'
        print('Generation {},  best cost {}'.format(i,abs(c_w)))
        #print(w)
        # Stop if the accuracy is above 90%
        if abs(c_w)>0.90:
            break
 
    # Print the search result
    print('Population size: {}. Maximum Iterations: {}'.format("10","20"))
    print('Stopped search after {} generation. Best accuracy reached is {}'.format(i,abs(c_w)))   
    print('Hyperparameters found:')
    print('nh1 = {}, nh2 = {}'.format(int(1+w[0]), int(1+w[1])))          
    print('alpha = {}, learning_rate_init = {}'.format(10**w[2],10**w[3]))
    print("-"*30)

    
# ----------------------------------------------------------------------------

def task_3():
    '''
    Place holder for Task 3    
    '''
    def my_task2(popsize,maxiter):
        '''
        Modified task_2. The code altered to accept parameters and 
        return result and accuracy statistics
        
        @params
        popsize : size of population for DE
        maxiter : maximum iteration for DE
        '''
        def eval_hyper(w):  
            '''
            Lost function. Returns negative accuracy
            w : heperparameters for MLPClassifier
            [nh1, nh2, alpha, learning_rate_init]
            '''
            import math
            nh1, nh2, alpha, learning_rate_init  = (
                    int(1+w[0]), # nh1
                    int(1+w[1]), # nh2
                    10**w[2], # alpha on a log scale
                    10**w[3]  # learning_rate_init  on a log scale
                    )
    
            clf = MLPClassifier(hidden_layer_sizes=(nh1, nh2), 
                                max_iter=100, 
                                alpha=alpha, #1e-4
                                learning_rate_init=learning_rate_init, #.001
                                solver='sgd', verbose=False , tol=1e-4, random_state=1,
                                )
            clf.fit(X_train_transformed, y_train)
            # compute the accurary on the test set
            mean_accuracy =  clf.score(X_test_transformed,y_test)
            if math.isnan(mean_accuracy):
                print("/*-A"*30)
                return 100000
                
            return -mean_accuracy
            
    # . . . end of eval_hyper . . . . . . . . . . . . . . . . . . . . . . . . .  

        # Load the dataset
        X_all = np.loadtxt('dataset_inputs.txt', dtype=np.uint8)[:1000]
        y_all = np.loadtxt('dataset_targets.txt',dtype=np.uint8)[:1000]    
        X_train, X_test, y_train, y_test = model_selection.train_test_split(
                                    X_all, y_all, test_size=0.4, random_state=42)
           
        # Preprocess the inputs with 'preprocessing.StandardScaler'
        
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train_transformed = scaler.transform(X_train)
        X_test_transformed =  scaler.transform(X_test) #'INSERT MISSING CODE HERE'
    
        #popsize=kwargs["popsize"]
        #maxiter=kwargs["maxiter"]
    
        
        bounds = [(1,100),(1,100),(-6,2),(-6,1)]  # bounds for hyperparameters
        history=[]
        
        de_gen = differential_evolution(
                eval_hyper, 
                bounds, 
                mut = 1,
                popsize=popsize, 
                maxiter=maxiter,
                verbose=True)
        print('Starting DE. Population size: {}. Maximum Iterations: {}'.format(popsize,maxiter))
        for i, p in enumerate(de_gen):
            w, c_w =p#    'INSERT MISSING CODE HERE'
            history.append(abs(c_w))
            print('Generation {},  best cost {}'.format(i,abs(c_w)))
            #print(w)
            # Stop if the accuracy is above 90%
            if abs(c_w)>0.90:
                break
     
        # Print the search result
        print('Population size: {}. Maximum Iterations: {}'.format("10","20"))
        print('Stopped search after {} generation. Best accuracy reached is {}'.format(i,abs(c_w)))   
        print('Hyperparameters found:')
        print('nh1 = {}, nh2 = {}'.format(int(1+w[0]), int(1+w[1])))          
        print('alpha = {}, learning_rate_init = {}'.format(10**w[2],10**w[3]))
        print("-"*30)      
        return w,c_w, history
#.......end of my_Task_2.......................................................    
    
    def my_grid_search(fobj,param_list):
        '''
        runs passed function with each set of parameters from the list
        returns list of results if the function and parameters
        
        @params
        fobj(param1,param2,...paramN)  -  the function to run
        param_list - list of tuples of parameters for fobj function
        [(param1_v1, param2_v1....paramN_v1),(param1_v2, param2_v2....paramN_v2),...]
        '''
        history=[]
        for i,params in enumerate(param_list):
            print("{} of {}".format(i+1,len(param_list)))
            #history.append([fobj(*params),params])
            fobj_return=fobj(*params)
            record={"fobj_return":fobj_return,
                    "params":params}
            history.append(record)
        return history
#......end of my_grid_search...................................................
        
    #Parameters to test
    experiments=[(5,40),(10,20),(20,10),(40,5)]
    #experiments=[(5,5),(10,5)]#,(20,5),(40,5)]
    #Run the search
    history=my_grid_search(my_task2,experiments)
    
    print('*'*30)
    print()
    #print the report and plot graphs
    for h in history:
        #print(h)
        print("DE params: popsize={}; maxiter={}".format(h["params"][0],h["params"][1]))
        print ("accuracy={}:".format(abs(h["fobj_return"][1])))
        print ("NN params: nh1={}; nh2 ={}; alpha={}; learning rate={}".format(
                1+round(h["fobj_return"][0][0]), 
               1+round(h["fobj_return"][0][1]),
               10**h["fobj_return"][0][2],10**h["fobj_return"][0][3]))
        x=np.arange(len(h["fobj_return"][2]))
        y=h["fobj_return"][2]
        plt.plot(x,y)
        plt.xlabel("Generation")
        plt.ylabel("Accuracy")
        plt.title('Accuracy evolution. DE params: popsize={}; maxiter={}'.format(
                h["params"][0],h["params"][1]))
        plt.show()
        print('-'*30)
        

# ----------------------------------------------------------------------------


if __name__ == "__main__":
#    pass
#    task_1()    
#     task_2()    
     task_3()    

