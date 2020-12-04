import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Display plots inline and change default figure size
#matplotlib.rcParams['figure.figsize'] = (5.0, 4.0)

font1 = { 'weight' : 'normal',
'size'   : 14,
}

font2 = { 'weight' : 'normal',
'size'   : 12,
}

def plot_decision_boundary(pred_func, X, y, color_ind, alph, linewd):
    # Set min and max values and give it some padding
    padd = 5
    x_min, x_max = X[:, 0].min() - padd,  padd
    y_min, y_max = X[:, 1].min() - padd,  padd
    h =0.04 
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, alpha = alph,  levels=14, linewidths=linewd, colors=color_ind)

 
def Compute_decision_boundary(pred_func, X, y):
    
    # Set min and max values and give it some padding
    padd = 5
    x_min, x_max = X[:, 0].min() - padd,  padd
    y_min, y_max = X[:, 1].min() - padd,  padd
    h = 0.04
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    return Z
 
    

    
# data initialization
X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y = np.array([[0,1,1,0]]).T



class BinNN(object):
    def __init__(self, n_x, n_h, n_y, ww, learning_rate=0.01):
        """
        initialize the NN model
        
        parameters
        ----------
        n_x: int
            number of neurons in input layer
        n_h: int
            number of neurons in hidden layer
        n_y: int
            number of neurons in output layer
        learning_rate: float
            a hyperparam used in gradient descent
        """
        self.learning_rate = learning_rate
        bin_w = np.array(ww)
        
        # assign 9 weights
        self.W1 = bin_w [:4].reshape(2,2)
        self.b1 = bin_w [4:6]
        self.W2 = bin_w [6:8].reshape(2,1)
        self.b2 = bin_w [8:9]


    def _sigmoid(self, Z):
        """a sigmoid output loss function
        """
        return (1 / (1 + np.exp(-Z)))
    
    def feedforward(self, X):
        """performing the feedforward pass
        """
        # first hidden layer
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.tanh(self.Z1)
        # second layer
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self._sigmoid(self.Z2)
        
        
        return self.A2 

    def predict(self, X):

        A2 = self.feedforward(X)
        predictions = np.round(A2)
        return predictions
    



################## In the following we plot the unique decision boundaries


"""
Unique decision boundaries for Prune binary weights without bi-half
"""        
prunenumber = [0, 1]    
len_prune_rate = len(prunenumber)
for signmask_index in range(len_prune_rate):
    prunerate =  prunenumber[signmask_index] 
    print("Without Bi-half by pruning: {}".format(prunerate))
    sum_rem = int(9 - prunerate)   
    from itertools import product
    l = [-1, 0, 1]
    combin = list(product(l, repeat = 9))
    combin_arr = np.array(combin)
    sum_all = np.abs(combin_arr).sum(1)
    indx = np.where(sum_all==sum_rem)  # pick up all possible weights combinations
    combin = combin_arr[indx[0]]
    combin_len = combin.shape[0]
    print("All combinations: {}".format(combin_len))   
    first = True
    first0 = True
    for ind, bin_w in enumerate(combin):      
        Binary_model = BinNN(n_x=2, n_h=2, n_y=1, ww = bin_w, learning_rate=0.01)  
        Z = Compute_decision_boundary(lambda x: Binary_model.predict(x), X, y)

        if Z.sum() != 0 and Z.sum() != Z.shape[0]:
            if first:
                all_decisionB = Z
                All_bin_w = bin_w
            else:
                all_decisionB = np.concatenate((all_decisionB, Z), 1)
                All_bin_w = np.vstack((All_bin_w, bin_w))
                
            first = False        
  
    all_decisionB = np.array(all_decisionB)
    signall_decisionB = np.sign((all_decisionB - 0.5))
    q = signall_decisionB.shape[0]
    distH = 0.5 * (q - np.dot(signall_decisionB.transpose(), signall_decisionB ))
    distH[distH == q] = 0
    distH_used = distH
    aa = distH == 0 
    U = np.triu(aa,1)
    U_sum = U.sum(0)
    Unique_decision = len(U_sum) - (U_sum != 0).sum()  # roughly estimate the unique decison boundaries 
    print("Unique decision boundaries: {}".format(Unique_decision))  
    
     
    index = np.where(U_sum == 0)[0]
    hist = np.zeros([1, len(index)])
    i = 0
    first = True
    for ii in index:
        aa = All_bin_w[distH_used[ii] == 0]   # pick up all weight combinations for one decision boundary
        hist[0, i] =(distH_used[ii] == 0).sum()
        i+=1
        
        if first:
            Unique_weight = aa[0]
        else:
            Unique_weight = np.vstack((Unique_weight, aa[0]))        
        first = False   

    # plot the unique decision boundaries    
    ax = plt.figure(figsize=(5.0, 4.0) )
    linewd = 1    
    alph = 1
    color = 'r'           
    for ind, bin_w in enumerate(Unique_weight):      
        Binary_model = BinNN(n_x=2, n_h=2, n_y=1, ww = bin_w, learning_rate=0.01)  # without training
        plot_decision_boundary(lambda x: Binary_model.predict(x), X, y, color, alph, linewd)
    
    x_ticks =[-5, 0, 5]   
    x_ticks_lable = [ '-5',  '0',  '5']
    plt.xticks(x_ticks, x_ticks_lable, fontsize=12)
    y_ticks =[-5, 0, 5]   
    y_ticks_lable = [ '-5',  '0',  '5']  
    plt.yticks(y_ticks, y_ticks_lable, fontsize=12)
  
    plt.xlabel(r'$\mathtt{x}_1$ input value',font1)
    plt.ylabel('x$_2$ input value',font1)
    plt.tick_params(labelsize=13.5) 
    plt.savefig('./fig/prune{}w_noprior.pdf'.format(prunerate)) 
    plt.show()
    

"""
Unique decision boundaries with prior bi-half by pruning 1 weight
"""  

prunenumber =  [1] 
len_prune_rate = len(prunenumber)
for signmask_index in range(len_prune_rate):
    prunerate =  prunenumber[signmask_index] 
    print("Bi-half by pruning: {}".format(prunerate))
    sum_rem = int(9 - prunerate)
    l = [-1, 0, 1]
    combin = list(product(l, repeat = 9))
    combin_arr = np.array(combin)
    sum_all = np.abs(combin_arr).sum(1)
    indx = np.where(sum_all==sum_rem) 
    combin = combin_arr[indx[0]]
    combin_arr = combin
    sum_all = (combin_arr).sum(1) 
    sum_all_abs = np.abs(sum_all)
    min_sum = sum_all_abs.min()
    indx = np.where(sum_all_abs==min_sum)  
    combin = combin_arr[indx[0]]
    combin_len = combin.shape[0]
    print("All combinations: {}".format(combin_len))
    first = True
    first0 = True
    for ind, bin_w in enumerate(combin):
      
        Binary_model = BinNN(n_x=2, n_h=2, n_y=1, ww = bin_w, learning_rate=0.01)  # without training
        Z = Compute_decision_boundary(lambda x: Binary_model.predict(x), X, y)

        if Z.sum() != 0 and Z.sum() != Z.shape[0]:
            if first:
                all_decisionB = Z
                All_bin_w = bin_w
            else:
                all_decisionB = np.concatenate((all_decisionB, Z), 1)
                All_bin_w = np.vstack((All_bin_w, bin_w))
                
            first = False        
  
    all_decisionB = np.array(all_decisionB)
    signall_decisionB = np.sign((all_decisionB - 0.5))
    q = signall_decisionB.shape[0]
    distH = 0.5 * (q - np.dot(signall_decisionB.transpose(), signall_decisionB ))
    distH[distH == q] = 0
    distH_used = distH
    aa = distH == 0 
    U = np.triu(aa,1)
    U_sum = U.sum(0)
    Unique_decision = len(U_sum) - (U_sum != 0).sum()
    print("Unique decision boundaries: {}".format(Unique_decision))
    
     
    index = np.where(U_sum == 0)[0]
    hist = np.zeros([1, len(index)])
    i = 0
    first = True
    for ii in index:
        aa = All_bin_w[distH_used[ii] == 0]   # pick up all weight combinations
        hist[0, i] =(distH_used[ii] == 0).sum()
        i+=1
        
        if first:
            Unique_weight_half = aa[0]
        else:
            Unique_weight_half = np.vstack((Unique_weight_half, aa[0]))   
            
        first = False
   
    # plot the unique decision boundaries with bihalf    
    ax = plt.figure(figsize=(5.0, 4.0) )
    linewd = 1    
    alph = 1
    color = 'r'           
    for ind, bin_w in enumerate(Unique_weight_half):      
        Binary_model = BinNN(n_x=2, n_h=2, n_y=1, ww = bin_w, learning_rate=0.01)  
        plot_decision_boundary(lambda x: Binary_model.predict(x), X, y, color, alph, linewd)
    
    x_ticks =[-5, 0, 5]   
    x_ticks_lable = [ '-5',  '0',  '5']
    plt.xticks(x_ticks, x_ticks_lable, fontsize=12)
    y_ticks =[-5, 0, 5]   
    y_ticks_lable = [ '-5',  '0',  '5']  
    plt.yticks(y_ticks, y_ticks_lable, fontsize=12)
  
    plt.xlabel(r'$\mathtt{x}_1$ input value',font1)
    plt.ylabel('x$_2$ input value',font1)
    plt.tick_params(labelsize=13.5) 
    plt.savefig('./fig/prune{}w_half.pdf'.format(prunerate)) 
    plt.show()
        