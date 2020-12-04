import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Display plots inline and change default figure size
matplotlib.rcParams['figure.figsize'] = (5.0, 4.0)



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
    h =1 
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
#    plt.contourf(xx, yy, Z, cmap="RdBu_r")
    plt.contour(xx, yy, Z, alpha = alph,  levels=14, linewidths=linewd, colors=color_ind)
# =============================================================================
#     plt.scatter(X[0, 0], X[0, 1], s=100, marker='^', c= 'r')   
#     plt.scatter(X[3, 0], X[3, 1],  s=100,marker='^', c= 'r')       
#     plt.scatter(X[1, 0], X[1, 1], s=100, c= 'g')       
#     plt.scatter(X[2, 0], X[2, 1],  s=100,c= 'g')             
#     
# =============================================================================


 
def Compute_decision_boundary(pred_func, X, y, color_ind):
    
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
 
    
class NN(object):
    def __init__(self, n_x, n_h, n_y, learning_rate=0.01):
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
        self.W1 = np.random.randn(n_x, n_h)
        self.b1 = np.zeros(shape=(1, n_h))
        self.W2 = np.random.randn(n_h, n_y)
        self.b2 = np.zeros(shape=(1, n_y))
        
    def _sigmoid(self, Z):
        """a sigmoid activation function
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
        
# =============================================================================
#         # first hidden layer
#         self.Z1 = np.dot(X, self.W1)
#         self.A1 = np.tanh(self.Z1)
#         # second layer
#         self.Z2 = np.dot(self.A1, self.W2)
#         self.A2 = self._sigmoid(self.Z2)        
# =============================================================================
        
        
        return self.A2 
    
    def compute_cost(self, A2, Y):
        """
        Computes the cross-entropy cost
        parameters
        ----------
        A2: np.ndarray
            the output generated by the output layer
        Y: np.ndarray
            the true labels
            
        return
        ----------
        cost: np.float64
            the cost per feedforward pass
        """
        m = Y.shape[0] # number of example
        logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
        cost = - np.sum(logprobs) / m
        cost = np.squeeze(cost)     # makes sure cost is the dimension we expect.
        return cost
    
    def backprop(self, X, Y):
        """
        performs the backpropagation algorithm
        
        parameters
        ------------
        X: np.ndarray
            representing the input that we feed to the neural net
        Y: np.ndarray
            the true label
        """
        m = X.shape[0]

        self.dZ2 = self.A2 - Y


        self.dW2 = (1. / m) * np.dot(self.A1.T, self.dZ2)
        self.db2 = (1. / m) * np.sum(self.dZ2, axis=0, keepdims=True)


        self.dZ1 = np.multiply(np.dot(self.dZ2, self.W2.T ), 1 - np.power(self.A1, 2))
        self.dW1 = (1. / m) * np.dot(X.T, self.dZ1) 
        self.db1 = (1. / m) * np.sum(self.dZ1, axis=0, keepdims=True)
        
    def update_parameters(self):
        """performs an update parameters for gradient descent
        """
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1 
        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2
        
    def predict(self, X):
        """
        an interface to generate prediction
        parameters
        ------------
        X: np.ndarray
           input features to our model
        
        return
        ------------
            np.ndarray - the predicted labels
        """
        A2 = self.feedforward(X)
        predictions = np.round(A2)
        return predictions
    
# data initialization
X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y = np.array([[0,1,1,0]]).T



plt.scatter(X[:,0], X[:,1], s=40, c=y.reshape((4, )), cmap=plt.cm.Spectral)




def build_model(X, y, num_hidden, learning_rate=0.01, num_iterations=50000, verbose=True):
    """
    an intermediate method to train our model
    parameters
    -------------------
    X: numpy.ndarray
        input data
    y: numpy.ndarray
        the real label
    num_hidden: int
        number of hidden neurons in the layer
    learning_rate: float
        hyperparam for gradient descent algorithm
    num_iterations: int
        number of passes, each pass using number of examples. 
    verbose: boolean
        optional. if True, it will print the cost per 1000 iteration
        
    return
    ---------------------
    model: an instance of NN object that represent the trained neural net model
    cost_history: a list containing the cost during training
    """
    model = NN(n_x=2, n_h=num_hidden, n_y=1, learning_rate=learning_rate)
    cost_history = []
    fir = 0
    for i in range(0, num_iterations):
        A2 = model.feedforward(X)
        cost = model.compute_cost(A2, y)
        model.backprop(X, y)
        model.update_parameters()
        if i % 100 == 0 and verbose:
            print ("Iteration %i Cost: %f" % (i, cost))
#            plot_decision_boundary(lambda x: model.predict(x), X, y, fir)
            fir += 1
            
        cost_history.append(cost)
    return model, cost_history

# =============================================================================
# 
# model, _ = build_model(X, y, 2,  num_iterations=10000)
# 
# 
# plot_decision_boundary(lambda x: model.predict(x), X, y, 'r')
# plt.title("2 neurons in hidden layer")
# 
#     
# plt.show()    
# 
# =============================================================================



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
        
        
        self.W1 = bin_w [:4].reshape(2,2)
        self.b1 = bin_w [4:6]
        self.W2 = bin_w [6:8].reshape(2,1)
        self.b2 = bin_w [8:9]


# =============================================================================
#         
#         self.W1 = np.random.randn(n_x, n_h)
#         self.b1 = np.zeros(shape=(1, n_h))
#         self.W2 = np.random.randn(n_h, n_y)
#         self.b2 = np.zeros(shape=(1, n_y))
# =============================================================================
        
    def _sigmoid(self, Z):
        """a sigmoid activation function
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
        
# =============================================================================
#         # first hidden layer
#         self.Z1 = np.dot(X, self.W1)
#         self.A1 = np.tanh(self.Z1)
#         # second layer
#         self.Z2 = np.dot(self.A1, self.W2)
#         self.A2 = self._sigmoid(self.Z2)        
# =============================================================================
        
        
        return self.A2 

    def predict(self, X):
        """
        an interface to generate prediction
        parameters
        ------------
        X: np.ndarray
           input features to our model
        
        return
        ------------
            np.ndarray - the predicted labels
        """
        A2 = self.feedforward(X)
        predictions = np.round(A2)
        return predictions
    




# =============================================================================
# 
# save_file_path1 = './Without_Prune_half_Unique_weight.npy'
# Unique_weight_prun0 = np.load(save_file_path1)
# 
# ax = plt.figure(figsize=(5.0, 4.0) )
# 
# 
# linewd = 0.5    
# alph = 1
# color = 'r'           
# for ind, bin_w in enumerate(Unique_weight_prun0):
#     print("current training ind: {} for training".format(ind))  
#   
#     Binary_model = BinNN(n_x=2, n_h=2, n_y=1, ww = bin_w, learning_rate=0.01)  # without training
#     plot_decision_boundary(lambda x: Binary_model.predict(x), X, y, color, alph, linewd)
# 
#     
# x_ticks =[-5, 0, 5]   
# x_ticks_lable = [ '-5',  '0',  '5']
# 
# plt.xticks(x_ticks, x_ticks_lable, fontsize=12)
# 
# y_ticks =[-5, 0, 5]   
# y_ticks_lable = [ '-5',  '0',  '5']
# 
# plt.yticks(y_ticks, y_ticks_lable, fontsize=12)
# 
# 
# plt.xlabel(r'$\mathtt{x}_1$ input value',font1)
# plt.ylabel('x$_2$ input value',font1)
# plt.tick_params(labelsize=13.5) 
# 
# 
# plt.savefig('./fig/Onecolor_prune0.pdf', bbox_inches = 'tight', pad_inches=0) 
# 
# plt.show()
# 
# 
# 
# 
# 
# 
# 
# 
# save_file_path1 = './Without_half_half_Unique_weight.npy'
# Unique_weight = np.load(save_file_path1)
# 
# ax = plt.figure(figsize=(5.0, 4.0) )
# linewd = 0.5    
# alph = 1
# color = 'r'           
# for ind, bin_w in enumerate(Unique_weight):
#     print("current training ind: {} for training".format(ind))  
#   
#     Binary_model = BinNN(n_x=2, n_h=2, n_y=1, ww = bin_w, learning_rate=0.01)  # without training
#     plot_decision_boundary(lambda x: Binary_model.predict(x), X, y, color, alph, linewd)
# 
#     
# x_ticks =[-5, 0, 5]   
# x_ticks_lable = [ '-5',  '0',  '5']
# 
# plt.xticks(x_ticks, x_ticks_lable, fontsize=12)
# 
# y_ticks =[-5, 0, 5]   
# y_ticks_lable = [ '-5',  '0',  '5']
# 
# plt.yticks(y_ticks, y_ticks_lable, fontsize=12)
# 
# 
# plt.xlabel(r'$\mathtt{x}_1$ input value',font1)
# plt.ylabel('x$_2$ input value',font1)
# plt.tick_params(labelsize=13.5) 
# 
# 
# plt.savefig('./fig/Onecolor_prune1_nohalf.pdf', bbox_inches = 'tight', pad_inches=0) 
# 
# plt.show()
# 
# 
# =============================================================================
        


save_file_path1 = './Without_half_half_Unique_weight.npy'
Unique_weight = np.load(save_file_path1)


save_file_path2 = './Half_half_Unique_weight.npy'        
Unique_weight_half = np.load(save_file_path2)




####################################------------Prune binary weights------------########################
# Display plots inline and change default figure size
alph = 1
#ax = plt.figure(figsize=(5.0, 4.0) )
color = 'b'  
first = True         
for ind, bin_w in enumerate(Unique_weight):
    print("current training ind: {} for training".format(ind))  
  
    Binary_model = BinNN(n_x=2, n_h=2, n_y=1, ww = bin_w, learning_rate=0.01)  # without training

    Z = Compute_decision_boundary(lambda x: Binary_model.predict(x), X, y, color)

    if Z.sum() != 0 and Z.sum() != Z.shape[0]:
        if first:
            all_decisionB = Z
            All_bin_w = bin_w
        else:
            all_decisionB = np.concatenate((all_decisionB, Z), 1)
            All_bin_w = np.vstack((All_bin_w, bin_w))
            
        first = False        
  
all_decisionB = np.array(all_decisionB)
signall_decisionB_without = np.sign((all_decisionB - 0.5))
q = signall_decisionB_without.shape[0]



####################################------------Prune half-half binary weights------------########################
# Display plots inline and change default figure size
color = 'r'  
first = True         
for ind, bin_w in enumerate(Unique_weight_half):
    print("current training ind: {} for training".format(ind))  
  
    Binary_model = BinNN(n_x=2, n_h=2, n_y=1, ww = bin_w, learning_rate=0.01)  # without training

    Z = Compute_decision_boundary(lambda x: Binary_model.predict(x), X, y, color)

    if Z.sum() != 0 and Z.sum() != Z.shape[0]:
        if first:
            all_decisionB0 = Z
            All_bin_w_half = bin_w
        else:
            all_decisionB0 = np.concatenate((all_decisionB0, Z), 1)
            All_bin_w_half = np.vstack((All_bin_w_half, bin_w))
            
        first = False        
  
all_decisionB0 = np.array(all_decisionB0)
signall_decisionB_half = np.sign((all_decisionB0 - 0.5))
q = signall_decisionB_half.shape[0]


distH = 0.5 * (q - np.dot(signall_decisionB_without.transpose(),  signall_decisionB_half))
distH[distH == q] = 0
distH_used = distH
aa = distH == 0 
bb= aa.sum(1)
index = np.where(bb == 0)[0]
first = True
for ii in index:
    aa = All_bin_w[ii]   # pick up all weight combinations
    
    if first:
        Unique_black = aa
    else:
        Unique_black = np.vstack((Unique_black, aa))   
        
    first = False









alph = 1
ax = plt.figure(figsize=(5.0, 4.0) )




####################################------------Prune half-half binary weights------------########################
# Display plots inline and change default figure size
color = 'r'   
linewd = 0.5        
for ind, bin_w in enumerate(Unique_weight_half):
    print("current training ind: {} for training".format(ind))  
  
    Binary_model = BinNN(n_x=2, n_h=2, n_y=1, ww = bin_w, learning_rate=0.01)  # without training
    plot_decision_boundary(lambda x: Binary_model.predict(x), X, y, color, alph, linewd)


####################################------------Prune binary weights------------########################
# Display plots inline and change default figure size
linewd = 0.5    
alph = 0.3
color = 'b'           
for ind, bin_w in enumerate(Unique_black):
    print("current training ind: {} for training".format(ind))  
  
    Binary_model = BinNN(n_x=2, n_h=2, n_y=1, ww = bin_w, learning_rate=0.01)  # without training
    plot_decision_boundary(lambda x: Binary_model.predict(x), X, y, color, alph, linewd)

# =============================================================================
# 
# ####################################------------Prune binary weights------------########################
# # Display plots inline and change default figure size
# linewd =1     
# 
# #color = 'b'  
# Unique_black_len = Unique_black.shape[0]
# color = plt.cm.Blues(np.linspace(0, 1, Unique_black_len))
#          
# for ind, bin_w in enumerate(Unique_black):
#     print("current training ind: {} for training".format(ind))  
#   
#     Binary_model = BinNN(n_x=2, n_h=2, n_y=1, ww = bin_w, learning_rate=0.01)  # without training
#     color_ind = color[ind]
#     plot_decision_boundary(lambda x: Binary_model.predict(x), X, y, color_ind.reshape(-1,4),  alph, linewd)
# 
# 
# 
# 
# =============================================================================

#######################################plot#######################################
    
x_ticks =[-5, 0, 5]   
x_ticks_lable = [ '-5',  '0',  '5']

plt.xticks(x_ticks, x_ticks_lable, fontsize=12)

y_ticks =[-5, 0, 5]   
y_ticks_lable = [ '-5',  '0',  '5']

plt.yticks(y_ticks, y_ticks_lable, fontsize=12)


plt.xlabel(r'$\mathtt{x}_1$ input value',font1)
plt.ylabel('x$_2$ input value',font1)
plt.tick_params(labelsize=13.5) 


plt.savefig('./fig/bluecolor_figure_half_half.pdf', bbox_inches = 'tight', pad_inches=0) 

plt.show()


        