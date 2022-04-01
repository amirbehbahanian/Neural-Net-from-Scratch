import numpy as np

	
def initialize_parameters_deep(layer_dims):
    
    np.random.seed(52)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):

        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters
	
def linear_forward(A, W, b):

    Z = np.dot(W, A) + b
    cache = (A, W, b)
    
    return Z, cache
	
def linear_activation_forward(A_prev, W, b, activation):

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
 
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z) 
    cache = (linear_cache, activation_cache)

    return A, cache
	
def L_model_forward(X, parameters):
    
    caches = []
    A = X
    L = len(parameters) // 2 
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache) 

    AL, cache = linear_activation_forward(A, parameters['W' + str(l+1)], parameters['b' + str(l+1)], "sigmoid")
    caches.append(cache)    
          
    return AL, caches
	
def compute_cost(AL, Y):
    
    m = Y.shape[1]
    cost = -1/m * ( np.dot( Y, np.log(AL.T) ) + np.dot( 1-Y, np.log(1-AL.T) ) )
    cost = np.squeeze(cost)      
 
    return cost
	
def linear_backward(dZ, cache):

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1/m * np.dot(dZ, A_prev.T)
    db = 1/m * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db
	
def linear_activation_backward(dA, cache, activation):

    linear_cache, activation_cache = cache
    
    if activation == "relu":

        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":

        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)     
    
    return dA_prev, dW, db
	
def L_model_backward(AL, Y, caches):

    grads = {}
    L = len(caches) 
    m = AL.shape[1]
    Y = Y.reshape(AL.shape) 

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp
    
    for l in reversed(range(L-1)):

        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp, current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads
	
def update_parameters(params, grads, learning_rate):

    parameters = params.copy()
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l+1)] = params['W' + str(l+1)] - learning_rate * grads["dW" + str(l+1)] 
        parameters["b" + str(l+1)] = params['b' + str(l+1)] - learning_rate * grads["db" + str(l+1)]     
    return parameters