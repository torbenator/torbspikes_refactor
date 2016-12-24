

glm_param_dict = {
    'optimizer' : ['adam','nadam'],
    'l1' : [0,0.001,0.01,0.1],
    'l2' : [0,0.001,0.01,0.1],
}


nn_param_dict_1l = {
    'dropout' : [0.2],
    'n_neurons' : [10,25,50,100],
    'act' : ['tanh','softmax'],
    'l1' : [0,0.001,0.01],
    'l2' : [0,0.001,0.01],
    'optimizer' : ['adam','nadam'],
}


nn_param_dict_2l = {
    'dropout1' : [0.2],
    'dropout2' : [0.2],
    'n_neurons_1l' : [100,200],
    'act1' : ['tanh','softmax'],
    'act2' : ['tanh','softmax'],
    'n_neurons_2l' : [25,50,100],
    'l1' : [0,0.001,0.01],
    'l2' : [0,0.001,0.01],
    'optimizer' : ['adam','nadam'],
}


rnn_param_dict = {
    'dropout' : [0,0.2,0.4],
    'nLSTM' : [5,10,25,50],
    'l1' : [0,0.001,0.01,0.1],
    'l2' : [0,0.001,0.01,0.1],
    'optimizer' : ['adam','nadam'],
}