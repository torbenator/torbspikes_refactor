import csv
import numpy as np
import os
import itertools
import datetime

import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.stats import sem

import models, utils

def _create_param_dicts(param_dict):

    all_dicts = []
    sorted_keys = sorted(param_dict)
    combinations = list(itertools.product(*(param_dict[key] for key in sorted_keys)))
    for c in xrange(len(combinations)):
        new_dict = dict()
        for i, key in enumerate(sorted_keys):
            new_dict[key]=combinations[c][i]
        all_dicts.append(new_dict)
    return all_dicts


def calculate_hp_performance(build_fn, param_grid, data,
    nb_epochs=[5,10,20], predicted_neuron=None, data_note=None,
    save_file=None, safe=True,n_replicate=3):

    if safe:
        if save_file is None:
            y = raw_input("This file will not be saved. Type y to continue.")
            assert y == 'y'

        elif os.path.exists(save_file):
            y = raw_input(save_file + " already exists. Data may be overwritten. Type y to continue.")
            assert y == 'y'

    print "Performance will be saved to save_file"

    init_t = datetime.datetime.now()
    X_train,X_test,y_train,y_test = data[0],data[1],data[2],data[3]

    # fitting using training epochs is additive.
    nb_epoch_diffs = np.diff(nb_epochs)
    nb_epoch_diffs = np.insert(nb_epoch_diffs,0,nb_epochs[0])

    all_dicts = _create_param_dicts(param_grid)

    performance_dict = OrderedDict()
    performance_dict['notes'] = dict()
    performance_dict['notes']['model']= str(build_fn)
    performance_dict['notes']['neuron']= predicted_neuron
    performance_dict['notes']['data_note']= data_note
    performance_dict['notes']['date']= init_t
    performance_dict['notes']['replicates']= n_replicate

    print 'Params: ' + str(param_grid.items())
    print 'Runs: ' + str(nb_epochs)
    print 'Replicates: ' + str(n_replicate)

    c=0
    for these_params in all_dicts:
        for replicate in xrange(n_replicate):

            this_model = build_fn(input_dim=X_train.shape[1], **these_params)

            t = datetime.datetime.now()

            total_epoch=0
            for nb_epoch in nb_epoch_diffs:

                print "Run: "+str(c+1) + "/" + str(len(all_dicts*len(nb_epochs)*n_replicate))

                this_model.fit(X_train, y_train, nb_epoch=nb_epoch, verbose=False)
                total_epoch+= nb_epoch

                Yr = this_model.predict(X_train)
                Yt = this_model.predict(X_test)
                train_pR2 = utils.poisson_pseudoR2(y_train,Yr)
                test_pR2 = utils.poisson_pseudoR2(y_test,Yt)

                performance_dict[c]=dict()
                performance_dict[c]['params'] = these_params
                performance_dict[c]['training_epochs'] = total_epoch
                performance_dict[c]['train_score'] = train_pR2
                performance_dict[c]['test_score'] = test_pR2
                performance_dict[c]['time'] = datetime.datetime.now() - t

                c+=1

    total_runtime = datetime.datetime.now() - init_t
    print "Total Runtime:" + str(total_runtime)
    performance_dict['notes']['total_runtime'] = total_runtime

    if save_file is not None:
        pickle.dump(performance_dict, open( save_file, "wb" ) )

    return performance_dict


def reorganize_performance_dict(performance_dict,top_n=3):
    numerical_keys = [i for i in performance_dict.keys() if isinstance(i,int)]

    unique_params = np.unique([performance_dict[i]['params'] for i in numerical_keys])
    unique_runs = np.unique([performance_dict[i]['training_epochs'] for i in numerical_keys])

    reorganized = []

    for param in unique_params:
        for run in unique_runs:
            replicates = [performance_dict[i]['test_score'] for i in numerical_keys if performance_dict[i]['params']==param and performance_dict[i]['training_epochs']==run]
            reorganized.append([param,run,np.mean(replicates),sem(replicates)])

    sorted_reorganized = sorted(reorganized, key=itemgetter(2)) # sort by mean score

    reorganized_performance_dict = OrderedDict()

    for i in sorted_reorganized:
        label=str(i[0])+' '+str(i[1])+' training epochs: '
        reorganized_performance_dict[label]=dict()
        reorganized_performance_dict[label]['mean']=i[2]
        reorganized_performance_dict[label]['sem']=i[3]
    top_keys = reorganized_performance_dict.keys()[-top_n:]
    top_scores = [reorganized_performance_dict[i]['mean'] for i in top_keys]

    return reorganized_performance_dict, zip(top_keys,top_scores)

