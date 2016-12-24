import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from collections import OrderedDict,defaultdict
from sklearn.decomposition import PCA

"""
Functions to organize and manipulate spiking data

"""

def load_dat(indir="/Users/Torben/Documents/Kording/GLMDeep/M1_Stevenson_Binned.mat"):
    monkey_dat = scipy.io.loadmat(indir)
    monkey_dat.keys()
    all_spikes = monkey_dat['spikes']
    return all_spikes

def _build_kernel(kernel_size,kernel_type="cos"):
    """
    Builds different kinds of kernals to use for smoothing spike trains.
    Kernel types:
    cos : 1/2 cosine where
    """
    if kernel_type == "cos":
        return (np.cos(np.linspace(np.pi,3*np.pi,kernel_size))+1)*0.5

    if kernel_type == "exp":
        exp = np.linspace(0,10,kernel_size)**2
        return exp/float(max(exp))
    if kernel_type == "shifted_cos":
        return np.concatenate(((np.cos(np.linspace(np.pi,3*np.pi,kernel_size))+1)*0.5, np.zeros(kernel_size)))
    else:
        print kernel_type + " not built yet"

def resample_data(all_spikes,winsize):
    """
    Resamples data into winsize ms bins assuming srate is winsize/1ms.

    """
    n_neurons = all_spikes.shape[0]
    n_wins = int(np.floor(all_spikes.shape[1]/winsize))
    resampled_data = np.zeros((n_neurons,n_wins))

    for n in xrange(n_neurons):
        for sw in xrange(n_wins):
            this_win = sw * winsize
            next_win = (sw+1) * winsize
            resampled_data[n,sw] = np.sum(all_spikes[n,this_win:next_win])

    return resampled_data

def filter_neurons(validation_mat,include_types='valid', sorted_labels=[1,2], invalid_label=255):
    """
    some neurons are not actually neurons. Invalid neurons have bad fits.
    """

    if include_types == 'valid':
        neuron_inds = [i for i,j in enumerate(validation_mat) if j[1] != invalid_label]

    if include_types == 'sorted':
        neuron_inds = [i for i,j in enumerate(validation_mat) if j[1] in sorted_labels]
    if include_types == 'invalid':
        neuron_inds = [i for i,j in enumerate(validation_mat) if j[1] == invalid_label]
    return neuron_inds


def organize_data(all_spikes,my_neuron,subsample=None,
                  train_test_ratio=0.9, subsample_time=None,
                  n_wins=None, convolve_params=None,
                  RNN_out=False, window_mean=False,
                  verbose=False, include_my_neuron=False,
                  shrink_X=None, to_binary=False, boxcar_simp=None,
                  include_avg=True, include_pca_dims=2, set_seed=False):

    """

    Monstrosity of a bookkeeping method that builds train and test sets for predicting a
    spike train using nearby spike trains

    Parameters
    ============
    all_spikes : np.array of n_spikes x bins
    my_neuron : index of spike train we're trying to guess
    subsample : number of neurons to use as parameters set to 0 or all_spikes.shape[1]
    to include all spiketrains except for the one we're guessing
    train_test_ratio : self explanatory
    convolve_params : dictionary of kernel sizes and types to convolve with data
    as well as whether to convolve them with features or predictor(optional)
    EXAMPLE: kernel_params = {"kernel_size":[5,10,15],"kernel_type":["cos","cos","cos"],"X":True,"y":False}
    RNN_out : Boolean. True if you want X to be (Features x Examples),
    False if you want X to be (Features x Example_d1 x Example_d2) --> needed for RNN so that each example
    can have a feature and time component
    shrink_X : multiplies the spike series by the integer you input to normalize spikes - helps with NN algorithms
    to_binary : converts spike counts into simple 1/0 spikes/no spikes matrix

    Returns
    ===========
    X_train,X_test,y_train,y_test

    """

    if RNN_out == True:
        window_mean = False

    if set_seed == True:
        np.random.seed(16) # another sanity check

    orig_shape = all_spikes.shape

    if subsample != None:
        these_neurons = subsample
    else:
        these_neurons = range(all_spikes.shape[0])

    if convolve_params is None:
        convolve_params = {"kernel_size":None,"X":False,"y":False}

    if include_avg == True:
        all_spikes = np.vstack((all_spikes,np.mean(all_spikes,0)))
        these_neurons.append(all_spikes.shape[0]-1)
        if verbose == True:
            print 'added avg firing rate: ' + str(these_neurons[-1]) + ' to list of features'

    if include_pca_dims is not None:
            pca = PCA(n_components=2)
            pca.fit(all_spikes[these_neurons,:])
            all_spikes = np.vstack((all_spikes,pca.components_))
            these_neurons.append(all_spikes.shape[0]-1)
            if verbose == True:
                print 'added PCA dim 1: ' + str(these_neurons[-1]) + ' to list of features'
            these_neurons.append(all_spikes.shape[0]-2)
            if verbose == True:
                print 'added PCA dim 2: ' + str(these_neurons[-1]) + ' to list of features'


    # should we include the neuron we're trying to guess in the train set?
    these_neurons = [i for i in these_neurons if i != my_neuron or include_my_neuron==True]
    if verbose:
        print "Using "+str(len(these_neurons))+" X neurons:"

    # building kernels to convolve spike train with
    if convolve_params["kernel_size"]:
        new_y_inds = [] # exclude these from feature matrix
        processed_dat = []
        for kernel_set in zip(convolve_params['kernel_size'],convolve_params['kernel_type']):
            kernel = _build_kernel(kernel_size=kernel_set[0],kernel_type=kernel_set[1])
            for i, row in enumerate(all_spikes):
                conv = np.convolve(row,kernel,'same')
                normalized_conv = conv/float(max(conv))
                if sum(np.isnan(normalized_conv)) > 0:
                    if verbose:
                        print "excluding spike train " + str(i) + ". Kernel " + str(kernel_set) + " created nans."
                else:
                    processed_dat.append(normalized_conv) # normalizing. big filters have big amps
                    if i == my_neuron:
                        new_y_inds.append(len(processed_dat)-1)

        processed_dat = np.array(processed_dat)
        these_neurons = [i for i in xrange(processed_dat.shape[0]) if i not in new_y_inds]

        if verbose:
            print "processed_dat shape: " + str(processed_dat.shape)
            print "excluded " + str(len(new_y_inds)) + " rows of processed matrix. " + str(len(these_neurons)) + " features to be used"

    # setting train and test inds
    # for training using only preceeding information
    if n_wins:

        print "using data "+ str(n_wins) +" windows preceeding spike bins."
        split_ind = int((all_spikes.shape[1] - n_wins) * train_test_ratio) + n_wins

        X_train = np.zeros((split_ind, n_wins, len(these_neurons)))
        X_train[:] = np.nan
        X_test= np.zeros((all_spikes.shape[1] - split_ind - n_wins, n_wins, len(these_neurons)))
        X_test[:] = np.nan

        if convolve_params["X"] == True:
            for n in xrange(X_train.shape[2]):
                for i in xrange(split_ind):
                    X_train[i,:,n] = processed_dat[n, i:i + n_wins]

            for n in xrange(X_train.shape[2]):
                for i in xrange(split_ind, all_spikes.shape[1] - n_wins):
                    X_test[i-split_ind,:,n] = processed_dat[n, i:i + n_wins]

        elif convolve_params["X"] == False:
            for n in xrange(X_train.shape[2]):
                for i in xrange(split_ind):
                    X_train[i,:,n] = all_spikes[n, i:i + n_wins]
            for n in xrange(X_train.shape[2]):
                for i in xrange(split_ind, all_spikes.shape[1] - n_wins):
                    X_test[i-split_ind,:,n] = all_spikes[n, i:i + n_wins]

        if convolve_params["y"] == True:
            y_train = processed_dat[new_y_inds,n_wins:split_ind+n_wins]
            y_test = processed_dat[new_y_inds,split_ind+n_wins:processed_dat.shape[1]]

        elif convolve_params["y"] == False:
            y_train = all_spikes[my_neuron,n_wins:split_ind+n_wins]
            y_test = all_spikes[my_neuron,split_ind+n_wins:all_spikes.shape[1]]


        if RNN_out == False and window_mean == True:
            X_train = np.mean(X_train,1)
            X_test = np.mean(X_test,1)

        if RNN_out == False and window_mean == False:
            X_train = np.reshape(X_train,[X_train.shape[0],X_train.shape[1]*X_train.shape[2]])
            X_test = np.reshape(X_test,[X_test.shape[0],X_test.shape[1]*X_test.shape[2]])
            print "X flattened. X shape = " + str(X_train.shape)

    # for training using any information

    elif boxcar_simp is not None:

        split_ind = int((all_spikes.shape[1] - np.max(boxcar_simp)) * train_test_ratio) + np.max(boxcar_simp)

        X_train = np.zeros((split_ind - np.max(boxcar_simp), len(boxcar_simp), len(these_neurons)))
        X_train[:] = np.nan
        X_test= np.zeros((all_spikes.shape[1] - split_ind-1, len(boxcar_simp), len(these_neurons)))
        X_test[:] = np.nan

        for bi,b in enumerate(boxcar_simp):
            for n in xrange(X_train.shape[2]):
                for i in xrange(np.max(boxcar_simp), split_ind):
                    X_train[i-np.max(boxcar_simp),bi,n] = np.sum(all_spikes[n,i-b:i])

        y_train = all_spikes[my_neuron, xrange(np.max(boxcar_simp), split_ind)]

        for bi,b in enumerate(boxcar_simp):
            for n in xrange(X_test.shape[2]):
                for i in xrange(split_ind, all_spikes.shape[1]-1):
                    X_test[i-split_ind,bi,n] = np.sum(all_spikes[n,i-b:i])
        y_test = all_spikes[my_neuron, xrange(split_ind, all_spikes.shape[1]-1)]

        if window_mean == True:
            X_train = np.reshape(X_train,[X_train.shape[0],X_train.shape[1]*X_train.shape[2]])
            X_test = np.reshape(X_test,[X_test.shape[0],X_test.shape[1]*X_test.shape[2]])
            print "X flattened. X shape = " + str(X_train.shape)

    else:
        train_inds = np.random.choice(all_spikes.shape[1],int(all_spikes.shape[1]*train_test_ratio),replace=False)
        test_inds = [i for i in range(all_spikes.shape[1]) if i not in train_inds]
        if verbose:
            print "length of train inds: " + str(len(train_inds))
            print "length of test inds: " + str(len(test_inds))

        # convolve X spikes?
        if convolve_params["X"] == True:
            X_train = np.array([processed_dat[i,train_inds] for i in these_neurons]).T
            X_test = np.array([processed_dat[i,test_inds] for i in these_neurons]).T
        else:
            X_train = np.array([all_spikes[i,train_inds] for i in these_neurons]).T
            X_test = np.array([all_spikes[i,test_inds] for i in these_neurons]).T

        # convolve y spikes?
        if convolve_params["y"] == True:
            y_train = processed_dat[my_neuron,train_inds].T
            y_test = processed_dat[my_neuron,test_inds].T
        else:
            y_train = all_spikes[my_neuron,train_inds].T
            y_test = all_spikes[my_neuron,test_inds].T

    if shrink_X:
        if verbose:
            print "shrinking X by factor of " + str(shrink_X)
        X_train_z = (X_train-np.mean(X_train))/np.std(X_train)
        X_train = (X_train_z-np.min(X_train_z))*shrink_X

        X_test_z = (X_test-np.mean(X_test))/np.std(X_test)
        X_test = (X_test_z-np.min(X_test_z))*shrink_X

    if to_binary:
        X_train = np.array(X_train>0,dtype=int)
        X_test = np.array(X_test>0,dtype=int)
        y_train = np.array(y_train>0,dtype=int)
        y_test = np.array(y_test>0,dtype=int)

    return X_train,X_test,y_train,y_test


def sort_spikes(all_spikes, method):
    """

    A method to sort neurons by their spike counts.

    Parameters
    ==========
    all_spikes : neuron x spike count matrix
    method : method to sort neurons by
        'sum' : sort neurons by total spike count
        'residual_distance' : sort neurons by how much their spiking deviates from the field
        'sum_corr' : sort neurons by how they correlate with the other neurons

    Returns:
    ==========
    inds : sorted indicies based on method that was used

    """

    if method == 'sum':
        inds = np.argsort(np.nansum(all_spikes,1))

    elif method == 'residual_distance':
        mean_time_series = np.mean(all_spikes,1)
        residuals = []
        for spike in all_spikes:
            diff = np.sum((spike-mean_time_series)**2)
            residuals.append(diff)
        inds = np.argsort(residuals)

    elif method == "sum_corr":
        mat = np.corrcoef(all_spikes)
        inds = np.argsort(np.nansum(mat,1))

    else:
        print method + " not built yet."
        return

    sums = np.sum(all_spikes,1)
    counts = [sums[i] for i in inds]

    return inds, counts

#def supplement_avg_spiking(X_train,X_test):


