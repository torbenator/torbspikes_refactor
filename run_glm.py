import numpy as np
import scipy.io
import bookkeeping, models, utils, plotting, optimization, params


if __name__ == "__main__":

    # load neural data
    indir="/Users/Torben/Documents/Kording/GLMDeep/M1_Stevenson_Binned.mat"
    monkey_dat = scipy.io.loadmat(indir)
    all_spikes = monkey_dat['spikes']

    # just valid neurons
    validation_mat = scipy.io.loadmat('/Users/Torben/Documents/Kording/GLMDeep/Stevenson_valids.mat')
    validation_data = validation_mat['a']
    run_these = bookkeeping.filter_neurons(validation_data)

    for neuron_n in run_these[0:1]:

        valid_neurons = bookkeeping.filter_neurons(validation_data)

        print "Running Neuron: " + str(neuron_n)
        print 'valid neurons: ' + str(valid_neurons)

        data = bookkeeping.organize_data(all_spikes=all_spikes,my_neuron=neuron_n,
                                                      subsample=valid_neurons,train_test_ratio=0.9,
                                                      n_wins=None, convolve_params=None)

        glm_grid = params.glm_param_dict
        model = models.create_GLM_model

        save_path = '/Users/Torben/Code/torbspikes_refactor/hp_optimization/simultaneus/NN2l/'
        save_file = save_path+"neuron_"+str(neuron_n)+'.p'
        data_note='simultaneus_50ms'
        performance_dict = optimization.calculate_hp_performance(model,glm_grid, data,
            predicted_neuron=neuron_n,data_note=data_note,n_replicate=3,save_file=save_file)
