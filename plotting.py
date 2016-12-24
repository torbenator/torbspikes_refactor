

def plot_performance_dict(performance_dict, verbose=False):

    reorganized_performance_dict,top_n = reorganize_performance_dict(performance_dict)
    plotkeys = reorganized_performance_dict.keys()
    fsize = (20,10)
    if len(plotkeys)>100:
        fsize = (len(plotkeys)/20,len(plotkeys)/40)
    fig = plt.figure(figsize=fsize)
    ax1 = fig.add_subplot(111)

    for i,k in enumerate(plotkeys):
        mean_score = reorganized_performance_dict[k]['mean']
        sem_score = reorganized_performance_dict[k]['sem']

        ax1.errorbar(mean_score, i, xerr=sem_score, color='k',linestyle='-', linewidth=3,
                     marker='s', markersize=5, markeredgecolor='k',
                     markerfacecolor='w')

    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    ytick_params = range(len(plotkeys))
    ax1.set_yticks([i for i in ytick_params])
    ax1.set_yticklabels([k for k in plotkeys])
    ax1.set_ylim([min(ytick_params)-.5,max(ytick_params)+0.25])
    ax1.set_xlabel('Test Score')

    fig.tight_layout()
    if verbose:
        print top_n
    return fig