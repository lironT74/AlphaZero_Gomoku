from multiprocessing import Pool, set_start_method
import pickle
import numpy             as np
import pandas            as pd
import matplotlib.pyplot as plt
import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from scipy import stats

from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

discription_dict = {
    "pt_6_6_4_p4_v27_training": "v27    25        1        yes",
    "pt_6_6_4_p4_v28_training": "v28    25        0        yes",
    "pt_6_6_4_p4_v29_training": "v29    25        1        no",
    "pt_6_6_4_p4_v30_training": "v30    25        0        no",
    "pt_6_6_4_p4_v23_training": "v23    50        1        yes",
    "pt_6_6_4_p4_v24_training": "v24    50        0        yes",
    "pt_6_6_4_p4_v25_training": "v25    50        1        no",
    "pt_6_6_4_p4_v26_training": "v26    50        0        no",
    "pt_6_6_4_p4_v31_training": "v31    100       1        yes",
    "pt_6_6_4_p4_v32_training": "v32    100       0        yes",
    "pt_6_6_4_p4_v33_training": "v33    100       1        no",
    "pt_6_6_4_p4_v34_training": "v34    100       0        no"
}



def boot_matrix(z, B):
    """Bootstrap sample

    Returns all bootstrap samples in a matrix"""

    n = len(z)  # sample size
    idz = np.random.randint(0, n, size=(B, n))  # indices to pick for all boostrap samples
    return z[idz]


def bootstrap_mean(x, B=100000, alpha=0.05, plot=False):
    """Bootstrap standard error and (1-alpha)*100% c.i. for the population mean

    Returns bootstrapped standard error and different types of confidence intervals"""

    x = np.array(x)
    # Deterministic things
    n = len(x)  # sample size
    orig = np.average(x)  # sample mean
    se_mean = np.std(x)/np.sqrt(n) # standard error of the mean
    qt = stats.t.ppf(q=1 - alpha/2, df=n - 1) # Student quantile

    # Generate boostrap distribution of sample mean
    xboot = boot_matrix(x, B=B)
    sampling_distribution = np.average(xboot, axis=1)

   # Standard error and sample quantiles
    se_mean_boot = np.std(sampling_distribution)
    quantile_boot = np.percentile(sampling_distribution, q=(100*alpha/2, 100*(1-alpha/2)))

    # # RESULTS
    # print("Estimated mean:", orig)
    # print("Classic standard error:", se_mean)
    # print("Classic student c.i.:", orig + np.array([-qt, qt])*se_mean)
    # print("\nBootstrap results:")
    # print("Standard error:", se_mean_boot)
    # print("t-type c.i.:", orig + np.array([-qt, qt])*se_mean_boot)
    # print("Percentile c.i.:", quantile_boot)
    # print("Basic c.i.:", 2*orig - quantile_boot[::-1])


    return quantile_boot


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def plot_smoothing_with_ci_aux(model, n_steps, model_loss_values, path):

    if not os.path.exists(f"{path}smoothing_data/{model}/steps_{n_steps}/"):
        os.makedirs(f"{path}smoothing_data/{model}/steps_{n_steps}/")

    if os.path.exists(f"{path}smoothing_data/{model}/steps_{n_steps}/under_line") and \
        os.path.exists(f"{path}smoothing_data/{model}/steps_{n_steps}/over_line")  and \
        os.path.exists(f"{path}smoothing_data/{model}/steps_{n_steps}/smooth_path"):

        return


    print(f"Starting model: {model}, n_steps = {n_steps}")

    windows = rolling_window(np.array(model_loss_values), n_steps)

    smooth_path = windows.mean(axis=1)

    under_line = []
    over_line = []

    for window in range(windows.shape[0]):
        l, h = bootstrap_mean(windows[window, :])
        under_line.append(l)
        over_line.append(h)


    pickle.dump(under_line, open(f"{path}smoothing_data/{model}/steps_{n_steps}/under_line", "wb"))
    pickle.dump(over_line, open(f"{path}smoothing_data/{model}/steps_{n_steps}/over_line", "wb"))
    pickle.dump(smooth_path, open(f"{path}smoothing_data/{model}/steps_{n_steps}/smooth_path", "wb"))

    print(f"Done model: {model}, n_steps = {n_steps}")


def plot_smoothing_with_ci(models_dfs, save_name="12_6x6_models"):
    n_steps = range(5, 505, 5)

    jobs = []

    for n_steps in n_steps:
        # Compute curves of interest:

        for index, (model, model_loss_values) in enumerate(models_dfs.items()):
            jobs.append((model, n_steps, model_loss_values, path))

    with Pool() as pool:
        pool.starmap(plot_smoothing_with_ci_aux, jobs)
        pool.close()
        pool.join()



    for n_steps in n_steps:
        # Compute curves of interest:

        colors_full = plt.cm.get_cmap("Blues", len(models_dfs.keys()))
        colors_non_full = plt.cm.get_cmap("Reds", len(models_dfs.keys()))

        for index, (model, model_loss_values) in enumerate(models_dfs.items()):

            under_line = pickle.load(open(f"{path}smoothing_data/{model}/steps_{n_steps}/under_line", "rb"))
            over_line = pickle.load(open(f"{path}smoothing_data/{model}/steps_{n_steps}/over_line", "rb"))
            smooth_path = pickle.load(open(f"{path}smoothing_data/{model}/steps_{n_steps}/smooth_path", "rb"))

            if discription_dict[model].endswith('yes'):
                color = colors_full(index)
            else:
                color = colors_non_full(index)


            plt.plot(smooth_path, linewidth=2, color=color)

            plt.fill_between(range(len(under_line)), under_line, over_line, color=color, alpha=.2)


        handles = []
        handles.append(mpatches.Patch(label="name simulations shutter full", color='none'))

        for index, value in enumerate(discription_dict.values()):

            if value.endswith('yes'):
                color = colors_full(index)
            else:
                color = colors_non_full(index)

            patch = mpatches.Patch(color=color, label=value)
            handles.append(patch)

        plt.legend(handles=handles, prop={'family': 'DejaVu Sans Mono', 'size': 8})
        plt.title(f"{save_name} {n_steps} steps smoothing")

        plt.savefig(f"{path}results/{save_name}_{n_steps}_steps_smoothing.png", bbox_inches='tight')
        # plt.show()
        plt.close('all')



def tabulate_events(dpath, path):

    final_out = {}

    runs = [f"pt_6_6_4_p4_v{i}_training" for i in range(23,35,1)]

    for dname in runs:

        print(f"{dname}")

        ea = EventAccumulator(os.path.join(dpath, dname)).Reload()
        # tags = ea.Tags()['scalars']

        tags = ['training_loss']


        for tag in tags:
            tag_values=[]
            wall_time=[]
            steps=[]

            for event in ea.Scalars(tag):
                tag_values.append(event.value)
                wall_time.append(event.wall_time)
                steps.append(event.step)

            # out[tag]=pd.DataFrame(data=dict(zip(steps,np.array([tag_values,wall_time]).transpose())), columns=steps,index=['value','wall_time'])

        # if len(tags)>0:
        #     df= pd.concat(out.values(),keys=out.keys())
        #     # df.to_csv(f'{dname}.csv')
        #     print("- Done")
        # else:
        #     print('- Not scalers to write')

        final_out[dname] = tag_values

    pickle.dump(final_out, open(f"{path}{save_name}_df_dict", "wb"))



if __name__ == '__main__':

    runs_path = "/home/lirontyomkin/AlphaZero_Gomoku/runs/"
    save_name = "12_6x6_models"
    path = f"/home/lirontyomkin/AlphaZero_Gomoku/smoothing_CI/"

    # steps = tabulate_events(runs_path, path)

    steps = pickle.load(open(f"{path}{save_name}_df_dict", "rb"))

    plot_smoothing_with_ci(steps, save_name=save_name)

    # all_result = pd.concat(steps.values(),keys=steps.keys())


