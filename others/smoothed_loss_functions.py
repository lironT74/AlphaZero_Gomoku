"""download loss functions graphs data from tensorboard and plot smoothed graphs.
It takes a long, long time to produce the Bootstrap smoothing graphs."""


from multiprocessing import Pool, set_start_method
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
from scipy import stats
from datetime import datetime
from dateutil.relativedelta import relativedelta as rd
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from aux_for_models_statistics import *


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

colors_dict = {
    "pt_6_6_4_p4_v27_training": 0,
    "pt_6_6_4_p4_v28_training": 2,
    "pt_6_6_4_p4_v29_training": 0,
    "pt_6_6_4_p4_v30_training": 2,
    "pt_6_6_4_p4_v23_training": 4,
    "pt_6_6_4_p4_v24_training": 6,
    "pt_6_6_4_p4_v25_training": 4,
    "pt_6_6_4_p4_v26_training": 6,
    "pt_6_6_4_p4_v31_training": 8,
    "pt_6_6_4_p4_v32_training": 10,
    "pt_6_6_4_p4_v33_training": 8,
    "pt_6_6_4_p4_v34_training": 10
}



def cur_time():
    now = datetime.now()
    cur_time = now.strftime("%d/%m/%Y %H:%M:%S")
    return cur_time



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


def plot_smoothing_with_ci_aux_BS(model, n_step, model_loss_values, path):

    if not os.path.exists(f"{path}smoothing_data/{model}/steps_{n_step}/"):
        os.makedirs(f"{path}smoothing_data/{model}/steps_{n_step}/")

    if os.path.exists(f"{path}smoothing_data/{model}/steps_{n_step}/under_line_BS") and \
        os.path.exists(f"{path}smoothing_data/{model}/steps_{n_step}/over_line_BS")  and \
        os.path.exists(f"{path}smoothing_data/{model}/steps_{n_step}/smooth_path_BS"):

        return


    print(f"Starting model: {model}, n_steps = {n_step} ({cur_time()})")

    windows = rolling_window(np.array(model_loss_values), n_step)

    smooth_path = windows.mean(axis=1)

    under_line = []
    over_line = []

    for window in range(windows.shape[0]):
        l, h = bootstrap_mean(windows[window, :])
        under_line.append(l)
        over_line.append(h)


    pickle.dump(under_line, open(f"{path}smoothing_data/{model}/steps_{n_step}/under_line_BS", "wb"))
    pickle.dump(over_line, open(f"{path}smoothing_data/{model}/steps_{n_step}/over_line_BS", "wb"))
    pickle.dump(smooth_path, open(f"{path}smoothing_data/{model}/steps_{n_step}/smooth_path_BS", "wb"))

    print(f"Done model: {model}, n_steps = {n_step} ({cur_time()})")


def plot_smoothing_with_ci_aux_NORM(model, n_step, model_loss_values, path):

    if not os.path.exists(f"{path}smoothing_data/{model}/steps_{n_step}/"):
        os.makedirs(f"{path}smoothing_data/{model}/steps_{n_step}/")

    if os.path.exists(f"{path}smoothing_data/{model}/steps_{n_step}/under_line_NORM") and \
        os.path.exists(f"{path}smoothing_data/{model}/steps_{n_step}/over_line_NORM")  and \
        os.path.exists(f"{path}smoothing_data/{model}/steps_{n_step}/smooth_path_NORM"):

        return


    print(f"Starting model: {model}, n_steps = {n_step} ({cur_time()})")

    windows = rolling_window(np.array(model_loss_values), n_step)

    smooth_path = windows.mean(axis=1)

    under_line = []
    over_line = []

    for window in range(windows.shape[0]):
        std = np.std(windows[window, :])
        mean = np.mean(windows[window, :])

        l = mean - 1.96 * std
        h = mean + 1.96 * std

        under_line.append(l)
        over_line.append(h)


    pickle.dump(under_line, open(f"{path}smoothing_data/{model}/steps_{n_step}/under_line_NORM", "wb"))
    pickle.dump(over_line, open(f"{path}smoothing_data/{model}/steps_{n_step}/over_line_NORM", "wb"))
    pickle.dump(smooth_path, open(f"{path}smoothing_data/{model}/steps_{n_step}/smooth_path_NORM", "wb"))

    print(f"Done model: {model}, n_steps = {n_step} ({cur_time()})")


def group_by_action(label, group_by):

    if group_by == 'full':
        return label.endswith('yes')

    elif group_by == 'shutter':
        return label.find('    1    ') != -1

    else:
        raise Exception(f'group by {group_by} is not valid')


def plot_smoothing_with_ci(path, models_dfs, save_name="12_6x6_models", method = "BS", group_by='shutter'):

    n_steps = range(5, 1005, 5)

    # jobs = []
    #
    # is_all_exists = True
    #
    # for n_step in n_steps:
    #     # Compute curves of interest:
    #
    #     for index, (model, model_loss_values) in enumerate(models_dfs.items()):
    #
    #         jobs.append((model, n_step, model_loss_values, path))
    #
    #         if not (os.path.exists(f"{path}smoothing_data/{model}/steps_{n_step}/under_line_{method}") and \
    #                 os.path.exists(f"{path}smoothing_data/{model}/steps_{n_step}/over_line_{method}") and \
    #                 os.path.exists(f"{path}smoothing_data/{model}/steps_{n_step}/smooth_path_{method}")):
    #
    #             is_all_exists = False
    #
    #
    # if is_all_exists:
    #     print(f"{method}: all models data saved!")
    #
    #
    # else:
    #
    #     with Pool() as pool:
    #         print(f"There are {len(jobs)} jobs")
    #
    #         if method == "BS":
    #             pool.starmap(plot_smoothing_with_ci_aux_BS, jobs)
    #             pool.close()
    #             pool.join()
    #         elif method == "NORM":
    #             pool.starmap(plot_smoothing_with_ci_aux_NORM, jobs)
    #             pool.close()
    #             pool.join()
    #
    #
    #
    # print(f"plotting {method}")

    colors_full = plt.cm.get_cmap("spring", len(models_dfs.keys()))
    colors_non_full = plt.cm.get_cmap("winter", len(models_dfs.keys()))


    for n_step in n_steps:
        # Compute curves of interest:

        for index, (model, model_loss_values) in enumerate(models_dfs.items()):

            if model not in [f"pt_6_6_4_p4_v{i}_training" for i in [23, 24, 27, 28, 31, 32]]:
                continue

            under_line = pickle.load(open(f"{path}smoothing_data/{model}/steps_{n_step}/under_line_{method}", "rb"))
            over_line = pickle.load(open(f"{path}smoothing_data/{model}/steps_{n_step}/over_line_{method}", "rb"))
            smooth_path = pickle.load(open(f"{path}smoothing_data/{model}/steps_{n_step}/smooth_path_{method}", "rb"))

            if group_by_action(discription_dict[model], group_by):

                color = colors_full(colors_dict[model])

            else:

                color = colors_non_full(colors_dict[model])


            plt.plot(smooth_path, linewidth=0.3, color=color)

            plt.fill_between(range(len(under_line)), under_line, over_line, color=color, alpha=.2)


        handles = []

        handles.append(mpatches.Patch(label="name simulations shutter full", color='none'))


        for index, (model, label) in enumerate(discription_dict.items()):

            if model not in [f"pt_6_6_4_p4_v{i}_training" for i in [23, 24, 27, 28, 31, 32]]:
                continue


            if group_by_action(label, group_by):
                color = colors_full(colors_dict[model])
            else:
                color = colors_non_full(colors_dict[model])

            patch = mpatches.Patch(color=color, label=label)
            handles.append(patch)


        plt.ylabel('Loss function')
        plt.xlabel('Training epoch')

        plt.legend(handles=handles, prop={'family': 'DejaVu Sans Mono', 'size': 8})
        # plt.title(f"{save_name} Loss functions {n_step} steps smoothing")
        plt.title(f"Full boards trained models loss functions {n_step} steps smoothing")


        # image_path = f'{path}{save_name}/group_by_{group_by}/{method}/'
        image_path = f'{path}{save_name}/group_by_{group_by}/{method}_only_full_models/'

        if not os.path.exists(image_path):
            os.makedirs(image_path)

        plt.savefig(f"{image_path}{n_step}_steps_smoothing.png", bbox_inches='tight')


        # plt.show()
        plt.close('all')



    # for index, (model, model_loss_values) in enumerate(models_dfs.items()):
    #     if discription_dict[model].endswith('yes'):
    #         color = colors_full(colors_dict[model])
    #     else:
    #         color = colors_non_full(colors_dict[model])
    #
    #     plt.plot(model_loss_values, linewidth=0.3, color=color)
    #
    # handles = []
    #
    # handles.append(mpatches.Patch(label="name simulations shutter full", color='none'))
    #
    # for index, (model, label) in enumerate(discription_dict.items()):
    #
    #     if group_by_action(label, group_by):
    #         color = colors_full(colors_dict[model])
    #     else:
    #         color = colors_non_full(colors_dict[model])
    #
    #     patch = mpatches.Patch(color=color, label=label)
    #     handles.append(patch)
    #
    # plt.legend(handles=handles, prop={'family': 'DejaVu Sans Mono', 'size': 8})
    # plt.title(f"{save_name} Loss functions no smoothing")
    #
    # plt.ylabel('Loss function')
    # plt.xlabel('Training epoch')
    #
    # if not os.path.exists(f"{path}{save_name}/group_by_{group_by}/"):
    #     os.makedirs(f"{path}{save_name}/group_by_{group_by}/")
    #
    # plt.savefig(f"{path}{save_name}/group_by_{group_by}/{method}_NO_smoothing_groupby_{group_by}.png", bbox_inches='tight')
    #
    # # plt.show()
    # plt.close('all')


def plot_loss_only_empty(group_by='shutter', n_playout=50):
    models_dfs = {}

    for path_5000, short_name, input_plains_num, _, _, limit_shutter in all_new_12_models_6_policies:
        path = path_5000.split('/')
        model_full_name = path[5]
        short_name = short_name.split("_")[0]

        with open(f'/home/lirontyomkin/AlphaZero_Gomoku/models/{model_full_name}/loss_only_empty_playout_{n_playout}', 'rb') as f:
            models_dfs[short_name] = pickle.load(f)


    colors_full = plt.cm.get_cmap("spring", len(models_dfs.keys()))
    colors_non_full = plt.cm.get_cmap("winter", len(models_dfs.keys()))


    for index, (model, model_loss_values) in enumerate(models_dfs.items()):
        if discription_dict[model].endswith('yes'):
            color = colors_full(colors_dict[model])
        else:
            color = colors_non_full(colors_dict[model])

        plt.plot(model_loss_values, linewidth=0.3, color=color)

    handles = []

    handles.append(mpatches.Patch(label="name simulations shutter full", color='none'))

    for index, (model, label) in enumerate(discription_dict.items()):

        if group_by_action(label, group_by):
            color = colors_full(colors_dict[model])
        else:
            color = colors_non_full(colors_dict[model])

        patch = mpatches.Patch(color=color, label=label)
        handles.append(patch)

    plt.legend(handles=handles, prop={'family': 'DejaVu Sans Mono', 'size': 8})
    plt.title(f"{save_name} Loss functions only empty board, no smoothing")

    plt.ylabel('Loss function')
    plt.xlabel('Training epoch')

    if not os.path.exists(f"{path}{save_name}/group_by_{group_by}/"):
        os.makedirs(f"{path}{save_name}/group_by_{group_by}/")

    plt.savefig(f"{path}{save_name}/group_by_{group_by}/Loss only on empry board.png", bbox_inches='tight')
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

    matplotlib.use('Agg')


    # runs_path = "/home/lirontyomkin/AlphaZero_Gomoku/runs/"
    # save_name = "12_6x6_models"
    #
    # path = f"/home/lirontyomkin/AlphaZero_Gomoku/smoothing_CI/"
    #
    # # steps = tabulate_events(runs_path, path)
    #
    # steps = pickle.load(open(f"{path}{save_name}_df_dict", "rb"))
    #
    #
    # group_by='shutter'
    #
    # plot_smoothing_with_ci(path, steps, save_name=save_name, method="BS", group_by=group_by)
    # # plot_smoothing_with_ci(path, steps, save_name=save_name, method="BS", group_by=group_by)


    plot_loss_only_empty(group_by='shutter', n_playout=50)