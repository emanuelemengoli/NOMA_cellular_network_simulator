import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from simulation_files.graphic_visualization import *

#----------------------For a single run - comparison of WDBO vs RNDM pick----------------------

def plot_metrics_comparison(sim_wdbo,sim_rndm,result_id:int): 
    """
    Compare metrics between WDBO and Random Pick simulations.

    Parameters:
        sim_wdbo (SIMULATION): Simulation object for WDBO.
        sim_rndm (SIMULATION): Simulation object for Random Pick.
        result_id (int): Identifier for saving output files.
    """
    # Store the results for Random Pick
    metrics_true = sim_rndm.history_logger.handlers[0].records
    sinr_list_true = sim_rndm.controller.accumulated_sinr
    avg_n_users_true = sim_rndm.avg_users

    # Store the results for WDBO
    metrics_false = sim_wdbo.history_logger.handlers[0].records
    sinr_list_false = sim_wdbo.controller.accumulated_sinr
    avg_n_users_false = sim_wdbo.avg_users

    out_dir = f'output/results/single_comparison/comparison{result_id}'
    os.makedirs(out_dir, exist_ok=True)
    log_file_path = f'{out_dir}infos_log.txt'
    
    flatten = lambda xss: [x for xs in xss for x in xs]
    to_dict = lambda lst: [json.loads(x) for x in lst]

    metrics_false = to_dict(metrics_false)
    metrics_true = to_dict(metrics_true)
    flattened_false = flatten(sinr_list_false)
    flattened_true = flatten(sinr_list_true)

    metric_keys = {'f(x)'}
    times_false = [log['time'] for log in metrics_false]
    times_true = [log['time'] for log in metrics_true]
    min_length = min(len(times_false), len(times_true))

    if len(times_false) == min_length:
        max_time = max(times_false)
    else:
        max_time = max(times_true)

    common_timeline = np.linspace(0, max_time, min_length)

    sorted_sinr_f = np.sort(flattened_false)
    cdf_sinr_f = np.arange(1, len(sorted_sinr_f) + 1) / len(sorted_sinr_f)

    sorted_sinr_t = np.sort(flattened_true)
    cdf_sinr_t = np.arange(1, len(sorted_sinr_t) + 1) / len(sorted_sinr_t)

    plot_id = 1

    with open(log_file_path, 'w') as log_file:
        log_file.write(f"Average Number of users in the system (Random Pick): {round(avg_n_users_true)}\n")
        log_file.write(f"Average Number of users in the system (WDBO): {round(avg_n_users_false)}\n")
        log_file.write(f"Timeline: 0 to {max_time}s - Interpolation between WDBO (0 - {max(times_false)}s) and Random Pick (0 - {max(times_true)}s)\n")
        log_file.write(f"Number of SINR values collected (WDBO): {len(flattened_false)}\n")
        log_file.write(f"Number of SINR values collected (Random Pick): {len(flattened_true)}\n")


    print(f"Average Number of users in the system (Random Pick): {round(avg_n_users_true)}")
    print(f"Average Number of users in the system (WDBO): {round(avg_n_users_false)}")
    print(f"Timeline: 0 to {max_time}s - Interpolation between WDBO (0 - {max(times_false)}s) and Random Pick (0 - {max(times_true)}s)")
    print(f"Number of SINR values collected (WDBO): {len(flattened_false)}")
    print(f"Number of SINR values collected (Random Pick): {len(flattened_true)}")

    for metric in metric_keys:
        plt.figure(figsize=(15, 5))

        m_f = [log[metric] for log in metrics_false if metric in log]
        m_t = [log[metric] for log in metrics_true if metric in log]
        min_length = min(len(m_f), len(m_t))
        m_f = m_f[:min_length]
        m_t = m_t[:min_length]

        # Line plot of the metric over time
        plt.subplot(1, 2, 1)
        plt.plot(common_timeline, m_f,marker='x', label=f'{metric.capitalize()} (WDBO)')
        plt.plot(common_timeline, m_t, marker='o', label=f'{metric.capitalize()} (Random Pick)')
        plt.xlabel('Time (s)')
        plt.ylabel('Metric Values')
        plt.title(f'{metric.capitalize()} Over Time (Comparison)')
        plt.legend()

        # Box plot of the metric values
        plt.subplot(1, 2, 2)
        data = [m_f, m_t]
        plt.boxplot(data, labels=['WDBO', 'Random Pick'])
        plt.xlabel('Model')
        plt.ylabel('Metric Values')
        plt.title(f'{metric.capitalize()} Box Plot (Comparison)')

        plt.tight_layout()
        plt.savefig(f'{out_dir}/{result_id}.{plot_id}.png')
        plot_id += 1
        plt.show()
        plt.close()
        

    plt.figure(figsize=(20, 6))

    # SINR CDF
    plt.plot(sorted_sinr_f, cdf_sinr_f,  color='b', label='WDBO') 
    plt.plot(sorted_sinr_t, cdf_sinr_t,  color='r', label='Random Pick') 
    plt.title('SINR: Cumulative Density Function ')
    plt.xlabel('SINR')
    plt.ylabel('Cumulative Probability')
    plt.xscale('log')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend(loc = 8)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/{result_id}.{plot_id}.png')
    plot_id += 1
    plt.show()
    plt.close()

#----------------------------------------------------------------------------------------------------------------------
#-----------------Visualization Renderer (check graphic_visualization.py for the full list of functions)---------------
    
def render_it(motion_model:str, optimization_model:str, id:int, bs_file:str, num_ues_samples:int):
    """
    Render the simulation visualization for given motion and optimization models.

    Parameters:
        motion_model (str): Type of motion model used in the simulation.
        optimization_model (str): Type of optimization model used in the simulation.
        id (int): Unique identifier for the simulation.
        bs_file (str): Filepath to the base station data.
        num_ues_samples (int): Number of user equipment samples to visualize.
    """
    strl2 = '_' + motion_model + '_' + optimization_model + f'{id}_result'
    pattern = f'sim{id}_{optimization_model}_{motion_model}'
    ue_file = f'output/results/{pattern}/trajectories/ues_traces_{pattern}.csv'
    bs_traces_file = f'output/results/{pattern}/trajectories/bss_traces_{pattern}.csv'
    render = GraphyRenderer(
                            id = strl2, bs_file= bs_file, ue_file = ue_file,bs_traces_file = bs_traces_file,
                            given_metric = 'm', metric_to_use= 'm', output_dir = 'output',sample_view = num_ues_samples,
                            scaling_factor = 1,origin = ORIGIN, dimensions = (NET_WIDTH*1000, NET_HEIGHT*1000),
                            eps_border = None, sim_time= 220)

    render2 = GraphyRenderer(
                            id = strl2, bs_file= bs_file, ue_file = ue_file,bs_traces_file = bs_traces_file,
                            given_metric = 'm', metric_to_use= 'm', output_dir = 'output',sample_view = num_ues_samples,
                            scaling_factor = 1,origin = ORIGIN, dimensions = (NET_WIDTH*1000, NET_HEIGHT*1000),
                            eps_border = EPS_BORDER*1000, sim_time= 220)

    render2.voronoi_gif_grid()
    render2.voronoi_gif_interactive_grid()
    render.gif_grid() 
    render.gif_interactive_grid()

#----------------------------------------------------------------------------------------------------------------------
#---------------------------------Utility functions for Multiple Comparison--------------------------------------------

def retrive_data(folder_path:str, file_type: str = '*.json'): #or '*.csv'
    """
    Retrieve and combine data from multiple JSON files in a specified folder.

    Parameters:
        folder_path (str): Path to the folder containing the data files.
        file_type (str): File type to search for (default is '*.json').

    Returns:
        pandas.DataFrame: DataFrame containing mean, standard deviation, and standard error for the retrieved data.
    """

    # Get a list of all JSON files in the folder
    _files = glob.glob(os.path.join(folder_path, file_type))

    if 'wdbo' in folder_path:
        t1 = 'wdbo'
    elif 'const' in folder_path:
        t1 = 'const'
    else:
        t1 = 'rndm'

    n = len(_files)
    print(f"f(x) {t1} - Number of replicates: {n}")

    # Read and combine all JSON files into a single DataFrame
    df_list = []
    for file in _files:
        df = pd.read_json(file, lines=True)
        df_list.append(df)

    # Concatenate all DataFrames
    combined_df = pd.concat(df_list).sort_values('time').reset_index(drop=True)

    # Group by 'time' and compute mean and standard error
    grouped_df = combined_df.groupby('time')['f(x)']

    mean_df = grouped_df.mean().reset_index(name='mean')
    std_df = grouped_df.std().reset_index(name='std')

    stats_df = pd.merge(mean_df, std_df, on='time')

    n = combined_df.groupby('time').size().reset_index(name='count')['count']
    stats_df['stderr'] = stats_df['std'] / np.sqrt(n)

    return stats_df


def retrieve_stats_sinr_cdf(folder_path:str, file_type:str='*.csv'):
    """
    Retrieve and calculate the mean and standard error of SINR values from CSV files.

    Parameters:
        folder_path (str): Path to the folder containing the data files.
        file_type (str): File type to search for (default is '*.csv').

    Returns:
        tuple: 
            - numpy.ndarray: Common CDF grid for interpolation.
            - numpy.ndarray: Mean SINR across all files.
            - numpy.ndarray: Standard error of SINR across all files.
    """

    file_list = glob.glob(os.path.join(folder_path, file_type))

    if 'wdbo' in folder_path:
        t1 = 'wdbo'
    elif 'const' in folder_path:
        t1 = 'const'
    else:
        t1 = 'rndm'

    n = len(file_list)
    print(f"SINR CDF {t1} - Number of replicates: {n}")

    sinrs = []
    cdf_values_list = []

    for file in file_list:
        df = pd.read_csv(file)
        
        sinr_l = np.sort(df['sinr'].values)
        sinrs.append(sinr_l)
        
        cdf_ = np.arange(1, len(sinr_l) + 1) / len(sinr_l)
        cdf_values_list.append(cdf_)

    # Create a common CDF grid for interpolation
    common_cdf_grid = np.linspace(0, 1, 10000)

    # Interpolate SINR values to the common CDF grid
    interpolated_sinrs = []
    for i in range(len(sinrs)):
        f_interp = interp1d(cdf_values_list[i], sinrs[i], bounds_error=False, fill_value='extrapolate')
        interpolated_sinrs.append(f_interp(common_cdf_grid))

    interpolated_sinrs = np.array(interpolated_sinrs)

    # Calculate mean SINR and standard error across all files
    mean_sinr_across_files = np.mean(interpolated_sinrs, axis=0)
    stderr_sinr_across_files = np.std(interpolated_sinrs, axis=0) / np.sqrt(len(sinrs))

    return common_cdf_grid, mean_sinr_across_files, stderr_sinr_across_files

def plotter(ax, args, label, color):
    """
    Plot the average CDF of SINR values and the area between ± one standard error.

    Parameters:
        ax (matplotlib.axes.Axes): Matplotlib axes object to plot on.
        args (tuple[str, str]): Tuple containing folder path and file type.
        label (str): Label for the plot.
        color (str): Color for the plot.
    """

    _folder,file_type = args

    unique_cdf_values, avg_sinrs, std_errors = retrieve_stats_sinr_cdf(_folder,file_type)

    # Plot the average CDF
    ax.plot(avg_sinrs, unique_cdf_values, label=f'mean sinr - {label}', color=color, linestyle='dashed', alpha= 0.3)
    
    # Fill the area between ± one standard error
    ax.fill_betweenx(unique_cdf_values, avg_sinrs - std_errors, avg_sinrs + std_errors, color=color, alpha=0.2, label=f'standard error - {label}')

def plot_and_fill_between(time: np.ndarray, mean: np.ndarray, stderr: np.ndarray, label: str, color: str):
    """
    Plot the mean and standard error filled area for a given metric.

    Parameters:
        time (np.ndarray): Array of time values.
        mean (np.ndarray): Array of mean values to plot.
        stderr (np.ndarray): Array of standard error values to fill between.
        label (str): Label for the plot.
        color (str): Color for the plot.
    """
    plt.plot(time, mean, label=f'mean f(x) - {label}', color=color)
    plt.fill_between(time, mean - stderr, mean + stderr, color=color, alpha=0.1, label=f'standard error - {label}')

def load_and_aggregate_data(file_pattern:str, metric:str = None, motion_model:str = None):
    """
    Load and aggregate data from multiple JSON files matching a given pattern.

    Parameters:
        file_pattern (str): Pattern to search for JSON files.
        metric (str,optional): Given metric for which aggregation is performed.
        motion_model (str,optional): Given motion model for which aggregation is performed.

    Returns:
        tuple: 
            - aggregated_df (pandas.DataFrame): DataFrame containing the aggregated data.
            - num_replicates (int): Number of replicates aggregated.
    """
    # Load and aggregate data across multiple JSON files
    all_dfs = []
    for file in glob.glob(file_pattern):
        df = pd.read_json(file, lines=True)
        df = df.fillna(0)
        all_dfs.append(df)
    num_replicates = len(all_dfs)
    check = (metric is not None) and (motion_model is not None)
    if check:
        print(f'Aggregated results for {num_replicates} replicates of "{metric}" metric for "{motion_model}" motion model.')

    # Concatenate all dataframes and reset index
    aggregated_df = pd.concat(all_dfs, ignore_index=True)
    return aggregated_df, num_replicates

def compute_mean_std(df: pd.DataFrame, base_stations: list[str], metric: str):
    """
    Compute mean, standard deviation, and standard error for a specified metric grouped by time and base station.

    Parameters:
        df (pandas.DataFrame): DataFrame containing the data.
        base_stations (list[str]): List of base stations to include in the computation.
        metric (str): Metric to compute (e.g., 'SINR').

    Returns:
        pandas.DataFrame: DataFrame containing mean, standard deviation, and standard error.
    """
    grouped = df[df['base station'].isin(base_stations)].groupby(['time', 'base station'])[metric]
    mean_std = grouped.agg(['mean', 'std']).reset_index()
    mean_std['stderr'] = mean_std['std'] / np.sqrt(len(base_stations))
    return mean_std

#----------------------------------------------------------------------------------------------------------------------
#-----------------Multiple Comparison - WDBO, RNDM, Const pick - Based on multiple runs--------------------------------

def safety_check_sinr_cdf(folder_path:str, ix:int, t1:str, file_type='*.csv'):
    """
    Create a plot of SINR CDF for multiple replicates in a specified folder.

    Parameters:
        folder_path (str): Path to the folder containing the data files.
        ix (int): Index for saving the output file.
        t1 (str): Type of simulation (e.g., 'wdbo', 'rndm').
        file_type (str): File type to search for (default is '*.csv').
    """

    # Get the list of files in the folder
    file_list = glob.glob(os.path.join(folder_path, file_type))
        
    fp = f'output/results/multiple_comparison/{ix}'
    os.makedirs(fp, exist_ok=True)

    
    fp += f'/full_sinr_hist_{t1}_{ix}.png'

    n = len(file_list)
    print(f"SINR CDF {t1} - Number of replicates: {n}")

    cmap = plt.get_cmap('viridis', n)
    
    plt.figure(figsize=(10, 6))

    for idx, file in enumerate(file_list):
        df = pd.read_csv(file)
        
        sinr_l = np.sort(df['sinr'].values)
        
        cdf_ = np.arange(1, len(sinr_l) + 1) / len(sinr_l)
        
        plt.plot(sinr_l, cdf_, color=cmap(idx), label=f'Replicate {idx+1}')

    plt.xscale('log')
    plt.title(f'SINR CDFs ({t1})')
    plt.xlabel('SINR')
    plt.ylabel('CDF')
    plt.legend(loc='best', fontsize='small', title='Replicates')
    plt.grid(True)
    plt.savefig(fp)
    plt.show()

def compare_metrics(random_pick_folder:str, wdbo_folder:str, const_folder:str, ix:int, metric:str, alpha_fair:float, prime:bool):
    """
    Compare metrics across different simulation configurations (WDBO, Random Pick, Constant Pick).

    Parameters:
        random_pick_folder (str): Folder path for Random Pick simulation results.
        wdbo_folder (str): Folder path for WDBO simulation results.
        const_folder (str): Folder path for Constant Pick simulation results.
        ix (int): Index for saving the output file.
        metric (str): Metric to compare (e.g., 'SINR').
        alpha_fair (float): Parameter indicating fairness in the comparison.
        prime (bool): Boolean indicating whether to use prime settings for Constant Pick.
    """
    fp = f'output/results/multiple_comparison/{ix}'
    os.makedirs(fp, exist_ok=True)
               
    fp += f'/{metric}_comparison_{ix}.png'

    if metric == 'sinr':
        file_type = '*.csv'

        fig, ax = plt.subplots(figsize=(10, 6))

        if wdbo_folder is not None:
            plotter(ax = ax, args = (wdbo_folder,file_type), color ='blue', label = 'wdbo' )
        
        if random_pick_folder is not None:
            plotter(ax = ax, args = (random_pick_folder,file_type), color ='red', label = 'random pick' )

        if const_folder is not None:
            if prime:
                label = 'constant pick prime' 
            else:
                label = label = 'constant pick' 
            plotter(ax = ax, args = (const_folder,file_type), color ='green', label= label)
            
        ax.set_xscale('log')
        ax.set_xlabel('SINR')
        ax.set_ylabel('CDF')
        ax.set_title('Mean SINR CDF')
        ax.legend(fontsize='small')
        ax.grid('True')
        fig.savefig(fp) 
        plt.show()
        
    else:
        file_type = '*.json'

        plt.figure(figsize=(10, 6))

        if wdbo_folder is not None:
            wdbo_df = retrive_data(wdbo_folder,file_type)
            plot_and_fill_between(wdbo_df['time'], wdbo_df['mean'], wdbo_df['stderr'], label='wdbo', color='blue')
        if random_pick_folder is not None:   
            rndm_df = retrive_data(random_pick_folder,file_type)
            plot_and_fill_between(rndm_df['time'], rndm_df['mean'], rndm_df['stderr'], label='random pick', color='red')
        if const_folder is not None:
            const_df = retrive_data(const_folder,file_type)
            if prime:
                label = 'constant pick prime' 
            else:
                label = label = 'constant pick'
            plot_and_fill_between(const_df['time'], const_df['mean'], const_df['stderr'], label=label, color='green')

        plt.xlabel('Time (s)')
        if alpha_fair != 1:
            mt = 'bps/user'
        else:
            mt = 'value'
        plt.ylabel(mt)
        plt.title('Objective function over time')
        plt.legend(fontsize='small')
        plt.grid(True)
        plt.savefig(fp)
        plt.show()

#----------------------------------------------------------------------------------------------------------------------
#-----------------WDBO's Hyperparameters visualization---------------------------------------------------------------

def plot_single_model_metrics(df: pd.DataFrame, m_model: str, num_replicates: int):
    """
    Plot metrics for a single model across multiple replicates.

    Parameters:
        df (pd.DataFrame): DataFrame containing the optimizers data.
        m_model (str): The motion model type being analyzed.
        num_replicates (int): Number of replicates for the model.
    """

    output_dir = f'output/optimizer_parameter_comparison/single_motion_model_aggregated/{m_model}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_stations = df['base station'].unique()
    
    def plot_with_shaded_area(metric, ylabel, title, filename):

        plt.figure(figsize=(10, 6))
        mean_std = compute_mean_std(df, base_stations, metric)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(base_stations)))
        
        for i, bs in enumerate(base_stations):
            bs_data = mean_std[mean_std['base station'] == bs]
            plt.plot(bs_data['time'], bs_data['mean'], label=f'BS {bs}', color=colors[i])
            plt.fill_between(bs_data['time'], 
                            bs_data['mean'] - bs_data['stderr'], 
                            bs_data['mean'] + bs_data['stderr'], 
                            color=colors[i], alpha=0.2)
        plt.xlabel('Time')
        plt.ylabel(ylabel)
        plt.title(title)
        
        plt.legend(title=f'BS ({m_model})', fontsize='small', title_fontsize='small',alignment='left')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
        plt.show()
        plt.close()

    # Aggregated Dataset Size vs Time
    plot_with_shaded_area('dataset Size', 'Dataset Size', 
                          f'Aggregated Dataset Size vs Time (across {num_replicates} replicates)', 
                          'aggregated_dataset_size_vs_time.png')

    # Aggregated Spatial Lengthscale vs Time
    plot_with_shaded_area('spatial lengthscale', 'Spatial Lengthscale', 
                          f'Aggregated Spatial Lengthscale vs Time (across {num_replicates} replicates)', 
                          'aggregated_spatial_lengthscale_vs_time.png')

    # Aggregated Temporal Lengthscale vs Time
    plot_with_shaded_area('temporal lengthscale', 'Temporal Lengthscale', 
                          f'Aggregated Temporal Lengthscale vs Time (across {num_replicates} replicates)', 
                          'aggregated_temporal_lengthscale_vs_time.png')

    # Aggregated views of each metric across all replicates
    metrics = ['dataset Size', 'spatial lengthscale', 'temporal lengthscale']
    colors = ['blue', 'red', 'green']
    for ix, metric in enumerate(metrics):
        
        plt.figure(figsize=(10, 6))
        mean_std = compute_mean_std(df, base_stations, metric)
        
        # Aggregating across all base stations
        overall_mean = mean_std.groupby('time')['mean'].mean()
        overall_stderr = mean_std.groupby('time')['stderr'].mean()

        plt.plot(overall_mean.index, overall_mean, label=f'Overall {metric}', color=colors[ix])
        plt.fill_between(overall_mean.index, 
                         overall_mean - overall_stderr, 
                         overall_mean + overall_stderr, 
                         color=colors[ix], alpha=0.2)

        plt.xlabel('Time')
        plt.ylabel(f'{metric} (aggregated)')
        plt.title(f'Aggregated {metric} across all base stations and {num_replicates} replicates - {m_model}')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'aggregated_{metric}_across_bs.png'), bbox_inches='tight')
        plt.show()
        plt.close()

def plot_aggregated_metrics_across_motion_models(models: list[str], metrics: list[str], rex_folder_path: str):
    """
    Plot aggregated metrics across different motion models.

    Parameters:
        models (list[str]): List of motion models to compare.
        metrics (list[str]): List of metrics to plot (e.g., 'SINR').
        rex_folder_path (str): Path to the folder containing results for each model.

    """

    output_dir = 'output/optimizer_parameter_comparison/multiple_motion_models_aggregated'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model_colors = ['blue', 'red', 'green']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        for idx, m_model in enumerate(models):

            file_pattern = f'{rex_folder_path}/{m_model}/*.json'
            
            df, num_replicates = load_and_aggregate_data(file_pattern=file_pattern, metric=metric, motion_model=m_model)
            
            base_stations = df['base station'].unique()

            mean_std = compute_mean_std(df, base_stations, metric)
            overall_mean = mean_std.groupby('time')['mean'].mean()
            overall_stderr = mean_std.groupby('time')['stderr'].mean()
            

            plt.plot(overall_mean.index, overall_mean, label=f'{m_model}', color=model_colors[idx])
            plt.fill_between(overall_mean.index, 
                             overall_mean - overall_stderr, 
                             overall_mean + overall_stderr, 
                             color=model_colors[idx], alpha=0.2)
        

        plt.xlabel('Time')
        plt.ylabel(f'{metric} (aggregated)')
        plt.title(f'Aggregated {metric} across all motion models')
        plt.legend(title='Motion Model', fontsize='small', title_fontsize='medium')
        plt.tight_layout()
        

        plt.savefig(os.path.join(output_dir, f'aggregated_{metric}_across_models.png'), bbox_inches='tight')
        plt.show()
        plt.close()