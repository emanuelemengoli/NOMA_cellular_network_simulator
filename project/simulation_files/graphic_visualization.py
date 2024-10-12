import os
import folium
import imageio
import shutil
import folium.plugins as plugins
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import inspect
import webbrowser
from geopy.distance import distance
from simulation_files.simulation_env import NET_WIDTH, NET_HEIGHT, ORIGIN, EPS_BORDER
import math 
from scipy.spatial import Voronoi, voronoi_plot_2d, ConvexHull
import io
from PIL import Image
from typing import Optional, List, Tuple
from simulation_files.utility import *
import ast
import selenium
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
import warnings
import re
warnings.filterwarnings("ignore")


@staticmethod
def open_html_file(file_path):
    """Opens the HTML file created by the gif_grid function."""
    file_url = 'file://' + os.path.abspath(file_path)
    print('Copy and paste in the web-browser:\n')
    print(file_url.replace(' ', '%20'))
    webbrowser.open(file_url)

@staticmethod
def list_visualization_functions():
    """Lists all functions available in the class that do not begin with an underscore, along with their first-line comments."""
    functions = inspect.getmembers(GraphyRenderer, predicate=inspect.isfunction)
    func_info = []
    
    for name, func in functions:
        if not name.startswith('_'):
            docstring = inspect.getdoc(func)
            comment = docstring.split("\n")[0] if docstring else "No description available."
            func_info.append({"name": name, "comment": comment})
    
    for func in func_info:
        print(f"Function: {func['name']}\nComment: {func['comment']}\n")

class GraphyRenderer:

    def __init__(self, id: str, bs_file: str, ue_file:str, bs_traces_file: str,
                 given_metric: str = 'm', metric_to_use: str = 'm', output_dir: str = 'output',
                   sample_view: int = 5, scaling_factor: int = 1,
                   origin: Tuple[float,float]= ORIGIN, dimensions: Tuple[float,float] = (NET_WIDTH, NET_HEIGHT), eps_border: float = None, sim_time:float=220):
       """
        Initializes the GraphyRenderer instance with the given parameters.

        Params:
            id (str): Identifier for the graph renderer.
            bs_file (str): Path to the base station CSV file.
            ue_file (str): Path to the user equipment CSV file.
            bs_traces_file (str): Path to the base station traces CSV file.
            given_metric (str): The initial metric to use ('m' for meters or 'km' for kilometers). Default is 'm'.
            metric_to_use (str): The target metric to convert to. Default is 'm'.
            output_dir (str): Directory where outputs will be stored. Default is 'output'.
            sample_view (int): Number of user samples to view. Default is 5.
            scaling_factor (int): Scaling factor for visualization. Default is 1.
            origin (Tuple[float, float]): The origin coordinates of the network. Default is ORIGIN.
            dimensions (Tuple[float, float]): Dimensions of the network area. Default is (NET_WIDTH, NET_HEIGHT).
            eps_border (float): Optional border value for EPS (Encapsulated PostScript) output. Default is None.
            sim_time (float): Simulation time duration. Default is 220.

        Raises:
            AssertionError: If the required columns are missing from the base station or user equipment dataframes.
        """
       self.id = id
       self.sim_time = sim_time
       self.sample_view = sample_view
       self.bs_df = pd.read_csv(bs_file)
       self.ue_df = pd.read_csv(ue_file).sort_values('time').reset_index(drop=True)
       self.ue_df['time'] = pd.to_datetime(self.ue_df['time']).apply(self._truncate_datetime)

       self.bs_traces_df = pd.read_csv(bs_traces_file)
       self.bs_traces_df['time'] = pd.to_datetime(self.bs_traces_df['time']).apply(self._truncate_datetime)
       
       if 'cluster' not in self.ue_df.columns or self.ue_df['cluster'].isnull().all():
           self.ue_df['cluster'] = 1  # Assign all users to cluster 1
 
       self.converter = self._get_converter(given_metric, metric_to_use)
       self.metric = metric_to_use
       self.output_dir = output_dir

       bs_required_columns = {'id', 'latitude', 'longitude', 'x', 'y'}
       assert bs_required_columns.issubset(self.bs_df.columns), \
        f"The BS DataFrame must contain the columns: {', '.join(bs_required_columns)}"
       
       ue_required_columns = {'time','id', 'latitude', 'longitude', 'x', 'y', 'cluster'}
       assert ue_required_columns.issubset(self.ue_df.columns), \
        f"The UEs DataFrame must contain the columns: {', '.join(ue_required_columns)}"
       
       traces_required_columns = {'time', 'id', 'r_in', 'r_out'}
       assert traces_required_columns.issubset(self.bs_traces_df.columns), \
        f"The BS traces DataFrame must contain the columns: {', '.join(traces_required_columns)}"

       
       self.visual_path = f'{self.output_dir}/visualization'
       self.processing_path = f'{self.output_dir}/processing'
       self._check_directory(self.output_dir)
       self._check_directory(self.visual_path)
       self._check_directory(self.processing_path)
       self.ue_grouped = None
       self._set_trace_magnitude()
       self.scaling_factor = scaling_factor
       self.origin = origin
       self.dimensions = dimensions
       self.eps_border = eps_border
       self.utility = Utility(d_metric=self.metric, origin=self.origin, dimensions=self.dimensions)
       #self.time_normalizer()

#---------------------------------------UTILITY FUNCTIONS---------------------------------------  
    
    def time_normalizer(self):
        """
        Normalizes the time column in both the base station and user equipment DataFrames.

        This method ensures that time stamps are uniformly spaced with a 10-second difference between them.
        """
        # Step 1: Calculate the total number of unique timestamps
        unique_bs_times = self.bs_traces_df['time'].unique()
        n_timestamps_bs = len(unique_bs_times)

        unique_ue_times = self.ue_df['time'].unique()
        n_timestamps_ue = len(unique_ue_times)

        print('Sanity check n_timestamps:', (n_timestamps_bs == n_timestamps_ue))

        # Step 2: Generate timestamps with a 10-second difference
        new_times_bs = pd.to_datetime(self.bs_traces_df['time'].min()) + pd.to_timedelta(np.arange(0, n_timestamps_bs * 10, 10), unit='s')
        new_times_ue = pd.to_datetime(self.ue_df['time'].min()) + pd.to_timedelta(np.arange(0, n_timestamps_ue * 10, 10), unit='s')

        new_times_bs = new_times_ue

        # Step 3: Create a mapping of original unique times to the new times
        time_mapping_bs = dict(zip(unique_bs_times, new_times_bs))
        time_mapping_ue = dict(zip(unique_ue_times, new_times_ue))

        # Step 4: Apply the mapping to the 'time' column
        self.bs_traces_df['time'] = self.bs_traces_df['time'].map(time_mapping_bs)
        self.ue_df['time'] = self.ue_df['time'].map(time_mapping_ue)

        # Step 5: Convert the 'time' columns to string format
        self.bs_traces_df['time'] = self.bs_traces_df['time'].astype(str)
        self.ue_df['time'] = self.ue_df['time'].astype(str)

        # Step 6: Print the 'time' columns
        print("BS Traces Time Column:")
        print(self.bs_traces_df['time'])

        print("\nUE Time Column:")
        print(self.ue_df['time'])

        print('Bs max time:', self.bs_traces_df['time'].max())
        print('UE max time:', self.ue_df['time'].max())

    
    def _check_directory(self, path):
        """
        Checks if a directory exists and creates it if it does not.

        Params:
            path (str): The path of the directory to check or create.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    # Function to truncate datetime to two decimal places
    def _truncate_datetime(self,t):
        """
        Truncates a datetime object to two decimal places (milliseconds).

        Params:
            t (pd.Timestamp): The datetime object to truncate.

        Returns:
            str: The truncated datetime string in the format '%Y-%m-%d %H:%M:%S.%f' (up to 2 decimal places).
        """
        t_str = t.strftime('%Y-%m-%d %H:%M:%S.%f')  # Convert to string with microseconds
        return t_str[:t_str.index('.')+3]  # Keep only two decimal places

    def _set_trace_magnitude(self):
        """
        Sets the trace magnitude by grouping user equipment data by 'cluster' and selecting a sample view.

        If the sample_view attribute is set, only a limited number of users (based on the sample_view) will be included.
        Otherwise, all users will be considered.
        """
        if self.sample_view is not None:
            selecter = lambda group: group[group['id'].isin(group['id'].unique()[:self.sample_view])]
            self.ue_grouped = self.ue_df.groupby('cluster').apply(selecter)
            self.ue_grouped = self.ue_grouped.reset_index(drop=True)
            self.ue_grouped = self.ue_grouped.sort_values(by = 'time', ignore_index = True).groupby('id')
        else:
            self.ue_grouped = self.ue_df.sort_values(by = 'time', ignore_index = True).groupby('id')

    def _create_colormap(self,animate_toggle: bool = False):
        """
        Creates a colormap that maps each user cluster to a specific color.

        Params:
            animate_toggle (bool): If True, creates a color map for animated output, using HEX colors.
                                   If False, uses RGBA colors for static visualization.

        Returns:
            dict: A dictionary mapping each cluster to a color (either HEX or RGBA, depending on animate_toggle).
        """
        # Get unique cluster values
        unique_clusters = self.ue_df['cluster'].unique()
        # Define a colormap
        cmap = plt.get_cmap('inferno', len(unique_clusters))
        # Create a dictionary mapping each cluster to a color
        if animate_toggle:
            cluster_colors = {cluster: mcolors.rgb2hex(cmap(i)[:3]) for i, cluster in enumerate(unique_clusters)}
        else:
            cluster_colors = {cluster: mcolors.to_rgba_array(cmap(i)) for i, cluster in enumerate(unique_clusters)}
        return cluster_colors
    
    def _grid_legend(self,ax):
        """
        Adds a grid legend to the provided matplotlib axis (ax).

        Params:
            ax: The axis object to which the legend should be added.
        """
        # Add a legend for markers
        """Adds a legend for markers to the given axis."""
        handles = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10, label='Base Station'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=6, label='User')
        ]
        ax.legend(handles=handles, loc='upper right')
    
    def _get_converter(self, current_metric, metric_to_use):
        """
        Returns a function to convert between the current metric and the desired metric.

        Params:
            current_metric (str): The current unit of measurement ('m' or 'km').
            metric_to_use (str): The target unit of measurement ('m' or 'km').

        Returns:
            function: A lambda function that converts the given value from the current metric to the target metric.
        """
        return (lambda x: x / KM_TO_M) if current_metric == 'm' and metric_to_use == 'km' else \
               (lambda x: x * KM_TO_M) if current_metric == 'km' and metric_to_use == 'm' else \
               (lambda x: x)
    
    def _anim_subroutine(self, map):
        """
        Adds a custom legend to the provided folium map for animation visualization.

        Params:
            map: The folium map object where the legend will be added.
        """
        # Legend HTML as defined before
        legend_html = '''
        <div style="position: fixed; 
                    top: 25px; right: 25px; width: 100px; height: 60px; 
                    border:2px solid grey; z-index:9999; font-size:10px;
                    background-color: white; opacity: 0.8; padding: 5px;
                    ">&nbsp; Legend <br>
                    &nbsp; <span style="font-size: 10px;"><i class="fa fa-caret-up" style="color: blue;"></i>&nbsp; Base Station <br>
                    &nbsp; <span style="height: 6px; width: 6px; background-color: red; border-radius: 50%; display: inline-block;"></span>&nbsp; User <br>
        </div>
        '''

        # Add the legend to the map
        map.get_root().html.add_child(folium.Element(legend_html))

#---------------------------------------SIMPLE VISUALIZATION---------------------------------------  

    def plot_static_grid(self):
        """Plots a static grid map of base stations and user equipment and saves it as a PNG image."""
        
        colormap = self._create_colormap(animate_toggle=False)
        file_path = f'{self.visual_path}/static_grid_sim{self.id}.png'

        fig, ax = plt.subplots()
        ax.scatter(self.bs_df['x'].apply(self.converter), self.bs_df['y'].apply(self.converter),
                     marker='^', c='orange',s=10, label='Base Station', alpha=0.7)

        print('n_bss:', len(self.bs_df['id']))
        if self.eps_border is not None:
            patch_min_dim = (self.converter(self.eps_border), self.converter(self.eps_border))
            patch_max_dim = (self.converter(self.dimensions[0] - self.eps_border), self.converter(self.dimensions[1] - self.eps_border))

            ax.add_patch(plt.Rectangle(patch_min_dim, patch_max_dim[0] - patch_min_dim[0], patch_max_dim[1] - patch_min_dim[1],
                                    fill=False, edgecolor='red', linewidth=2))

        for _, group in self.ue_grouped:
            color = colormap[group['cluster'].iloc[0]]
            ax.plot(group['x'].apply(self.converter), group['y'].apply(self.converter), marker='.', linestyle=':',c = color,alpha=0.75, label='End-user')#marker = '.'
        
        ax.set_title('Static Map of BS and UE')
        ax.set_xlabel(f'Distance {self.metric}')
        ax.set_ylabel(f'Distance {self.metric}')
        ax.set_xlim(0, self.converter(self.dimensions[0]))
        ax.set_ylim(0, self.converter(self.dimensions[1]))
        self._grid_legend(ax)

        plt.savefig(file_path)

        plt.show()


    def plot_interactive_grid(self):
        """Plots an interactive grid map of base stations and user equipment and saves it as an HTML file."""

        colormap = self._create_colormap(animate_toggle=True)
        file_path = f'{self.visual_path}/interactive_grid_sim{self.id}.html'

        m = folium.Map(location=[self.bs_df['latitude'].mean(), self.bs_df['longitude'].mean()], zoom_start=14)

        for _, row in self.bs_df.iterrows():
            folium.Marker(location=[row["latitude"], row["longitude"]],popup=row["id"],
                            icon=folium.DivIcon(html=f"""<div style="font-size: 10px; color: blue;"><i class="fa fa-caret-up"></i></div>""")
                        ).add_to(m)

        print('n_bss:', len(self.bs_df['id']))
        if self.eps_border is not None:
            lower_bnd = self.utility.get_geo_coordinate((self.eps_border, self.eps_border))
            upp_bound = self.utility.get_geo_coordinate((self.dimensions[0] - self.eps_border, self.dimensions[1] - self.eps_border))
            bounds = [
                [lower_bnd[0], lower_bnd[1]],
                [upp_bound[0],upp_bound[1]]
            ]
            folium.Rectangle(bounds=bounds, color='red', fill=False).add_to(m)

        for _, group in self.ue_grouped:

            trace = group[['latitude', 'longitude']].values.tolist()

            initial_pos = trace[0]

            final_pos = trace[-1]

            color = colormap[group['cluster'].iloc[0]]

            folium.PolyLine(trace, color=color, weight=2, opacity=1).add_to(m)

            folium.Marker(location=initial_pos,popup=group['id'],
                          icon=folium.DivIcon(html=f"""<div style="font-size: 5px; color: red;"><i class="fa fa-circle"></i></div>""")
                          ).add_to(m)
            
            folium.Marker(location=final_pos,popup=group['id'],
                          icon=folium.DivIcon(html=f"""<div style="font-size: 5px; color: red;"><i class="fa fa-circle"></i></div>""")
                          ).add_to(m)
            
        self._anim_subroutine(map = m)

        m.save(file_path)

        display(m)
    
    def gif_grid(self):
        """Creates an animated GIF of user equipment movement in a static grid."""

        gif_path = f'{self.visual_path}/gif_grid_sim{self.id}.html'
        colormap = self._create_colormap(animate_toggle=False)

        fig, ax = plt.subplots()

        num_steps = self.ue_grouped.apply(lambda g: g['time'].nunique()).max()

        def init():
            # Set up the base plot
            ax.scatter(self.bs_df['x'].apply(self.converter), self.bs_df['y'].apply(self.converter), marker='^', c='orange', s=10, label='Base Station', alpha=0.7)
            print('n_bss:', len(self.bs_df['id']))
            if self.eps_border is not None:
                patch_min_dim = (self.converter(self.eps_border), self.converter(self.eps_border))
                patch_max_dim = (self.converter(self.dimensions[0] - self.eps_border), self.converter(self.dimensions[1] - self.eps_border))
                
                rectangle = plt.Rectangle(
                    patch_min_dim, 
                    patch_max_dim[0] - patch_min_dim[0], 
                    patch_max_dim[1] - patch_min_dim[1],
                    fill=False, 
                    edgecolor='red', 
                    linewidth=2
                )
                ax.add_patch(rectangle)
            ax.set_xlabel(f'Distance {self.metric}')
            ax.set_ylabel(f'Distance {self.metric}')
            ax.set_xlim(0, self.converter(self.dimensions[0]))
            ax.set_ylim(0, self.converter(self.dimensions[1]))
            self._grid_legend(ax)
            return ax

        def get_trajectories():
            """Get the full trajectory."""
            return [group[['x', 'y']].map(self.converter).values for _, group in self.ue_grouped]

        
        def get_cluster_color():
            """Get the cluster color."""
            colors = []
            for _, group in self.ue_grouped:
                ue_colors = []
                for cluster in group['cluster']:
                    ue_colors.append(colormap[cluster])
                # Pad the list with the last color if necessary
                ue_colors += [ue_colors[-1]] * (num_steps - len(ue_colors)) #color padding
                colors.append(ue_colors)
            return colors



        def update_lines(num, walks, lines, scatters, colors):
            "Updates the gif visualization w. new a frame."
            for line, scatter, walk, color in zip(lines, scatters, walks, colors):
                scatter.set_offsets(walk[num:num+1])
                line.set_data(walk[:num+1, 0], walk[:num+1, 1])
                line.set_color(color[num])
                scatter.set_color(color[num])
            return lines + scatters

        walks = get_trajectories()
        colors = get_cluster_color()
        
        #initizialize the grid
        init()
        # Create lines initially without data
        lines = [ax.plot([], [], linestyle=':')[0] for _ in walks]
        scatters = [ax.scatter([], [], s=5, marker='.') for _ in walks]
        # Creating the Animation object
        anim = FuncAnimation(
            fig, update_lines, frames=num_steps, fargs=(walks, lines, scatters, colors), interval=500)
        
        anim.save(gif_path, writer='html', fps=5)  #'pillow' Adjust FPS as needed

        open_html_file(gif_path)

    def gif_interactive_grid(self):
        """Creates an animated GIF of user equipment movement on an interactive map."""

        gif_path = f'{self.visual_path}/gif_interactive_sim{self.id}.html'
        colormap = self._create_colormap(animate_toggle=True)
        

        def init_map():
            m = folium.Map(location=[self.bs_df['latitude'].mean(), self.bs_df['longitude'].mean()], zoom_start=14)
            for _, row in self.bs_df.iterrows():
                folium.Marker(location=[row["latitude"], row["longitude"]],popup=row["id"],
                                icon=folium.DivIcon(html=f"""<div style="font-size: 8px; color: blue;"><i class="fa fa-caret-up"></i></div>""")
                            ).add_to(m)
                

            print('n_bss:', len(self.bs_df['id']))
            if self.eps_border is not None:
                lower_bnd = self.utility.get_geo_coordinate((self.eps_border, self.eps_border))
                upp_bound = self.utility.get_geo_coordinate((self.dimensions[0] - self.eps_border, self.dimensions[1] - self.eps_border))
                bounds = [
                    [lower_bnd[0], lower_bnd[1]],
                    [upp_bound[0],upp_bound[1]]
                ]
                folium.Rectangle(bounds=bounds, color='red', fill=False).add_to(m)

            return m

        lines = []
        points = []

        # Iterate over each group
        for ue_id, group in self.ue_grouped:
            group = group.sort_values('time').reset_index(drop=True)  # Sort by time and reset index
            for i in range(len(group) - 1):  # Iterate through each group
                row = group.iloc[i]
                next_row = group.iloc[i + 1]
                line = {
                    "times": [row['time'], next_row['time']],
                    "coordinates": [[float(row['longitude']), float(row['latitude'])],
                                    [float(next_row['longitude']), float(next_row['latitude'])]],
                    "color": colormap[row['cluster']]
                }
                lines.append(line)

                point = {
                    "ue":ue_id,
                    "time":row['time'], 
                    "coordinates":[float(row['longitude']), float(row['latitude'])], 
                    "color": "red"
                    }
                points.append(point)

        features = [
            {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': point["coordinates"],
                },
                'properties': {
                    'time': point['time'],
                    'popup': point['ue'],
                    'icon': 'circle',
                    'iconstyle': {
                        'fillColor': point['color'],
                        'fillOpacity': 0.6,
                        'stroke': 'false',
                        'color': '#ff0000',
                        'radius': 3,
                        'weight': 1,
                    }
                }

            }
            for point in points
        ]

        for line in lines:
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': line["coordinates"]
                },
                'properties': {
                    'times': line['times'],
                    'style': {
                        'color': line['color'],#
                        'weight': 3,
                    }
                }
            })

        m = init_map()
        # Add TimestampedGeoJson to the map
        plugins.TimestampedGeoJson({
            'type': 'FeatureCollection',
            'features': features
        }, period='PT10S', add_last_point=False, transition_time=200, auto_play=False, loop=False,loop_button=True, time_slider_drag_update=True,max_speed=30).add_to(m) #max_speed=1, loop_button=True, time_slider_drag_update=True

        self._anim_subroutine(map = m)

        m.save(gif_path)

        open_html_file(gif_path)

        display(m)

#---------------------------------------WAYPOINTS VISUALIZATION---------------------------------------  

    def _bs_colormap(self,bs_file):
        """
        Creates a colormap mapping each unique cluster to a distinct color.

        Params:
            bs_file (DataFrame): A DataFrame containing base station data, including cluster information.

        Returns:
            dict: A dictionary mapping each cluster to a hex color code.
        """
        # Get unique cluster values
        unique_clusters = bs_file['cluster'].unique()
        # Define a colormap
        cmap = plt.get_cmap('Dark2', len(unique_clusters))
        # Create a dictionary mapping each cluster to a color
        cluster_colors = {cluster: mcolors.rgb2hex(cmap(i)[:3]) for i, cluster in enumerate(unique_clusters)}

        return cluster_colors
    
    def _preprocess_centroid(self,centroid: str):
        """
        Preprocess the centroid string to replace spaces with commas and parse it to a numeric tuple or list.

        Params:
            centroid (str): The centroid string to preprocess.

        Returns:
            tuple or list: The parsed numeric centroid.

        Raises:
            ValueError: If the parsed centroid is not a tuple or list.
        """
        if isinstance(centroid, str):
            # Preprocess the centroid string to replace spaces with commas
            # Remove any surrounding square brackets and replace multiple spaces with a single comma
            centroid = re.sub(r'\s+', ',', centroid.strip('[]'))
            # Parse the centroid string to a numeric tuple or list
            centroid = ast.literal_eval(centroid)
            if not isinstance(centroid, (tuple, list)):
                raise ValueError(f"Centroid must be a tuple or list, got {type(centroid)}")
        return centroid
    
    def way_points_static_grid(self, bs_clusters_file: str):
        """
        Plots way-points considering the BSs clusters, then saves the plot as a PNG image.

        Params:
            bs_clusters_file (str): Path to the CSV file containing base station cluster data.
        
        Raises:
            AssertionError: If the required columns are missing from the CSV file.
        """

        bs_file = pd.read_csv(bs_clusters_file)

        bs_required_columns = {'id', 'latitude', 'longitude', 'x', 'y', 'cluster', 'centroid', 'radius'}
        assert bs_required_columns.issubset(bs_file.columns), \
        f"The BS DataFrame must contain the columns: {', '.join(bs_required_columns)}"

        file_path = f'{self.processing_path}/way_points_static_grid.png'

        color_map = self._bs_colormap(bs_file)

        fig, ax = plt.subplots(figsize=(10, 10))
        
        for cluster in bs_file['cluster'].unique():
            cluster_data = bs_file[bs_file['cluster'] == cluster]
            radius = cluster_data['radius'].iloc[0]
            color = color_map[cluster]
            centroid = self._preprocess_centroid(cluster_data['centroid'].iloc[0])
            
            plt.scatter(cluster_data['x'].apply(self.converter), 
                        cluster_data['y'].apply(self.converter), 
                        marker='^', c=color, s=10, alpha=0.7)

            # Draw a circle for each cluster
            if cluster != -1 and cluster != 0:
                ax.add_patch(plt.Circle(tuple(self.converter(v) for v in centroid), radius=self.converter(radius), fill=False, edgecolor=color, linewidth=2)) #label=f'BS Cluster {cluster}'
            
        ax.set_title('BS density based clustered')

        ax.set_xlabel(f'Distance ({self.metric})')
        ax.set_ylabel(f'Distance ({self.metric})')

        ax.set_xlim(0, self.converter(self.dimensions[0]))
        ax.set_ylim(0, self.converter(self.dimensions[1]))

        plt.legend(loc='upper left')

        plt.savefig(file_path)
        plt.show()
    
    def way_points_interactive_grid(self, bs_clusters_file: str):
        """
        Plots way-points considering the BSs clusters using Folium, 
        with the ability to visualize clusters and centroids on a web-based map.

        Params:
            bs_clusters_file (str): Path to the CSV file containing base station cluster data.

        Raises:
            AssertionError: If the required columns are missing from the CSV file.
        """

        file_path = f'{self.processing_path}/way_points_interactive_grid.png'

        bs_file = pd.read_csv(bs_clusters_file)

        bs_required_columns = {'id', 'latitude', 'longitude', 'x', 'y', 'cluster', 'centroid', 'radius'}
        assert bs_required_columns.issubset(bs_file.columns), \
        f"The BS DataFrame must contain the columns: {', '.join(bs_required_columns)}"

        def _anim_subroutine(map):
            """
            Adds a legend to the provided Folium map, displaying icons and their meanings.

            Params:
                map (folium.Map): The Folium map object to which the legend will be added.
            """
            # Legend HTML as defined before
            legend_html = '''
            <div style="position: fixed; 
                        top: 25px; right: 25px; width: 100px; height: 60px; 
                        border:2px solid grey; z-index:9999; font-size:10px;
                        background-color: white; opacity: 0.8; padding: 5px;
                        ">&nbsp; Legend <br>
                        &nbsp; <span style="font-size: 10px;"><i class="fa fa-caret-up" style="color: blue;"></i>&nbsp; Base Station <br>
            </div>
            '''

            # Add the legend to the map
            map.get_root().html.add_child(folium.Element(legend_html))

        def _get_geo_radius(cluster_data):
            """
            Calculates the geographic centroid and radius of the cluster in meters.

            Params:
                cluster_data (DataFrame): Data for a specific cluster including centroid and geographic coordinates.

            Returns:
                tuple: The geographic coordinates of the centroid.
                float: The radius of the cluster in meters.
            """
            c = self._preprocess_centroid(cluster_data['centroid'].iloc[0])
            centroid = self.utility.get_geo_coordinate(c)
            radius = self.converter(cluster_data[['latitude', 'longitude']].apply(lambda row: distance(row, centroid).km, axis=1).max())*1000
            return centroid, radius

        def save_map(m):
            """
            Saves the current Folium map as a PNG image.

            Params:
                m (folium.Map): The Folium map object to be saved.
            """ 
            img_data = m._to_png(5)
            img = Image.open(io.BytesIO(img_data))
            img.save(file_path)

        def plot():
            """
            Creates and displays an interactive map using Folium, visualizing base stations, 
            clusters, centroids, and their radii.
            """
            m = folium.Map(location=[bs_file['latitude'].mean(), bs_file['longitude'].mean()], zoom_start=13)

            color_map = self._bs_colormap(bs_file)

            for cluster in bs_file['cluster'].unique():
                cluster_data = bs_file[bs_file['cluster'] == cluster]
                color = color_map[cluster]
                
                for _, row in cluster_data.iterrows():
                    folium.Marker(
                        location=[row["latitude"], row["longitude"]],
                        popup=f'Cluster {cluster}',
                        icon=folium.DivIcon(html=f"""<div style="font-size: 10px; color: {color};"><i class="fa fa-caret-up"></i></div>""")
                    ).add_to(m)
                
                if cluster != -1 and cluster != 0:
                    centroid, radius = _get_geo_radius(cluster_data)
                    folium.Circle(
                        location=centroid,
                        radius=radius,  # Convert degree difference to meters approximately
                        color=color,
                        fill=False
                    ).add_to(m)

            _anim_subroutine(map = m)
            m.save(file_path)
            #save_map(m)
            display(m)
        
        plot()

 #---------------------------------------Cell Breathing Behavior - Convex hull visualization---------------------------------------  
 
    def _generate_hull_points(self,center, radius, num_points=15):
        """
        Generate points around a center to form a convex hull.

        Params:
        - center (tuple): A tuple representing the (x, y) coordinates of the center point.
        - radius (float): The radius for the points to be generated around the center.
        - num_points (int): The number of points to generate (default is 15).

        Returns:
        - np.ndarray: An array of shape (num_points, 2) containing the generated (x, y) points.
        """
        angles = np.linspace(0, 2 * np.pi, num_points)
        points = np.array([(center[0] + np.cos(angle) * radius/self.scaling_factor, center[1] + np.sin(angle) * radius/self.scaling_factor) for angle in angles])
        return points 

    def _get_convex_hull(self, center, radius):
        """
        Calculate the convex hull of points generated around a center.

        Params:
        - center (tuple): A tuple representing the (x, y) coordinates of the center point.
        - radius (float): The radius for generating points around the center.

        Returns:
        - tuple: Two numpy arrays containing the x and y coordinates of the convex hull points.
        """
        points = self._generate_hull_points(center, radius, num_points=15)
        hull = ConvexHull(points)
        hull_points = np.append(hull.vertices, hull.vertices[0])  # Append the first point to close the hull
        return points[hull_points, 0], points[hull_points, 1]
    
    def _get_geo_convex_hull(self,center, radius, gif_flag):
            """
            Generate geographical convex hull points from center and radius.

            Params:
            - center (tuple): A tuple representing the (x, y) coordinates of the center point.
            - radius (float): The radius for generating points around the center.
            - gif_flag (bool): Flag indicating whether to flip latitude and longitude for geo coordinates.

            Returns:
            - list: A list of geographical coordinates (latitude, longitude) forming the convex hull.
            """
            floater = lambda point: [float(point[0]),float(point[1])]
            points = self._generate_hull_points(center, radius, num_points=15)
            hull = ConvexHull(points)
            hull_points = points[hull.vertices].tolist()
            hull_points.append(hull_points[0])  # Append the first point to close the hull
            geo_coordinates = [floater(self.utility.get_geo_coordinate(point)) for point in hull_points]

            if gif_flag:
                flip_lat_lon = lambda coord: (coord[1], coord[0])
                # Apply the lambda function to each coordinate pair
                flipped_geo_coordinates = [flip_lat_lon(coord) for coord in geo_coordinates]
                return flipped_geo_coordinates
            
            else:
                return geo_coordinates

    def voronoi_gif_grid(self):
        """Creates an animated GIF of user equipment movement in a static grid."""

        gif_path = f'{self.visual_path}/voronoi_gif_grid_sim{self.id}.html'
        colormap = self._create_colormap(animate_toggle=False)

        fig, ax = plt.subplots()
        num_steps = self.ue_grouped.apply(lambda g: g['time'].nunique()).max()
        print('n_bss:', len(self.bs_df['id']))

        def init():
            # Set up the base plot
            ax.scatter(self.bs_df['x'].apply(self.converter), self.bs_df['y'].apply(self.converter), marker='^', c='orange', s=10, label='Base Station', alpha=0.7)
            if self.eps_border is not None:
                patch_min_dim = (self.converter(self.eps_border), self.converter(self.eps_border))
                patch_max_dim = (self.converter(self.dimensions[0] - self.eps_border), self.converter(self.dimensions[1] - self.eps_border))
                
                rectangle = plt.Rectangle(
                    patch_min_dim, 
                    patch_max_dim[0] - patch_min_dim[0], 
                    patch_max_dim[1] - patch_min_dim[1],
                    fill=False, 
                    edgecolor='red', 
                    linewidth=2
                )
                ax.add_patch(rectangle)
            ax.set_xlabel(f'Distance {self.metric}')
            ax.set_ylabel(f'Distance {self.metric}')
            ax.set_xlim(0, self.converter(self.dimensions[0]))
            ax.set_ylim(0, self.converter(self.dimensions[1]))
            self._grid_legend(ax)
            return ax

        def get_trajectories():
            """Get the full trajectory of user equipment."""
            return [group[['x', 'y']].map(self.converter).values for _, group in self.ue_grouped]

        def get_cluster_color():
            """Get the color associated with each cluster."""
            colors = []
            for _, group in self.ue_grouped:
                ue_colors = []
                for cluster in group['cluster']:
                    ue_colors.append(colormap[cluster])
                # Pad the list with the last color if necessary
                ue_colors += [ue_colors[-1]] * (num_steps - len(ue_colors)) #color padding
                colors.append(ue_colors)
            return colors

        def update_lines(num, walks, lines, scatters, colors,hulls_in, hulls_out,bs_traces):
            """
            Update the lines and scatter points for each frame in the animation.

            Params:
            - num (int): The current frame number.
            - walks (list): A list of trajectories for user equipment.
            - lines (list): A list of line objects to be updated.
            - scatters (list): A list of scatter plot objects to be updated.
            - colors (list): A list of colors for each user equipment.
            - hulls_in (list): A list of inner hull plot objects to be updated.
            - hulls_out (list): A list of outer hull plot objects to be updated.
            - bs_traces (DataFrameGroupBy): Grouped base station traces data.

            Returns:
            - list: A list of updated line and scatter objects.
            """

            for line, scatter, walk, color in zip(lines, scatters, walks, colors):
                th = 3
                if num>=th:
                    line.set_data(walk[th:num+1, 0], walk[th:num+1, 1])
                    line.set_color(color[num])
                else:
                    scatter.set_offsets(walk[num:num+1])
                    scatter.set_color(color[num])
            
            for hull_in, hull_out, (bs_id, bs_group) in zip(hulls_in, hulls_out, bs_traces):

                bs_time_step = bs_group.iloc[num]

                bs_coordinate = self.bs_df[self.bs_df['id'] == bs_id][['x', 'y']].values[0]

                center = self.converter(bs_coordinate)

                r_in = bs_time_step['r_in']
                r_out = bs_time_step['r_out']

                # Replace 0 with 0.5
                r_in = 0.5 if r_in == 0 else r_in
                r_out = 0.5 if r_out == 0 else r_out

                r_in = self.converter(r_in)
                r_out = self.converter(r_out)

                if r_in > 0:  # Ensure the radius is positive
                    hull_in_x, hull_in_y = self._get_convex_hull(center, r_in)
                    hull_in.set_data(hull_in_x, hull_in_y)

                if r_out > 0:  # Ensure the radius is positive
                    hull_out_x, hull_out_y = self._get_convex_hull(center, r_out)
                    hull_out.set_data(hull_out_x, hull_out_y)

            return lines + scatters + hulls_in + hulls_out

        walks = get_trajectories()
        colors = get_cluster_color()

        init()
        # Create lines initially without data
        lines = [ax.plot([], [], linestyle=':')[0] for _ in walks]
        scatters = [ax.scatter([], [], s=5, marker='.') for _ in walks]

        bs_traces = self.bs_traces_df.groupby('id')
        hulls_in = [ax.plot([], [], color='turquoise')[0] for _ in bs_traces]
        hulls_out = [ax.plot([], [], color='darkcyan')[0] for _ in bs_traces]
        
        # Creating the Animation object
        anim = FuncAnimation(
            fig, update_lines, frames=num_steps, fargs=(walks, lines, scatters, colors, hulls_in, hulls_out,bs_traces), interval=500)
        
        anim.save(gif_path, writer='html', fps=5)  #'pillow' Adjust FPS as needed

        open_html_file(gif_path)

    def voronoi_static_grid(self):
        """Plots a static grid map of base stations and user equipment and saves it as a PNG image."""
        
        colormap = self._create_colormap(animate_toggle=False)
        file_path = f'{self.visual_path}/voronoi_static_grid_sim{self.id}.png'

        bs_traces = self.bs_traces_df.groupby('id')
        bs_coord = self.bs_df.groupby('id')
        
        fig, ax = plt.subplots()
        ax.scatter(self.bs_df['x'].apply(self.converter), self.bs_df['y'].apply(self.converter),
                     marker='^', c='orange',s=10, label='Base Station', alpha=0.7)

        print('n_bss:', len(self.bs_df['id']))
        if self.eps_border is not None:

            patch_min_dim = (self.converter(self.eps_border), self.converter(self.eps_border))
            patch_max_dim = (self.converter(self.dimensions[0] - self.eps_border), self.converter(self.dimensions[1] - self.eps_border))
            
            rectangle = plt.Rectangle(
                patch_min_dim, 
                patch_max_dim[0] - patch_min_dim[0], 
                patch_max_dim[1] - patch_min_dim[1],
                fill=False, 
                edgecolor='red', 
                linewidth=2
            )
            ax.add_patch(rectangle)

        # Plot convex hulls
        for id, group in bs_traces:
            center = bs_coord.get_group(id)[['x', 'y']].apply(self.converter).values[0]
            r_in = group['r_in'].iloc[-1]
            r_out = group['r_out'].iloc[-1]

            r_in = self.converter(r_in)
            r_out = self.converter(r_out)

            if r_in > 0:
                x_in, y_in = self._get_convex_hull(center, r_in)
                ax.plot(x_in, y_in, color='turquoise')

            if r_out > 0: 
                x_out, y_out = self._get_convex_hull(center, r_out)
                ax.plot(x_out, y_out, color='darkcyan')

        for _, group in self.ue_grouped:
            color = colormap[group['cluster'].iloc[0]]
            ax.plot(group['x'].apply(self.converter), group['y'].apply(self.converter), marker='.', linestyle=':',c = color,alpha=0.75, label='End-user')#marker = '.'
        
        ax.set_title('Static Map of BS and UE')

        ax.set_xlabel(f'Distance {self.metric}')
        ax.set_ylabel(f'Distance {self.metric}')

        ax.set_xlim(0, self.converter(self.dimensions[0]))
        ax.set_ylim(0, self.converter(self.dimensions[1]))

        self._grid_legend(ax)

        plt.savefig(file_path)

        plt.show()  


    def voronoi_interactive_grid(self):
        """Plots an interactive grid map of base stations and user equipment and saves it as an HTML file."""
        
        colormap = self._create_colormap(animate_toggle=True)
        file_path = f'{self.visual_path}/voronoi_interactive_grid_sim{self.id}.html'

        m = folium.Map(location=[self.bs_df['latitude'].mean(), self.bs_df['longitude'].mean()], zoom_start=14)
 
        bs_traces = self.bs_traces_df.groupby('id')
        bs_coord = self.bs_df.groupby('id')
        

        for _, row in self.bs_df.iterrows():
            folium.Marker(location=[row["latitude"], row["longitude"]],popup=row["id"],
                            icon=folium.DivIcon(html=f"""<div style="font-size: 10px; color: blue;"><i class="fa fa-caret-up"></i></div>""")
                        ).add_to(m)
        
        print('n_bss:', len(self.bs_df['id']))
        if self.eps_border is not None:
            lower_bnd = self.utility.get_geo_coordinate((self.eps_border, self.eps_border))
            upp_bound = self.utility.get_geo_coordinate((self.dimensions[0] - self.eps_border, self.dimensions[1] - self.eps_border))
            bounds = [
                [lower_bnd[0], lower_bnd[1]],
                [upp_bound[0],upp_bound[1]]
            ]
            folium.Rectangle(bounds=bounds, color='red', fill=False).add_to(m)
        
        for id, group in bs_traces:
            center = bs_coord.get_group(id)[['x', 'y']].values[0]
            r_in = group['r_in'].iloc[-1]
            r_out = group['r_out'].iloc[-1]

            r_in = self.converter(r_in)
            r_out = self.converter(r_out)

            if r_in > 0:
                hull_points = self._get_geo_convex_hull(center=center, radius=r_in, gif_flag=False)
                folium.Polygon(locations=hull_points, color='turquoise', weight=2, opacity=0.8, fill=False, fill_opacity=0.1).add_to(m)
            if r_out > 0:
                hull_points = self._get_geo_convex_hull(center=center, radius=r_out, gif_flag=False)
                folium.Polygon(locations=hull_points, color='darkcyan', weight=2, opacity=0.8, fill=False, fill_opacity=0.1).add_to(m) 

        for _, group in self.ue_grouped:

            trace = group[['latitude', 'longitude']].values.tolist()

            initial_pos = trace[0]

            final_pos = trace[-1]

            color = colormap[group['cluster'].iloc[0]]

            folium.PolyLine(trace, color=color, weight=2, opacity=1).add_to(m)

            folium.Marker(location=initial_pos,popup=group['id'],
                          icon=folium.DivIcon(html=f"""<div style="font-size: 5px; color: red;"><i class="fa fa-circle"></i></div>""")
                          ).add_to(m)
            
            folium.Marker(location=final_pos,popup=group['id'],
                          icon=folium.DivIcon(html=f"""<div style="font-size: 5px; color: red;"><i class="fa fa-circle"></i></div>""")
                          ).add_to(m)
            
        self._anim_subroutine(map = m)

        m.save(file_path)

        display(m)

    def voronoi_gif_interactive_grid(self):
        """Creates an animated GIF of user equipment movement on an interactive map."""

        gif_path = f'{self.visual_path}/voronoi_gif_interactive_sim{self.id}.html'
        colormap = self._create_colormap(animate_toggle=True)

        def init_map():
            m = folium.Map(location=[self.bs_df['latitude'].mean(), self.bs_df['longitude'].mean()], zoom_start=14)
            for _, row in self.bs_df.iterrows():
                folium.Marker(location=[row["latitude"], row["longitude"]],popup=row["id"],
                            icon=folium.DivIcon(html=f"""<div style="font-size: 8px; color: blue;"><i class="fa fa-caret-up"></i></div>""")
                            ).add_to(m)
            
            print('n_bss:', len(self.bs_df['id']))
            if self.eps_border is not None:
                lower_bnd = self.utility.get_geo_coordinate((self.eps_border, self.eps_border))
                upp_bound = self.utility.get_geo_coordinate((self.dimensions[0] - self.eps_border, self.dimensions[1] - self.eps_border))
                bounds = [
                    [lower_bnd[0], lower_bnd[1]],
                    [upp_bound[0],upp_bound[1]]
                ]
                folium.Rectangle(bounds=bounds, color='red', fill=False).add_to(m)
            return m

        bs_traces = self.bs_traces_df.sort_values('time').reset_index(drop=True)
        bs_traces = bs_traces.groupby('id')
        bs_coord = self.bs_df.groupby('id')
        lines = []
        points = []
        hull_in = []
        hull_out = []
        

        for id, group in bs_traces:
            group = group.sort_values('time').reset_index(drop=True)  # Sort by time and reset index
            center = bs_coord.get_group(id)[['x', 'y']].values[0]
            
            for i in range(len(group)):  # Iterate through each group
                row = group.iloc[i]
                r_in = row['r_in']
                r_out = row['r_out']

                hull_in_i = {
                    "bs": id,
                    "time": row['time'],
                    "coordinates": [self._get_geo_convex_hull(center=center, radius=r_in, gif_flag=True)],
                    "color": "turquoise"
                }
                hull_in.append(hull_in_i)

                hull_out_i = {
                    "bs": id,
                    "time": row['time'],
                    "coordinates": [self._get_geo_convex_hull(center=center, radius=r_out, gif_flag=True)],
                    "color": "darkcyan",
                }
                hull_out.append(hull_out_i)

        for ue_id, group in self.ue_grouped:
            for i in range(len(group) - 1):  # Iterate through each group
                row = group.iloc[i]
                next_row = group.iloc[i + 1]
                line = {
                    "times": [row['time'], next_row['time']],
                    "coordinates": [[float(row['longitude']), float(row['latitude'])],
                                    [float(next_row['longitude']), float(next_row['latitude'])]],
                    "color": colormap[row['cluster']]
                }
                lines.append(line)

                point = {
                    "ue": ue_id,
                    "time": row['time'], 
                    "coordinates": [float(row['longitude']), float(row['latitude'])], 
                    "color": "red"
                }
                points.append(point)

        features = []

        for point in points:
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': point["coordinates"],
                },
                'properties': {
                    'time': point['time'],
                    'popup': point['ue'],
                    'icon': 'circle',
                    'iconstyle': {
                        'fillColor': point['color'],
                        'fillOpacity': 0.6,
                        'stroke': 'false',
                        'color': '#ff0000',
                        'radius': 3,
                        'weight': 1,
                    }
                }
            })

        for line in lines:
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': line["coordinates"]
                },
                'properties': {
                    'times': line['times'],
                    'style': {
                        'color': line['color'],
                        'weight': 3,
                    }
                }
            })

        for hull in hull_in:
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': hull["coordinates"] # Note the extra wrapping list
                },
                'properties': {
                    'time': hull["time"],
                    'style': {
                        'color': hull["color"],
                        'weight': 2,
                    }
                }
            })

        for hull in hull_out:
            features.append({
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': hull["coordinates"]  # Note the extra wrapping list
                },
                'properties': {
                    'time': hull["time"],
                    'style': {
                        'color': hull["color"],
                        'weight': 2,
                    }
                }
            })

        m = init_map()

        # Add TimestampedGeoJson to the map
        plugins.TimestampedGeoJson({
            'type': 'FeatureCollection',
            'features': features
        }, period='PT10S', add_last_point=False, transition_time=200, auto_play=False, loop=False, loop_button=True, duration='PT5S',
        max_speed=1, time_slider_drag_update=True).add_to(m)

        self._anim_subroutine(map=m)
        m.save(gif_path)
        open_html_file(gif_path)
        display(m)
      
