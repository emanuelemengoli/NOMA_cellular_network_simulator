from simulation_files.libs.libraries import *
from simulation_files.utility import *
from simulation_files.user import *
from simulation_files.base_station import *
from simulation_files.controller import *
from simulation_files.mobility.mobility_class import *

class SIMULATION():
    """
    A simulation framework for evaluating the performance of Non-Orthogonal Multiple Access (NOMA) in a cell-free context. 
    The simulation is based on dynamic user equipment (UE) behavior and base station (BS) interaction.
    """   

    def __init__(self, hot_spots: list = None, id:int=1, alpha_fairness: float = 1,
                 verbose: bool = False,given_metric: str = DISTANCE_METRIC,metric_to_use:str = 'm',
                 lambda_: float = (1/3), mu: float = (1/10), epsilon: float = 0.3,move_interval: float = 2, 
                 optimizer_kwargs:dict = None): #rho = lambda/mu < 1
        
        """
        Initializes the simulation environment and sets parameters for the simulation. 

        Parameters:
        - hot_spots (list, optional): A list of way-point coordinates for user equipment (UE).
        - id (int): Unique identifier for the simulation instance.
        - alpha_fairness (float): Fairness parameter for the objective function in the simulation.
        - verbose (bool): If True, enables verbose output during the simulation.
        - given_metric (str): The metric system for distance (e.g., 'm' for meters, 'km' for kilometers).
        - metric_to_use (str): The metric system to be used for conversions (e.g., 'm', 'km').
        - lambda_ (float): Rate of UE arrivals in the simulation.
        - mu (float): Rate of UE departures in the simulation.
        - epsilon (float): Probability of pausing in UE cluster movement.
        - move_interval (float): Time interval in seconds for each movement update for UEs.
        - optimizer_kwargs (dict): Additional keyword arguments for the optimizer.
        """

        self.id = id
        self.verbose = verbose
        if self.verbose:
            print('----------------------------------------------\n')
            print(f'Simulation ID: {self.id}','\n')
        self.queue = []
        self.lambda_ = lambda_  # User birth rate
        self.mu = mu  # User lifetime rate parameter
        self.move_interval = move_interval  # Time interval for user movement
        self.epsilon = epsilon  # Probability of pausing in movement

        self.origin = None
        self.dimensions = None
        self.converter = None
        self.num_tiles_x = None
        self.num_tiles_y = None
        self.n_tiles = None
        self.tile_size = None
        self.optim_round = 1
        self.avg_users = 0
        self.n_death = 0
        self.tag = None
        self.ue_gen = None

        self.user_id_lister = []        
        self.clusters = None
        self.ues_traces = []
        self.bss_traces = []
        self.current_time = 0
        self.ue_distribution = None #tile granularity
        self.bs_distribution = None #tile granularity
        self.sim_time = None
        self.max_v = None 
        self.min_v = None
        self.bs_max_range = None
        self.t=[]

        self.given_metric = given_metric.lower()
        self.metric_to_use = metric_to_use.lower()

        self.converter = self._get_converter(given_metric, metric_to_use)

        self._set_grid_params()

        self.hot_spots = hot_spots

        self.alpha_fair = alpha_fairness
        self.optimizer_kwargs = optimizer_kwargs

        self.out_dir = f'output/results'

        self.time_converter = lambda x: x #modify this to change the unit time (currently seconds)

        self.rndm_generator = np.random.default_rng(seed=30)

    
    def _initialize_run(self):
        """
        Initializes the run environment by setting up the controller and utility instances, and logging the simulation history.
        """

        self.controller = Controller(alpha_fairness = self.alpha_fair, verbose=self.verbose,
                                    bs_max_range = self.bs_max_range, optimizer_kwargs = self.optimizer_kwargs, id=self.id)

        self.utility = Utility(verbose = self.verbose, n_tiles = self.n_tiles,\
                               num_tiles_x = self.num_tiles_x, num_tiles_y = self.num_tiles_y,
                               origin = self.origin, dimensions = self.dimensions, tile_size = self.tile_size, d_metric = self.metric_to_use)
        
        self.history_logger = self.utility._utlity_init_logger(name = f'hs{self.id}')

        self.utility._check_directory(self.out_dir)


    def _get_converter(self, current_metric, metric_to_use):
        """
        Returns a conversion function based on the current and target metrics.
        
        Parameters:
        - current_metric (str): The current metric system (e.g., 'm', 'km').
        - metric_to_use (str): The target metric system to convert to (e.g., 'm', 'km').

        Returns:
        - function: A lambda function for metric conversion.
        """
        rounder = lambda x: round(x,3)
        return (lambda x: rounder(x / KM_TO_M)) if current_metric == 'm' and metric_to_use == 'km' else \
               (lambda x: rounder(x * KM_TO_M)) if current_metric == 'km' and metric_to_use == 'm' else \
               (lambda x: rounder(x))

    def _set_grid_params(self):
        """
        Sets grid parameters for the simulation based on the given and target metrics.
        """
        self.origin = ORIGIN
        self.dimensions = self.converter(NET_WIDTH), self.converter(NET_HEIGHT) #x,y
        self.tile_size = self.converter(TILE_SIZE)
        self.num_tiles_x = int(self.dimensions[0] // self.tile_size)
        self.num_tiles_y = int(self.dimensions[1] // self.tile_size)
        self.n_tiles = self.num_tiles_x * self.num_tiles_y
        self.bs_max_range = self.converter(BS_MAX_RANGE)
        self.border = self.converter(EPS_BORDER)
    
    def generate_bs(self, bs_filepath: str = None, n_bss: int = None, dbscan_args: dict =None, get_waypoints:bool=False):
        """
        Loads base station (BS) data from a CSV file and initializes BS objects within the controller.

        Parameters:
        - bs_filepath (str, optional): File path to a CSV containing BS data.
        - n_bss (int, optional): Number of BSs to generate if no file is provided.
        - dbscan_args (dict, optional): Additional arguments for DBSCAN clustering.
        - get_waypoints (bool): If True, retrieves waypoints for user equipment (UE) displacement.

        Raises:
        - ValueError: If 'n_bss' is not provided when 'bs_filepath' is not specified.
        """
        if bs_filepath is not None:
            bs_df = pd.read_csv(bs_filepath)
            bs_df[['x', 'y']] = bs_df[['x', 'y']].map(self.converter)

        else:
            if n_bss is None:
                raise ValueError("'n_bss' must be provided if 'bs_filepath' is not specified.")
            
            make_array = lambda lst: np.array([np.array([*i]) for i in lst]) # Given a list of tuples

            def generate_positions(n_bss, min_distance):
                """
                Generates random positions for base stations, ensuring a minimum distance between them.

                Parameters:
                - n_bss (int): Number of base stations to generate.
                - min_distance (float): Minimum distance to maintain between base stations.

                Returns:
                - np.ndarray: An array of generated positions.
                """
                positions = []
                while len(positions) < n_bss:
                    new_position = self.utility.sample_position(tile_loc=None, eps_border=self.border)
                    if all(l2_norm(new_position, pos) >= min_distance for pos in positions):
                        positions.append(new_position)
                return make_array(positions)

            positions = generate_positions(n_bss = n_bss, min_distance = 200) #suitable distance should be computed w.r.t. bs nominal range

            ids = list(range(len(positions)))
            geo_positions = make_array([self.utility.get_geo_coordinate(position = p) for p in positions])

            bs_df = pd.DataFrame(data={'id': ids,
                                       'latitude' :geo_positions[:,0],
                                       'longitude':geo_positions[:,1],
                                       'x':positions[:,0],
                                       'y':positions[:,1]})
            
            out = 'simulation_datasets/base_stations_generated.csv'
            bs_df.to_csv(out, index=False)
        
        if self.metric_to_use == 'm':
            r_def = 5
        else:
            r_def = 0.005

        self.controller.BSs = [BS(bs_id=row.id, position=(row.x, row.y), controller = self.controller, default_r = r_def) for _, row in bs_df.iterrows()]
        self.controller.bs_max_range = self.bs_max_range
        self.controller.set_sectors()
        self.controller.set_border_filter(dimensions = self.dimensions, eps_border = self.border)
        if self.verbose:
            print('num_BSs: ', len(self.controller.BSs),'\n')
            print('BS areal density:', round(len(self.controller.BSs)/(self.dimensions[0]*self.dimensions[1]/10**6),3),'\n')
        
        if (get_waypoints) and (self.hot_spots is None):
            if dbscan_args is None:
                #tune this depending on n. of waypoints desired and number of BS in the network
                dbscan_args = {'eps': self.converter(0.3), 'min_samples': 1, 'metric': 'euclidean'} 
            self.hot_spots = self.utility.get_way_points(bs_df = bs_df, dbscan_args = dbscan_args)
    
    def generate_ues(self, total_population: int = None, ue_distribution_params: dict=None):
        """
        Generates user equipment (UE) based on the specified total population or minimum user threshold.
        This function can create UEs either directly using a defined count or through a statistical distribution.

        Parameters:
        - total_population (int, optional): Total number of UEs to generate.
        - ue_distribution_params (dict, optional): Parameters for the distribution to use for generating UEs.
            - Expected keys include:
                - 'distr_type' (str): The type of distribution ('poisson' or others).
                - 'lambda_' (float): The rate parameter for the Poisson distribution if used.
        """ 

        def uniform_sampling(ue_count:int, tile_loc:int=None):
            """
            Uniformly samples positions for UEs and initializes them.

            Parameters:
            - ue_count (int): Number of UEs to generate and initialize.
            - tile_loc (int, optional): Specific tile location to sample positions for UEs. If None, samples from the entire area.
            """

            for _ in range(ue_count):

                ue_id = len(self.user_id_lister) + 1  

                pos = self.utility.sample_position(eps_border = 0, tile_loc=None) if tile_loc is None else self.utility.sample_position(tile_loc=tile_loc, eps_border = 0)

                new_ue = UE(ue_id = ue_id, position = pos,utility = self.utility, max_velocity = self.max_v, min_velocity = self.min_v)

                if not self.controller.add_ue(ue = new_ue):
                    new_ue.set_inactive() 
                self.user_id_lister.append(ue_id)
        
        if total_population is not None:
            uniform_sampling(ue_count=total_population)
        else:
            if ue_distribution_params is None:
                ue_distribution_params = {
                    'distr_type' : 'poisson',
                    'lambda_': 0.5*1e-2,
                }

            dict_distributions = self.utility.get_distributions_list() #returns dict

            distr_type = ue_distribution_params.pop('distr_type', 'poisson').lower()

            if distr_type not in dict_distributions:
                raise ValueError(f"Invalid distribution type '{distr_type}'. Valid types are: {list(dict_distributions.keys())}")
            
            if distr_type == 'poisson' and 'lambda_' not in ue_distribution_params:
                ue_distribution_params['lambda_'] = 3

            if self.verbose:
                print(f'Generating UEs following a {distr_type} distribution.')
        
            distr = dict_distributions[distr_type](**ue_distribution_params) 
            total_population = np.sum(distr)
            print('Total Potential Population:', total_population)

            for tile_loc, ue_count in enumerate(distr):
                    uniform_sampling(ue_count=ue_count, tile_loc=tile_loc)

        if self.verbose:  
            print('num_active_users: ', len(self.controller.UEs), '\n')  
 
    def _group_generator(self,grouped):
        """
        A generator function that yields groups of data, resetting their index for each group.

        Parameters:
        - grouped (iterable): An iterable of groups to yield.

        Yields:
        - DataFrame: Each group of data with reset index.
        """
        for _, group in grouped:
            yield group.reset_index(drop=True)
    
    def iter_ue_trace(self, randomPick:bool, const_pick:bool):
        """
        Iterates over user equipment (UE) traces, updating positions and handling handovers and deaths.

        Parameters:
        - randomPick (bool): If True, uses random power levels picking strategy.
        - const_pick (bool): If True, uses constant power levels picking strategy.
        """

        next_batch = next(self.ue_gen)
        ues_alive = next_batch['id'].unique()

        if not next_batch.empty:
            ue_trace = self._group_generator(next_batch.groupby(by='id'))
            id_list = [ue.get_id() for ue in self.controller.UEs]
            for ud_df in ue_trace:
                ue_id_ = ud_df.iloc[-1].id
                pos = ud_df.iloc[-1].position
                if ue_id_ in id_list:
                    ue_ = self.controller.ue_lookup(ue_id = ue_id_)
                    ue_.position = pos
                else:
                    new_ue = UE(ue_id=ue_id_, position=pos,
                     utility=self.utility, max_velocity=self.max_v, min_velocity=self.min_v)
                
                    if not self.controller.add_ue(ue=new_ue):
                        new_ue.set_inactive()
                    
                    self.user_id_lister.append(ue_id_)

            handover_events = [(ue, self.controller.bs_handover(ue=ue)) for ue in self.controller.UEs]
                    
            for tuple_ in handover_events:
                ue, death = tuple_
                if not death:
                    self.ue_death(ue)
            
            #Kill user born with a B&D process
            id_list_ = [ue.get_id() for ue in self.controller.UEs]
            for ue_id_ in id_list_:
                if ue_id_ not in ues_alive:
                    ue_p = self.controller.ue_lookup(ue_id = ue_id_)
                    self.ue_death(ue_p)
            
            bs_l = [ue.bs_id for ue in self.controller.UEs]
            tmp = list(dict.fromkeys(bs_l))
            bs_l = [self.controller.bs_lookup(bs_id = bs_id) for bs_id in tmp]

            for bs in bs_l:
                bs.intra_cell_assignment()

            self.controller.build_ues_db(ues_traces=self.ues_traces, bss_traces=self.bss_traces)
            self.metrics_collection(randomPick=randomPick, const_pick=const_pick)         
        
                
    def set_ue_trace(self, ue_file_path:str):
        """
        This function reads the UE trace data from a CSV file, processes it, and initializes a generator for iterating through the traces.

        Parameters:
        - ue_file_path (str): Path to the CSV file containing UE trace data.
        """
        ue_df = pd.read_csv(ue_file_path)
        ue_df = ue_df.sort_values(by='time', ignore_index=True) # Sort data by time

        unique_ue_times = ue_df['time'].unique()
        time_mapping = {time: i for i, time in enumerate(unique_ue_times)} # Map times to unique indices
        ue_df['time'] = ue_df['time'].map(time_mapping) # Replace original times with mapped indices
        
        grouped = ue_df.groupby('time')

        # Process position columns for UEs
        if 'x' in ue_df.columns and 'y' in ue_df.columns:
            ue_df['position'] = ue_df.apply(
                lambda row:tuple(row[['x', 'y']]), axis=1
            ) # Create position tuples if x and y are available
        else:
            ue_df['position'] = ue_df.apply(
                    lambda row: self.utility.get_grid_position(geo_coordinates=tuple(row[['latitude', 'longitude']])), axis=1
                ) # Generate grid positions from geo coordinates
        ue_df = ue_df.fillna(0)

        # Create the generator
        self.ue_gen = self._group_generator(grouped)
      

    def ue_arrival(self, motion_params:dict, tile_loc:int=None, n_users:int=1): 
        """
        Simulates the arrival of new  (UEs) over time and intializes associated instances.

        Parameters:
        - motion_params (dict): Parameters for UE movement.
        - tile_loc (int, optional): Specific tile location for UEs. If None, UEs will be placed randomly within the grid.
        - n_users (int): Number of users to generate.
        """

        for _ in range(n_users):

            ue_id = len(self.user_id_lister) + 1
            
            pos = self.utility.sample_position(eps_border = 0, tile_loc=None) if tile_loc is None else self.utility.sample_position(tile_loc=tile_loc, eps_border = 0)

            lifetime = self.time_converter(self.rndm_generator.exponential(self.mu))

            lifetime += self.current_time 

            new_ue = UE(ue_id = ue_id, position = pos,utility = self.utility, max_velocity = self.max_v, min_velocity = self.min_v, death_time = lifetime)

            if not self.controller.add_ue(ue = new_ue):
                    new_ue.set_inactive()   
                    
            self.user_id_lister.append(ue_id)

            if new_ue.active:
                # If the UE is active, assign it to the nearest cluster if clusters is not None
                if self.clusters is not None:
                    try:
                        c_ix = np.argmin(np.array([l2_norm(new_ue.position,c.destination) for c in self.clusters]))
                        self.clusters[c_ix].add_ue(ue = new_ue)
                        new_ue.cluster = self.clusters[c_ix].get_id()
                    except TypeError as e:
                        print(f"TypeError occurred: {e}")
                        print(f"new_ue.position {new_ue.position}, type: {type(new_ue.position)}")
                        print(f"c.destination {self.clusters[c_ix].destination}, type: {type(self.clusters[c_ix].destination)}")
                        raise  # Re-raise the exception after printing the types
                else:
                    # Initialize mobility for the UE if no clusters are present
                    new_ue._init_mobility(params = motion_params,dimensions = tuple(self.dimensions), border_margin = 0)#self.converter(EPS_BORDER)) 
    
    def monitor_users(self):
        """
        Schedules the next user arrival based on a random interval and updates the queue of user arrivals.
        """
        # Schedule the next user arrival
        if len(self.queue) > 0:
            queue_elements = sum(tple[0] for tple in self.queue)
        else:
            queue_elements = 0
        proj_users = len(self.controller.UEs)+queue_elements

        # Generate a random delta for the number of new users
        delta = self.rndm_generator.integers(low = 1, high = 100)
        # Calculate the inter-arrival time for new users
        inter_arrival_time = self.time_converter(self.rndm_generator.exponential(self.lambda_))
        #Sanity check
        #print(delta, 'new users in', round(inter_arrival_time, 2), 's')

        inter_arrival_time += self.current_time 
        to_gen = (delta, inter_arrival_time)
        self.queue.append(to_gen)
                
    def ue_translation(self, randomPick:bool=False,const_pick:bool=False):
        """
        Triggers the displacement of UEs based on their motion model (entity-based or cluster-based).
        Checks for BS handover based on UE displacement exceeding the maximum coverage radius of the current base station.

        Parameters:
        - randomPick (bool, optional): If True, uses random power levels picking strategy.
        - const_pick (bool, optional): If True, uses constant power levels picking strategy.
        """

        if len(self.controller.UEs) != 0:

            if self.verbose:
                print(f'== Event translation, time: {round(self.current_time, 2)}s ==')
                print(f'== Active users:', len(self.controller.UEs), '==','\n')

            # Move UEs based on clusters or individually
            if self.clusters is not None:
                for c in self.clusters:
                    if self.rndm_generator.random() >= self.epsilon:
                        c.move_destination()
            else:
                for ue in self.controller.UEs:
                    ue.move()

            # Handle handover events
            handover_events = [(ue, self.controller.bs_handover(ue=ue)) for ue in self.controller.UEs]
            for tuple_ in handover_events:
                ue, death = tuple_
                if not death:
                    self.ue_death(ue)

            # Collect base station IDs and process intra-cell assignments
            bs_l = [ue.bs_id for ue in self.controller.UEs]
            tmp = list(dict.fromkeys(bs_l))
            bs_l = [self.controller.bs_lookup(bs_id = bs_id) for bs_id in tmp]

            for bs in bs_l:
                bs.intra_cell_assignment()

            # Build the UEs database and collect metrics
            self.controller.build_ues_db(ues_traces=self.ues_traces, bss_traces=self.bss_traces)
            self.metrics_collection(randomPick=randomPick, const_pick=const_pick)

    def metrics_collection(self, randomPick:bool=False, const_pick:bool=False):
        """
        Periodically collects Signal-to-Interference-plus-Noise Ratio (SINR) metrics across all BS-UE pairs in the simulation.

        Parameters:
        - randomPick (bool, optional): If True, uses random power levels picking strategy.
        - const_pick (bool, optional): If True, uses constant power levels picking strategy.
        """
        if len(self.controller.UEs) != 0:

            if self.verbose:
                print(f'== Round: {self.optim_round}, Optimizing power, time: {round(self.current_time, 2)}s ==')
                print(f'== Active users:', len(self.controller.UEs),'==')
                print('\n')

            ct = self.current_time/self.sim_time    
            self.controller.optimize_power(ctime = ct, randomPick=randomPick, const_pick=const_pick)
            
            sinr_metrics = self.controller.gather_metrics()
            self.t.append(round(self.current_time, 2))

            self.history_logger.info({
                'time': round(self.current_time, 2),
                **sinr_metrics  
            })
            #Sanity check
            # print('time', len(self.t))
            # print('accumulated_sinr:', len(self.controller.accumulated_sinr))
            
            for bs in self.controller.BSs:
                bs_metrics = {
                    'time': round(self.current_time, 2),
                    'bs_id': bs.get_id(),
                    'power levels (in, out)': (bs.p_tx_in, bs.p_tx_out),
                    'Served UEs': [
                        {
                            'ue_id': ue.get_id(),
                            'inner_region': ue.inner_region,
                            'd (m)': round(bs.get_distance(ue), 3),
                            'sinr': self.controller.compute_sinr(
                                bs=bs,
                                ue=ue,
                                interference_set=None,
                                pairing=False,
                                inner_test=False,
                                outer_test=False
                            )
                        } for ue in bs.served_UEs
                    ]
                }

                # Convert bs_metrics to Python serializable types
                bs_metrics = json.loads(json.dumps(bs_metrics, cls=NumpyEncoder))

                # Log the BS metrics
                self.controller.logger.info(bs_metrics)


            if self.verbose:
                print(
                    '----------------------------------------------','\n',
                    f'Round: {self.optim_round},',
                    'time:', round(self.current_time, 2), sinr_metrics,'\n', 
                    '----------------------------------------------','\n', 
                )

    def ue_death(self, ue: UE):
        """
        Simulates the departure of a UE after its lifetime expires.

        Parameters:
        - ue (UE): UE object representing the UE that is departing.
        """
        if self.clusters is not None:
            cluster = self.controller._lookup(self.clusters, ue.cluster)
            cluster.remove_ue(ue_id=ue.ue_id)
        self.controller.kill_ue(ue)
        self.n_death +=1

    def cluster_UEs(self, n_clusters:int):
        """
        Clusters UEs based on their spatial positions using K-means clustering.

        Parameters:
        - n_clusters (int): Number of clusters to create for grouping UEs.
        """
        # Extract user positions from UE objects
        users = self.controller.UEs
        user_positions = np.array([ue.position for ue in users])

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=56).fit(user_positions)

        self.clusters = [Cluster(id = i, destination=tuple(centroid), dimensions= tuple(self.dimensions), rndm_generator=self.utility.motion_generator)
                          for i,centroid in enumerate(kmeans.cluster_centers_)]

        # Assign cluster labels to UEs and add them to their corresponding Cluster
        for ue, label in zip(users, kmeans.labels_):
            ue.cluster = label
            self.clusters[label].add_ue(ue)

        for i,cluster in enumerate(self.clusters):
            cluster.set_destinations(hot_points = self.hot_spots)
            cluster._init_motion(border_margin = 0, position= kmeans.cluster_centers_[i]) #self.converter(EPS_BORDER)
    
    def write_log(self, out_dir,file_str, logger):
        """
        Writes log data to a specified directory and file.

        Parameters:
        - out_dir (str): Directory path where the log file will be saved.
        - file_str (str): File name for the log file (should include the desired file extension).
        - logger (logging.Logger): Logger instance containing log records to be written.
        """

        self.utility._check_directory(out_dir)

        tmp = f'{out_dir}/{file_str}'

        with open(tmp, 'w') as log_file:
            for record in logger.handlers[0].records:
                log_file.write(record + '\n')

            
    def check_user_density(self, density_thr):
        """
        Checks if the density of UEs served by BSs meets a specified threshold.

        Parameters:
        - density_thr (float): Threshold value for the acceptable user density ratio.

        Returns:
        bool: True if the measured user density is greater than or equal to the threshold, False otherwise.
        """

        bss_ = [b for b in self.controller.BSs if self.controller.filter_border_bss(bs=b)]
        
        measured_ues = sum([len(bs.served_UEs) for bs in bss_])
        tot_ues = len(self.controller.UEs)

        measured_ues_density = measured_ues/tot_ues

        #Sanity check
        if self.verbose:
            print('UEs - BS density',measured_ues_density, '(w.r.t BSs inside border perimeter.)')

        check = measured_ues_density >= density_thr

        return  check
    
    def run(self,sim_time: int = None, n_users: int = None, n_clusters: int = None,n_bss: int = None,
            ue_distribution_params: dict = None,bs_filepath: str = None,
            ue_max_velocity:float = 0.011,ue_min_velocity: float= 1.7*10**(-3),
            motion_params:dict = {'mobility_model': 'biased_random_walk'},
            randomPick:bool= False, dbscan_args:dict =None, min_rounds:int = 110,
            ue_file_path:str =None,const_pick:bool= False,density_thr:float=0.6): 


        """
        Runs the entire simulation setup, managing UE births, movements, and SINR collection.

        Parameters:
        - sim_time (int, optional): Total simulation time in seconds.
        - n_users (int, optional): Initial number of UEs.
        - n_clusters (int, optional): Number of clusters to create for UEs.
        - n_bss (int, optional): Number of BSs to generate.
        - ue_distribution_params (dict, optional): Parameters for UE distribution.
        - bs_filepath (str, optional): File path to a CSV containing BS data.
        - ue_max_velocity (float, optional): Maximum UE velocity in units of distance per time.
        - ue_min_velocity (float, optional): Minimum UE velocity in units of distance per time.
        - motion_params (dict, optional): Parameters for UE movement, including the mobility model.
        - randomPick (bool, optional):  If True, uses random power levels picking strategy.
        - dbscan_args (dict, optional): Additional arguments for BSs DBSCAN clustering.
        - min_rounds (int, optional): Minimum number of rounds for the simulation if `sim_time` is not provided.
        - ue_file_path (str, optional): Path to a CSV file containing UE traces.
        - const_pick (bool,optional): If True, uses constant power levels picking strategy.
        - density_thr (float,optional): Threshold for checking user density relative to BSs.
        """

        # Initialize the simulation run
        self._initialize_run()
        m = motion_params['mobility_model']
            
        if randomPick:
            self.tag = f'_rndm_{m}'
        elif const_pick:
            self.tag = f'_const_{m}'
        else:
            self.tag = f'_wdbo_{m}'
         
        self.out_dir += f'/sim{self.id}{self.tag}'

        out_dir = f'{self.out_dir}/trajectories'

        self.utility._check_directory(out_dir)
        self.sim_time = sim_time

        # File paths for storing UE and BS traces
        fp_ue = f'{out_dir}/ues_traces_sim{self.id}{self.tag}.csv'
        fp_bs = f'{out_dir}/bss_traces_sim{self.id}{self.tag}.csv'

        self.max_v = self.converter(ue_max_velocity)
        self.min_v = self.converter(ue_min_velocity) 

        # Determine simulation time
        if self.sim_time is None:
            if min_rounds is not None:
                self.sim_time = min_rounds * self.move_interval
            else:
                raise ValueError("Either 'sim_time' or 'min_rounds' must be provided.")
        

        def initialize():
            """
            Initializes the simulation by setting metrics, generating BSs, and populating UEs.
            """
            if self.verbose:
                metric_domain = ['m','km']
                if self.given_metric in metric_domain and self.metric_to_use in metric_domain:
                    p_ = '(w.r.t simulation_env.py)'
                    if self.given_metric == self.metric_to_use:
                        print(f'Reference distance metric:', self.metric_to_use,p_, '\n')
                    else:
                        print(f'Reference distance metric changed from {self.given_metric} to: {self.metric_to_use}',p_,'\n')     
                else:
                    raise ValueError('Invalid reference distance metric', 'Possible values: km, m')
                
                pw_m = POWER_METRIC
                if 'dBm' in POWER_METRIC:
                    pw_m = 'mW'
                elif 'dB' in POWER_METRIC:
                    pw_m = 'W'
                
                translating_factor = (3.6/self.move_interval) # Factor to convert to km/h
                print('Motion model:', m, '\n')
                print(f'Power metric changed from {POWER_METRIC} to:', pw_m, '(w.r.t simulation_env.py)','\n')
                print('Network area: ', round(self.dimensions[0]*self.dimensions[1]/10**6,3), f'km^2', '\n') 
                print('n_tiles: ', self.n_tiles, f'({self.tile_size} {self.metric_to_use})', '\n')
                print('Displacement interval: ', round(self.move_interval,2), 's', '\n')
                print(f'Translation domain: [{round(self.min_v,2)},{round(self.max_v,2)}] {self.metric_to_use}', '\n')
                print(f'User velocity domain: [{round(self.min_v * translating_factor, 2)},{round(self.max_v  * translating_factor, 2)}] km/h', '\n')
                print('BS max range',self.bs_max_range,f'{self.metric_to_use}', '\n')

            # Determine if clustering of UEs is required
            get_waypoints = n_clusters is not None
            #Populate the network with base stations
            self.generate_bs(bs_filepath = bs_filepath, n_bss = n_bss, dbscan_args=dbscan_args, get_waypoints=get_waypoints)
            for bs in self.controller.BSs:
                    bs.power_adjustment(const_pick=True, random_ = False, power_levels = None) 

            if self.verbose:
                min_power = min(bs.min_power for bs in self.controller.BSs)
                max_power = max(bs.max_power for bs in self.controller.BSs)
                print(f'Power Domain [{round(min_power,3)}, {round(max_power,3)}] ({pw_m})', '\n')

            #init optimization
            self.controller.init_optimization()

            # Intial static user population of the network
            if ue_file_path is None:
                self.generate_ues(total_population=n_users, ue_distribution_params=ue_distribution_params)
                # Cluster UEs if required
                if n_clusters is not None:
                    if self.verbose:
                        print('Clustering UEs...', '\n')
                    self.cluster_UEs(n_clusters = n_clusters)
                else:
                    for ue in self.controller.UEs:
                        ue._init_mobility(params = motion_params, dimensions = tuple(self.dimensions), border_margin = 0) #alternatively self.converter(EPS_BORDER))
            else:
                print('UEs reading from a CSV file', '\n')
                self.set_ue_trace(ue_file_path = ue_file_path)             
        
        initialize()

        if self.verbose:
            print(f'Simulation started at: {round(self.current_time, 2)}s', '\n')
            print('End time:', self.sim_time,'s', '\n')
            print('----------------------------------------------\n')
            
        user_cnt = []

        move_int = self.time_converter(self.move_interval) + self.current_time

        early_exit = False # Flag for early termination

        while self.current_time <= self.sim_time:

            if ue_file_path is None:

                counter = 0
                for ue in self.controller.UEs:
                    if self.current_time >= ue.death_time:
                        self.ue_death(ue = ue) # Remove UEs that have reached their death time
                        counter +=1
                #Sanity check        
                #print('Killed', counter, 'users at',  round(self.current_time, 2), 's')

                if len(self.queue) > 0:
                    min_queue_time = min(item[1] for item in self.queue)
                    if self.current_time >= min_queue_time:
                        to_gen = 0
                        iter_queue = copy.deepcopy(self.queue)
                        for i in iter_queue:
                            if self.current_time >= i[1]: # Check if it's time to generate new UEs
                                to_gen += i[0]
                                self.queue.remove(i)
                        self.ue_arrival(n_users=to_gen,motion_params = motion_params)
                        #Sanity check
                        #print('Generated', to_gen, 'users at',  round(self.current_time, 2), 's')


                if self.current_time >= move_int:
                    move_int = self.time_converter(self.move_interval) + self.current_time  # Update movement interval

                    #Sanity check
                    #print('move interval:',  f'{move_int} s')

                    self.ue_translation(randomPick = randomPick, const_pick= const_pick) # Update UEs' positions
                    # Check density condition for early exit
                    if const_pick:
                        if not self.check_user_density(density_thr = density_thr):
                            print('Early exit - UEs-BSs density is the set threshold. Most of the UEs are associated with filtered BSs.', '\n')
                            early_exit = True
                            break
                    self.optim_round +=1

                self.monitor_users() # Monitor the status of UEs

                user_cnt.append(len(self.controller.UEs)) # Track user count
                self.current_time += 1

            else:
                try:
                    self.current_time += self.move_interval
                    self.iter_ue_trace(randomPick = randomPick, const_pick= const_pick) # Iterate through UE traces
                    self.optim_round +=1
                    user_cnt.append(len(self.controller.UEs))
                    
                except StopIteration:
                    print("Simulation over - End of UE trace data reached.", '\n')
                    gc.collect()
                    break # Exit loop if trace data is exhausted
        
        # Post-simulation processing
        if not early_exit:
            self.avg_users = round(sum(user_cnt) / len(user_cnt))
            if self.verbose:
                print(f'Simulation ended at: {round(self.current_time, 2)}s', '\n')
                print(f'Average number of users during the simulation: {self.avg_users}', '\n')
                print(f'Average death per round: ', round((self.n_death/self.optim_round),2), '\n')
            
            self.controller.build_ues_db(ue_filename=fp_ue, ues_traces=self.ues_traces,bs_filename= fp_bs, bss_traces=self.bss_traces)

            if 'wdbo' in self.tag:
                t = 'wdbo'
            elif 'const' in self.tag:
                t = 'const'
            else:
                t = 'rndm'
            
            m_model = self.tag.rsplit(f'{t}_', 1)[-1]

            self.write_log(out_dir = f'output/results/performance_logs/obj_{t}/{m_model}', logger = self.history_logger, file_str=f'metrics_logs_sim{self.id}{self.tag}.json')
            
            self.write_log(out_dir = f'{self.out_dir}/trajectories', logger = self.controller.logger,file_str=f'snapshots_logs_sim{self.id}{self.tag}.json')

            if 'wdbo' in self.tag: 
                self.write_log(out_dir = f'output/optimizers_logs/{m_model}', logger = self.controller.optim_logger,file_str=f'optimizer_logs_sim{self.id}{self.tag}.json')

            self.plot_metrics_history() # Plot the collected metrics
        
    def plot_metrics_history(self):
        """
        Plots the historical metrics collected during the simulation.
        """

        flatten = lambda xss: [x for xs in xss for x in xs]

        out_dir = f'{self.out_dir}/metrics/plots'

        self.utility._check_directory(out_dir)

        handler = self.history_logger.handlers[0]
        if not hasattr(handler, 'records') or not handler.records:
            print("No data to plot.")
            return

        # Parse records assuming they are in JSON format
        records = [json.loads(record) for record in handler.records]  # Adjusting how records are parsed
        times = [log['time'] for log in records]
        metric_keys = {'f(x)'}

        cdf_data = {}
        #Sanity check
        # print('times', len(times))
        # print('sinr_lists', len(self.controller.accumulated_sinr))
        for ix, t in enumerate(times):
            try:
                sorted_values = self.controller.accumulated_sinr[ix]
            except IndexError:
                print(f"IndexError: index {ix} is out of range for accumulated_sinr. Change ID to the current simulation or delete the log for the simulation with same ID ({self.id})")
            cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
            cdf_data[t] = (sorted_values, cdf)

        # Convert cdf_data dictionary to DataFrame
        cdf_list = []
        for t, (sorted_values, cdf) in cdf_data.items():
            for sv, c in zip(sorted_values, cdf):
                cdf_list.append([t, sv, c])

        cdf_df = pd.DataFrame(cdf_list, columns=['time', 'sinr', 'cdf'])

        # Export DataFrame to CSV
        if 'wdbo' in self.tag:
            t = 'wdbo'
        elif 'const' in self.tag:
            t = 'const'
        else:
            t = 'rndm'
        
        m_model = self.tag.rsplit(f'{t}_', 1)[-1]
    
        tp = f'output/results/performance_logs/sinr_{t}/{m_model}'
        self.utility._check_directory(tp)
        cdf_df.to_csv(f'{tp}/cdf_data_sim{self.id}{self.tag}.csv', index=False)

        # Plot the CDFs
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create a colormap from the inferno color scale
        colormap = plt.cm.viridis
        normalize = mcolors.Normalize(vmin=min(times), vmax=max(times))
        colors = [colormap(normalize(t)) for t in times]

        for i,(time,color) in enumerate(zip(times, colors)):
            sorted_values, cdf = cdf_data[time]
            label = None
            plt.plot(sorted_values, cdf, color=color, label=label)
            print(f'time: {time}, median_sinr: {np.median(sorted_values)}')
    

        fp_ = f'{out_dir}/sinr_cdf_sim{self.id}{self.tag}.png'
        
        sm = mpl.cm.ScalarMappable(cmap=colormap, norm=normalize)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='vertical', label='Time (s)')

        ax.set_title('SINR: Cumulative Density Function over Time')
        ax.set_xlabel('SINR')
        ax.set_ylabel('Cumulative Probability')
        ax.set_xscale('log')
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True)

        plt.savefig(fp_)
        plt.show()

        for metric in metric_keys:
            fp_ = f'{out_dir}/{metric}_sim{self.id}{self.tag}.png'
            plt.figure(figsize=(10, 5))
            metric_values = [log[metric] for log in records if metric in log]  # Safety check for key existence
            plt.plot(times, metric_values, marker='x', label=f'{metric.capitalize()}')
            plt.xlabel('Time (s)')
            if self.alpha_fair != 1:
                mt = '(bps/user)'
            else:
                mt = ''
            plt.ylabel(f'{metric.capitalize()} {mt}')
            plt.title(f'{metric.capitalize()} Over Time')
            plt.legend()
            plt.savefig(fp_)
            plt.show()
        