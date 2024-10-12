from simulation_files.libs.libraries import *
from simulation_files.simulation_env import *

EARTH_RADIUS_KM = 6371.0  # Earth radius in kilometers
KM_TO_M = 10**3

class NumpyEncoder(json.JSONEncoder):
    """ 
    Custom encoder for NumPy data types to be JSON serializable.
    It handles the conversion of NumPy data types like int, float, and arrays into standard Python types.
    """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        return super().default(obj)

class MemoryHandler(logging.Handler):

    """ 
    Custom logging handler to store log records in memory instead of writing to a file.
    """
    
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        """
        Overwrites the 'emit' method to append formatted records to memory.
        
        Parameters:
        - record: The log record being processed.
        """
        self.records.append(self.format(record))

class JsonFormatter(logging.Formatter):
    """
    Custom formatter to output log messages as JSON-formatted strings.
    """
    def format(self, record):
        """
        Formats the log record as a JSON object.
        
        Parameters:
        - record: The log record to be formatted.
        
        Returns:
        - str: JSON formatted log message.
        """
        message = record.msg
        if isinstance(message, dict):
            return json.dumps(message)
        return json.dumps({"message": message})


class Utility():
    """
    Utility class for managing grid dimensions, motion generators, and distribution methods.
    """
    def __init__(self, n_tiles: int = 0,num_tiles_x:int = 0, num_tiles_y:int = 0, verbose:bool=False, 
                 origin: Tuple[float,float]= ORIGIN, dimensions: Tuple[float,float] = (NET_WIDTH, NET_HEIGHT), tile_size: float = TILE_SIZE,d_metric:str = 'm' ) -> None:
        
        """
        Initializes the Utility object with grid parameters, random generators, and metrics.
        
        Parameters:
        - n_tiles (int): Total number of tiles in the grid.
        - num_tiles_x (int): Number of tiles along the x-axis.
        - num_tiles_y (int): Number of tiles along the y-axis.
        - verbose (bool): If True, display detailed output.
        - origin (Tuple[float, float]): The origin coordinates (latitude, longitude).
        - dimensions (Tuple[float, float]): Dimensions of the grid (width, height).
        - tile_size (float): The size of each tile in the grid.
        - d_metric (str): Distance metric used, either 'm' (meters) or 'km' (kilometers).
        """

        self.verbose = verbose
        self.n_tiles = n_tiles
        self.num_tiles_x = num_tiles_x
        self.num_tiles_y = num_tiles_y
        self.origin = origin
        self.dimensions = dimensions
        self.tile_size = tile_size
        self.d_metric = d_metric
        self.dstr_generator = np.random.default_rng(seed=21)
        self.motion_generator = np.random.default_rng(seed=42)
        

    def _check_directory(self, path):
        """
        Checks if the directory exists, and creates it if it doesn't.
        
        Parameters:
        - path (str): Directory path to check/create.
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def _utlity_init_logger(self, name):
        """
        Initializes a logger to store records in memory.

        Parameters:
        - name (str): Name of the logger.
        
        Returns:
        - logging.Logger: Configured logger object.
        """
        history_logger = logging.getLogger(name)
        memory_handler = MemoryHandler()
        memory_handler.setLevel(logging.INFO)
        formatter = JsonFormatter()
        memory_handler.setFormatter(formatter)
        history_logger.addHandler(memory_handler)
        history_logger.setLevel(logging.INFO)
        history_logger.propagate = False
        return history_logger

    def get_distributions_list(self):
        """
        Returns a dictionary of available distribution functions.

        Returns:
        - dict: Mapping of distribution names to corresponding methods.
        """
        return {
            'poisson': self._poisson_point_process,
            'log_gaussian_cox': self._log_gaussian_cox_process,
            'rbf':  self._radial_basis_prob_map,
        }

    def get_geo_coordinate(self,position: Tuple[float, float])->Tuple[float, float]:
        """
        Converts grid position (x, y) to geographical coordinates (latitude, longitude).

        Parameters:
        - position (Tuple[float, float]): The position in the grid (x, y).

        Returns:
        - Tuple[float, float]: The corresponding latitude and longitude.
        """
        # NB
        # source: http://www.edwilliams.org/avform147.htm
        # and https://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters/2980#2980?newreg=0bedc72752ea4e629440a761d6f4a231
        # ORIGIN = (Latitude, Longitude) 
        # self.position = (x, y) in km or m x=>longitude, y=>latitude
        if self.d_metric.lower() not in ('km', 'm'):
            raise ValueError('distance_metric must be either "km" or "m"')
 
        earth_radius = EARTH_RADIUS_KM if self.d_metric.lower() == 'km' else EARTH_RADIUS_KM*KM_TO_M

        # Convert latitude and longitude to radians
        lat_rad = math.radians(self.origin[0])
        # lon_rad = math.radians(ORIGIN[1])

        # Calculate the change in latitude
        delta_lat = position[1] / earth_radius

        # Calculate the change in longitude
        delta_lon = position[0] / (earth_radius * math.cos(lat_rad))

        # Convert the changes to degrees
        delta_lat_deg = math.degrees(delta_lat)
        delta_lon_deg = math.degrees(delta_lon)

        # Calculate the new latitude and longitude
        new_latitude = self.origin[0] + delta_lat_deg
        new_longitude = self.origin[1] + delta_lon_deg

        return new_latitude, new_longitude

    def get_grid_position(self,geo_coordinates: Tuple[float, float])->Tuple[float, float]: #geo_coordinates(Lat,Long)
        """
        Converts geographical coordinates (latitude, longitude) to grid position (x, y).

        Parameters:
        - geo_coordinates (Tuple[float, float]): The geographical coordinates (latitude, longitude).

        Returns:
        - Tuple[float, float]: The corresponding grid position (x, y).
        """
        x = round(distance((self.origin[0], geo_coordinates[1]), self.origin).km, 3) 
                                                                                                   
        y = round(distance((geo_coordinates[0], self.origin[1]), self.origin).km, 3) 
                                                                                                            
        if self.d_metric.lower() == 'm':
            x = x * KM_TO_M
            y = y * KM_TO_M
        return (x,y)

    def get_tile_geo_coordinate(self,tile_loc: int)->Tuple[float, float]:
        """
        Gets the geographical coordinates for the center of a tile.

        Parameters:
        - tile_loc (int): The tile index.

        Returns:
        - Tuple[float, float]: The latitude and longitude of the tile center.
        """
        return self.get_geo_coordinate(self.get_tile_coordinate(tile_loc = tile_loc))

    def get_tile_coordinate(self,tile_loc: int)->Tuple[float, float]:
        """
        Returns the (x, y) grid coordinates of a tile.

        Parameters:
        - tile_loc (int): The tile index.

        Returns:
        - Tuple[float, float]: The (x, y) coordinates of the tile.
        """
        row_index = tile_loc // self.num_tiles_x
        col_index = tile_loc % self.num_tiles_x

        tile_base_x = col_index * self.tile_size
        tile_base_y = row_index * self.tile_size

        return tile_base_x, tile_base_y

    def _radial_basis_prob_map(self,**kwargs):
        """
        Generates a probability distribution map based on a radial basis function.

        Parameters:
        - total_users (int): The total number of users to distribute across the grid.
        - c (int): The center tile for the distribution.
        - gamma (float): The spread parameter for the radial basis function.

        Returns:
        - np.ndarray: An array representing the number of users allocated to each tile.
        """
        if 'total_users' not in kwargs:
            raise ValueError("Missing required parameters 'total_users' for RBF distribution")
        
        total_users = kwargs['total_users']

        if 'c' not in kwargs:
            c = self.dstr_generator.choice(range(self.n_tiles))
        else:
            c = kwargs['c']
        
        if 'gamma' not in kwargs:
            gamma = 1/self.n_tiles
        else:
            gamma = kwargs['gamma']

        if c <= 0:
            raise ValueError(f"'c' must be greater in [0,{self.n_tiles}]")

        densifier = lambda i, c: np.exp(- gamma * l2_norm(self.get_tile_coordinate(tile_loc = i),c))
        normalizer_p = lambda lst: np.array(lst)/np.sum(lst)
        prob_map_ = normalizer_p([densifier(i, c) for i in range(self.n_tiles)])

        #allocate each a proportional amoiunt of user to each tile accordingly to the distribution
        allocated_users = np.ceil(prob_map_ * total_users).astype(int)

        if self.verbose:
            plt.imshow(prob_map_.reshape((int(self.num_tiles_y), int(self.num_tiles_x))))
            plt.colorbar()
            plt.title('Probability Distribution Across Grid')
            plt.show()
        
        return allocated_users
    
    def _poisson_point_process(self, **kwargs): 
        """
        Generates a Poisson point process distribution over the grid tiles.

        Parameters:
        - lambda_ (float): The expected number of points per unit area.

        Returns:
        - np.ndarray: An array representing the number of points in each tile.
        """
        if 'lambda_' not in kwargs:
            raise ValueError("Missing required parameter 'lambda_' for Poisson distribution.")
        
        lambda_ = kwargs['lambda_']
        if lambda_ <= 0:
            raise ValueError("lambda_ must be greater than 0 for Poisson distribution.")
        
        return self.dstr_generator.poisson(lambda_, self.n_tiles)
    
    def _log_gaussian_cox_process(self, **kwargs):
        """
        Generates a Log-Gaussian Cox process distribution over the grid tiles.

        Parameters:
        - mu (float): Mean of the Gaussian distribution.
        - sigma (float): Standard deviation of the Gaussian distribution.

        Returns:
        - np.ndarray: An array representing the number of points in each tile.
        """
        # Sample a Gaussian random variable for each tile
        if 'mu' not in kwargs or 'sigma' not in kwargs:
            raise ValueError("Missing required parameters 'mu' and 'sigma' for Log-Gaussian Cox process.")
        
        mu = kwargs['mu']
        sigma = kwargs['sigma']

        if sigma <= 0:
            raise ValueError("sigma must be greater than 0 for Gaussian distribution.")

        gaussian_samples = self.dstr_generator.normal(mu, sigma, self.n_tiles)

        # Calculate intensity and sample the number of UEs per tile
        intensity = self.dstr_generator.exp(gaussian_samples)
        ue_counts = self.dstr_generator.poisson(intensity)
        return ue_counts 

    def get_way_points(self,bs_df, dbscan_args, max_cap = None):

        """
        Identifies waypoints based on clustered base station locations using DBSCAN.

        Parameters:
        - bs_df (pd.DataFrame): DataFrame of base station locations with required columns.
        - dbscan_args (dict): Arguments to pass to the DBSCAN clustering algorithm.
        - max_cap (int, optional): Maximum number of waypoints to return.

        Returns:
        - list: A list of hot spots (centroids) of the clusters.
        """

        fp = '../simulation_datasets'

        self._check_directory(fp)
        fp_clusters = f'{fp}/bs_clusters.csv'
        fp_way_points = f'{fp}/way_points.csv'

        bs_required_columns = {'id', 'latitude', 'longitude', 'x', 'y'}
        assert bs_required_columns.issubset(bs_df.columns), \
            f"The DataFrame has these columns {', '.join(bs_df.columns)}.\n It must contain the columns: {', '.join(bs_required_columns)}."

        # Function to compute centroid
        get_centroid = lambda df: df[['x', 'y']].mean().round(3).values

        def get_radius(df):
            centroid = np.array(df.iloc[0]['centroid'])
            return df[['x', 'y']].apply(lambda row: np.linalg.norm(row - centroid), axis=1).max()

        clustering = DBSCAN(**dbscan_args).fit(bs_df[['x', 'y']])

        bs_df['cluster'] = clustering.labels_

        # Compute centroids for each cluster
        centroids = bs_df.groupby('cluster').apply(get_centroid)

        # Map centroids back to the original DataFrame
        bs_df['centroid'] = bs_df['cluster'].map(centroids)

        # Compute the radius for each cluster
        radii = bs_df.groupby('cluster').apply(get_radius)

        # Map radii back to the original DataFrame
        bs_df['radius'] = bs_df['cluster'].map(radii)

        condition = (bs_df['cluster'] != -1) & (bs_df['cluster'] != 0)
        filtered_bs_df = bs_df[condition]

        if max_cap is not None:
            # Select the first 'max_cap' base stations
            filtered_bs_df = filtered_bs_df.iloc[:max_cap]

        hot_spots = filtered_bs_df['centroid'].drop_duplicates().tolist()

        # Save the clusters and hot spots to CSV files
        filtered_bs_df.to_csv(fp_clusters, index=False)

        filtered_bs_df['centroid'].drop_duplicates().to_csv(fp_way_points, index=False)

        return hot_spots
    
    def sample_position(self, tile_loc: Optional[int] = None, eps_border: Optional[int] = None) -> Tuple[float, float]:
        """
        Samples a random position within a tile for placing a User Equipment (UE).

        Parameters:
        - tile_loc (int, optional): The tile index. If None, sample a random position across the grid.
        - eps_border (int, optional): Margin to avoid sampling close to tile borders.

        Returns:
        - Tuple[float, float]: Random position (x, y) within the tile or grid.
        """
        def random_offset(min_dim:Tuple[float, float], max_dim:Tuple[float, float]):
            if eps_border is not None:
                min_dim = [min_dim[0]+eps_border,min_dim[1]+eps_border]
                max_dim = [max_dim[0]-eps_border,max_dim[1]-eps_border]

            return self.dstr_generator.uniform(low = min_dim,high=max_dim, size=(1,2))[0]

        if tile_loc is not None:
            
            min_x, min_y = self.get_tile_coordinate(tile_loc=tile_loc)
            new_position = random_offset(min_dim=[min_x, min_y],max_dim=[self.tile_size, self.tile_size])

        else:
            new_position = random_offset(min_dim=[0, 0],max_dim=[self.dimensions[0], self.dimensions[1]])
        
        return tuple(new_position)