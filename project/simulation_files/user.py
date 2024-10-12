from simulation_files.libs.libraries import *
from simulation_files.simulation_env import *
from simulation_files.utility import *
from simulation_files.mobility.mobility_class import *


class UE:
    def __init__(self, ue_id: int, position: Tuple[float, float], utility:Utility, max_velocity: float, min_velocity: float,death_time:float = float('inf')):
        """
        Initialize a User Equipment (UE) object.

        Parameters:
        - ue_id (int): The ID of the UE.
        - position (tuple): The initial position of the UE (x, y coordinates).
        - utility (Utility): The utility function associated with the UE.
        - max_velocity (float): The maximum velocity of the UE.
        - min_velocity (float): The minimum velocity of the UE.
        - death_time (float): The time the UE will stop functioning. Defaults to infinity.
        """
        
        self.ue_id = ue_id
        self.utility = utility
        self.position = position
        self.active = False 
        self.inner_region = False
        self.bs_id = None
        self.cluster = None
        self.generator = None
        self.max_velox = max_velocity
        self.min_velox = min_velocity
        self.death_time = death_time
    
    def get_id(self):
        """
        Get the ID of the UE.

        Returns:
        - int: The ID of the UE.
        """
        return self.ue_id

    def set_active(self):
        """ 
        Set the UE as active.
        """
        self.active = True

    def set_inactive(self):
        """ 
        Set the UE as inactive and clear its attributes.
        """
        self.active = False
        # Sanity check
        if self.bs_id is not None: self.bs_id = None
        # Clear variables
        self.inner_region = False
        self.generator = None
        self.tile = None
        self.cluster = None
        self.utility = None
        self.max_velox = None
        self.min_velox = None
        self.death_time = None
    
    def ue_trace(self, ctime):
        """
        Get the trace of the UE, including its geographical coordinates, position, and cluster information.

        Parameters:
        - ctime (float): The current time.

        Returns:
        - dict: A dictionary containing the UE's time, ID, latitude, longitude, x, y coordinates, and cluster.
        """

        latitude, longitude = self.utility.get_geo_coordinate(position = self.position)
        
        x, y = self.position
        return {
            'time': ctime, 
            'id': self.ue_id,
            'latitude': latitude,
            'longitude': longitude,
            'x': x,
            'y': y,
            'cluster': self.cluster,
        }

    def _init_mobility(self, params, dimensions, border_margin):
        """
        Initialize the mobility model for the UE. The possible motion models are:

        User-based mobility models:
        - Random walk:
            - v and θ sampled from a uniform distribution.
        - Biased random walk:
            - v, θ, θ', β sampled from a uniform distribution.
        - Lévy flight:
            - 0.1 <= α < 2.
        - Truncated Lévy flight:
            - finite-variance model.
        - Brownian motion:
            - Lévy flight with α = 2.

        Parameters:
        - params (dict): A dictionary containing mobility model parameters (e.g., 'mobility_model' = 'biased_random_walk').
        - dimensions (tuple): The dimensions of the network grid (width, height).
        - border_margin (float): The margin around the borders of the grid to constrain movement.

        """

        model = params.get('mobility_model', 'biased_random_walk')

        if model == "random_walk":
            self.generator = random_walk(position=self.position, dimensions=dimensions,border_margin=border_margin, max_velox=self.max_velox, min_velox=self.min_velox, rndm_gen = self.utility.motion_generator)
        elif model == "biased_random_walk":
            self.generator = biased_random_walk(position=self.position, dimensions=dimensions, border_margin = border_margin, max_velox=self.max_velox, min_velox=self.min_velox, rndm_gen = self.utility.motion_generator)
        elif model == "levy_walk":
            alpha = params.get('alpha', 1)
            self.generator = levy_walk(position=self.position, dimensions=dimensions, border_margin = border_margin,alpha=alpha, max_step_length=None, rndm_gen = self.utility.motion_generator)
        elif model == "truncated_levy_walk":
            max_step_length = self.utility.tile_size
            alpha = params.get('alpha', 1)
            self.generator = levy_walk(position=self.position, dimensions=dimensions, alpha=alpha, max_step_length=max_step_length, border_margin=border_margin, rndm_gen = self.utility.motion_generator)  # Use max_step_length to control truncation
        elif model == "brownian_motion":
            self.generator = brownian_motion(position=self.position, dimensions=dimensions,border_margin=border_margin, rndm_gen = self.utility.motion_generator)
        elif model == "random_waypoint":
            wt_max = params.get('wt_max', 1)
            self.generator = random_waypoint(nr_nodes=1, dimensions=dimensions, velocity=(self.min_velox, self.max_velox), wt_max=wt_max)#RandomWaypoint
        else:
            raise ValueError(f"Unknown mobility model: {model}")
    
    def move(self):
        """
        Move the UE based on the generator associated with the UE's mobility model. This function updates the UE's position.
        """
        
        self.position = next(self.generator)

        if isinstance(self.position, np.ndarray):

            if self.position.shape == (1,2): #RWP model
                self.position = self.position[0]

            self.position = tuple(self.position)


class Cluster():
    def __init__(self,rndm_generator, id: int,destination: Tuple[float,float],dimensions: Tuple[float,float]):
        """
        Initialize a Cluster object that contains a group of UEs with a common destination.

        Parameters:
        - rndm_generator (Generator): A random number generator for stochastic operations.
        - id (int): The ID of the cluster.
        - destination (tuple): The destination of the cluster (x, y coordinates).
        - dimensions (tuple): The dimensions of the network grid (width, height).
        """

        self.id = id
        self.paired_ues = []
        self.destination = destination
        self.generator = None
        self.history = [] #to record trajectory
        self.hot_points = []
        self.dimensions = dimensions
        self.rndm_generator = rndm_generator
    
    def get_id(self):
        """
        Get the ID of the cluster.

        Returns:
        - int: The ID of the cluster.
        """
        return self.id

    def add_ue(self, ue: UE):
        """
        Add a UE to the cluster's list of paired UEs.

        Parameters:
        - ue (UE): The UE object to be added to the cluster.
        """
        self.paired_ues.append(ue)
    
    def remove_ue(self, ue_id: int):
        """
        Remove a UE from the cluster based on its ID.

        Parameters:
        - ue_id (int): The ID of the UE to be removed.
        """
        #remove ue with matching UE_id
        for ue in self.paired_ues:
            if ue.ue_id == ue_id:
                self.paired_ues.remove(ue)

    def _init_motion(self, border_margin, position: np.ndarray):
        """
        Set the initial destination for the cluster and initialize the movement generator.

        Parameters:
        - border_margin (float): The margin around the borders of the grid to constrain movement.
        - position (np.ndarray): The initial position of the cluster centroid.
        """
        avg_max_v = max(ue.max_velox for ue in self.paired_ues)
        avg_min_v = min(ue.min_velox for ue in self.paired_ues)

        self.generator = hybrid_gmm(position = position,destination_sets = self.hot_points,
                                        dimensions = self.dimensions,border_margin=border_margin, max_velox= avg_max_v, min_velox = avg_min_v,
                                        rndm_gen= self.rndm_generator) #m/transition 

    def move_destination(self):
        """
        Update the cluster's destination using the movement generator and randomly adjust the UEs' positions relative to the destination.
        """
        self.destination = next(self.generator)
        self.history.append(self.destination)

        for ue in self.paired_ues:

            angle = self.rndm_generator.uniform(0, 2 * np.pi)
        
            # Generate a random radius with uniform distribution within the circle of radius 100m
            radius = self.rndm_generator.uniform(0, 1) * 100 #m
            
            # Convert polar coordinates to Cartesian coordinates
            delta_x = radius * np.cos(angle)
            delta_y = radius * np.sin(angle)
            
            # Set the UE position relative to the destination
            ue.position = (self.destination[0] + delta_x, self.destination[1]+ delta_y)


    def set_destinations(self, hot_points: list):
        """
        Set the hot points (waypoints) for the cluster to follow during its movement.

        Parameters:
        - hot_points (list): A list of waypoints for the cluster.
        """
        self.hot_points = hot_points
            
        