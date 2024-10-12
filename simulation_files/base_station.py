from simulation_files.libs.libraries import *
from simulation_files.simulation_env import *
from simulation_files.utility import *
from simulation_files.user import *

class BS():

    def __init__(self, controller, default_r: float, bs_id: int, position = Tuple[float, float]):

        """
        Initializes a new instance of the BS (Base station) class, representing a BS in a cellular network.

        Parameters:
        - controller (Controller): The controller managing the BS.
        - default_r (float): Default radius for the BS's coverage area (for visualization purpose).
        - bs_id (int): Identifier for the BS.
        - position (Tuple[float, float]): Geographic coordinates of the BS (latitude, longitude).

        """

        self.bs_id = bs_id
        self.position = position
        self.cntr = controller
        self.served_UEs = [] #List[UE]
        self.snr_thr = None
        self.sector = None
        self.optimizer = None
        self.bw = W
        self.min_power = None
        self.max_power = None
        self.noise = None
        self.alpha = ALPHA
        self.gain = None
        self.p_tx_in, self.p_tx_out = None, None
        self.default_r = default_r
        self.lin_operator = lambda x_db: 10**(x_db/10)
        self.db_operator = lambda x_lin: 10*np.log10(x_lin)
        self.init_transmission_params()
        

    def init_transmission_params(self):
        """
        Initialises transmission parameters: minimum and maximum transmission power,
        noise and gain with respect to reference loss. The initialisation is compliant with the
        to the unit of measurement indicated in ‘POWER_METRIC’ (from the simulation.env file). 

        """

        if 'dBm' in POWER_METRIC or 'dB' in POWER_METRIC:
            self.min_power = self.lin_operator(10) #lin(dBm) = mW
            self.max_power = self.lin_operator(BS_P_TX) #lin(dBm) = mW
            self.noise = self.lin_operator(N)*self.bw #lin(dBm/Hz) * Hz = mW
            self.gain = self.lin_operator(-L_ref) #lin(dBm) = mW 
        elif 'mW' in POWER_METRIC or 'W' in POWER_METRIC: #all given in lin-scale 
            self.min_power = self.lin_operator(10) #lin(dBm) = mW
            self.max_power = BS_P_TX #mW
            self.noise = N*self.bw #(mW/Hz * Hz) = mW
            self.gain = 1/L_ref #mW
        else:
            raise ValueError('Invalid power metric', 'Possible values: dBm, dB, mW, W')

    def get_id(self):
        """
        Retrieves the unique identifier of the BS.

        Outputs:
        - bs_id (int): The unique identifier of the BS.
        """
        return self.bs_id
    
    def get_power_levels(self):
        """
        Retrieves the current transmission power levels.

        Outputs:
        - (float, float): A tuple containing the current outer and inner power levels (p_tx_out, p_tx_in respectively).
        """
        return self.p_tx_out, self.p_tx_in
    
    def ue_sorting(self):
        """
        Sorts the list of UEs  served by the BS in descending order 
        based on their signal-to-noise ratio (SNR).

        """
        self.served_UEs = sorted(self.served_UEs, key=lambda ue: self.compute_snr(ue), reverse=True)
    
    def power_adjustment(self, power_levels: Optional[List[float]]= None, verbose: bool = False, random_:bool = False, const_pick:bool = False):
        """
        Adjusts the power output levels for the BS based on specified parameters or randomly selected values.

        Parameters:
        - power_levels (Optional[List[float]]): A parameter vector used to compute the power adjustment (default is None).
        - verbose (bool): If True, outputs additional debugging information (default is False).
        - random_ (bool): If True, randomizes the power levels (default is False).
        - const_pick (bool): If True, picks/keeps constant power values (default is False).

        """
        if const_pick:
            if self.p_tx_in is None and self.p_tx_out is None:

                min_p = self.db_operator(self.min_power)
                max_p = self.db_operator(self.max_power)

                x1 = np.random.uniform(low = min_p, high=max_p)
                x1_l = self.lin_operator(x1)
                x2 = np.random.uniform(low = 0.5, high=0.99)
                self.p_tx_in = x1_l*(1-x2)
                self.p_tx_out = x1_l * x2

        elif random_:
            min_p = self.db_operator(self.min_power)
            max_p = self.db_operator(self.max_power)

            x1 = np.random.uniform(low = min_p, high=max_p)
            x1_l = self.lin_operator(x1)
            x2 = np.random.uniform(low = 0.5, high=0.99)
            self.p_tx_in = x1_l*(1-x2)
            self.p_tx_out = x1_l * x2

        else:
            if power_levels is not None:
                median_pl = np.median(np.array(power_levels), axis = 0)
                x1_db, x2 = median_pl[0],median_pl[1]
                x1_lin = self.lin_operator(x1_db)
                new_p_out =  x1_lin * x2 #ensure p_tx_out > p_tx_in
                new_p_in = x1_lin*(1-x2)

                #Debug - Safety check
                if (new_p_out ==0) or (new_p_in ==0):
                    print('power levels', np.array(power_levels))
                    print('median power levels', np.median(np.array(power_levels), axis = 0))

                if new_p_in != 0:
                    self.p_tx_in = new_p_in
                if new_p_out != 0:
                    self.p_tx_out = new_p_out
            else:
                raise ValueError("Missing power_levels")
                

    def retrieve_snr_thr(self): 
        """
        It calculates the SNR threshold by determining the SNR of the UE that maximizes the objective function,
        based on a 2-partition configuration of all served UEs. 
        This threshold is then used to determine whether a UE is inside or outside the coverage area of the BS.

        """
        assert self.served_UEs, 'Empty user set'

        self.ue_sorting() #sort by descending snr values
        
        ue_trace = []
        max_obj_value = float('-inf')
        ue_max = None

        def objective_for_trace(ue):
            ue_trace.append(ue)
            g = self.cntr.objective_function(bss=[self], ues_test=ue_trace)[0]
            return g
    

        ue_objs = [(ue, objective_for_trace(ue)) for ue in self.served_UEs]

        # Find the UE with the maximum objective value
        for ue, obj_value in ue_objs:
            if obj_value >= max_obj_value:
                max_obj_value = obj_value
                ue_max = ue

        self.snr_thr = self.compute_snr(ue_max, inner_flag=True)

        #Find the index of ue_max in self.served_UEs
        ue_max_index = self.served_UEs.index(ue_max)

        # Set the UEs before ue_max to be inner, and the rest to be outer
        for i, ue in enumerate(self.served_UEs):
            if i <= ue_max_index:
                ue.inner_region = True  # Assuming there's a method to set the region
            else:
                ue.inner_region = False
            
    def add_ue(self, ue: UE): 
        """
        Adds a UE to the list of UEs served by the BS. 
        Additionally, assigns the BS ID to the UE and marks the UE as being in the outer region by default.

        Parameters:
        - ue (UE) : The UE instance to be added to the BS.

        """
        self.served_UEs.append(ue)
        ue.inner_region = False 
        ue.bs_id = self.get_id()

    def intra_cell_assignment(self):
        """
        Assigns UEs to the inner or outer regions of the BS based on the computed SNR threshold.

        Parameters:
        - verbose (bool): If True, outputs additional information during the assignment process (default is True).
        """

        self.retrieve_snr_thr()
                
        
    def remove_ue(self, ue: UE):
        """
        Removes a UE  from the list of UEs served by the BS.

        Parameters:
        - ue (UE): The UE instance to be removed.
        """
        self.served_UEs.remove(ue)
    
    def get_distance(self, ue:UE):
        """
        Calculates the Euclidean distance between the BS and a given UE.

        Parameters:
        - ue (UE): The UE instance whose distance to the BS is to be calculated.

        Outputs:
        - distance (float): The computed Euclidean distance between the BS and the UE, with a minimum value of 1 unit to avoid very small distances.
        """
        try:
            d = math.sqrt(sum((x1 - x2) ** 2 for x1, x2 in zip(self.position, ue.position)))
            return max(d,1) #cap the distance to 1 if the user get to close to the BS (i.e. d<=1)
        except TypeError as e:
            print(f"TypeError occurred: {e}")
            print(f"self.position {self.position}, type: {type(self.position)}")
            print(f"ue.position {ue.position}, type: {type(ue.position)}")
            raise  
    
    def compute_link_gain(self, ue:UE):
        """
        Computes the link gain between the BS and a UE based on the path loss model.

        Parameters:
        - ue (UE): The UE instance for which the link gain is to be calculated.

        Outputs:
        - gain (float): The computed link gain based on the distance between the BS and the UE.
        """
        d = self.get_distance(ue)
        gain = self.gain * ((1000/d)**(self.alpha)) #standard path loss model in nominal, LOS conditions(note L_ref w.r.t 1km)
        return gain

    def compute_snr(self, ue: UE, inner_flag: bool = None):
        """
        Computes the SNR for a given UE, considering whether it is in the inner or outer BS region.

        Parameters:
        - ue (UE): The UE instance for which the SNR is to be computed.
        - inner_flag (bool): An optional parameter to explicitly define whether the computation should consider the inner (`True`) or outer (`False`) region. If None, the SNR is computed based on the UE's current region.

        Outputs:
        - snr (float): The computed SNR value for the given UE.
        """
        if inner_flag is None:
            pw = self.p_tx_in if ue.inner_region else self.p_tx_out
        elif inner_flag:
            pw = self.p_tx_in
        else:
            pw = self.p_tx_out

        return (pw * self.compute_link_gain(ue)) / self.noise

    
    def set_optimizer(self, kwargs):
        """
        Configures the optimizer for the BS using the specified keyword arguments, including the number of initial observations and kernel parameters.

        Parameters:
        - kwargs (dict): A dictionary of configuration options, including:
        - 'n_initial_observations' (int): The number of initial observations to use for optimization (default is 15).
        - 'spatial_kernel_args' (list[float]): Arguments for the spatial kernel (default is [2.5]).
        - 'temporal_kernel_args' (list[float]): Arguments for the temporal kernel (default is [1.5]).

        """

        if kwargs is None:
            kwargs = {}

        n_initial_observations = kwargs.get('n_initial_observations', 15)
        spatial_kernel_args = kwargs.get('spatial_kernel_args', [2.5])
        temporal_kernel_args = kwargs.get('temporal_kernel_args', [1.5])

        spatial_domain = np.array([[self.db_operator(self.min_power), self.db_operator(self.max_power)],[0.5, 0.99]])

        self.optimizer = WDBOOptimizer(
            spatial_domain,
            gpytorch.kernels.MaternKernel, gpytorch.kernels.MaternKernel, #(nu = 1.5)
            spatial_kernel_args=spatial_kernel_args, temporal_kernel_args=temporal_kernel_args,
            n_initial_observations=n_initial_observations,
            alpha = 0.25
        )

    def bs_trace(self, ctime):
        """
        Collects the trace information for the BS at a specific timestamp, including current power levels, served UEs, and coverage radii.

        Parameters:
        - ctime (float): The current timestamp for which the BS trace is to be retrieved.

        Outputs:
        - trace (dict): A dictionary containing the following information:
        - 'time' (float): The current timestamp.
        - 'id' (int): The identifier of the BS.
        - 'r_in' (float): The inner coverage radius of the BS.
        - 'r_out' (float): The outer coverage radius of the BS.
        """

        print('time:', ctime, 'bs_id:', self.get_id(),'power levels (in,out):', (self.p_tx_in,self.p_tx_out),'\n')
        print('Served UEs:', [{'ue_id': ue.get_id(),'inner_region': ue.inner_region, 'd':round(self.get_distance(ue),3),
                         'sinr': self.cntr.compute_sinr(bs = self, ue = ue, interference_set=None, pairing= False, inner_test = False, outer_test= False)}
                           for ue in self.served_UEs],'\n')
        print('---------------', '\n')
        
        active_ues = self.served_UEs

        r_in = max((self.get_distance(ue) for ue in active_ues if ue.inner_region), default= self.default_r)
        r_out = max((self.get_distance(ue) for ue in active_ues if not ue.inner_region), default = r_in + self.default_r)

        return {
            'time': ctime,
            'id': self.bs_id,
            'r_in': r_in,
            'r_out': r_out,
        }