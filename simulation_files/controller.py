from simulation_files.libs.libraries import *
from simulation_files.utility import *
from simulation_files.user import *
from simulation_files.base_station import *

class Controller:
    def __init__(self, alpha_fairness:float, verbose:bool,bs_max_range:float, optimizer_kwargs:dict, id:int):
        """
        Initializes a new instance of the Controller class, responsible for managing 
        user equipment (UEs) and base stations (BSs) in a cellular network system.
        
        Parameters:
        - alpha_fairness (float): A parameter for the alpha-fairness in the system's resource allocation.
        - verbose (bool): Flag to enable detailed logging output.
        - bs_max_range (float): Maximum communication range for the BSs.
        - optimizer_kwargs (dict): Additional parameters for the optimizer controlling BS power levels.
        - id (int): Identifier for the Controller instance.
        """
        self.UEs = []  # List[UE]
        self.BSs = []  # List[BS]
        self.alpha_fairness = alpha_fairness
        self.verbose = verbose
        self.optimizer_kwargs = optimizer_kwargs
        self.bs_max_range = bs_max_range
        self.accumulated_sinr = []
        self.filter = lambda bs_list : [bs for bs in bs_list if len(bs.served_UEs) != 0] #filter non empty bss
        self.filter_border_bss = None
        self.trace_count = 0
        self.trace_time = None
        if self.verbose:
            # Initialize logging for general events and optimizer-specific events.
            self.logger = Utility()._utlity_init_logger(name= f'timestamp{id}')
            self.optim_logger = Utility()._utlity_init_logger(name= f'optimizers{id+1}')
            
        self.gen_rng = np.random.default_rng(id)

        print('alpha_fairness:', round(self.alpha_fairness,3),'\n')
    

    def set_border_filter(self, dimensions: Tuple[float, float], eps_border: float):
        """
        Sets a filter for BSs that are within a specified distance from the network's borders.
        
        Parameters:
        - dimensions (Tuple[float, float]): The dimensions of the area.
        - eps_border (float): The distance from the borders within which the filter applies.
        """
        min_dim = (eps_border, eps_border)
        max_dim = (dimensions[0] - eps_border, dimensions[1] - eps_border)
        
        def filter_func(bs):
            return (min_dim[0] <= bs.position[0] <= max_dim[0]) and (min_dim[1] <= bs.position[1] <= max_dim[1])
        
        self.filter_border_bss = filter_func
        
    def _lookup(self, items: List, identifier: int):
        """
        General-purpose method to retrieve an object from a list using its identifier.
        
        Parameters:
        - items (List): List of items to search through.
        - identifier (int): The identifier of the item to retrieve.
        
        Returns:
        - Optional: The item object if found, None otherwise.
        """
        if items:
            return next((item for item in items if item.get_id() == identifier), None)
        return None

    def bs_lookup(self, bs_id: int = None) -> Optional[BS]:
        """
        Retrieves a BS object from the list of BSs using its identifier.
        
        Parameters:
        - bs_id (int): The identifier of the BS to retrieve.
        
        Returns:
        - Optional[BS]: The BS object if found, None otherwise.
        """
        return self._lookup(self.BSs, bs_id) if bs_id is not None else None
    
    def ue_lookup(self, ue_id: int = None) -> Optional[UE]:
        """
        Retrieves a UE object from the list of UEs using its identifier.
        
        Parameters:
        - ue_id (int): The identifier of the UE to retrieve.
        
        Returns:
        - Optional[UE]: The UE object if found, None otherwise.
        """
        return self._lookup(self.UEs, ue_id) if ue_id is not None else None

    def bs_handover(self, ue: UE):
        """
        Manages the handover of a UE from one BS to another based on signal-to-noise ratio (SINR) metrics.
        
        Parameters:
        - ue (UE): The UE undergoing handover.
        
        Returns:
        - bool: A flag indicating whether the UE is in range of any BS.
        """
        
        def subroutine(ue:UE, eligible_bs: List[BS], current_bs: BS = None):
            bs = max(eligible_bs, key=lambda bs: self.compute_sinr(bs = bs, ue = ue, interference_set=eligible_bs, pairing= True, inner_test = False, outer_test= False))
            if current_bs is not None:
                if current_bs.get_id() != bs.get_id():
                    current_bs.remove_ue(ue)
                    bs.add_ue(ue)
            else:
                bs.add_ue(ue) 
            

        ue_in_range = True

        current_bs = self.bs_lookup(ue.bs_id)

        eligible_bs = [b for b in self.BSs]

        if len(eligible_bs)==0:
            ue_in_range = False # Early exit if no BS is eligible

        else:
            subroutine(ue = ue,eligible_bs =eligible_bs, current_bs = current_bs)

        return ue_in_range
    
    def compute_sinr(self, bs: BS, ue: UE, pairing:bool, interference_set: List[BS] = None, inner_test: bool = False, outer_test: bool = False)->float:
        """
        Computes the downlink SINR for a given UE at a specified BS.
        
        Parameters:
        - bs (BS): The BS.
        - ue (UE): The UE.
        - pairing (bool): Whether to include intra-cell interference.
        - interference_set (List[BS], optional): List of BSs to consider for inter-cell interference. Defaults to None.
        - inner_test (bool, optional): Flag to perform inner region SINR calculation. Defaults to False.
        - outer_test (bool, optional): Flag to perform outer region SINR calculation. Defaults to False.
        
        Returns:
        - float: The computed SINR.
        """

        dl_sinr = .0 

        if interference_set is None:
            bs_interference_set = [b for b in self.BSs if b!=bs]
        else:
            bs_interference_set = [b for b in interference_set if b!=bs]

        inter_cell_interf = sum([b.compute_link_gain(ue)*(b.p_tx_in + b.p_tx_out) for b in bs_interference_set]) 

        interference = bs.noise + inter_cell_interf

        link_gain = bs.compute_link_gain(ue)

        if pairing:

            intra_cell_interf = link_gain*bs.p_tx_in

            interference += intra_cell_interf 

            dl_sinr = (link_gain*bs.p_tx_out)/interference

        elif inner_test:
              
            dl_sinr = (link_gain*bs.p_tx_in)/interference
        
        elif outer_test:

            intra_cell_interf = link_gain*bs.p_tx_in

            interference += intra_cell_interf 

            dl_sinr = (link_gain*bs.p_tx_out)/interference

        else:
            if ue.inner_region:
                dl_sinr = (link_gain*bs.p_tx_in)/interference
            else:
                intra_cell_interf = link_gain*bs.p_tx_in

                interference += intra_cell_interf 

                dl_sinr = (link_gain*bs.p_tx_out)/interference

        return float(min(dl_sinr,256)) #cap at 256 for a UE very close to BS, avoid large outliers
    
    def kill_ue(self, ue: UE):
        """
        Removes a UE from the system, including dissociating it from the BS and marking it as inactive.
        
        Parameters:
        - ue (UE): The UE to be removed.
        """
        if ue in self.UEs:
            current_bs = self.bs_lookup(bs_id = ue.bs_id)
            if current_bs is not None:
                current_bs.remove_ue(ue) 
            self.UEs.remove(ue)
            ue.set_inactive()

    def add_ue(self, ue:UE):
        """
        Adds a new UE to the system, sets it as active, and performs an initial BS assignment.
        
        Parameters:
        - ue (UE): The UE to be added.
        
        Returns:
        - bool: A flag indicating whether the UE was successfully added to a BS.
        """
        b = self.bs_handover(ue)
        if b: 
            self.UEs.append(ue)
            ue.set_active()
        return b

    def gather_metrics(self):
        """
        Collects SINR metrics (median, mean, max SINR) for each active UE in the network and calculates the system's objective function value.
        
        Returns:
        - dict: A dictionary containing the following metrics:
            - 'f(x)' (float): The objective function value.
            - 'median_sinr' (float): The median SINR across the network.
            - 'mean_sinr' (float): The average SINR across the network.
            - 'max_sinr' (float): The maximum SINR across the network.
        """
        rounder = lambda x: round(x,5)

        bss_l = self.filter(self.BSs)

        bss_ = [b for b in bss_l if self.filter_border_bss(bs=b)]

        active_ues = []
        for bs in bss_:
            active_ues.extend(bs.served_UEs)

        metrics_ = [self.compute_sinr(bs=self.bs_lookup(ue.bs_id), ue=ue, pairing=False,inner_test = False, outer_test= False) for ue in active_ues]

        obj = np.array([np.sum(self.objective_function(bss=self.filter(bs.sector))) for bs in bss_])

        y = np.sum(obj)/len(active_ues) 

        median_sinr = rounder(np.median(metrics_, axis = 0))
        mean_sinr = rounder(np.mean(metrics_, axis = 0))
        sinr_cdf = np.sort(metrics_)

        self.accumulated_sinr.append(sinr_cdf)
    
        return {'f(x)': rounder(y), 'median_sinr': median_sinr, 'mean_sinr': mean_sinr,'max_sinr': max(sinr_cdf)}


    def objective_function(self, bss: List[BS], ues_test: List[UE] = None):

        """
        Defines the objective function for BSs, calculating the resource scheduling and Shannon capacity for the UEs they serve.
        
        Parameters:
        - bss (List[BS]): List of BSs to calculate the objective function for.
        - ues_test (List[UE], optional): List of test UEs for SINR testing. Defaults to None.
        
        Returns:
        - List[float]: The objective function values for each BS.
        """

        alpha_raiser = lambda lst: [capacity ** ((1-self.alpha_fairness) / self.alpha_fairness)for capacity in lst]

        normalizer = lambda lst: np.array(lst)/np.sum(lst)

        hadamard = lambda scheduling_lst, capacity_lst: [s*c for s,c in zip(scheduling_lst, capacity_lst)]

        def optimal_scheduling(capacity_lst):
            scheduling = None
            if self.alpha_fairness == 0:
                max_index = np.argmax(np.array(capacity_lst))
                scheduling = [0] * len(capacity_lst)
                scheduling[max_index] = 1
            else:
                scheduling = normalizer(alpha_raiser(capacity_lst))

            if scheduling is None:
                raise ValueError("Error - Scheduling in None.")

            return scheduling

        def shannon_capacity(bs, ue):

            if ues_test is not None:
                if ue in ues_test:
                    inner_test = True
                    outer_test = False
                else:
                    inner_test = False
                    outer_test = True
            else:
                inner_test = False
                outer_test = False

            sinr = self.compute_sinr(bs=bs, ue=ue, pairing=False, inner_test=inner_test, outer_test=outer_test)
            
            try:
                shannon = bs.bw * math.log2(1 + sinr)
            except ValueError:
                print(f"ValueError encountered while computing Shannon: sinr = {sinr}")
            
            #Debug - Sanity check
            if shannon == 0.0:
                print('BS_id', bs.get_id(), 'UE_id:', ue.get_id(),'distance:',bs.get_distance(ue), 'sinr', sinr)
            
            return shannon
        
        def obj_f(bs):

            obj = None

            elig_ues = bs.served_UEs

            capacity_lst = [shannon_capacity(bs, ue) for ue in elig_ues]
            scheduler = optimal_scheduling(capacity_lst)

            if len(capacity_lst)==0:
                raise ValueError("Shannon capacity is an empty list - No user to serve.")
            if len(scheduler)==0:
                raise ValueError("Scheduling is an empty list - No user to serve.")
            
            if len(scheduler) != len(capacity_lst):
                raise ValueError(f"Lenght mismatch beween 'scheduler' and 'capacity_lst', must be the same length. scheduler = {scheduler},capacity_lst = {capacity_lst} ")
            
            prod = hadamard(capacity_lst = capacity_lst, scheduling_lst=scheduler)
            if 0 in prod:
                print('capacity_lst', capacity_lst)
                print('scheduler', scheduler)
                print('bs.served_UEs',len(bs.served_UEs))
                print('prod',prod)
                raise ValueError("prod contains zeros.")

            if self.alpha_fairness == 1:
                obj = np.sum(np.log(prod))
            else:
                alpha_c = 1 - self.alpha_fairness
                obj = np.sum(np.power(prod, alpha_c)) / alpha_c
            if obj is None:
                raise ValueError("Error - obj in None.")
            
            return obj/len(bs.sector)
            
        return [obj_f(bs) for bs in bss]


    def set_sectors(self):
        """
        Sets the sector (interference set) for each BS in the network based on their distances.
        """

        for bs in self.BSs:
            bs.sector = [b for b in self.BSs if bs.get_distance(b) < self.bs_max_range]
    
    def init_optimization(self):
        """
        Initializes the optimization process for each BS in the network by setting up their optimizers.
        """
        for bs in self.BSs:
            bs.set_optimizer(self.optimizer_kwargs)
    

    def optimize_power(self, ctime, randomPick:bool = False, const_pick:bool = False):
        """
        Optimizes the power levels for each BS at a given time.
        
        Parameters:
        - ctime (float): The current time at which the optimization is performed.
        - randomPick (bool, optional): If True, selects power levels randomly. Defaults to False.
        - const_pick (bool, optional): If True, sets power levels to constant values. Defaults to False.
        """

        bss_ = self.filter(self.BSs)

        if const_pick:
            for bs in bss_:
                bs.power_adjustment(random_ = False, verbose=self.verbose, power_levels = None, const_pick = True)
    
        elif randomPick:
            for bs in bss_:
                bs.power_adjustment(random_ = True, verbose=self.verbose, power_levels = None, const_pick = False)
        else:

            def mapper(tple):
                """
                Maps the tuple of power levels to the considered bijection.
                """
                db_operator = lambda x_lin: 10*np.log10(x_lin)
                p_out,p_in = tple
                x1 = db_operator(p_out + p_in)
                x2 = p_out / (p_out + p_in)
                return [x1, x2] 
            
            list_mapper = lambda lst_arr: [mapper(tple) for tple in lst_arr]

            # Step 1: Get the next power level for each BS
            next_power_levels = [bs.optimizer.next_query(ctime) for bs in bss_]

            # Step 2: Optimize BS power levels
            for bs in bss_:
                power_levels = [next_power_levels[ix] for ix, bs2 in enumerate(bss_) if bs2 in self.filter(bs.sector)]
                bs.power_adjustment(power_levels=power_levels, verbose=self.verbose, random_ = False, const_pick = False)

            # Step 3: Compute the objective function for each BS
            y_ni = [np.sum(self.objective_function(bss=self.filter(bs.sector))) for bs in bss_]  # bs.sector = N_i
            x_p = list_mapper([bs.get_power_levels() for bs in bss_])  # (p_out, p_in)

            # # Step 4: Run the optimization
            start_t = time()

            # Using ThreadPoolExecutor for tell operations
            with ThreadPoolExecutor(max_workers=2) as executor_t:
                futures_t = [
                    executor_t.submit(bs.optimizer.tell, x_i, ctime, y_i, self.verbose)
                    for x_i, y_i, bs in zip(x_p, y_ni, bss_)
                ]
                for future_t in as_completed(futures_t):
                    try:
                        future_t.result()  # Check if the function raised an exception
                    except Exception as e:
                        print(f"Exception occurred during tell: {e}")

            # Using ThreadPoolExecutor for clean operations
            with ThreadPoolExecutor(max_workers=2) as executor_c:
                futures_c = [
                    executor_c.submit(bs.optimizer.clean, ctime, self.verbose)
                    for bs in bss_
                ]
                for future_c in as_completed(futures_c):
                    try:
                        future_c.result()  # Check if the function raised an exception
                    except Exception as e:
                        print(f"Exception occurred during clean: {e}")

            
            #dump the data in a log file:
            for bs in bss_:
                self.optim_logger.info({
                    'time': ctime,
                    'base station': bs.get_id(),
                    'dataset Size': bs.optimizer.dataset_size(),
                    'spatial lengthscale': bs.optimizer._lS,
                    'temporal lengthscale': bs.optimizer._lT,
                })
                if self.verbose:
                    i = bs.get_id()
                    print(f'Optimizer {i} - Current Time: {ctime}')
                    print(f'Optimizer {i} - Dataset Size: {bs.optimizer.dataset_size()}')
                    print(f'Optimizer {i} - lt: {bs.optimizer._lT}')
                    print(f'Optimizer {i} - ls: {bs.optimizer._lS}')
                    print("\n")
            

            end_t = time()
            t_ = end_t - start_t
            print(f'== Iteration time: {round(t_, 2)}s ==','\n')        

    def build_ues_db(self, ues_traces: list, bss_traces:list, ue_filename: Optional[str] = None,bs_filename: Optional[str] = None):
        """
        Builds and exports a database of UEs and BSs with their current trace information to CSV files.
        
        Parameters:
        - ues_traces (list): A list of UE trace data.
        - bss_traces (list): A list of BS trace data.
        - ue_filename (Optional[str]): The path to the CSV file for UE data export. If None, the data will be stored internally.
        - bs_filename (Optional[str]): The path to the CSV file for BS data export. If None, the data will be stored internally.
        """

        if self.trace_count == 0:
            self.trace_time = datetime.datetime.now()
        else:
            #add 10 seconds to trace_time and convert it to pd.datetime (for gif-based visualization purpose)
            self.trace_time += datetime.timedelta(seconds=10)
            self.trace_time = pd.Timestamp(self.trace_time)

        
        if ue_filename is not None:
            ues_db = pd.DataFrame(ues_traces)
            ues_db.to_csv(ue_filename, index=False)
        else:
            data = [ue.ue_trace(ctime=self.trace_time) for ue in self.UEs]
            ues_traces.extend(data)
        
        #build base station trace
        if bs_filename is not None:
            bss_db = pd.DataFrame(bss_traces)
            bss_db.to_csv(bs_filename, index=False)
        else:
            data = [bs.bs_trace(ctime=self.trace_time) for bs in self.BSs]
            bss_traces.extend(data)
        
        self.trace_count +=1
        