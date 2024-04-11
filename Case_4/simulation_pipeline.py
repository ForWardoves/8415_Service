import math
import ciw
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

class queue:
    '''
    A queue system class. 
    '''
    def __init__(self, lbda: float, mu: float, c: float, mode = str, tandem_queue: bool = False,
                cv_a: float = None,  cv_s: float = None, unit: str = 'years'): 
        '''
        Initialize necessary parameter for a new queue system class.
        \n==INPUT==
        `lbda`: `float`
            𝜆: Arrival Rate
        `mu`: `float`
            µ: Service Rate
        `c`: `float`
            Number of servers
        `mode`: `str`
            Queue system configuration
        `tandem_queue`: `bool`, default `False`
            Whether the servers are sequentially aligned where a user must finish one queue before joining another. 
        `cv_a`, `cv_s`: `float`, default `None`
            Exclusive for `mode = 'G/G'. Coefficient of variation for arrival rate and service rate.
        `unit`: `str`, default `years`
            Unit for annotation usage only. 
        '''
        # Input check and conditional properties
        self.mode = mode
        if self.mode not in ['M/M', 'G/G']:
            raise ValueError(f'{mode} queue system not currently supported. Queue system can be either "M/M" or "G/G". ')
        elif self.mode == 'G/G':
            if cv_s is None or cv_a is None: 
                raise ValueError ('Coefficient of Variations cv_s and cv_s must not be none when mode is G/G')
            else: 
                self.cvs = cv_s
                self.cva = cv_a
        # Universal properties
        self.lbda = lbda
        self.mu = mu
        self.c = c
        self.tandem = tandem_queue
        self.verbose = 0
        self.unit = unit

    def factorial(self, x):
        if (x == 1) or (x==0):
            return 1
        else:
            # recursive call to the function
            return (x * self.factorial(x-1)) 

    def user_defined_sum(self, c,rho):
        sum = 0
        for n in range(c):
            sum += (c*rho)**n/self.factorial(n) 

        return 1/(sum + (c*rho)**c/(self.factorial(c)*(1-rho)))

    def get_params(self):
        '''
        Get the estimated steady state params of the new queue system using steady state analytical formulas. 
        '''
        if self.mode == 'M/M': 
            self.rho = self.lbda/(self.c * self.mu)                       # Utilization rate
            self.p0 = self.user_defined_sum(self.c, self.rho)             # Empty queue probability
            self.lq = (self.c * self.rho) ** self.c * self.p0 * self.rho\
                    / (self.factorial(self.c) * (1-self.rho)**2)          # Length of queue
            self.l = self.lq + self.lbda/self.mu                          # Length of system
            self.wq = self.lq / self.lbda                                 # time in queue
            self.w = self.l / self.lbda                                   # time in system
        elif self.mode == 'G/G': 
            # Only in G-G Model: 
            self.s = 1 / self.mu # Average service time
            self.a = 1 / self.lbda # Inter-arrival time
            self.cvai = 1 / self.cva # CV of inter-arrival time
            self.cvsi = 1 / self.cvs # CV of service time
            # Other features: 
            self.rho = self.s / (self.c * self.a)          # Utilization rate
            self.wq = self.s / self.c * (self.cvai**2 + self.cvsi**2)/2 * self.rho ** (-1+math.sqrt(2*(self.c+1)))/(1-self.rho) # Time in queue
            self.w = self.wq + self.s # time in system
            self.lq = self.wq * self.lbda # length of queue
            self.l = self.w * self.lbda # length of system
        if self.tandem == True: 
            self.wq = self.wq * 2
            self.w = self.w * 2

        if self.verbose ==1: 
            print(f'Queue system configuration: {self.mode}, arrival rate: {self.lbda:.2f}, service rate: {self.mu:.2f}, total {self.c} server(s).')
            print(f'Is current queue tandem: {self.tandem}')
            print(f'Utilization rate: {self.rho:.3f}')
            print(f'Length of queue: {self.lq:.3f}, length of system: {self.l:.3f}')
            print(f'Time in queue: {self.wq:.3f} {self.unit}, time in system: {self.w:.3f} {self.unit}')
            print('------------------------------------------------------------------------------------------------------------')
    
    def discrete_sim_exp(self, seed:int = 0): 
        '''
        Create a discrete event simulation (`self.sim`) using the given network with exponential simulation on both arrival and service distributions.
        \r Note that this will overwrite any previously-establish `self.sim` instances. 
        \n==INPUT==
        `seed`: `int`, default `0`
            Random state seed to be passed to `ciw.seed()`
        '''
        # Define simulation network: 
        N = ciw.create_network(
            arrival_distributions=[ciw.dists.Exponential(rate=self.lbda)],
            service_distributions=[ciw.dists.Exponential(rate=self.mu)],
            number_of_servers=[self.c])
        ciw.seed(seed)
        # Set up simulation:
        Q = ciw.Simulation(N, tracker=ciw.trackers.SystemPopulation())
        self.sim = Q
        self.sim_type = 'exponential'
    
    def discrete_sim_gamma(self, seed:int = 0): 
        '''
        Create a discrete event simulation (`self.sim`) using the given network with gamma simulation on both arrival and service distributions. 
        \r Note that this will overwrite any previously-establish `self.sim` instances. 
            \r **Only works for G/G/c queue instance!**
        \n==INPUT==
        `seed`: `int`, default `0`
            Random state seed to be passed to `ciw.seed()`
        '''
        # Check for queue instance mode: 
        if self.mode != 'G/G':
            raise ValueError('gamma discrete simulation cannot be initiated under the current instance. Queue system must be created with G/G mode.')
        # Calculate gamma distribution parameters:
        #### Arrival: 
        self.alpha_arrival = (1/self.cva)**2
        self.beta_arrival = self.alpha_arrival / self.lbda
        #### Service:
        self.alpha_service = (1/self.cvs)**2
        self.beta_service = self.alpha_service / self.mu
        # Define simulation network: 
        N = ciw.create_network(
            arrival_distributions=[ciw.dists.Gamma(shapes = self.alpha_arrival, scale = self.beta_arrival)],
            service_distributions=[ciw.dists.Gamma(shapes = self.alpha_service, scale = self.beta_service)],
            number_of_servers=[self.c])
        ciw.seed(seed)
        # Set up simulation:
        Q = ciw.Simulation(N, tracker=ciw.trackers.SystemPopulation())
        self.sim = Q
        self.sim_type = 'gamma'
    
    def get_sim_record(self, stop_criteria: str = 'customer', stop_count: int = 100000): 
        '''
        Run `self.sim` for one time and generate event records `self.sim_record`.
        \n==INPUT==
        `stop_criteria`: `str`, default `customer`
            Simulation stop criteria. Can be:
            \t > `customer` (stopped when `stop_count` number of customers has reached the exit node)
            \r \t > `time` (stop when simulation time has reached `stop_count`)
        `stop_count`: `int`, default `100000`
            Quantity threshold of stop criteria.
        '''
        # Run simulation:
        if stop_criteria == 'customer':
            self.sim.simulate_until_max_customers(stop_count, progress_bar=True)
        elif stop_criteria == 'time': 
            self.sim.simulate_until_max_time(stop_count, progress_bar=True)
        else: 
            raise ValueError('Stop criteria must be either "customer" or "time". ')
        # Fetch simulation records
        recs = self.sim.get_all_records()
        df = pd.DataFrame(recs)
        df.sort_values(by='arrival_date',inplace=True)
        df['inter_arrival'] = df.arrival_date - df.arrival_date.shift(1,fill_value=0)
        df['system_time'] = df.exit_date - df.arrival_date
        df = df[['id_number',
                 'server_id',
                 'arrival_date',
                 'waiting_time',
                 'system_time',
                 'service_start_date',
                 'server_id',
                 'service_time',
                 'service_end_date',
                 'exit_date',
                 'queue_size_at_arrival',
                 'queue_size_at_departure']]
        # Upload to class instance: 
        self.sim_record = df
    
    def get_sim_params(self, df:pd.DataFrame = None): 
        '''
        Print out calculate necessary simulation params.
        \nGenerate the following quantities: 
        \n > `self.sim_wq` Average time in queue; 
        \n > `self.sim_w` Average time; 
        \n > `self.sim_util_rate` Average utilization rate of all servers.
        '''
        if df is None: 
            df = self.sim_record
            self.sim_wq = round(df['waiting_time'].mean(),3)
            self.sim_w = round(df['system_time'].mean(),3)
            self.sim_util_rate = [srv.utilisation for srv in self.sim.nodes[1].servers]
        else: 
            # Only used for map_wait_time
            sim_wq = round(df['waiting_time'].mean(),3)
            return sim_wq
        if self.verbose == 1: 
            print(f'{self.sim_type} Simulation result: ')
            print(f'Time in queue: {self.sim_wq:.3f}, time in system: {self.sim_w:.3f}')
            print(f'Server utilization rate: {[round(rate,3) for rate in self.sim_util_rate]}')
            print('------------------------------------------------------------------------------------------------------------')
    
    
    def get_steady_state(self, start_sim_time: int, sim_time_increments:int, steps: int, 
                          plot:bool = True, show_plot:bool = True):
        '''
        Used to find the minimal simulation time to reach steady state in terms of average wait time.
        \n==INPUT==
        `start_sim_time`: `int`
            Simulation time to start the iteration from
        `sim_time_increments`: `int`
            Increment in simulation time per iteration
        `steps`: `int`
            Number of iterations
        `plot`: `bool`, default `True`
            Whether to plot the result as a wait_time vs. simulation time chart (stored as `self.sim_steady_state_plot`)
        `show_plot`: bool, default `True`
            Whether to show `self.sim_steady_state_plot` immediately after this method
        '''
        self.sim_avg_queue_time = {}
        for time in tqdm(np.arange(start_sim_time, sim_time_increments*steps+start_sim_time, sim_time_increments), desc = 'Iterating day scenarios..', total=steps):
            self.sim.simulate_until_max_time(time)
            recs = self.sim.get_all_records()
            mwp_df = pd.DataFrame(recs)
            wait_time = self.get_sim_params(df=mwp_df)
            self.sim_avg_queue_time[time] = wait_time
        if plot is True:
            fig,ax = plt.subplots(figsize=(12,4))
            ax.plot(list(self.sim_avg_queue_time.keys()), list(self.sim_avg_queue_time.values()))
            ax.set_ylabel("Average Wait Time ", fontsize=14)
            ax.set_xlabel("Simultion Stop Time (yrs)", fontsize=14)
            self.sim_steady_state_plot = fig
            #plt.legend(bbox_to_anchor=(1.6, .5))
            plt.title(f'Steady State Behavior of Avg. Wait Time - sim under {self.sim_type} dist.')
            if show_plot is True:
                plt.show() 
