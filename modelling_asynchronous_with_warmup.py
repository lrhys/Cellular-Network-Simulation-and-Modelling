# Imports
import numpy as np
import pandas as pd
from scipy.stats import expon, norm
from heapq import heappush, heappop
from tqdm import tqdm
import logging
import concurrent.futures

class VariableGenerator:
    # Generated variables
    @staticmethod
    def generate_car_direction(seed=None):
        if seed:
            np.random.seed(seed)
        # if np.random.uniform() < 0.5:
        #     return -1 # Going left
        # else:
        #     return 1 # Going right
        
        # np.random.choice is uniform by default
        return np.random.choice([-1, 1])

    @staticmethod
    def generate_car_position(cell_diameter_km, seed=None):
        if seed:
            np.random.seed(seed)
        return np.random.uniform(0, cell_diameter_km)

    @staticmethod
    def generate_car_speed(seed=None):
        # Based on fitted parameters from input analysis
        mean = 120.07209801685805
        std = 9.018606933727629

        if seed:
            np.random.seed(seed)
            return norm.rvs(loc=mean, scale=std, random_state=seed)
        
        return norm.rvs(loc=mean, scale=std)
    
    @staticmethod
    def generate_call_interarrival_time(seed=None):
        # Based on fitted parameters from input analysis
        a = 2.5090155759244226e-05
        b = 1.3697918363207653

        if seed:
            np.random.seed(seed)
            return expon.rvs(loc=a, scale=b, random_state=seed)

        return expon.rvs(loc=a, scale=b)
    
    @staticmethod
    def generate_call_duration(seed=None):
        # Based on fitted parameters from input analysis
        a = 10.003951603252272
        b = 99.83194913549542

        if seed:
            np.random.seed(seed)
            return expon.rvs(loc=a, scale=b, random_state=seed)

        return expon.rvs(loc=a, scale=b)

    @staticmethod
    def generate_initial_base_station(no_of_base_stations, seed=None):
        
        if seed:
            np.random.seed(seed)
            return np.random.randint(0, no_of_base_stations - 1)

        return np.random.randint(0, no_of_base_stations - 1)


# Call_Initiation_Event Class
class Call_Initiation:
    def __init__(self, time, base_station, car_position, car_speed, car_direction, call_duration):
        self.time = time
        self.base_station = base_station
        self.car_position = car_position
        self.car_speed = car_speed
        self.car_direction = car_direction
        self.call_duration = call_duration


# Call_Termination_Event Class
class Call_Termination:
    def __init__(self, time, base_station):
        self.time = time
        self.base_station = base_station


# Call_Handover_Event Class
class Call_Handover:
    def __init__(self, time, prev_base_station, base_station, car_speed, car_direction, call_duration):
        self.time = time
        self.prev_base_station = prev_base_station
        self.base_station = base_station
        self.car_speed = car_speed
        self.car_direction = car_direction
        self.call_duration = call_duration


class Simulator:
    def __init__(self):
        logging.info("Initialized new simulator")

    def reset_state(self):
        # Initialize simulation clock, state variables, event list and statistical counters. 
        self.sim_clock = 0

        # Initialize state variables
        self.time_of_last_event = 0
        self.cell_availabilities = [no_of_channels] * no_of_base_stations 

        # Initialize future event list (FEL)
        self.future_event_list = []

        # Initialize statistical counters
        self.total_number_of_calls = 0
        self.dropped_calls = 0
        self.blocked_calls = 0

    def schedule_call_initiation(self):
        # Calculate next arrival time
        time_of_next_call = self.sim_clock + VariableGenerator.generate_call_interarrival_time()

        # Generate variables
        base_station = VariableGenerator.generate_initial_base_station(no_of_base_stations)
        car_position = VariableGenerator.generate_car_position(cell_diameter_km)
        car_speed = VariableGenerator.generate_car_speed()
        car_direction = VariableGenerator.generate_car_direction()
        call_duration = VariableGenerator.generate_call_duration()
        
        # Push event to FEL
        logging.info(f"Scheduling call initation event at time {round(time_of_next_call, 2)}")
        heappush(self.future_event_list, (time_of_next_call, Call_Initiation(time_of_next_call, base_station, car_position, car_speed, car_direction, call_duration)))

    def schedule_call_handover(self, handover_time, prev_base_station, base_station, car_speed, car_direction, remaining_call_duration):
        # Push event to FEL
        logging.info(f"Scheduling call handover event at time {round(handover_time, 2)}")
        heappush(self.future_event_list, (handover_time, Call_Handover(handover_time, prev_base_station, base_station, car_speed, car_direction, remaining_call_duration)))

    def schedule_call_termination(self, terminating_time, terminating_base_station):
        # Push event to FEL
        logging.info(f"Scheduling call terminating event at time {round(terminating_time, 2)}")
        heappush(self.future_event_list, (terminating_time, Call_Termination(terminating_time, terminating_base_station)))

    def process_event(self, event):
        if isinstance(event, Call_Initiation):
            logging.info(f"Processing call initation event at time {round(event.time, 2)}")
            self.handle_call_initiation(event)
        elif isinstance(event, Call_Handover):
            logging.info(f"Processing call handover event at time {round(event.time, 2)}")
            self.handle_call_handover(event)
        elif isinstance(event, Call_Termination):
            logging.info(f"Processing call termination event at time {round(event.time, 2)}")
            self.handle_call_termination(event)
        else:
            raise Exception("Invalid event detected inside simulation...")
 
    def handle_call_initiation(self, event):
        # Schedule next call_initiation event
        self.schedule_call_initiation()

        # Update statistical counters
        self.total_number_of_calls += 1

        if self.cell_availabilities[event.base_station] - no_of_reserved_channels <= 0:
            # Call is blocked
            logging.warning("Call is blocked during initiation")
            self.blocked_calls += 1
        else:
            # Update cell availabilties
            self.cell_availabilities[event.base_station] -= 1

            # Calculate remaining time in cell
            # Assume leftmost point in cell diameter to be 0
            if event.car_direction == 1: 
                # Car is going right
                remaining_time_in_cell = ((cell_diameter_km - event.car_position) / event.car_speed) * 3600
            else:
                # Car is going left
                remaining_time_in_cell = (event.car_position / event.car_speed) * 3600

            if event.call_duration <= remaining_time_in_cell:  
                # Schedule termination
                self.schedule_call_termination(self.sim_clock + event.call_duration, event.base_station)
            else:
                next_base_station = event.base_station + event.car_direction
                handover_time = self.sim_clock + remaining_time_in_cell
                # Check if car crosses the end of highway
                if next_base_station < 0 or next_base_station > 19:
                    # Schedule termination
                    self.schedule_call_termination(handover_time, event.base_station)
                else:
                    # Schedule Handover
                    remaining_call_duration = event.call_duration - remaining_time_in_cell
                    self.schedule_call_handover(handover_time, event.base_station, next_base_station, event.car_speed, event.car_direction, remaining_call_duration)

        self.time_of_last_event = event.time
        logging.info(f"Current cell availabilities: {self.cell_availabilities}")


    def handle_call_handover(self, event):
        # Free up availability at previous base station
        self.cell_availabilities[event.prev_base_station] += 1

        # Check availability of current cell
        if self.cell_availabilities[event.base_station] <= 0:
            # Call is dropped
            logging.warning("Call is dropped during handover")
            self.dropped_calls += 1
        else:
            self.cell_availabilities[event.base_station] -= 1
            # Car enters from either ends of the cell
            remaining_time_in_cell = (cell_diameter_km / event.car_speed) * 3600

            if event.call_duration <= remaining_time_in_cell:
                # Schedule Termination
                self.schedule_call_termination(self.sim_clock + event.call_duration, event.base_station)
            else:
                # calculate the next station
                next_base_station = event.base_station + event.car_direction
                handover_time = self.sim_clock + remaining_time_in_cell
                # Check if car crosses the end of highway
                if next_base_station < 0 or next_base_station > 19:
                    # Schedule termination
                    self.schedule_call_termination(handover_time, event.base_station)
                else:
                    remaining_call_duration = event.call_duration - remaining_time_in_cell
                    self.schedule_call_handover(handover_time, event.base_station, next_base_station, event.car_speed, event.car_direction, remaining_call_duration)

        self.time_of_last_event = event.time
        logging.info(f"Cell availabilities at {round(event.time, 2)}: {self.cell_availabilities}")

    def handle_call_termination(self, event):
        # Free up channel
        self.cell_availabilities[event.base_station] += 1

        # Update statistical counter
        self.time_of_last_event = event.time

    def run_simulation(self, max_steps=100, iterations=1, warmup=0, save_output=False):
        results = []
        detailed_results = []

        # for i in range(1, iterations +  1):
        for i in tqdm(range(1, iterations + 1)):
            logging.info(f"Resetting state for simulation {i}...")
            self.reset_state()
            
            # Run simulation as long as there is events in queue or for predefined number of runs
            logging.info("=================================================================================================================")
            logging.info(f"Running simulation {i}...")
            logging.info("=================================================================================================================")
            # Schedule the first call_initiation event into FEL
            if self.sim_clock == 0:
                self.schedule_call_initiation()
                
            for j in tqdm(range(1, max_steps + 1)):
                if not self.future_event_list:
                    logging.warning("FEL is empty.")
                    break
                
                event = heappop(self.future_event_list)[1]
                self.sim_clock = event.time

                # Reset counters after warmup period
                if j == warmup:
                    self.blocked_calls = 0 
                    self.dropped_calls = 0
                    self.total_number_of_calls = 1

                # Process corresponding event 
                self.process_event(event)
                
                if save_output and j >= warmup:
                    # "Simulation {i} | Step: {j} | Percentage of blocked calls | Percentage of dropped calls"
                    detailed_results.append([i , j, self.total_number_of_calls, self.blocked_calls, self.dropped_calls, (self.blocked_calls / self.total_number_of_calls) * 100, (self.dropped_calls / self.total_number_of_calls) * 100])


            logging.info("=================================================================================================================")
            logging.info(f"End of Simulation {i} Statistical Counters")
            logging.info("=================================================================================================================")
            logging.info(f"Total number of calls: {self.total_number_of_calls}")
            logging.info(f"Total blocked calls: {self.blocked_calls}")
            logging.info(f"Total dropped calls: {self.dropped_calls}")
            logging.info(f"Percentage of blocked calls: {round((self.blocked_calls / self.total_number_of_calls) * 100, 3)}%")
            logging.info(f"Percentage of dropped calls: {round((self.dropped_calls / self.total_number_of_calls) * 100, 3)}%")
            if save_output:
                results.append([(self.blocked_calls / self.total_number_of_calls) * 100, (self.dropped_calls / self.total_number_of_calls) * 100])

        return results, detailed_results

if __name__ == '__main__':
    # Global Defined variables
    no_of_base_stations = 20
    no_of_channels = 10
    cell_diameter_km = 2 
    no_of_reserved_channels = 0

    # Parameters
    steps = 100000
    iterations = 1
    samples = 30
    warmup = 50000
    debug = True

    # Logging setup
    logging.basicConfig(filename=f'simulation_report_{steps}Steps_{iterations}Iterations_{samples}Samples_{no_of_channels}Channels_{no_of_reserved_channels}Reserved_{warmup}Warmup.log', filemode='w', format='%(levelname)s - %(message)s', level=logging.INFO)
    # Disable logging
    logging.disable(logging.CRITICAL)
    # logging.disable(logging.NOTSET)


    def draw_sample(i):
        logging.info(f"=================================================================================================================")
        logging.info(f"                                        Sample {i}                                                               ")
        logging.info(f"=================================================================================================================")
        logging.info(f"----------------------Running simulation for {iterations} iterations with {steps} max_steps----------------------")

        # Initialize new simulation
        simulator = Simulator()
        results, detailed_results = simulator.run_simulation(max_steps=steps, iterations=iterations, warmup=warmup, save_output=True)
        logging.info(f"----------------------Ended simulation successfully----------------------")

        if results:
            df = pd.DataFrame(results, columns=['Percentage of blocked calls', 'Percentage of dropped calls'])
            logging.info(f"----------------------Saving sample results----------------------")
            simulation_results.append([df['Percentage of blocked calls'].mean(), df['Percentage of dropped calls'].mean()])

        if detailed_results and debug:
            # Used for analysing statistical counter values within a sample. Can be a sample of multiple iterations. 
            stats_df = pd.DataFrame(detailed_results, columns=['Iteration No.', 'Step No.', 'Total calls', 'Total blocked calls', 'Total dropped calls', 'Percentage of blocked calls', 'Percentage of dropped calls'])
            stats_df.insert(loc=0, column='Sample No.', value=i)

            # Save simulation stats individually by sample
            # stats_df.to_csv(f'simulation_stats_sample_{i}.csv', index=Non                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         e)

            # Save simulation stats for all samples
            simulation_stats_dfs.append(stats_df)


    simulation_results = []
    simulation_stats_dfs = [] 

    
    # multiprocessing.Process(simulator.run_simulation(max_steps=steps, iterations=iterations, save_output=True)).start()

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=samples)

    for i in tqdm(range(1, samples + 1)):
        pool.submit(draw_sample, i)
        # draw_sample()

    pool.shutdown(wait=True)

    logging.info(f"----------------------Saving simulation results to CSV----------------------")
    df = pd.DataFrame(simulation_results, columns=['Percentage of blocked calls (Sample Average)', 'Percentage of dropped calls (Sample Average)'])
    # df.to_csv(f'simulation_results_{steps}Steps_{iterations}Iterations_{samples}Samples_{no_of_channels}Channels_{no_of_reserved_channels}Reserved.csv', index=None)
    df.to_csv(f'simulation_results_{steps}Steps_{iterations}Iterations_{samples}Samples_{no_of_channels}Channels_{no_of_reserved_channels}Reserved_{warmup}Warmup.csv', mode='a', index=None)
    

    # Save simulation stats for all samples
    if debug:
        simulation_stats_all_samples = pd.concat(simulation_stats_dfs)
        # simulation_stats_all_samples.to_csv(f'simulation_stats_{steps}Steps_{iterations}Iterations_{samples}Samples_{no_of_channels}Channels_{no_of_reserved_channels}Reserved.csv', index=None)
        simulation_stats_all_samples.to_csv(f'simulation_stats_{steps}Steps_{iterations}Iterations_{samples}Samples_{no_of_channels}Channels_{no_of_reserved_channels}Reserved_{warmup}Warmup.csv', mode='a', index=None)