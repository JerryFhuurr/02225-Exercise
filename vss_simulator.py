
import math
from functools import reduce
import sys
import copy
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import numpy as np
import argparse
import logging
import statistics
import logging
import os


def lcm(a, b):
    """ Calculate the least common multiple (LCM) of two numbers """
    # Calculate the absolute value of the product of the two numbers
    # Divide the absolute value of the product by the greatest common divisor (GCD) of the two numbers
    return abs(a * b) // math.gcd(a, b)


def lcm_of_list(numbers):
    """ Calculate the least common multiple of all numbers in the list """
    return reduce(lcm, numbers)


def generate_execution_time_workload(bcet: int, wcet: int, period: int, alpha: float) -> int:
    """
    Calculate the task execution time 'C' :
    - `C = α * T`
    - Make sure `C ∈ [1, WCET]`
        
    Parameters:
    - period (int): Task period (T)
    - alpha (float): load Factor (CPU Utilization Factor)
    """
    # Calculate the execution time based on the load factor and period
    execution_time = alpha * period
    # Return the execution time, ensuring it is within the range of 1 and WCET
    return max(1, min(wcet, round(execution_time)))


def generate_execution_time_truncnorm(bcet: int, wcet: int) -> int:
    """ Generate a random execution time using a truncated normal distribution """
    # If the best case execution time (bcet) is equal to the worst case execution time (wcet), return the bcet
    if bcet == wcet:  
            return max(1, bcet)
    # Calculate the mean of the truncated normal distribution
    mean = (bcet + wcet) / 2
    # Calculate the standard deviation of the truncated normal distribution
    std_dev = max((wcet - bcet) / 3, 0.1)  
    # Calculate the lower and upper bounds of the truncated normal distribution
    a, b = (bcet - mean) / std_dev, (wcet - mean) / std_dev
    # Generate a random execution time using the truncated normal distribution
    execution_time = round(truncnorm.rvs(a, b, loc=mean, scale=std_dev))
    # Return the execution time, ensuring it is between 1 and the wcet
    return max(1, min(wcet, execution_time))


class Task:
    """ Task Class """
    def __init__(self, task_id, bcet, wcet, period, deadline, priority, method="workload"):
        # Initialize the task with the given parameters
        self.task_id = task_id
        self.period = period
        self.deadline = deadline
        self.priority = priority
        self.bcet = bcet
        self.wcet = wcet
        self.method = method  
        self.alpha = 0.0  

        # Initialize the task's execution time, release time, remaining time, completion time, response time, wcrt, and finish time
        self.execution_time = 0
        self.release_time = 0                 
        self.remaining_time = 0  
        self.completion_time = None           
        self.response_time = None             
        self.wcrt = 0
        self.finish_time = 0  

        # Reset the execution time of the task
        self.reset_execution_time()

    def reset_execution_time(self):
        """Compute the (new) execution_time for this job instance."""
        # If the method is "workload", generate the execution time using the workload method
        if self.method == "workload":
            self.execution_time = generate_execution_time_workload(self.bcet, self.wcet, self.period, self.alpha)
        # If the method is "truncnorm", generate the execution time using the truncnorm method
        elif self.method == "truncnorm":
            self.execution_time = generate_execution_time_truncnorm(self.bcet, self.wcet)
        # Set the remaining time to the execution time
        self.remaining_time = self.execution_time

    def release_new_job(self, current_time):
        """Release a new job of this task at current_time."""
        # Set the release time of the new job to the current time
        self.release_time = current_time
        # Reset the execution time of the new job
        self.reset_execution_time()
        # Set the completion time of the new job to None
        self.completion_time = None
        # Set the response time of the new job to None
        self.response_time = None
    
    def is_ready(self, current_time):
        """ Check if the task is ready to execute """
        # Check if the current time is greater than or equal to the release time and the remaining time is greater than 0
        return current_time >= self.release_time and self.remaining_time > 0
    
    def execute(self, time_units=1):
        """ Execute the task for a given number of time units """
        # Subtract the given number of time units from the remaining time
        self.remaining_time = max(0, self.remaining_time - time_units)  
        # Return True if the remaining time is 0, otherwise return False
        return self.remaining_time == 0  

    def calculate_response_time(self, finish_time):
        """Calculate response time = finish_time - release_time, and update wcrt."""
        # Set the completion time to the finish time
        self.completion_time = finish_time
        # Calculate the response time by subtracting the release time from the finish time
        self.response_time = self.completion_time - self.release_time
        # Update the worst-case response time (wcrt) if the current response time is greater
        self.wcrt = max(self.wcrt, self.response_time)
        # Set the finish time to the finish time
        self.finish_time = finish_time  


def load_tasks_from_csv(filename, method="workload"):
    """ Load task list from CSV file with selected execution time method """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)
    tasks = []
    # Iterate through each row in the DataFrame
    for _, row in df.iterrows():
        # Create a Task object for each row
        task = Task(
            task_id=row["Task"],
            bcet=row["BCET"],
            wcet=row["WCET"],
            period=row["Period"],
            deadline=row["Deadline"],
            priority=row["Priority"],
            method=method  
        )
        # Add the Task object to the list
        tasks.append(task)
    # Return the list of Task objects
    return tasks


def assign_alpha(tasks, U_target=None, method="workload"):
    # Check if the method is not "workload"
    if method != "workload":
        
        return tasks
    
    n = len(tasks)
    # Check if U_target is not None
    if U_target is not None:  
        
        total_T = sum(task.period for task in tasks)
        # Calculate the alpha value for each task
        for task in tasks:
            task.alpha = (task.period / total_T) * U_target  
    else:
        
        # Generate random alpha values for each task
        alphas = np.random.rand(n)
        # Normalize the alpha values
        alphas = alphas / alphas.sum()  
        # Assign the alpha values to each task
        for task, alpha in zip(tasks, alphas):
            task.alpha = alpha
    
        # Print the total CPU load
        print(f"Total CPU load: Σα = {sum(task.alpha for task in tasks):.2f}")
    return tasks


def advance_time(current_time, job_release_times, active_jobs):
    """ Advance time to the next event """
    # If there are active jobs, return 1
    if active_jobs:
        return 1
    # Get the future release times that are greater than the current time
    future = [t for t in job_release_times.values() if t > current_time]
    # Return the difference between the minimum future release time and the current time
    return min(future) - current_time if future else 1


def rate_monotonic_scheduling(tasks, simulation_time, verbose=False, log_file=None):
    """ 
    Rate Monotonic Scheduling 
    """
    # Initialize current time, active jobs, job release times, and schedule log
    current_time = 0
    active_jobs = []
    job_release_times = {task.task_id: 0 for task in tasks}
    schedule_log = []

    # Loop through the simulation time
    while current_time < simulation_time:
        
        # For each task, if the current time is equal to the job release time, release the job
        for task in tasks:
            if current_time == job_release_times[task.task_id]:
                task.release_new_job(current_time)
                active_jobs.append(task)
                job_release_times[task.task_id] += task.period
                msg = f"[Time {current_time}] Task {task.task_id} Released, ExecTime: {task.execution_time}"
                if verbose:
                    print(msg)
                if log_file:
                    log_file.write(msg + "\n")
        
        # Sort the active jobs by priority
        active_jobs.sort(key=lambda t: t.priority)
        # If there are active jobs, execute the highest priority job
        if active_jobs:
            current_job = active_jobs[0]
            finished = current_job.execute(1)
            schedule_log.append((current_time, current_job.task_id))
            msg = f"[Time {current_time}] Task {current_job.task_id} Running, Remaining: {current_job.remaining_time}"
            if verbose:
                print(msg)
            if log_file:
                log_file.write(msg + "\n")
            # If the job is finished, calculate the response time and remove it from the active jobs
            if finished:
                
                finish_t = current_time + 1
                current_job.calculate_response_time(finish_t)
                
                current_job.finish_time = finish_t
                current_job.calculate_response_time(current_time + 1)
                active_jobs.remove(current_job)
                msg = f"[Time {current_time+1}] Task {current_job.task_id} Completed, Response: {current_job.response_time}"
                if verbose:
                    print(msg)
                if log_file:
                    log_file.write(msg + "\n")
        # If there are no active jobs, add an idle entry to the schedule log
        else:
            schedule_log.append((current_time, "Idle"))
        # Advance the current time
        current_time += advance_time(current_time, job_release_times, active_jobs)
    
    # If verbose is True, print the worst-case response time for each task
    if verbose:
        print("\n=== Worst-Case Response Time (WCRT) ===")
        for task in tasks:
            print(f"Task {task.task_id}: WCRT = {task.wcrt}")
    # Return the schedule log
    return schedule_log


def plot_gantt_chart(schedule_log, save_path=None):
    """ Draw a Gantt chart """
    # Create a figure with a size of 10x5
    plt.figure(figsize=(10, 5))
    # Create a dictionary to store the colors of each task
    task_colors = {}
    # Create a dictionary to store the y position of each task
    y_pos = {}
    
    # Get a set of unique tasks from the schedule log, excluding "Idle"
    unique_tasks = set(entry[1] for entry in schedule_log if entry[1] != "Idle")
    # Loop through the unique tasks and assign a color and y position to each
    for i, task in enumerate(sorted(unique_tasks)):
        task_colors[task] = plt.colormaps["tab10"](i)
        y_pos[task] = i
    # Loop through the schedule log and plot a bar for each task
    for start_time, task in schedule_log:
        if task != "Idle":
            plt.barh(y_pos[task], 1, left=start_time, color=task_colors[task], edgecolor="black")
    # Set the y ticks to the sorted keys of the y_pos dictionary
    plt.yticks(range(len(y_pos)), sorted(y_pos.keys()))
    # Set the x and y labels
    plt.xlabel("Time")
    plt.ylabel("Tasks")
    # Set the title
    plt.title("Rate Monotonic Schedule - Gantt Chart")
    # Set the grid to only show the x axis
    plt.grid(axis="x")
    
    # If a save path is provided, save the plot to that path
    if save_path:
        plt.savefig(save_path)
        print(f"Gantt chart saved to {save_path}")
    
    # Show the plot
    plt.show()


def run_single_simulation(tasks, simulation_time, verbose=False, log_file=None):
    # Run a single simulation of the given tasks for the given simulation time
    rate_monotonic_scheduling(tasks, simulation_time, verbose=verbose, log_file=log_file)
    # Create an empty dictionary to store the results
    result = {}
    # Iterate through each task
    for t in tasks:
        
        # Store the worst-case response time and finish time of each task in the result dictionary
        result[t.task_id] = (t.wcrt, t.finish_time)
    # Return the result dictionary
    return result


def run_multiple_simulations(original_tasks, simulation_time, num_runs=10, verbose=False, log_filename=None):
    
    # Create a dictionary to store the results of each task
    results = {task.task_id: [] for task in original_tasks}

    # Run the simulation multiple times
    for run in range(num_runs):
        if verbose:
            print(f"\n=== Run {run+1} ===")
        
        
        # If a log filename is provided, open the file and write the run number
        if log_filename:
            with open(log_filename, "w") as log_file:
                if verbose:
                    log_file.write(f"=== Run {run+1} ===\n")
                # Create a copy of the original tasks to run the simulation
                tasks_copy = copy.deepcopy(original_tasks)
                # Run the single simulation and store the results
                run_result = run_single_simulation(tasks_copy, simulation_time, verbose=verbose, log_file=log_file)
        else:
            # Create a copy of the original tasks to run the simulation
            tasks_copy = copy.deepcopy(original_tasks)
            # Run the single simulation and store the results
            run_result = run_single_simulation(tasks_copy, simulation_time, verbose=verbose, log_file=None)
        
        # Store the results of each task in the results dictionary
        for task_id, (wcrt, finish_time) in run_result.items():
            results[task_id].append(wcrt)
    
    
    # Calculate the statistics for each task
    stats = {}
    for task_id, wcrt_list in results.items():
        avg = sum(wcrt_list) / len(wcrt_list)
        var = sum((x - avg) ** 2 for x in wcrt_list) / len(wcrt_list)
        median = statistics.median(wcrt_list)
        percentile_95 = np.percentile(wcrt_list, 95)
        stats[task_id] = {
            "average": avg,
            "variance": var,
            "median": median,
            "max": max(wcrt_list),
            "95th": percentile_95
        }
    return stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSS Simulator with extended statistics and logging")
    parser.add_argument("csv_filename", help="CSV file containing task parameters")
    parser.add_argument("--U_target", type=float, default=None, help="Target CPU utilization (0,1)")
    parser.add_argument("--method", choices=["workload", "truncnorm"], default="workload", help="Execution time generation method")
    parser.add_argument("--runs", type=int, default=1, help="Number of simulation runs")
    parser.add_argument("--simtime", type=int, default=None, help="Simulation time (if not provided, use LCM of task periods)")
    parser.add_argument("--verbose", action="store_true", help="Output detailed log to console")
    
    parser.add_argument("--logfile", action="store_true", help="If set, enable logging to a default file 'sim.log'")
    
    args = parser.parse_args()

    images_dir = "output/images"
    logs_dir = "output/logs"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    original_tasks = load_tasks_from_csv(args.csv_filename, args.method)
    assign_alpha(original_tasks, args.U_target, method=args.method)
    
    
    for task in original_tasks:
        if args.method == "workload":
            task.execution_time = generate_execution_time_workload(
                task.bcet, task.wcet, task.period, task.alpha
            )
        else:
            task.execution_time = generate_execution_time_truncnorm(
                task.bcet, task.wcet
            )
    
    
    if args.simtime:
        simulation_time = args.simtime
    else:
        simulation_time = lcm_of_list([task.period for task in original_tasks])
    print(f"Simulation Time: {simulation_time}")
    
    
    log_file_path = None
    if args.logfile:
        
        log_file_path = os.path.join(logs_dir, "sim.log")

    
    if args.runs == 1:
        schedule_log = rate_monotonic_scheduling(
            original_tasks, simulation_time,
            verbose=args.verbose,
            log_file=(open(log_file_path, "w") if log_file_path else None)
        )
        
        gantt_path = os.path.join(images_dir, "gantt_chart.png")
        plot_gantt_chart(schedule_log, save_path=gantt_path)
    else:
        
        stats = run_multiple_simulations(
            original_tasks,
            simulation_time,
            num_runs=args.runs,
            verbose=args.verbose,
            log_filename=log_file_path
        )
        print("\n=== Simulation Statistics ===")
        for task_id, stat in stats.items():
            print(
                f"Task {task_id}: Average WCRT = {stat['average']:.2f}, "
                f"Median = {stat['median']}, Variance = {stat['variance']:.2f}, "
                f"95th Percentile = {stat['95th']}, Max WCRT = {stat['max']}"
            )