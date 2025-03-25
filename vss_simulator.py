# vss_simulator.py
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
import random


def lcm(a, b):
    """ Calculate the least common multiple (LCM) of two numbers """
    return abs(a * b) // math.gcd(a, b)


def lcm_of_list(numbers):
    """ Calculate the least common multiple of all numbers in the list """
    return reduce(lcm, numbers)


def generate_execution_time_uniform(bcet: int, wcet: int) -> int:
    """
    Generate a random execution time using a uniform distribution (integer values)
    in the interval [bcet, wcet]. This supports BCET = 0.
    """
    if bcet == wcet:
        return bcet
    return random.randint(bcet, wcet)


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
        """Compute the new execution_time for this job instance using uniform distribution."""
        self.execution_time = generate_execution_time_uniform(self.bcet, self.wcet)
        self.remaining_time = self.execution_time

        
    def release_new_job(self, current_time):
        """Release a new job of this task at current_time."""
        self.release_time = current_time
        self.reset_execution_time()
        self.completion_time = None
        self.response_time = None
    
    def is_ready(self, current_time):
        """ Check if the task is ready to execute """
        return current_time >= self.release_time and self.remaining_time > 0
    
    def execute(self, time_units=1):
        """
        Execute the task for a given number of time units.
        Returns True if the task finishes (remaining_time becomes 0).
        """
        self.remaining_time = max(0, self.remaining_time - time_units)
        return self.remaining_time == 0  

    def calculate_response_time(self, finish_time):
        """Calculate response time and update worst-case response time (WCRT)."""
        self.completion_time = finish_time
        self.response_time = self.completion_time - self.release_time
        self.wcrt = max(self.wcrt, self.response_time)
        self.finish_time = finish_time


def load_tasks_from_csv(filename):
    """ Load task list from CSV file with selected execution time method """
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
            priority=row["Priority"]
        )
        # Add the Task object to the list
        tasks.append(task)
    # Return the list of Task objects
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
    Rate Monotonic Scheduling (RMS) simulation.
    如果当前最高优先级任务的 remaining_time == 0，则立即完成并不增加 current_time，
    不会记录到 schedule_log，从而在 Gantt 图中“跳过”0 执行时间的任务。
    """
    current_time = 0
    active_jobs = []
    job_release_times = {task.task_id: 0 for task in tasks}
    schedule_log = []

    while current_time < simulation_time:
        # 1) 释放新的 job
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

        # 2) 按优先级升序排序 (priority值越小优先级越高)
        active_jobs.sort(key=lambda t: t.priority)

        if active_jobs:
            current_job = active_jobs[0]
            # 如果任务的 remaining_time == 0，立即完成并不推进时间
            if current_job.remaining_time == 0:
                current_job.calculate_response_time(current_time)
                active_jobs.remove(current_job)
                msg = (f"[Time {current_time}] Task {current_job.task_id} "
                       f"Completed Immediately (0 exec time), Response: {current_job.response_time}")
                if verbose:
                    print(msg)
                if log_file:
                    log_file.write(msg + "\n")
                # 不记录到 schedule_log，也不增加 current_time
                # 直接继续 while 循环，可能还有别的任务也为0
                continue
            else:
                # 正常执行 1 个 time unit
                finished = current_job.execute(1)
                schedule_log.append((current_time, current_job.task_id))
                msg = (f"[Time {current_time}] Task {current_job.task_id} Running, "
                       f"Remaining: {current_job.remaining_time}")
                if verbose:
                    print(msg)
                if log_file:
                    log_file.write(msg + "\n")

                if finished:
                    finish_t = current_time + 1
                    current_job.calculate_response_time(finish_t)
                    active_jobs.remove(current_job)
                    msg = (f"[Time {finish_t}] Task {current_job.task_id} "
                           f"Completed, Response: {current_job.response_time}")
                    if verbose:
                        print(msg)
                    if log_file:
                        log_file.write(msg + "\n")

                # 只有当我们真正执行了 1 time unit，才推进 current_time
                current_time += 1
        else:
            # 没有活动任务 => idle
            schedule_log.append((current_time, "Idle"))
            # idle时，可能要跳到下一个release时间
            jump = advance_time(current_time, job_release_times, active_jobs)
            current_time += jump
    
    if verbose:
        print("\n=== Worst-Case Response Time (WCRT) ===")
        for task in tasks:
            print(f"Task {task.task_id}: WCRT = {task.wcrt}")

    return schedule_log


def plot_gantt_chart(schedule_log, save_path=None):
    """ Draw a Gantt chart """
    plt.figure(figsize=(10, 5))
    task_colors = {}
    y_pos = {}
    
    unique_tasks = set(entry[1] for entry in schedule_log if entry[1] != "Idle")
    for i, task in enumerate(sorted(unique_tasks)):
        task_colors[task] = plt.colormaps["tab10"](i)
        y_pos[task] = i

    for start_time, task in schedule_log:
        if task != "Idle":
            plt.barh(y_pos[task], 1, left=start_time, color=task_colors[task], edgecolor="black")

    plt.yticks(range(len(y_pos)), sorted(y_pos.keys()))
    # Set the x and y labels
    plt.xlabel("Time")
    plt.ylabel("Tasks")
    # Set the title
    plt.title("Rate Monotonic Schedule - Gantt Chart")
    # Set the grid to only show the x axis
    plt.grid(axis="x")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Gantt chart saved to {save_path}")
    
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
    parser.add_argument("--runs", type=int, default=1, help="Number of simulation runs")
    parser.add_argument("--simtime", type=int, default=None, help="Simulation time (if not provided, use LCM of task periods)")
    parser.add_argument("--verbose", action="store_true", help="Output detailed log to console")
    parser.add_argument("--logfile", action="store_true", help="If set, enable logging to a default file 'sim.log'")
    
    args = parser.parse_args()

    images_dir = "output/images"
    logs_dir = "output/logs"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    original_tasks = load_tasks_from_csv(args.csv_filename)
    
    for task in original_tasks:
        task.execution_time = generate_execution_time_uniform(task.bcet, task.wcet)
    
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