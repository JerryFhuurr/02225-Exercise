import math
from functools import reduce
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
import numpy as np


def load_tasks_from_csv(filename, method="workload"):
    """ Load task list from CSV file with selected execution time method """
    df = pd.read_csv(filename)
    tasks = []
    for _, row in df.iterrows():
        task = Task(
            task_id=row["Task"],
            bcet=row["BCET"],
            wcet=row["WCET"],
            period=row["Period"],
            deadline=row["Deadline"],
            priority=row["Priority"],
            method=method  # 传递执行时间计算方法
        )
        tasks.append(task)
    return tasks


def lcm(a, b):
    """ Calculate the least common multiple (LCM) of two numbers """
    return abs(a * b) // math.gcd(a, b)


def lcm_of_list(numbers):
    """ Calculate the least common multiple of all numbers in the list """
    return reduce(lcm, numbers)


def assign_alpha(tasks, U_target=None):
    """
    Compute the α (CPU load factor) of the task, which supports two cases:
    1. If 'U_target' is given, normalize by 'T' such that Σ α_i ≈ U_target (may exceed 1).
    2. If 'U_target' is not given, assign 'α_i' randomly, and normalize to ensure that Σ α_i = 1 (does not exceed 100% CPU).

    Parameters:
    - tasks (list): indicates a list of tasks. Each task contains the "period" key.
    - U_target (float, optional): indicates the target CPU usage. The value range is 0,1.

    Back:
    - Updated 'tasks' including' alpha '.
    """
    n = len(tasks)
    
    if U_target:  
        # If 'U_target' is given, 'α' is normalized by 'T'
        total_T = sum(task.period for task in tasks)
        for task in tasks:
            task.alpha = (task.period / total_T) * U_target  # normalization
    else:
        # If 'U_target' is not given, 'α' is randomly assigned and normalized
        alphas = np.random.rand(n)
        alphas = alphas / alphas.sum()  # Normalized so that Σ α_i = 1
        for task, alpha in zip(tasks, alphas):
            task.alpha = alpha
    
    # Output total CPU load
    total_utilization = sum(task.alpha for task in tasks)
    print(f"Total CPU load: Σ α_i = {total_utilization:.2f}")

    return tasks


def generate_execution_time_workload(bcet: int, wcet: int, period: int, alpha: float) -> int:
    """
    Calculate the task execution time 'C' :
    - `C = α * T`
    - Make sure `C ∈ [1, WCET]`
        
    Parameters:
    - period (int): Task period (T)
    - alpha (float): load Factor (CPU Utilization Factor)
    """
    execution_time = alpha * period
    return max(1, min(wcet, round(execution_time)))


def generate_execution_time_truncnorm(bcet: int, wcet: int) -> int:
    """ Generate a random execution time using a truncated normal distribution """
    if bcet == wcet:  
            return max(1, bcet)

    mean = (bcet + wcet) / 2
    std_dev = max((wcet - bcet) / 3, 0.1)  # Make sure std_dev is not 0

    a, b = (bcet - mean) / std_dev, (wcet - mean) / std_dev
    execution_time = round(truncnorm.rvs(a, b, loc=mean, scale=std_dev))

    return max(1, min(wcet, execution_time))


class Task:
    """ Task Class """
    def __init__(self, task_id, bcet, wcet, period, deadline, priority, method="workload"):
        self.task_id = task_id
        self.period = period
        self.deadline = deadline
        self.priority = priority
        self.bcet = bcet
        self.wcet = wcet
        self.method = method  # 选择执行时间计算方法
        self.alpha = 0.0  # 默认 CPU 负载因子
        
        # 计算任务执行时间 C
        if self.method == "workload":
            self.execution_time = generate_execution_time_workload(self.bcet, self.wcet, self.period, self.alpha)
        elif self.method == "truncnorm":
            self.execution_time = generate_execution_time_truncnorm(self.bcet, self.wcet)

        self.release_time = 0                 # task release time
        self.remaining_time = self.execution_time  # task remaining execution time
        self.completion_time = None           # task completion time
        self.response_time = None             # task response time
        self.wcrt = 0

    def release_new_job(self, current_time):
        """ Release a new task instance (new job) """
        self.release_time = current_time
        # 任务执行时间按选择的方法重新计算
        if self.method == "workload":
            self.execution_time = generate_execution_time_workload(self.bcet, self.wcet, self.period, self.alpha)
        elif self.method == "truncnorm":
            self.execution_time = generate_execution_time_truncnorm(self.bcet, self.wcet)

        self.remaining_time = self.execution_time  # Reset remaining time
        self.completion_time = None  # Completion time not yet calculated
        self.response_time = None  # Response time not yet calculated
    
    def is_ready(self, current_time):
        """ Check if the task is ready to execute """
        return current_time >= self.release_time and self.remaining_time > 0
    
    def execute(self, time_units=1):
        """ Execute the task for a given number of time units """
        self.remaining_time = max(0, self.remaining_time - time_units)  # Subtract time units
        return self.remaining_time == 0  # Return True if the task is finished

    def calculate_response_time(self, current_time):
        """ Calculate the response time of the task """
        self.completion_time = current_time
        self.response_time = self.completion_time - self.release_time
        self.wcrt = max(self.wcrt, self.response_time)


def AdvanceTime(current_time, tasks, job_release_times, active_jobs):
    """ Advance time to the next event """
    if active_jobs:
        return 1  # If there are active jobs, advance 1 time unit
    
    future_release_times = [t for t in job_release_times.values() if t > current_time]  # Get future release times
    if future_release_times:
        return min(future_release_times) - current_time  # Advance to the next release time
    
    return 1  # If no future release times, advance 1 time unit


def rate_monotonic_scheduling(tasks, simulation_time):
    """ Rate Monotonic Scheduling Algorithm """
    current_time = 0
    
    active_jobs = []    # Store the active jobs

    job_release_times = {task.task_id: 0 for task in tasks} # Store the release times of each task

    schedule_log = []

    while current_time < simulation_time:
        # --- Step 1: Release new jobs ---
        for task in tasks:
            if current_time == job_release_times[task.task_id]:
                task.release_new_job(current_time)
                active_jobs.append(task)
                job_release_times[task.task_id] += task.period
                print(f"[Time {current_time}] Task {task.task_id} Release，Execution Time {task.execution_time}")

        # --- Step 2: Select the task with the highest priority ---
        active_jobs = sorted(active_jobs, key=lambda t: t.priority)  # Sort tasks by priority
        if active_jobs:
            current_job = active_jobs[0]
            finished = current_job.execute(1)  # Execute the task for 1 time unit
            schedule_log.append((current_time, current_job.task_id))
            
            print(f"[Time {current_time}] Task {current_job.task_id} Running，Remaining Time {current_job.remaining_time}")

            # --- Step 3: Check if the task is finished ---
            if finished:
                current_job.calculate_response_time(current_time + 1)
                active_jobs.remove(current_job)
                print(f"[Time {current_time+1}] Task {current_job.task_id} Completed，Response_Time {current_job.response_time}")

        else:
            # If no active jobs, add an "Idle" entry to the schedule log
            schedule_log.append((current_time, "Idle"))

        # --- Step 4: Advance time to the next event ---
        current_time += AdvanceTime(current_time, tasks, job_release_times, active_jobs)
        
    # -- Step 5: Calculate Worst-Case Response Time (WCRT)
    print("\n=== Worst-Case Response Time (WCRT) ===")

    for task in tasks:
        print(f"Task {task.task_id}: WCRT = {task.wcrt}")

    return schedule_log


def plot_gantt_chart(schedule_log):
    """ Draw a Gantt chart """
    plt.figure(figsize=(10, 5))

    task_colors = {}
    y_pos = {}

    # Generate colors and calculate the task Y-axis position
    unique_tasks = set([entry[1] for entry in schedule_log])
    for i, task in enumerate(sorted(unique_tasks)):
        task_colors[task] = plt.colormaps["tab10"](i)
        y_pos[task] = i

    for start_time, task in schedule_log:
        if task != "Idle":
            plt.barh(y_pos[task], 1, left=start_time, color=task_colors[task], edgecolor="black")

    plt.yticks(range(len(y_pos)), sorted(y_pos.keys()))
    plt.xlabel("Time")
    plt.ylabel("Tasks")
    plt.title("Rate Monotonic Schedule - Gantt Chart")
    plt.grid(axis="x")
    plt.show()


if __name__ == "__main__":
    """ Main Function """
    csv_filename = sys.argv[1]

    # 检查 `U_target` 和 `method`
    U_target = None
    method = "workload"  # 默认使用 `workload` 方法
    for arg in sys.argv[2:]:
        if "U_target=" in arg:
            try:
                U_target = float(arg.split("=")[1])
                if U_target <= 0:
                    print(f"Warning: U_target={U_target} Invalid，Ignored")
                    U_target = None
            except ValueError:
                print("Error: Invalid U_target value, default value used")
                U_target = None
        elif "method=" in arg:
            method = arg.split("=")[1]
            if method not in ["workload", "truncnorm"]:
                print(f"Warning: method={method} Invalid，'workload' by default")
                method = "workload"

    tasks = load_tasks_from_csv(csv_filename, method)

    # 分配 `α`
    tasks = assign_alpha(tasks, U_target)

     # 计算 `C` (执行时间)
    for task in tasks:
        task.alpha = task.alpha  # 任务类中 `alpha` 需要赋值
        if method == "workload":
            task.execution_time = generate_execution_time_workload(
                bcet=task.bcet, wcet=task.wcet, period=task.period, alpha=task.alpha
            )
        elif method == "truncnorm":
            task.execution_time = generate_execution_time_truncnorm(
                bcet=task.bcet, wcet=task.wcet
            )

    periods = [task.period for task in tasks]   #  Get the periods of all tasks
    simulation_time = lcm_of_list(periods) # Calculate the simulation time (LCM of all periods)
    print(f"Simulation Time (LCM of all periods): {simulation_time}")

    schedule_log = rate_monotonic_scheduling(tasks, simulation_time)

    plot_gantt_chart(schedule_log)  # Draw a Gantt chart