import math
from functools import reduce
import sys
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


def load_tasks_from_csv(filename):
    """ Load task list from CSV file """

    df = pd.read_csv(filename)
    tasks = []
    for _, row in df.iterrows():
        task = Task(
            task_id=row["Task"],
            bcet=row["BCET"],
            wcet=row["WCET"],
            period=row["Period"],
            deadline=row["Deadline"],
            priority=row["Priority"]
        )
        tasks.append(task)
    return tasks


def lcm(a, b):
    """ Calculate the least common multiple (LCM) of two numbers """
    return abs(a * b) // math.gcd(a, b)


def lcm_of_list(numbers):
    """ Calculate the least common multiple of all numbers in the list """
    return reduce(lcm, numbers)


def generate_execution_time(bcet: int, wcet: int) -> int:
    """ 在 [BCET, WCET] 之间生成任务执行时间 C """

    if bcet == wcet:  
            return max(1, bcet)

    mean = (bcet + wcet) / 2
    std_dev = max((wcet - bcet) / 3, 0.1)  # Make sure std_dev is not 0

    a, b = (bcet - mean) / std_dev, (wcet - mean) / std_dev
    execution_time = round(truncnorm.rvs(a, b, loc=mean, scale=std_dev))

    return max(1, execution_time)  # Ensure that the execution time is at least 1

class Task:
    """ Task Class """

    def __init__(self, task_id, bcet, wcet, period, deadline, priority):
        self.task_id = task_id
        self.period = period
        self.deadline = deadline
        self.priority = priority
        self.bcet = bcet
        self.wcet = wcet
        self.execution_time = generate_execution_time(bcet, wcet)  # Generate task execution time C
        self.release_time = 0                 # task release time
        self.remaining_time = self.execution_time  # task remaining execution time
        self.completion_time = None           # task completion time
        self.response_time = None             # task response time
        self.wcrt = 0

    def release_new_job(self, current_time):
        """ Release a new task instance (new job) """

        self.release_time = current_time
        self.execution_time = generate_execution_time(self.bcet, self.wcet)  # Generate a new execution time
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

    tasks = load_tasks_from_csv(csv_filename)

    periods = [task.period for task in tasks]   #  Get the periods of all tasks
    simulation_time = lcm_of_list(periods) # Calculate the simulation time (LCM of all periods)
    print(f"Simulation Time (LCM of all periods): {simulation_time}")

    schedule_log = rate_monotonic_scheduling(tasks, simulation_time)

    plot_gantt_chart(schedule_log)  # Draw a Gantt chart
