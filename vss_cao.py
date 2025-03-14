import csv
import random
from math import gcd
from functools import reduce

class Task:
    def __init__(self, name, bcet, wcet, period, deadline, priority):
        self.name = name
        self.bcet = bcet
        self.wcet = wcet
        self.period = period
        self.deadline = deadline
        self.priority = priority
        self.worst_response_time = 0
        self.jobs = []

class Job:
    def __init__(self, task, release_time):
        self.task = task
        self.release_time = release_time
        # Randomly generate execution time between BCET and WCET
        self.remaining_time = random.uniform(task.bcet, task.wcet)
        self.initial_time = self.remaining_time
        self.finished = False
        self.response_time = None

def read_tasks_from_csv(filename):
    tasks = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) >= 6 and row[0] != "Task":  # Skip header if present
                task_name = row[0]
                bcet = float(row[1])
                wcet = float(row[2])
                period = int(float(row[3]))  # Convert to int after float conversion
                deadline = float(row[4])
                priority = int(row[5])
                tasks.append(Task(task_name, bcet, wcet, period, deadline, priority))
    return tasks

def generate_jobs(tasks, simulation_time):
    """Generate all jobs for all tasks within the simulation time."""
    all_jobs = []
    
    for task in tasks:
        release_time = 0
        while release_time < simulation_time:
            job = Job(task, release_time)
            task.jobs.append(job)
            all_jobs.append(job)
            release_time += task.period
            
    return all_jobs

def get_ready_jobs(jobs, current_time):
    """Get jobs that have been released but not finished by current_time."""
    ready_jobs = [job for job in jobs if (job.release_time <= current_time) and not job.finished]
    # Sort by priority (lower number means higher priority)
    ready_jobs.sort(key=lambda job: job.task.priority)
    return ready_jobs

def advance_time(jobs, current_time):
    """Simple time advancement - just move forward by 1 time unit."""
    return 1

def calculate_hyperperiod(tasks):
    """Calculate the least common multiple of all task periods."""
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Convert periods to integers for GCD calculation
    periods = [int(task.period) for task in tasks]
    
    if not periods:
        return 0
    
    result = periods[0]
    for period in periods[1:]:
        result = lcm(result, period)
    
    return result

def simulate(tasks, simulation_time):
    jobs = generate_jobs(tasks, simulation_time)
    current_time = 0
    
    while current_time < simulation_time and any(not job.finished for job in jobs):
        ready_jobs = get_ready_jobs(jobs, current_time)
        
        if ready_jobs:
            # Get highest priority job (lowest priority number)
            current_job = ready_jobs[0]
            
            # Advance time
            delta = advance_time(jobs, current_time)
            current_time += delta
            
            # Decrement remaining execution time for the current job
            current_job.remaining_time -= delta
            
            # Check if job is completed
            if current_job.remaining_time <= 0:
                current_job.finished = True
                current_job.response_time = current_time - current_job.release_time
                
                # Update worst-case response time for the task
                if current_job.response_time > current_job.task.worst_response_time:
                    current_job.task.worst_response_time = current_job.response_time
        else:
            # No ready jobs, advance to the next job release
            unreleased_jobs = [job for job in jobs if not job.finished and job.release_time > current_time]
            if unreleased_jobs:
                next_release = min(job.release_time for job in unreleased_jobs)
                current_time = next_release
            else:
                # No more jobs to process
                break
    
    return tasks

def print_results(tasks):
    print("\nTask Simulation Results:")
    print("Task Name\tWorst-Case Response Time")
    print("-" * 40)
    for task in tasks:
        print(f"{task.name}\t\t{task.worst_response_time:.2f}")

def main():
    # Read tasks from CSV file
    csv_file = "C:\\Users\\fhuur\\OneDrive\\DTU\\02225 DRS\\Taskset-Generator-Exercise\\test_examples\\schedulable\\High_Utilization_Unique_Periods_taskset.csv"
  # Change this to your CSV file path
    tasks = read_tasks_from_csv(csv_file)
    
    if not tasks:
        print("No tasks found in the CSV file.")
        return
    
    # Calculate hyperperiod for simulation length suggestion
    hyperperiod = calculate_hyperperiod(tasks)
    suggested_sim_time = hyperperiod * 2  # Run for 2 hyperperiods
    
    print(f"Hyperperiod: {hyperperiod}")
    print(f"Suggested simulation time: {suggested_sim_time}")
    
    # Use the suggested simulation time
    simulation_time = suggested_sim_time
    
    # Run simulation
    tasks = simulate(tasks, simulation_time)
    
    # Print results
    print_results(tasks)

if __name__ == "__main__":
    main()