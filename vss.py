import csv
import argparse
import random
from typing import List, Dict, Optional
from collections import deque

# --------------------------
# Data Model Definitions
# --------------------------
class Task:
    """Represents a periodic task with timing constraints"""
    def __init__(self, name: str, bcet: int, wcet: int, period: int, deadline: int, priority: int):
        """
        Args:
            name: Task identifier
            bcet: Best-case execution time
            wcet: Worst-case execution time
            period: Task period
            deadline: Relative deadline
            priority: Task priority (lower value = higher priority)
        """
        self.name = name
        self.bcet = bcet
        self.wcet = wcet
        self.period = period
        self.deadline = deadline
        self.priority = priority
        self.worst_response_time = 0

    def __repr__(self):
        return f"Task({self.name}, P={self.priority}, C=[{self.bcet}-{self.wcet}], T={self.period}, D={self.deadline})"

class Job:
    """Represents a specific instance of a task"""
    def __init__(self, task: Task, release_time: int, execution_time: int, job_id: int):
        """
        Args:
            task: The task this job belongs to
            release_time: Time when the job is released
            execution_time: Actual execution time for this job instance
            job_id: Unique identifier for this job instance
        """
        self.task = task
        self.release_time = release_time
        self.execution_time = execution_time
        self.remaining_time = execution_time
        self.job_id = job_id
        self.response_time = None
        self.completion_time = None

    def __repr__(self):
        return f"Job({self.task.name}#{self.job_id}, R={self.release_time}, C={self.execution_time})"

# --------------------------
# Very Simple Simulator
# --------------------------
class VerySimpleSimulator:
    """Implements a simple fixed-priority scheduler simulator"""
    
    def __init__(self, tasks: List[Task]):
        """
        Initialize the simulator with a set of tasks
        
        Args:
            tasks: List of Task objects
        """
        self.tasks = sorted(tasks, key=lambda t: t.priority)
        self.jobs = []
        self.current_time = 0
        self.completed_jobs = []
        
    def initialize_jobs(self, simulation_length: int):
        """
        Initialize all jobs for the simulation period
        
        Args:
            simulation_length: Total simulation time
        """
        self.jobs = []
        
        for task in self.tasks:
            job_id = 1
            # Generate all job instances for this task within the simulation period
            for release_time in range(0, simulation_length + 1, task.period):
                # Generate random execution time between BCET and WCET
                execution_time = random.randint(task.bcet, task.wcet)
                job = Job(task, release_time, execution_time, job_id)
                self.jobs.append(job)
                job_id += 1
                
        # Sort jobs by release time and priority
        self.jobs.sort(key=lambda j: (j.release_time, j.task.priority))
        
    def get_ready_jobs(self, current_time: int) -> List[Job]:
        """
        Get all jobs that are ready to execute at the current time
        
        Args:
            current_time: Current simulation time
            
        Returns:
            List of ready jobs
        """
        ready_jobs = []
        for job in self.jobs:
            if job.release_time <= current_time and job.remaining_time > 0:
                ready_jobs.append(job)
        
        # Sort by priority (lower value = higher priority)
        ready_jobs.sort(key=lambda j: j.task.priority)
        return ready_jobs
        
    def advance_time(self, ready_jobs: List[Job]) -> int:
        """
        Determine how much time to advance
        
        Args:
            ready_jobs: List of ready jobs
            
        Returns:
            Time delta to advance
        """
        if ready_jobs:
            # If there are ready jobs, advance by one time unit
            return 1
        else:
            # If no ready jobs, find the next job release time
            next_release_times = [job.release_time for job in self.jobs 
                                if job.release_time > self.current_time and job.remaining_time > 0]
            
            if next_release_times:
                return min(next_release_times) - self.current_time
            else:
                # No future jobs, advance to end of simulation
                return 1  # Just advance by 1 to ensure progress
        
    def simulate(self, simulation_length: int):
        """
        Run the simulation for the specified length
        
        Args:
            simulation_length: Total simulation time
        """
        self.initialize_jobs(simulation_length)
        self.current_time = 0
        self.completed_jobs = []
        
        while self.current_time <= simulation_length and any(job.remaining_time > 0 for job in self.jobs):
            ready_jobs = self.get_ready_jobs(self.current_time)
            
            if ready_jobs:
                current_job = ready_jobs[0]  # Highest priority job
                
                # Advance time
                delta = self.advance_time(ready_jobs)
                self.current_time += delta
                
                # Update job execution
                current_job.remaining_time -= delta
                
                # Check if job completed
                if current_job.remaining_time <= 0:
                    current_job.completion_time = self.current_time
                    current_job.response_time = current_job.completion_time - current_job.release_time
                    
                    # Update worst-case response time for the task
                    if current_job.response_time > current_job.task.worst_response_time:
                        current_job.task.worst_response_time = current_job.response_time
                    
                    self.completed_jobs.append(current_job)
            else:
                # No ready jobs, advance time
                delta = self.advance_time(ready_jobs)
                self.current_time += delta
        
    def print_results(self):
        """Print simulation results"""
        print("\nVery Simple Simulator Results")
        print("-----------------------------")
        
        # Print task-level results
        print("\nTask-level Results:")
        print("Task  WCRT  Deadline  Status")
        print("----  ----  --------  ------")
        
        for task in sorted(self.tasks, key=lambda t: t.name):
            status = "✓" if task.worst_response_time <= task.deadline else "✗"
            print(f" {task.name:<4} {task.worst_response_time:<4} {task.deadline:<8} {status}")
        
        # Print job-level results
        print("\nJob-level Results:")
        print("Job   Release  Completion  Response  Deadline  Status")
        print("----  -------  ----------  --------  --------  ------")
        
        for job in sorted(self.completed_jobs, key=lambda j: (j.task.name, j.job_id)):
            status = "✓" if job.response_time <= job.task.deadline else "✗"
            print(f"{job.task.name}#{job.job_id:<2}  {job.release_time:<7}  {job.completion_time:<10}  {job.response_time:<8}  {job.task.deadline:<8}  {status}")

# --------------------------
# CSV File Handling
# --------------------------
def load_tasks(filename: str) -> List[Task]:
    """
    Load tasks from CSV file
    
    Expected CSV format:
        Task,BCET,WCET,Period,Deadline,Priority
        
    Args:
        filename: Path to CSV file
        
    Returns:
        List of Task objects
    """
    tasks = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            tasks.append(Task(
                name=row['Task'],
                bcet=int(row['BCET']),
                wcet=int(row['WCET']),
                period=int(row['Period']),
                deadline=int(row['Deadline']),
                priority=int(row['Priority'])
            ))
    return tasks

# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Very Simple Simulator (VSS) for Fixed-Priority Scheduling")
    parser.add_argument("input_file", help="CSV file containing task parameters (e.g., exercise-TC1.csv)")
    parser.add_argument("--cycles", type=int, default=100, help="Number of simulation cycles (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for execution time generation (default: 42)")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Load tasks from CSV
    tasks = load_tasks(args.input_file)

    # Create and run simulator
    simulator = VerySimpleSimulator(tasks)
    simulator.simulate(args.cycles)
    
    # Print results
    simulator.print_results()