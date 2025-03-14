import csv
import argparse
from typing import List, Dict

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

    def __repr__(self):
        # Calculate task-specific utilization (WCET/Period)
        utilization = self.wcet / self.period
        return (
            f"Task({self.name}, BCET={self.bcet}, WCET={self.wcet}, "
            f"Period={self.period}, Deadline={self.deadline}, "
            f"Utilization={utilization:.2f}, Priority={self.priority}, "
        )

# --------------------------
# Scheduling Algorithm Detection
# --------------------------
def detect_scheduling_algorithm(tasks: List[Task]) -> str:
    """Detects if the scheduling algorithm is RM or DM"""
    if all(task.deadline == task.period for task in tasks):
        return "RateMonotonic"
    return "DeadlineMonotonic"

# --------------------------
# Response Time Analysis (RTA)
# --------------------------
class RTAAnalyzer:
    """Implements worst-case response time analysis using iterative method"""
    @staticmethod
    def calculate_wcrt(tasks: List[Task]) -> Dict[str, int]:
        """
        Calculate worst-case response times for all tasks
        
        Args:
            tasks: List of Task objects (must be sorted by priority)
            
        Returns:
            Dictionary mapping task names to WCRT values
        """
        # Sorts tasks by ascending priority (lower value = higher priority)
        sorted_tasks = sorted(tasks, key=lambda x: x.priority)
        wcrt = {}

        for i, task in enumerate(sorted_tasks):
            # Includes current task + higher-priority tasks (hp_tasks[:i] = higher-priority only)
            hp_tasks = sorted_tasks[:i+1]  # Current task + higher priority tasks
            R = task.wcet  # Initial response time
            
            while True:
                prev_R = R
                # Calculates interference using CEIL(R/T_j) via integer math trick
                interference = sum(((prev_R + t.period - 1) // t.period) * t.wcet for t in hp_tasks[:i])
                R = task.wcet + interference

                # Early termination if deadline is missed
                if R > task.deadline:
                    raise ValueError(
                        f"Task {task.name} is not schedulable")  # Raise an exception Task is unschedulable

                # Loop until response time converges
                if R == prev_R:
                    break  # Convergence
            wcrt[task.name] = R
        return wcrt

# --------------------------
# Utilization Calculation
# --------------------------
def calculate_utilization(tasks: List[Task]) -> float:
    """Calculates total CPU utilization"""
    return sum(task.wcet / task.period for task in tasks)

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
    parser = argparse.ArgumentParser(description="Fixed-Priority Scheduling Analysis Tool")
    parser.add_argument("input_file", help="CSV file containing task parameters (e.g., exercise-TC1.csv)")
    args = parser.parse_args()

    # Load tasks from CSV
    tasks = load_tasks(args.input_file)

    # Detect scheduling algorithm
    scheduling_algorithm = detect_scheduling_algorithm(tasks)
    
    # Compute hyperperiod (LCM of task periods)
    hyperperiod = max(task.period for task in tasks)

    # Compute utilization
    utilization = calculate_utilization(tasks)

    # Run RTA analysis
    rta_results = RTAAnalyzer.calculate_wcrt(tasks)
    schedulable = rta_results is not None

    # ----------------------------------
    # Added TaskSet Printout
    # ----------------------------------
    print("\nTaskSet:")
    print(f"Hyperperiod = {hyperperiod}")
    print(f"CPU Worst Case Utilization = {utilization:.2f}")
    for task in tasks:
        print(task)
    # ----------------------------------

    # Print Analysis Results
    print("\nResponse Time Analysis")
    print(f"  Scheduling Algorithm: {scheduling_algorithm}")
    print(f"  Schedulable: {'True' if schedulable else 'False'}")
    print(f"  Hyperperiod: {hyperperiod}")
    print(f"  Utilization: {utilization:.2f}")
    print(f"  Status: {'✓=schedulable, ✗=not schedulable'}\n")

    if schedulable:
        print("Task  WCRT  Deadline  Status")
        print("----  ----  --------  ------")
        for task in sorted(tasks, key=lambda t: t.name):
            wcrt = rta_results[task.name]
            status = "✓" if wcrt <= task.deadline else "✗"
            print(f" {task.name:<4} {wcrt:<4} {task.deadline:<8} {status}")
        print("----  ----  --------  ------")
    else:
        print("System is not schedulable. Some tasks miss their deadlines.")