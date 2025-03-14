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
        return f"Task({self.name}, P={self.priority}, C=[{self.bcet}-{self.wcet}], T={self.period}, D={self.deadline})"

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
        sorted_tasks = sorted(tasks, key=lambda x: x.priority)
        wcrt = {}

        for i, task in enumerate(sorted_tasks):
            hp_tasks = sorted_tasks[:i+1]  # Current task + higher priority tasks
            R = task.wcet  # Initial response time
            
            while True:
                prev_R = R
                # Calculate interference from higher priority tasks
                interference = sum(((prev_R + t.period - 1) // t.period) * t.wcet for t in hp_tasks[:i])
                R = task.wcet + interference
                
                if R > task.deadline:
                    return None  # Task is unschedulable
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
        for task in sorted(tasks, key=lambda t: t.name):  # Ensure correct order T1, T2, ...
            wcrt = rta_results[task.name]
            status = "✓" if wcrt <= task.deadline else "✗"
            print(f" {task.name:<4} {wcrt:<4} {task.deadline:<8} {status}")
    else:
        print("System is not schedulable. Some tasks miss their deadlines.")
