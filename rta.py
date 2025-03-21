import csv
import argparse
import math
from functools import reduce
from typing import List, Dict

class Task:
    """Represents a periodic task with timing constraints"""
    def __init__(self, name: str, bcet: int, wcet: int, period: int, deadline: int, priority: int):
        # Task parameters initialization
        self.name = name  # Task identifier
        self.bcet = bcet  # Best-case execution time
        self.wcet = wcet  # Worst-case execution time
        self.period = period  # Task period
        self.deadline = deadline  # Relative deadline
        self.priority = priority  # Priority (lower value = higher priority)

    def __repr__(self):
        # 让任务打印时带上更多信息，如标准答案所示 | Create detailed string representation
        utilization = self.wcet / self.period
        return (f"Task({self.name}, BCET={self.bcet}, WCET={self.wcet}, "
                f"Period={self.period}, Deadline={self.deadline}, Utilization={utilization:.2f} "
                f"Core=0, Priority={self.priority}, Type=TT, MIT=0, Server=None)")

def detect_scheduling_algorithm(tasks: List[Task]) -> str:
    """Detects if the scheduling algorithm is RM or DM"""
    if all(task.deadline == task.period for task in tasks):
        return "RateMonotonic"
    return "DeadlineMonotonic"

class RTAAnalyzer:
    """Implements worst-case response time analysis using iterative method"""
    @staticmethod
    def calculate_wcrt(tasks: List[Task]) -> Dict[str, float]:
        """
        Calculate worst-case response times for all tasks,
        even if some tasks exceed their deadline.
        No exception is raised here.
        """
        # 按优先级排序 | Sort tasks by priority (ascending order)
        sorted_tasks = sorted(tasks, key=lambda x: x.priority)
        wcrt = {}

        for i, task in enumerate(sorted_tasks):
            hp_tasks = sorted_tasks[:i]  # higher-priority tasks only
            R = task.wcet # Initial response time
            while True:
                prev_R = R
                # 干扰：ceil(R / T_j) * C_j | Interference calculation
                interference = sum(((prev_R + t.period - 1) // t.period) * t.wcet for t in hp_tasks)
                R = task.wcet + interference
                if R == prev_R:
                    break
            wcrt[task.name] = float(R)
        return wcrt

def calculate_utilization(tasks: List[Task]) -> float:
    # Calculate total CPU utilization
    return sum(task.wcet / task.period for task in tasks)

def lcm(a, b):
    return abs(a*b) // math.gcd(a, b) if a and b else 0

def lcm_of_list(numbers):
    return reduce(lcm, numbers, 1)

def load_tasks(filename: str) -> List[Task]:
    # Load tasks from CSV file
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

def numeric_task_name(task_name: str) -> int:
    """
    提取任务名中的数字部分，用于自定义排序：
    例如 'T1' -> 1, 'T10' -> 10, 'T11' -> 11, 'T2' -> 2
    """
    # 假设任务名以 'T' 开头，后面是数字
    return int(task_name.lstrip("T"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed-Priority Scheduling Analysis Tool (modified)")
    parser.add_argument("input_file", help="CSV file containing task parameters (e.g., TC2.csv)")
    args = parser.parse_args()

    # 1. Load tasks from CSV
    tasks = load_tasks(args.input_file)

    # 2. Detect scheduling algorithm
    scheduling_algorithm = detect_scheduling_algorithm(tasks)
    
    # 3. Compute hyperperiod by LCM of all tasks' periods
    hyperperiod = lcm_of_list([task.period for task in tasks])

    # 4. Compute utilization
    utilization = calculate_utilization(tasks)
    # 打印时保留小数 | Format utilization with 2 decimal places
    utilization_str = f"{utilization:.2f}"
    # 或者只想显示整数 | Round to integer for status display
    utilization_int = int(round(utilization))

    # 5. Run RTA analysis (no exception even if WCRT>deadline)
    rta_results = RTAAnalyzer.calculate_wcrt(tasks)

    # 6. 判断是否有任何任务不可调度 | Determine overall schedulability
    schedulable = True
    for task in tasks:
        if rta_results[task.name] > task.deadline:
            schedulable = False
            break

    # 7. Print TaskSet
    print("\nTaskSet:")
    print(f"Hyperperiod = {hyperperiod}")
    print(f"CPU Worst Case Utilization = {utilization_str}")
    for task in tasks:
        print(task)

    # 8. Print Analysis Results
    print("\nResponse Time Analysis")
    print(f"  Scheduling Algorithm: {scheduling_algorithm}")
    print(f"  Schedulable: {'True' if schedulable else 'False'}")
    print(f"  Hyperperiod: {hyperperiod}")
    print(f"  Utilization: {utilization_int}")
    print(f"  Status: (✓=schedulable, ✗=not schedulable)\n")

    print("Task  WCRT   Deadline  Status")
    print("----  -----  --------  ------")

    # 9. 自定义排序：T1, T2, ..., T10, T11 | Sort tasks numerically (T1, T2,...)
    sorted_tasks = sorted(tasks, key=lambda t: numeric_task_name(t.name))

    # 10. 按顺序输出 | Format output results
    for task in sorted_tasks:
        wcrt_val = rta_results[task.name]
        status_char = "✓" if wcrt_val <= task.deadline else "✗"
        # 左边留1格空，保证与“标准答案”对齐
        # Task名占4格宽度，如 " T1 ", " T2 ", "T10"
        print(f" {task.name:<4} {wcrt_val:<6.1f} {task.deadline:<8} {status_char}")
    print("----  -----  --------  ------")
