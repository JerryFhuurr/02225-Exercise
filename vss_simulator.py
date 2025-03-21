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


# ----- 辅助函数 -----
def lcm(a, b):
    """ Calculate the least common multiple (LCM) of two numbers """
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


# ----- 任务类 -----
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


# ----- 数据加载与 α 分配 -----
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
        print(f"Total CPU load: Σα = {sum(task.alpha for task in tasks):.2f}")
    return tasks


# ----- 时间推进与调度 -----
def advance_time(current_time, job_release_times, active_jobs):
    """ Advance time to the next event """
    if active_jobs:
        return 1
    future = [t for t in job_release_times.values() if t > current_time]
    return min(future) - current_time if future else 1


def rate_monotonic_scheduling(tasks, simulation_time, verbose=False, log_file=None):
    """ 
    Rate Monotonic Scheduling 
    verbose 为 True 时将详细日志打印，并写入 log_file（若指定）。
    """
    current_time = 0
    active_jobs = []
    job_release_times = {task.task_id: 0 for task in tasks}
    schedule_log = []

    while current_time < simulation_time:
        # 任务到达：释放新作业
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
        # 按优先级调度（数值越小优先级越高）
        active_jobs.sort(key=lambda t: t.priority)
        if active_jobs:
            current_job = active_jobs[0]
            finished = current_job.execute(1)
            schedule_log.append((current_time, current_job.task_id))
            msg = f"[Time {current_time}] Task {current_job.task_id} Running, Remaining: {current_job.remaining_time}"
            if verbose:
                print(msg)
            if log_file:
                log_file.write(msg + "\n")
            if finished:
                current_job.calculate_response_time(current_time + 1)
                active_jobs.remove(current_job)
                msg = f"[Time {current_time+1}] Task {current_job.task_id} Completed, Response: {current_job.response_time}"
                if verbose:
                    print(msg)
                if log_file:
                    log_file.write(msg + "\n")
        else:
            schedule_log.append((current_time, "Idle"))
        current_time += advance_time(current_time, job_release_times, active_jobs)
    # 打印 WCRT 结果
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
    # 根据 schedule_log 中的任务（不包括 "Idle"）生成颜色和 Y 轴位置
    unique_tasks = set(entry[1] for entry in schedule_log if entry[1] != "Idle")
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
    
    # 如果提供了保存路径，则保存图片
    if save_path:
        plt.savefig(save_path)
        print(f"Gantt chart saved to {save_path}")
    
    # 最后弹出图窗
    plt.show()
    

# ----- 多次仿真与统计 -----
def run_single_simulation(tasks, simulation_time, verbose=False, log_file=None):
    """
    对任务集进行一次仿真，并返回每个任务的 WCRT。
    注意：传入的 tasks 应该是一个独立的拷贝
    """
    # 调用已有的调度函数，运行仿真
    rate_monotonic_scheduling(tasks, simulation_time, verbose=verbose, log_file=log_file)
    # 收集每个任务的 WCRT
    result = {task.task_id: task.wcrt for task in tasks}
    return result


def run_multiple_simulations(original_tasks, simulation_time, num_runs=10, verbose=False, log_filename=None):
    """
    进行多次仿真运行，统计每个任务的平均 WCRT、方差及最大 WCRT。
    
    Parameters:
      - original_tasks: 原始任务列表（从 CSV 读取后的任务对象）
      - simulation_time: 仿真总时间
      - num_runs: 仿真运行次数
    Returns:
      - stats: 一个字典，格式例如 { task_id: {"average": avg, "variance": var, "max": max_wcrt}, ... }
    """
    # 用来收集每个任务多次仿真得到的 WCRT 列表
    results = {task.task_id: [] for task in original_tasks}
    
    # 如果启用日志记录，则打开日志文件
    log_file = open(log_filename, "w") if log_filename else None
    
    for run in range(num_runs):
        if verbose:
            print(f"\n=== Run {run+1} ===")
            if log_file:
                log_file.write(f"=== Run {run+1} ===\n")
        tasks_copy = copy.deepcopy(original_tasks)
        run_result = run_single_simulation(tasks_copy, simulation_time, verbose=verbose, log_file=log_file)
        for task_id, wcrt in run_result.items():
            results[task_id].append(wcrt)
    if log_file:
        log_file.close()
    
    # 计算统计指标
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


# ----- 主函数 -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSS Simulator with extended statistics and logging")
    parser.add_argument("csv_filename", help="CSV file containing task parameters")
    parser.add_argument("--U_target", type=float, default=None, help="Target CPU utilization (0,1)")
    parser.add_argument("--method", choices=["workload", "truncnorm"], default="workload", help="Execution time generation method")
    parser.add_argument("--runs", type=int, default=1, help="Number of simulation runs")
    parser.add_argument("--simtime", type=int, default=None, help="Simulation time (if not provided, use LCM of task periods)")
    parser.add_argument("--verbose", action="store_true", help="Output detailed log to console")
    parser.add_argument("--plot", action="store_true", help="Generate Gantt chart for single simulation")
    parser.add_argument("--logfile", default=None, help="File to save detailed simulation log")
    parser.add_argument("--outdir", type=str, default="output", help="Base directory to save images/logs")

    
    args = parser.parse_args()
    
    # 1. 如果主输出目录不存在，创建之
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)

    # 2. 创建 images/ 和 logs/ 子目录
    images_dir = os.path.join(args.outdir, "images")
    logs_dir = os.path.join(args.outdir, "logs")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 载入任务及分配 α
    original_tasks = load_tasks_from_csv(args.csv_filename, args.method)
    original_tasks = assign_alpha(original_tasks, args.U_target)
    
    # 根据方法重新计算执行时间
    for task in original_tasks:
        if args.method == "workload":
            task.execution_time = generate_execution_time_workload(task.bcet, task.wcet, task.period, task.alpha)
        else:
            task.execution_time = generate_execution_time_truncnorm(task.bcet, task.wcet)
    
    # 计算仿真时间
    if args.simtime:
        simulation_time = args.simtime
    else:
        simulation_time = lcm_of_list([task.period for task in original_tasks])
    print(f"Simulation Time: {simulation_time}")

    # 如果 runs==1, 单次仿真
    if args.runs == 1:
        log_file_path = None
        if args.logfile:
            # 把日志文件放到 logs/ 下
            log_file_path = os.path.join(logs_dir, args.logfile)
        schedule_log = rate_monotonic_scheduling(
            original_tasks, simulation_time, verbose=args.verbose,
            log_file=(open(log_file_path, "w") if log_file_path else None)
        )
        if args.plot:
            # 把甘特图保存到 images/gantt_chart.png
            gantt_path = os.path.join(images_dir, "gantt_chart.png")
            plot_gantt_chart(schedule_log, save_path=gantt_path)
    else:
        # 多次仿真
        log_file_path = None
        if args.logfile:
            log_file_path = os.path.join(logs_dir, args.logfile)
        stats = run_multiple_simulations(
            original_tasks, simulation_time, num_runs=args.runs,
            verbose=args.verbose, log_filename=log_file_path
        )
        print("\n=== Simulation Statistics ===")
        for task_id, stat in stats.items():
            print(f"Task {task_id}: Average WCRT = {stat['average']:.2f}, "
                  f"Median = {stat['median']}, Variance = {stat['variance']:.2f}, "
                  f"95th Percentile = {stat['95th']}, Max WCRT = {stat['max']}")
            