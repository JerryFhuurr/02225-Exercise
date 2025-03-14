import math
from functools import reduce
import sys
import random
import pandas as pd
import plotly.express as px


def load_tasks_from_csv(filename):
    """ 从 CSV 文件中加载任务列表 """
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
    """计算两个数的最小公倍数 (LCM)"""
    return abs(a * b) // math.gcd(a, b)


def lcm_of_list(numbers):
    """计算列表中所有数的最小公倍数"""
    return reduce(lcm, numbers)


class Task:
    """任务类"""

    def __init__(self, task_id, bcet, wcet, period, deadline, priority):
        self.task_id = task_id
        self.period = period
        self.deadline = deadline
        self.priority = priority
        self.bcet = bcet
        self.wcet = wcet
        self.execution_time = wcet
        # random.randint(bcet, wcet)  # 随机选择一个执行时间 C
        self.release_time = 0                 # 任务何时释放
        self.remaining_time = self.execution_time  # 剩余执行时间
        self.completion_time = None           # 任务完成时间
        self.response_time = None             # 响应时间
        self.wcrt = 0

    def release_new_job(self, current_time):
        """ 释放一个新任务实例（新的 job）"""
        self.release_time = current_time
        self.execution_time = self.wcet
        # random.randint(self.bcet, self.wcet)  # 重新随机生成执行时间 C
        self.remaining_time = self.execution_time  # 重新设置剩余执行时间
        self.completion_time = None  # 还未完成
        self.response_time = None  # 还未计算响应时间
    
    def is_ready(self, current_time):
        """ 任务是否已释放 (可执行) """
        return current_time >= self.release_time and self.remaining_time > 0
    
    def execute(self, time_units=1):
        """ 执行任务 """
        self.remaining_time -= time_units
        if self.remaining_time <= 0:
            return True  # 任务完成
        return False  # 任务未完成

    def calculate_response_time(self, current_time):
        """ 计算任务的响应时间 (完成时间 - 释放时间) """
        self.completion_time = current_time
        self.response_time = self.completion_time - self.release_time
        self.wcrt = max(self.wcrt, self.response_time)


def AdvanceTime(current_time, tasks, job_release_times, active_jobs):
    """ 跳过 CPU 空闲时间，直接前进到下一个任务释放时间 """
    if active_jobs:
        return 1  # 如果有任务就绪，则推进 1 个时间单位
    
    future_release_times = [t for t in job_release_times.values() if t > current_time]  # 计算下一个任务的释放时间
    if future_release_times:
        return min(future_release_times) - current_time  # 跳到最近的释放时间
    
    return 1  # 没有任务时，推进 1


def rate_monotonic_scheduling(tasks, simulation_time):
    """ 运行 Rate Monotonic (RM) 调度算法 """
    current_time = 0
    
    active_jobs = []    # 存储当前正在运行或等待的任务（就绪任务）

    job_release_times = {task.task_id: 0 for task in tasks} # 存储每个任务的下次释放时间

    schedule_log = []

    while current_time < simulation_time:
        # --- 步骤 1: 释放新任务 ---
        for task in tasks:
            if current_time == job_release_times[task.task_id]:
                task.release_new_job(current_time)
                active_jobs.append(task)
                job_release_times[task.task_id] += task.period
                print(f"[时间 {current_time}] 任务 {task.task_id} 释放，执行时间 {task.execution_time}")

        # --- 步骤 2: 选择优先级最高的任务 ---
        active_jobs = sorted(active_jobs, key=lambda t: t.priority)  # 按优先级排序
        if active_jobs:
            current_job = active_jobs[0]
            finished = current_job.execute(1)  # 执行 1 个时间单位
            schedule_log.append((current_time, current_job.task_id))
            
            print(f"[时间 {current_time}] 任务 {current_job.task_id} 运行，剩余时间 {current_job.remaining_time}")

            # --- 步骤 3: 任务执行完成，计算响应时间 ---
            if finished:
                current_job.calculate_response_time(current_time + 1)
                active_jobs.remove(current_job)
                print(f"[时间 {current_time+1}] 任务 {current_job.task_id} 完成，响应时间 {current_job.response_time}")

        else:
            # 没有任务就绪，CPU 空闲
            schedule_log.append((current_time, "Idle"))
            # print(f"[时间 {current_time}] CPU 空闲")

        # --- 步骤 4: 时间前进 ---
        # current_time += 1

        # 使用 `AdvanceTime()` 跳过空闲时间
        current_time += AdvanceTime(current_time, tasks, job_release_times, active_jobs)
        
    # --- 步骤 5: 计算 Worst-Case Response Time (WCRT) ---
    print("\n=== Worst-Case Response Time (WCRT) ===")
    for task in tasks:
        print(f"任务 {task.task_id}: WCRT = {task.wcrt}")

    return schedule_log


if __name__ == "__main__":
    """ 主函数 """
    if len(sys.argv) < 2:
        print("使用方法: python simulator.py <csv文件名>")
        sys.exit(1)

    csv_filename = sys.argv[1]

    tasks = load_tasks_from_csv(csv_filename)

    periods = [task.period for task in tasks]   # 计算最小公倍数 (LCM) 作为 simulation_time
    simulation_time = lcm_of_list(periods)
    print(f"Simulation Time (LCM of all periods): {simulation_time}")

    schedule_log = rate_monotonic_scheduling(tasks, simulation_time)