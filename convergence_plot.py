#!/usr/bin/env python3
import sys
import copy
import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import statistics
from vss_simulator import load_tasks_from_csv, assign_alpha, lcm_of_list, run_single_simulation
from rta_2 import load_tasks, RTAAnalyzer

# 按任务名数字排序，如 T1, T2, ... T10, T11
def numeric_task_name(task):
    return int(task.name.lstrip("T"))


def simulate_convergence(original_tasks, simulation_time, num_runs):
    """
    对每个任务，记录每次运行后累计的最大 WCRT 值。
    
    返回一个字典，键为任务ID（task.task_id），值为列表，
    列表中第 i 个元素表示第 i 次仿真后该任务的累计最大 WCRT。
    """
    cumulative_max = {task.task_id: [] for task in original_tasks}
    current_max = {task.task_id: 0 for task in original_tasks}
    
    for run in range(num_runs):
        # 使用任务的深拷贝，确保每次仿真状态独立
        tasks_copy = copy.deepcopy(original_tasks)
        run_result = run_single_simulation(tasks_copy, simulation_time, verbose=False, log_file=None)
        for task_id, wcrt in run_result.items():
            current_max[task_id] = max(current_max[task_id], wcrt)
            cumulative_max[task_id].append(current_max[task_id])
    return cumulative_max


def plot_convergence(cumulative_max, num_runs, rta_results, save_path=None, extra_title=""):
    """
    绘制每个任务的累计最大 WCRT 收敛图，并叠加 RTA 得到的理论 WCRT 作为参考线（同色）。
    """
    runs = list(range(1, num_runs + 1))
    plt.figure(figsize=(10, 6))
    
    # 遍历 cumulative_max，先画任务收敛曲线，再画 RTA 虚线
    for task_id, cum_max in cumulative_max.items():
        # 先画收敛曲线，并获取线对象
        line, = plt.plot(runs, cum_max, label=f"Task {task_id}")
        # 获取该条线的颜色
        line_color = line.get_color()
        
        # 如果有 RTA 结果，使用相同颜色画虚线
        if task_id in rta_results:
            plt.hlines(
                rta_results[task_id],
                xmin=1, xmax=num_runs,
                colors=line_color,        # 与收敛曲线同色
                linestyles="dashed",
                label=f"RTA for {task_id}"
            )
    
    plt.xlabel("Number of Simulation Runs")
    plt.ylabel("Cumulative Max WCRT")
    plt.title(f"Convergence of VSS WCRT over Simulation Runs {extra_title}")

    # 让 matplotlib 自动做一次布局
    plt.tight_layout()
    
    # 再手动在右侧预留更大空间，例如0.7，留30%的空间给图例
    plt.subplots_adjust(left=0.12, right=0.85)

    # 去重：图例会包含两份（任务 + RTA），可能导致重复标签
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # 图例放在绘图区外右侧，稍微贴近一点
    plt.legend(by_label.values(), by_label.keys(),
           bbox_to_anchor=(1.0, 1), loc="upper left")
    
    # 设置 x 轴范围
    plt.xlim(0, num_runs)

    # 找到所有收敛曲线和 RTA 的最大值
    max_wcrt_data = 0
    for task_id, cum_max in cumulative_max.items():
        if cum_max:  # 避免空列表
            local_max = max(cum_max)
            max_wcrt_data = max(max_wcrt_data, local_max)

    max_wcrt_rta = 0
    if rta_results:
        max_wcrt_rta = max(rta_results.values())

    global_max = max(max_wcrt_data, max_wcrt_rta)

    # 让纵轴从 0 到 global_max * 1.05
    plt.ylim(0, global_max * 1.05)
    
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"Convergence plot saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate convergence plot for VSS WCRT")
    parser.add_argument("csv_filename", help="CSV file containing task parameters")
    parser.add_argument("--U_target", type=float, default=None, help="Target CPU utilization (0,1)")
    parser.add_argument("--method", choices=["workload", "truncnorm"], default="workload",
                        help="Execution time generation method")
    parser.add_argument("--runs", type=int, default=1000, help="Number of simulation runs for convergence")
    parser.add_argument("--simtime", type=int, default=None, help="Simulation time (if not provided, use LCM of task periods)")
    parser.add_argument("--save", type=str, default=None, help="Path to save the convergence plot (e.g., plot.png)")
    parser.add_argument("--outdir", type=str, default="output", help="Base directory to save images/logs")

    args = parser.parse_args()
    
    # 若主输出目录不存在，则创建
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
    # 创建 images/ 和 logs/ 子目录
    images_dir = os.path.join(args.outdir, "images")
    logs_dir = os.path.join(args.outdir, "logs")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 从 vss_simulator 中加载任务并分配 CPU 负载因子
    try:
        tasks_vss = load_tasks_from_csv(args.csv_filename, args.method)
        tasks_vss = assign_alpha(tasks_vss, args.U_target)
    except Exception as e:
        print(f"Error loading tasks from CSV: {e}")
        sys.exit(1)

    # 仿真总时间：如果未提供，则使用所有任务周期的最小公倍数
    if args.simtime:
        simulation_time = args.simtime
    else:
        simulation_time = lcm_of_list([task.period for task in tasks_vss])
    print(f"Simulation Time: {simulation_time}")
    
    # 同时使用 rta 模块加载任务，并计算理论上的 WCRT
    tasks_rta = load_tasks(args.csv_filename)
    # 按优先级排序后调用 RTA
    sorted_rta_tasks = sorted(tasks_rta, key=lambda t: t.priority)
    rta_results = RTAAnalyzer.calculate_wcrt(sorted_rta_tasks)
    
    # 如果任意任务的 WCRT > Deadline，则系统不可调度
    schedulable = True
    for t in tasks_rta:
        if rta_results[t.name] > t.deadline:
            schedulable = False
            break
    
    print(f"  Schedulable: {'True' if schedulable else 'False'}")
    
    sorted_by_name = sorted(tasks_rta, key=numeric_task_name)
    
    print("Task  WCRT   Deadline  Status")
    print("----  -----  --------  ------")
    for t in sorted_by_name:
        wcrt_val = rta_results[t.name]
        status_char = "✓" if wcrt_val <= t.deadline else "✗"
        print(f" {t.name:<4} {wcrt_val:<6.1f} {t.deadline:<8} {status_char}")
    print("----  -----  --------  ------")
    
    # 多次仿真收敛
    cumulative_max = simulate_convergence(tasks_vss, simulation_time, args.runs)
    
    # 若用户没指定 --save，则默认放到 images/convergence.png
    if args.save is None:
        save_path = os.path.join(images_dir, "convergence.png")
    else:
        # 如果指定了 --save, 我们把它放到 images/ 里
        # 也可以直接把 args.save 当绝对路径/相对路径用, 看你需求
        # 这里示例把文件名放到 images_dir
        base_name = os.path.basename(args.save)
        save_path = os.path.join(images_dir, base_name)
    
    extra_title = f"(U_target={args.U_target}, method={args.method}, runs={args.runs})"

    plot_convergence(cumulative_max, args.runs, rta_results, save_path=save_path, extra_title=extra_title)


if __name__ == "__main__":
    main()
