# Description: Generate convergence plot for VSS WCRT
import sys
import copy
import argparse
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import statistics

from vss_simulator import (
    load_tasks_from_csv,
    assign_alpha,
    lcm_of_list,
    run_single_simulation
)
from rta import load_tasks, RTAAnalyzer


def numeric_task_name(task):
    # 提取任务名中的数字部分（例如 T10 => 10）
    return int(re.sub(r"\D", "", task.name))


def simulate_convergence(original_tasks, simulation_time, num_runs, rta_results, log_file=None):
    cumulative_max = {t.task_id: [] for t in original_tasks}
    current_max    = {t.task_id: 0  for t in original_tasks}
    discovery_run  = {t.task_id: 0  for t in original_tasks}
    discovery_time = {t.task_id: 0  for t in original_tasks}

    for run_index in range(num_runs):
        # 每次都复制任务列表，确保仿真独立
        tasks_copy = copy.deepcopy(original_tasks)

        # 调用单次仿真
        run_result = run_single_simulation(
            tasks_copy, 
            simulation_time, 
            verbose=False, 
            log_file=log_file
        )

        # 更新 current_max
        for task_id, (wcrt_val, finish_val) in run_result.items():
            old_val = current_max[task_id]
            new_val = max(old_val, wcrt_val)
            current_max[task_id] = new_val
            cumulative_max[task_id].append(new_val)

            # 若出现新的 worst WCRT，则打印
            if new_val > old_val:
                # 说明出现了新的 worst WCRT
                # 计算全局时间 = run_index * simulation_time + finish_val
                global_t = run_index * simulation_time + finish_val
                discovery_run[task_id]  = run_index + 1  # 第几轮（人类可读）
                discovery_time[task_id] = global_t

                msg = (f"[Run {run_index+1}] New worst WCRT for {task_id}: "
                       f"{new_val:.1f}, discovered_global_time={global_t:.1f}")
                print(msg)
                if log_file:
                    log_file.write(msg + "\n")

        # 打印本轮结束后的全局最差任务
        if log_file:
            worst_task = max(current_max, key=current_max.get)
            worst_val  = current_max[worst_task]
            log_file.write(f"End of run {run_index+1}: "
                           f"worst so far => {worst_task}={worst_val:.1f}\n")

    # 所有仿真结束后，打印/写日志最终结果
    print("=== Final Convergence Result ===")
    if log_file:
        log_file.write("=== Final Convergence Result ===\n")
        
    for task_id in cumulative_max:
        final_max = cumulative_max[task_id][-1]  # 最后一轮结束时的累积最坏
        rta_val   = rta_results.get(task_id, 0.0)
        ratio     = (final_max / rta_val) if rta_val else 0.0
        final_run = discovery_run[task_id]
        final_gtime = discovery_time[task_id]

        msg = (f"Task {task_id}: final max WCRT={final_max:.1f}, "
               f"RTA WCRT={rta_val:.1f}, ratio={ratio:.2f}, "
               f"found in run={final_run}, global_time={final_gtime:.1f}")
        print(msg)
        if log_file:
            log_file.write(msg + "\n")

    return cumulative_max


def plot_convergence(cumulative_max, num_runs, rta_results, save_path="output/images/convergence.png", extra_title=""):
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
                colors=line_color,
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

    # 找到所有曲线和 RTA 的最大值
    max_wcrt_data = 0
    for task_id, cum_max in cumulative_max.items():
        if cum_max:
            local_max = max(cum_max)
            max_wcrt_data = max(max_wcrt_data, local_max)
    max_wcrt_rta = max(rta_results.values()) if rta_results else 0
    global_max = max(max_wcrt_data, max_wcrt_rta)
    plt.ylim(0, global_max * 1.05)

    plt.grid(True)
    plt.savefig(save_path)
    print(f"Convergence plot saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate convergence plot for VSS WCRT.")
    parser.add_argument("csv_filename", help="CSV file containing task parameters")
    parser.add_argument("--U_target", type=float, default=None,
                        help="Target CPU utilization (0,1) used only if method=workload")
    parser.add_argument("--method", choices=["workload", "truncnorm"], default="workload",
                        help="Execution time generation method for VSS tasks")
    parser.add_argument("--runs", type=int, default=1000,
                        help="Number of simulation runs for convergence")
    parser.add_argument("--simtime", type=int, default=None,
                        help="Simulation time (if not provided, use LCM of tasks' periods)")
    parser.add_argument("--logfile", action="store_true",
                        help="If set, enable logging to 'output/logs/convergence.log'")
    args = parser.parse_args()

    # 1) 创建输出目录
    images_dir = "output/images"
    logs_dir = "output/logs"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # 2) 加载 VSS 任务
    try:
        tasks_vss = load_tasks_from_csv(args.csv_filename, args.method)
        # 仅当 method=workload 时，才会真正分配 alpha；truncnorm 不会分配
        tasks_vss = assign_alpha(tasks_vss, args.U_target, method=args.method)
    except Exception as e:
        print(f"Error loading tasks from CSV: {e}")
        sys.exit(1)

    # 3) 若未指定 simtime，则使用所有任务周期的 LCM
    if args.simtime:
        simulation_time = args.simtime
    else:
        simulation_time = lcm_of_list([task.period for task in tasks_vss])
    print(f"Simulation Time: {simulation_time}")

    # 4) 加载 RTA 任务（与 VSS 任务可能只差执行时间生成方式）
    tasks_rta = load_tasks(args.csv_filename)
    # 根据优先级排序，进行 RTA 分析
    sorted_rta_tasks = sorted(tasks_rta, key=lambda t: t.priority)
    rta_results = RTAAnalyzer.calculate_wcrt(sorted_rta_tasks)

    # 5) 判断系统可调度性
    schedulable = True
    for t in tasks_rta:
        if rta_results[t.name] > t.deadline:
            schedulable = False
            break
    print(f"  Schedulable: {'True' if schedulable else 'False'}")

    # 6) 简要打印 RTA 结果
    tasks_rta_name_sorted = sorted(tasks_rta, key=numeric_task_name)
    print("Task  WCRT   Deadline  Status")
    print("----  -----  --------  ------")
    for t in tasks_rta_name_sorted:
        wcrt_val = rta_results[t.name]
        status_char = "✓" if wcrt_val <= t.deadline else "✗"
        print(f" {t.name:<4} {wcrt_val:<6.1f} {t.deadline:<8} {status_char}")
    print("----  -----  --------  ------")

    # 7) 若启用 --logfile，则打开日志文件
    log_file_path = None
    log_file_obj = None
    if args.logfile:
        log_file_path = os.path.join(logs_dir, "convergence.log")
        log_file_obj = open(log_file_path, "w")

    # 8) 进行多次仿真，记录收敛过程
    #    注意：simulate_convergence 需要把 rta_results 也传进去，以便最终输出 ratio 对比
    cumulative_max = simulate_convergence(
        tasks_vss,
        simulation_time,
        args.runs,
        rta_results=rta_results,
        log_file=log_file_obj
    )

    # 9) 若打开了日志文件，用完要关闭
    if log_file_obj:
        log_file_obj.close()

    # 10) 绘制收敛图
    save_path = os.path.join(images_dir, "convergence.png")
    extra_title = f"(U_target={args.U_target}, method={args.method}, runs={args.runs})"
    plot_convergence(cumulative_max, args.runs, rta_results,
                     save_path=save_path, extra_title=extra_title)

if __name__ == "__main__":
    main()
