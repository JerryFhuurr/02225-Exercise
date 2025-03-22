# Description: 从 CSV 文件中加载任务参数，使用 RTA 和 VSS 进行对比分析。
import sys
import copy
import os
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np

from rta import load_tasks, RTAAnalyzer, calculate_utilization
from vss_simulator import (
    load_tasks_from_csv,
    assign_alpha,
    lcm_of_list,
    run_multiple_simulations,
    plot_gantt_chart,
    rate_monotonic_scheduling
)


# 定义辅助函数，提取 "T10" 中的数字 10 用于排序
def numeric_task_name(task_name: str) -> int:
    # 假设任务名以 'T' 开头，后面是数字
    return int(re.sub(r"\D", "", task_name))

def plot_comparison_chart(rta_results, stats, save_path="output/images/comparison_chart.png"):
    # 获取所有任务标识，按照 rta_results 的 key 排序
    tasks = sorted(rta_results.keys(), key=numeric_task_name)

    rta_wcrt = [rta_results[t] for t in tasks]
    vss_avg = [stats[t]["average"] if t in stats else 0 for t in tasks]

    x = np.arange(len(tasks))  # x 轴位置
    width = 0.35  # 条形宽度

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, rta_wcrt, width, label='RTA WCRT')
    rects2 = ax.bar(x + width/2, vss_avg, width, label='VSS Avg WCRT')

    # 添加轴标签和标题
    ax.set_ylabel('WCRT')
    ax.set_title('Comparison of RTA WCRT and VSS Average WCRT')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.legend()

    # 在每个条形上添加数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移 3 个点
                        textcoords="offset points",
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    
    # 保存图表到文件
    plt.savefig(save_path)
    print(f"Comparison chart saved to {save_path}")
    plt.show()   
 

def compare_rta_vs_vss(csv_filename, U_target=None, method="workload", runs=50, logfile=False):
    # 打印运行次数信息
    print(f"Running simulation for {runs} runs...")
        
    # 固定输出目录
    images_dir = "output/images"
    logs_dir   = "output/logs"
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 使用 rta 模块加载任务（CSV 文件格式应保持一致）
    tasks_rta = load_tasks(csv_filename)
    # 使用 vss_simulator 模块加载任务
    tasks_vss = load_tasks_from_csv(csv_filename, method)

    # 为 VSS 部分任务分配 CPU 负载因子 α
    tasks_vss = assign_alpha(tasks_vss, U_target=U_target, method=method)
    # 仿真总时间默认按所有任务周期的最小公倍数计算
    simulation_time = lcm_of_list([task.period for task in tasks_vss])
    
    # 运行 RTA 分析（任务按优先级排序，RTA 使用 task.name 作为标识）
    rta_results = RTAAnalyzer.calculate_wcrt(sorted(tasks_rta, key=lambda t: t.priority))

    if runs == 1:
        # 单次仿真：运行调度算法并获取调度日志
        schedule_log = rate_monotonic_scheduling(tasks_vss, simulation_time)
        # 保存甘特图到 images/gantt_chart.png
        gantt_path = os.path.join(images_dir, "gantt_chart.png")
        plot_gantt_chart(schedule_log, save_path=gantt_path)
    else:
        # 多次仿真
        # 若启用 --logfile，则写到 output/logs/compare.log
        log_file_path = os.path.join(logs_dir, "compare.log") if logfile else None
        
        # 多次仿真，输出扩展统计指标（平均、中位数、方差、最大值、95百分位）
        stats = run_multiple_simulations(
            tasks_vss,
            simulation_time,
            num_runs=runs,
            verbose=False,
            log_filename=log_file_path
        )
        
        print("\n=== Comparison of RTA and VSS ===")
        
        # （A）先判断可调度性
        schedulable = True
        for task in tasks_rta:
            if rta_results[task.name] > task.deadline:
                schedulable = False
                break

        print(f"  Schedulable: {'True' if schedulable else 'False'}")

        # （B）打印每个任务的对比：RTA WCRT & VSS 统计
        # 同时打印该任务是否可调度 (✓ / ✗)
        print("Task  RTA_WCRT  Deadline  Status  VSS_Avg_WCRT")
        print("----  --------  --------  ------  -----------")

        # 按照任务名中的数字排序
        sorted_tasks = sorted(tasks_rta, key=lambda t: numeric_task_name(t.name))

        for t in sorted_tasks:
            wcrt_rta = rta_results[t.name]   # RTA 得到的WCRT
            avg_wcrt_vss = stats.get(t.name, {}).get('average', 0.0)
            status_char = "✓" if wcrt_rta <= t.deadline else "✗"
            # 为了对齐，与“标准答案”接近，可在任务名前留1空格
            print(f" {t.name:<4} {wcrt_rta:<6.1f} {t.deadline:<8} {status_char:<6} {avg_wcrt_vss:<.2f}")

        print("----  -----  --------  ------   -----------")

        # ---- 可视化：RTA vs. VSS 条形图 ----
        comparison_path = os.path.join(images_dir, "comparison_chart.png")
        plot_comparison_chart(rta_results, stats, save_path=comparison_path)


def main():
    parser = argparse.ArgumentParser(description="Compare RTA and VSS with optional Gantt chart generation")
    parser.add_argument("csv_filename", help="CSV file containing task parameters")
    parser.add_argument("--U_target", type=float, default=None, help="Target CPU utilization (0,1)")
    parser.add_argument("--method", choices=["workload", "truncnorm"], default="workload",
                        help="Execution time generation method")
    parser.add_argument("--runs", type=int, default=50, help="Number of simulation runs")
    parser.add_argument("--logfile", action="store_true",
                        help="If set, enable logging to 'output/logs/compare.log'")

    args = parser.parse_args()

    compare_rta_vs_vss(
        csv_filename=args.csv_filename,
        U_target=args.U_target,
        method=args.method,
        runs=args.runs,
        logfile=args.logfile
    )

if __name__ == "__main__":
    main()