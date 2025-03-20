import sys
import copy
from rta import load_tasks, RTAAnalyzer, calculate_utilization
from vss_simulator import load_tasks_from_csv, assign_alpha, lcm_of_list, run_multiple_simulations, plot_gantt_chart

def compare_rta_vs_vss(csv_filename, U_target=None, method="workload", runs=50):
    # 使用 rta 模块加载任务（CSV 文件格式应保持一致）
    tasks_rta = load_tasks(csv_filename)  
    # 使用 vss_simulator 模块加载任务
    tasks_vss = load_tasks_from_csv(csv_filename, method)

    # 为 VSS 部分任务分配 CPU 负载因子 α
    tasks_vss = assign_alpha(tasks_vss, U_target)
    # 仿真总时间默认按所有任务周期的最小公倍数计算
    simulation_time = lcm_of_list([task.period for task in tasks_vss])
    
    # 运行 RTA 分析（先按优先级排序，RTA 使用 task.name 作为标识）
    rta_results = RTAAnalyzer.calculate_wcrt(sorted(tasks_rta, key=lambda t: t.priority))
    
    # 运行 VSS 多次仿真，统计扩展指标（平均、中位数、方差、最大值、95百分位）
    stats = run_multiple_simulations(tasks_vss, simulation_time, num_runs=runs, verbose=False, log_filename=None)
    
    # 输出对比结果
    print("\n=== Comparison of RTA and VSS ===")
    for task in tasks_rta:
        key = task.name  # rta 模块中的任务标识
        rta_wcrt = rta_results.get(key, None)
        vss_stat = stats.get(key, {})
        print(f"Task {key}:")
        print(f"  RTA WCRT: {rta_wcrt}")
        print(f"  VSS Avg WCRT: {vss_stat.get('average', 'N/A'):.2f}, Median: {vss_stat.get('median', 'N/A')}, "
              f"95th Percentile: {vss_stat.get('95th', 'N/A')}, Max WCRT: {vss_stat.get('max', 'N/A')}, "
              f"Variance: {vss_stat.get('variance', 'N/A'):.2f}")
    
    # 可选：如需生成单次仿真的甘特图，可取消以下注释
    # schedule_log = rate_monotonic_scheduling(tasks_vss, simulation_time)
    # plot_gantt_chart(schedule_log)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare.py <csv_filename> [U_target=<value>] [method=workload|truncnorm] [runs=<num_runs>]")
        sys.exit(1)
    csv_filename = sys.argv[1]
    U_target = None
    method = "workload"
    runs = 50
    for arg in sys.argv[2:]:
        if arg.startswith("U_target="):
            try:
                U_target = float(arg.split("=")[1])
                if U_target <= 0:
                    U_target = None
            except ValueError:
                U_target = None
        elif arg.startswith("method="):
            m = arg.split("=")[1]
            if m in ["workload", "truncnorm"]:
                method = m
        elif arg.startswith("runs="):
            try:
                runs = int(arg.split("=")[1])
            except ValueError:
                runs = 50
 
    compare_rta_vs_vss(csv_filename, U_target, method, runs)
