# compare.py - Compare RTA and VSS with optional Gantt chart generation
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
    lcm_of_list,
    run_multiple_simulations,
    plot_gantt_chart,
    rate_monotonic_scheduling
)


def numeric_task_name(task_name: str) -> int:
    #Use the re.sub function to remove all non-digit characters from the string
    return int(re.sub(r"\D", "", task_name))

def plot_comparison_chart(rta_results, stats, save_path="output/images/comparison_chart.png"):
    
    # Sort the tasks by their numeric task name
    tasks = sorted(rta_results.keys(), key=numeric_task_name)

    # Get the RTA WCRT for each task
    rta_wcrt = [rta_results[t] for t in tasks]
    
    # Get the VSS average WCRT for each task
    vss_avg = [stats[t]["average"] if t in stats else 0 for t in tasks]

    # Create a bar chart with the RTA WCRT and VSS average WCRT
    x = np.arange(len(tasks))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, rta_wcrt, width, label='RTA WCRT')
    rects2 = ax.bar(x + width/2, vss_avg, width, label='VSS Avg WCRT')

    # Set the y-axis label and title
    ax.set_ylabel('WCRT')
    ax.set_title('Comparison of RTA WCRT and VSS Average WCRT')
    
    # Set the x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    
    # Add a legend
    ax.legend()

    
    def autolabel(rects):
        # Loop through each bar in the bar chart
        for rect in rects:
            # Get the height of the bar
            height = rect.get_height()

            # Annotate the height of the bar on the chart
            ax.annotate(f'{height:.1f}',  # Format the height to one decimal place
                        xy=(rect.get_x() + rect.get_width() / 2, height),  # Position the text at the center of the bar
                        xytext=(0, 3),  # Offset the text 3 points above the bar
                        textcoords="offset points",  # Use the offset points as the text coordinates
                        ha='center', va='bottom')
    autolabel(rects1)
    autolabel(rects2)

    # Adjust the layout
    fig.tight_layout()
    
    # Save the chart
    plt.savefig(save_path)
    print(f"Comparison chart saved to {save_path}")
    
    # Show the chart
    plt.show()   
 

def compare_rta_vs_vss(csv_filename, U_target=None, method="workload", runs=50, logfile=False, simtime=None):
    # Print the number of runs
    print(f"Running simulation for {runs} runs...")
        
    # Define the directories for images and logs
    images_dir = "output/images"
    logs_dir   = "output/logs"
    
    # Create the directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Load the tasks from the csv file
    tasks_rta = load_tasks(csv_filename)
    tasks_vss = load_tasks_from_csv(csv_filename)
    
    # Set the simulation time
    if simtime is None:
        simulation_time = lcm_of_list([task.period for task in tasks_vss])
    else:
        simulation_time = simtime
        
    # Print the simulation time
    print(f"Simulation Time: {simulation_time}")
    
    # Calculate the worst-case response times for the tasks using RTA
    rta_results = RTAAnalyzer.calculate_wcrt(sorted(tasks_rta, key=lambda t: t.priority))

    # If only one run, schedule the tasks and plot the gantt chart
    if runs == 1:
        schedule_log = rate_monotonic_scheduling(tasks_vss, simulation_time)
        gantt_path = os.path.join(images_dir, "gantt_chart.png")
        plot_gantt_chart(schedule_log, save_path=gantt_path)
    else:
        # Set the log file path
        log_file_path = os.path.join(logs_dir, "compare.log") if logfile else None
        # Run multiple simulations
        stats = run_multiple_simulations(
            tasks_vss,
            simulation_time,
            num_runs=runs,
            verbose=False,
            log_filename=log_file_path
        )
        
        # Print the comparison of RTA and VSS
        print("\n=== Comparison of RTA and VSS ===")
        
        # Set the schedulable flag
        schedulable = True
        
        # Check if the tasks are schedulable
        for task in tasks_rta:
            if rta_results[task.name] > task.deadline:
                schedulable = False
                break
        
        # Print the schedulable status
        print(f"  Schedulable: {'True' if schedulable else 'False'}")

        # Print the task names, RTA WCRT, deadlines, status, and VSS average WCRT
        print("Task  RTA_WCRT  Deadline  Status  VSS_Avg  VSS_Median  VSS_95th  VSS_Max")
        print("----  --------  --------  ------  -------  ---------  --------  -------")


        # Sort the tasks by name
        sorted_tasks = sorted(tasks_rta, key=lambda t: numeric_task_name(t.name))
        
        # Print the task information
        for t in sorted_tasks:
            wcrt_rta = rta_results[t.name]
            task_stats = stats.get(t.name, {})
            avg_wcrt_vss = task_stats.get('average', 0.0)
            median_wcrt_vss = task_stats.get('median', 0.0)
            percentile_95 = task_stats.get('95th', 0.0)
            max_wcrt_vss = task_stats.get('max', 0.0)
            status_char = "✓" if wcrt_rta <= t.deadline else "✗"
            print(f" {t.name:<4} {wcrt_rta:<8.1f} {t.deadline:<8} {status_char:<8} {avg_wcrt_vss:<8.2f} {median_wcrt_vss:<9.2f} {percentile_95:<8.2f} {max_wcrt_vss:<8.2f}")


        # Print the end of the task information
        print("----  -----  --------  ------   -----------")

        # Plot the comparison chart
        comparison_path = os.path.join(images_dir, "comparison_chart.png")
        plot_comparison_chart(rta_results, stats, save_path=comparison_path)


def main():
    parser = argparse.ArgumentParser(description="Compare RTA and VSS with optional Gantt chart generation")
    parser.add_argument("csv_filename", help="CSV file containing task parameters")
    parser.add_argument("--U_target", type=float, default=None, help="Target CPU utilization (0,1)")
    parser.add_argument("--simtime", type=int, default=None, help="Total simulation time to use instead of LCM of task periods")
    parser.add_argument("--runs", type=int, default=50, help="Number of simulation runs")
    parser.add_argument("--logfile", action="store_true",
                        help="If set, enable logging to 'output/logs/compare.log'")

    # Parse the arguments
    args = parser.parse_args()

    # Call the compare_rta_vs_vss function with the parsed arguments
    compare_rta_vs_vss(
    csv_filename=args.csv_filename,
    U_target=args.U_target,
    runs=args.runs,
    logfile=args.logfile,
    simtime=args.simtime
)

if __name__ == "__main__":
    main()