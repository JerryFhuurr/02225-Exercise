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
    #Use the re.sub function to remove any non-digit characters from the task.name string
    return int(re.sub(r"\D", "", task.name))


def simulate_convergence(original_tasks, simulation_time, num_runs, rta_results, log_file=None):
    # Initialize dictionaries to store the cumulative maximum WCRT for each task, the current maximum WCRT, the run and time of discovery for each task
    cumulative_max = {t.task_id: [] for t in original_tasks}
    current_max    = {t.task_id: 0  for t in original_tasks}
    discovery_run  = {t.task_id: 0  for t in original_tasks}
    discovery_time = {t.task_id: 0  for t in original_tasks}

    # Loop through the number of runs
    for run_index in range(num_runs):
        
        # Create a deep copy of the original tasks
        tasks_copy = copy.deepcopy(original_tasks)

        # Run a single simulation
        run_result = run_single_simulation(
            tasks_copy, 
            simulation_time, 
            verbose=False, 
            log_file=log_file
        )

        # Loop through the results of the simulation
        for task_id, (wcrt_val, finish_val) in run_result.items():
            # Get the old and new values of the WCRT
            old_val = current_max[task_id]
            new_val = max(old_val, wcrt_val)
            # Update the current maximum WCRT
            current_max[task_id] = new_val
            # Append the new value to the cumulative maximum WCRT
            cumulative_max[task_id].append(new_val)

            # If the new value is greater than the old value
            if new_val > old_val:
                
                # Calculate the global time
                global_t = run_index * simulation_time + finish_val
                # Update the run and time of discovery
                discovery_run[task_id]  = run_index + 1  
                discovery_time[task_id] = global_t

                # Print and log the new worst WCRT
                msg = (f"[Run {run_index+1}] New worst WCRT for {task_id}: "
                       f"{new_val:.1f}, discovered_global_time={global_t:.1f}")
                print(msg)
                if log_file:
                    log_file.write(msg + "\n")

        # Log the worst task and value at the end of each run
        if log_file:
            worst_task = max(current_max, key=current_max.get)
            worst_val  = current_max[worst_task]
            log_file.write(f"End of run {run_index+1}: "
                           f"worst so far => {worst_task}={worst_val:.1f}\n")

    # Print and log the final convergence result
    print("=== Final Convergence Result ===")
    if log_file:
        log_file.write("=== Final Convergence Result ===\n")
        
    # Loop through the cumulative maximum WCRT
    for task_id in cumulative_max:
        # Get the final maximum WCRT, RTA WCRT, ratio, run and global time
        final_max = cumulative_max[task_id][-1]  
        rta_val   = rta_results.get(task_id, 0.0)
        ratio     = (final_max / rta_val) if rta_val else 0.0
        final_run = discovery_run[task_id]
        final_gtime = discovery_time[task_id]

        # Print and log the final convergence result
        msg = (f"Task {task_id}: final max WCRT={final_max:.1f}, "
               f"RTA WCRT={rta_val:.1f}, ratio={ratio:.2f}, "
               f"found in run={final_run}, global_time={final_gtime:.1f}")
        print(msg)
        if log_file:
            log_file.write(msg + "\n")

    # Return the cumulative maximum WCRT
    return cumulative_max


def plot_convergence(cumulative_max, num_runs, rta_results, save_path="output/images/convergence.png", extra_title=""):
    # Create a list of runs from 1 to num_runs
    runs = list(range(1, num_runs + 1))
    # Create a figure with a size of 10x6
    plt.figure(figsize=(10, 6))
    
    # Loop through each task_id and its corresponding cumulative_max
    for task_id, cum_max in cumulative_max.items():
        
        # Plot the cumulative_max for each task_id
        line, = plt.plot(runs, cum_max, label=f"Task {task_id}")
        
        # Get the color of the line
        line_color = line.get_color()
        
        # If the task_id is in rta_results, plot the RTA for that task_id
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

    plt.tight_layout()
    
    plt.subplots_adjust(left=0.12, right=0.85)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Create the legend with the handles and labels
    plt.legend(by_label.values(), by_label.keys(),
           bbox_to_anchor=(1.0, 1), loc="upper left")
    
    # Set the x-axis limits
    plt.xlim(1, num_runs)

    # Initialize the max_wcrt_data to 0
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
    # Create an argument parser to parse command line arguments
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
    # Parse the arguments
    args = parser.parse_args()

    # Create directories for images and logs if they don't exist
    images_dir = "output/images"
    logs_dir = "output/logs"
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    # Load tasks from the CSV file
    try:
        tasks_vss = load_tasks_from_csv(args.csv_filename, args.method)
        
        # Assign alpha values to the tasks
        tasks_vss = assign_alpha(tasks_vss, args.U_target, method=args.method)
    except Exception as e:
        # Print an error message if there is an issue loading the tasks
        print(f"Error loading tasks from CSV: {e}")
        sys.exit(1)

    # Set the simulation time
    if args.simtime:
        simulation_time = args.simtime
    else:
        simulation_time = lcm_of_list([task.period for task in tasks_vss])
    print(f"Simulation Time: {simulation_time}")

    # Load tasks from the CSV file
    tasks_rta = load_tasks(args.csv_filename)
    
    # Sort the tasks by priority
    sorted_rta_tasks = sorted(tasks_rta, key=lambda t: t.priority)
    # Calculate the WCRT for the tasks
    rta_results = RTAAnalyzer.calculate_wcrt(sorted_rta_tasks)

    # Check if the tasks are schedulable
    schedulable = True
    for t in tasks_rta:
        if rta_results[t.name] > t.deadline:
            schedulable = False
            break
    print(f"  Schedulable: {'True' if schedulable else 'False'}")

    # Sort the tasks by name
    tasks_rta_name_sorted = sorted(tasks_rta, key=numeric_task_name)
    # Print the WCRT, deadline, and status for each task
    print("Task  WCRT   Deadline  Status")
    print("----  -----  --------  ------")
    for t in tasks_rta_name_sorted:
        wcrt_val = rta_results[t.name]
        status_char = "✓" if wcrt_val <= t.deadline else "✗"
        print(f" {t.name:<4} {wcrt_val:<6.1f} {t.deadline:<8} {status_char}")
    print("----  -----  --------  ------")

    # Set up logging if enabled
    log_file_path = None
    log_file_obj = None
    if args.logfile:
        log_file_path = os.path.join(logs_dir, "convergence.log")
        log_file_obj = open(log_file_path, "w")

    # Simulate the convergence of the tasks
    cumulative_max = simulate_convergence(
        tasks_vss,
        simulation_time,
        args.runs,
        rta_results=rta_results,
        log_file=log_file_obj
    )

    # Close the log file if it was opened
    if log_file_obj:
        log_file_obj.close()

    # Save the convergence plot
    save_path = os.path.join(images_dir, "convergence.png")
    extra_title = f"(U_target={args.U_target}, method={args.method}, runs={args.runs})"
    plot_convergence(cumulative_max, args.runs, rta_results,
                     save_path=save_path, extra_title=extra_title)

if __name__ == "__main__":
    main()