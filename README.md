# README

## 1. Overview

This project implements **fixed-priority scheduling** analysis and simulation for a set of periodic tasks. It provides:

1. ​**Static Analysis (RTA)**​: Calculates the Worst-Case Response Time (WCRT) of each task using an iterative Response-Time Analysis approach.
2. ​**Dynamic Simulation (VSS Simulator)**​: Simulates tasks under Rate Monotonic scheduling with variable execution times. Supports multiple runs to estimate statistical properties of the WCRT.
3. ​**Convergence Analysis**​: Repeatedly runs the simulation to show how the measured WCRT converges to a stable value compared to the RTA result.
4. ​**Comparison**​: Compares RTA-derived WCRT vs. simulated WCRT (averages, etc.) and generates visual charts.

The code is written in Python and uses ​**pandas**​, ​**numpy**​, ​**matplotlib**​, **scipy** for data handling, plotting, and statistical distributions.

---

## 2. Directory Structure

project/
│
├── data/
│   ├── Full_Utilization_NonUnique_Periods_taskset.csv
│   ├── High_Utilization_NonUnique_Periods_taskset.csv
│   ├── Low_Utilization_NonUnique_Periods_taskset.csv
│   ├── TC1.csv
│   ├── TC2.csv
│   └── TC3.csv
│
├── </span><span>output</span><span>/
│   ├── images/
│   │   ├── comparison_chart.png
│   │   ├── convergence.png
│   │   └── gantt_chart.png
│   └── logs/
│       ├── compare.</span><span>log</span><span>
│       ├── convergence.</span><span>log</span><span>
│       └── sim.</span><span>log</span><span>
│
├── .gitignore
├── compare.py
├── convergence_plot.py
├── README.md
├── requirements.txt
├── rta.py
└── vss_simulator.py

* ​**data/**​: Contains CSV test case files defining tasks (BCET, WCET, Period, Deadline, Priority, etc.).
* ​**output/**​: Directory where all generated images and logs are stored.
  * ​**images/**​: Stores figures like Gantt charts, comparison charts, and convergence plots.
  * ​**logs/**​: Stores log files generated by simulations (if enabled).
* ​**compare.py**​: Compares RTA results with VSS simulation results (single or multiple runs) and can produce charts.
* ​**convergence\_plot.py**​: Generates convergence plots by running multiple simulations and comparing each task’s WCRT with the RTA WCRT over successive runs.
* ​**rta.py**​: Implements the static Response-Time Analysis for a given CSV file.
* ​**vss\_simulator.py**​: Simulates tasks with variable execution times under Rate Monotonic scheduling and can produce Gantt charts or extended statistics over multiple runs.
* ​**requirements.txt**​: Lists all Python dependencies.

---

## 3. Requirements and Installation

* **Python 3.7+** (tested on Python 3.8/3.9+)
* Packages listed in `requirements.txt`:
  * `numpy`
  * `pandas`
  * `matplotlib`
  * `scipy`

​**Installation Steps**​:

1. Clone or download this project folder.
2. (Optional but recommended) Create a virtual environment:
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # or
    venv\Scripts\activate     # Windows
3. Install required packages:
    pip install -r requirements.txt

---

## 4. How to Run Each Script

### 4.1 `rta.py`

* ​**Purpose**​: Runs Response-Time Analysis (RTA) on a CSV file of tasks and prints the results.
* ​**Usage**​:
  python rta.py &lt;input_file.csv&gt;
* ​**Example**​:
  python rta.py data/TC1.csv
  
  This will:
  1. Load tasks from `TC1.csv`
  2. Compute the WCRT of each task using RTA
  3. Print out whether the set is schedulable (i.e., WCRT <= Deadline for all tasks)
  4. Print out each task’s WCRT vs. deadline

### 4.2 `vss_simulator.py`

* ​**Purpose**​: Simulates tasks under Rate Monotonic scheduling with variable execution times. Can run single or multiple simulations.
* ​**Usage**​:
  python vss_simulator.py &lt;csv_filename&gt; [options]
* ​**Key Options**​:
  * `--U_target`: Target CPU utilization (0 < U < 1) used for the “workload” method of execution-time generation.
  * `--method`: Execution time generation method: `workload` (default) or `truncnorm`.
  * `--runs`: Number of simulation runs (default=1).
  * `--simtime`: Simulation time. If not provided, uses LCM of all tasks’ periods.
  * `--verbose`: Print detailed log to console.
  * `--logfile`: If set, write logs to `output/logs/sim.log`.
* ​**Examples**​:
  1. **Single run** with default method (`workload`) and automatically computed simulation time:
     
     python vss_simulator.py data/TC1.csv --runs 1
     
     This will produce:
     * A Gantt chart saved to `output/images/gantt_chart.png`
     * Logs in `output/logs/sim.log` if `--logfile` is used.

  2. **Multiple runs** (e.g., 50 times) with target CPU utilization of 0.8:
     
     python vss_simulator.py data/TC1.csv --U_target 0.8 --method workload --runs 50 --logfile
     
     This will:
     * Run 50 simulations
     * Generate logs in `output/logs/sim.log`
     * Print out extended statistics (average, variance, max, etc.)

### 4.3 `compare.py`

* ​**Purpose**​: Compares **RTA** vs. **VSS** results for each task, showing WCRT from RTA and average WCRT from multiple simulation runs. Produces a bar chart comparison.

* ​**Usage**​:
  python compare.py &lt;csv_filename&gt; [options]

* ​**Key Options**​:
  * `--U_target`: Target CPU utilization for the VSS “workload” method.
  * `--method`: Execution time generation method: `workload` or `truncnorm`.
  * `--simtime`: Simulation time (default = LCM of periods).
  * `--runs`: Number of simulation runs (default=50).
  * `--logfile`: If set, write logs to `output/logs/compare.log`.
* ​**Example**​:
  
  python compare.py data/TC1.csv --U_target 0.6 --method workload --runs 50 --logfile
  
  This will:
  1. Run RTA on `TC1.csv`.
  2. Run 50 VSS simulations on the same CSV.
  3. Compare each task’s WCRT (RTA vs. average from simulation).
  4. Print a summary table to console and save a bar chart to `output/images/comparison_chart.png`.

### 4.4 `convergence_plot.py`

* ​**Purpose**​: Focuses on the **convergence** of the worst-case response times in simulation over multiple runs. It plots the **cumulative maximum** WCRT after each simulation run, overlaid with the RTA WCRT as a dashed line.
* ​**Usage**​:
  python convergence_plot.py &lt;csv_filename&gt; [options]
* ​**Key Options**​:
  * `--U_target`: Target CPU utilization for “workload” method.
  * `--method`: Execution time generation method: `workload` or `truncnorm`.
  * `--runs`: Number of simulation runs (default=1000).
  * `--simtime`: Simulation time (default = LCM of periods).
  * `--logfile`: If set, write logs to `output/logs/convergence.log`.
* ​**Example**​:
  python convergence_plot.py data/TC1.csv --U_target 0.7 --method workload --runs 500 --logfile
  
  This will:
  1. Load tasks and run RTA.
  2. Perform 500 simulation runs, each time measuring new WCRT if it exceeds the previous maximum.
  3. Plot a line for each task’s cumulative max WCRT vs. the run index, with RTA as a dashed line.
  4. Save the figure to `output/images/convergence.png`.

---

## 5. Test Cases

* ​**data/TC1.csv**​, ​**data/TC2.csv**​, ​**data/TC3.csv**​: Example task sets with different periods, WCET, deadlines, and priorities.
* ​**data/Full\_Utilization\_NonUnique\_Periods\_taskset.csv**​, ​**High\_Utilization\_NonUnique\_Periods\_taskset.csv**​, ​**Low\_Utilization\_NonUnique\_Periods\_taskset.csv**​: Additional examples for testing higher/lower CPU utilization and non-unique periods.

You can use these files directly with any of the scripts above. Each file contains columns:

Task, BCET, WCET, Period, Deadline, Priority

where **Task** is a label (e.g., T1, T2, …).

---

## 6. Output Files and Interpretation

1. **Logs** (`.log` files) in `output/logs/`:
   * Detailed information on each simulation run, including when tasks are released, completed, and their final WCRT.
2. **Images** in `output/images/`:
   * ​**gantt\_chart.png**​: Shows a Gantt chart of a single simulation run (if `--runs=1` in `vss_simulator.py`).
   * ​**comparison\_chart.png**​: Bar chart comparing RTA WCRT vs. VSS average WCRT (generated by `compare.py`).
   * ​**convergence.png**​: Line chart showing WCRT convergence over multiple runs (generated by `convergence_plot.py`).
3. ​**Console Output**​:
   * Each script prints a summary of tasks, deadlines, WCRT, and (for multi-run simulations) statistical measures like average WCRT, variance, 95th percentile, etc.

---

## 7. Example Workflow

A typical workflow might be:

1. **Run RTA** on a specific CSV:
   
   python rta.py data/TC1.csv

2. **Simulate** the same CSV for multiple runs:
   
   python vss_simulator.py data/TC1.csv --runs 50 --logfile
   
   * Check `output/logs/sim.log` for details and `output/images/gantt_chart.png` if `runs=1`.
3. **Compare** RTA vs. VSS results:
   
   python compare.py data/TC1.csv --runs 50 --logfile
   
   * View `comparison_chart.png` for a bar chart comparison.
4. **Check Convergence** of WCRT:
   
   python convergence_plot.py data/TC1.csv --runs 500 --logfile
   
   * View `convergence.png` to see if simulation WCRTs are approaching the RTA limit.

---

## 8. Packaging and Submission

If you need to submit this project:

1. **Zip** the entire folder (including all source code, CSV files, and output files).
2. Include this `README.md`.
3. Provide instructions to the grader or colleague to run:
   
   pip install -r requirements.txt
   python rta.py data/TC1.csv
   
   for any script you wish to demonstrate.

---

## 9. Troubleshooting

* ​**No module named ...**​: Make sure you installed all dependencies via `pip install -r requirements.txt`.
* ​**Permission denied / Cannot open log file**​: Check your folder permissions or run Python with the appropriate rights.
* ​**Plots not showing**​: By default, the scripts save plots to `output/images/` and also show them in a pop-up window. If running on a headless server, you may not see the pop-up, but the file will still be saved.