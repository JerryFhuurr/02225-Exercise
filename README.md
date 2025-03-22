# Real-Time Scheduling and Analysis Project

This project contains several Python scripts designed to analyze and simulate fixed-priority real-time systems. Specifically, it covers:

- **Worst-Case Response Time Analysis (RTA)**
- **Rate Monotonic (RM) scheduling** simulation, including multiple runs
- **Convergence testing** (comparing multiple simulation runs against RTA results)
- **Comparisons** between RTA and simulation results (average WCRT vs. theoretical WCRT)

By providing different CSV task files, you can evaluate the schedulability of your system, the worst-case response times, and the statistical performance under real execution scenarios. The scripts also generate logs and charts for further analysis.

---

## Project Structure

```plaintext
.
├── data/
│   ├── Full_Utilization_NonUnique_Periods_taskset.csv
│   ├── High_Utilization_NonUnique_Periods_taskset.csv
│   ├── Low_Utilization_NonUnique_Periods_taskset.csv
│   ├── TC1.csv
│   ├── TC2.csv
│   └── TC3.csv
│
├── output/
│   ├── images/
│   │   ├── comparison_chart.png
│   │   ├── convergence.png
│   │   └── gantt_chart.png
│   └── logs/
│       ├── compare.log
│       ├── convergence.log
│       └── sim.log
│
├── .gitignore
├── compare.py
├── convergence_plot.py
├── README.md
├── requirements.txt
├── rta.py
└── vss_simulator.py

```plaintext


- **data/**: Contains sample or test CSV task files.
output/: Stores generated images (in images/) and logs (in logs/) after each run.

compare.py: Compares RTA results with multiple simulation runs.

convergence_plot.py: Runs multiple VSS simulations to show how WCRT converges, comparing against RTA results.

rta.py: Performs fixed-priority Worst-Case Response Time Analysis (RTA).

vss_simulator.py: Simulates tasks under Rate Monotonic scheduling, supports multiple runs, Gantt charts, and logs.

requirements.txt: Lists the third-party Python libraries needed.

README.md: This documentation file.

- **Worst-Case Response Time Analysis (RTA)**
- **Rate Monotonic (RM) scheduling** simulation, including multiple runs
- **Convergence testing** (comparing multiple simulation runs against RTA results)
- **Comparisons** between RTA and simulation results (average WCRT vs. theoretical WCRT)

By providing different CSV task files, you can evaluate the schedulability of your system, the worst-case response times, and the statistical performance under real execution scenarios. The scripts also generate logs and charts for further analysis.

