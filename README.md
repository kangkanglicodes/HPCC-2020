1. This project was developed by Yitao Qiu, an author of our paper accepted by IEEE HPCC-2020. This project is used for backup.

2. This project is developed in Python3. To run the programs, you need to install two python library named numpy and matplotlib.

3. When you open the project, you need to open the HPCC-2020 as the root directory, the module Simulator and Visualizer are dependent.

4. The entry function file is Simulator/Bootstrap.py.
   Function online_simulate_with_FCFS uses the FCFS strategy and function online_simulate_with_SJF
   uses the SJF strategy.
   
5. Simulator/Wireless.py implements the wireless HPC system with 1D torus inter-cabinet wireless interconnect and 1D torus intra-cabinet cable interconnect.

6. Simulator/Conventional.py implememts the conventional Blue Waters system with 3D torus interconnect

7. Visualization/System.py implements the visualization of the system utilization of a HPC system with 3D torus interconnect.

8. If you use PyCharm to run the Visualization/System.py, you may need to edit the configuration before run.
