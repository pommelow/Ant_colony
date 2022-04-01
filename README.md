# Ant_Colony_ST7_SHP_2021_2022_Intel_Group_2

# Motivation

Our project is to apply Ant colony optimization on the [iso3Dfd problem](https://www.intel.com/content/www/us/en/developer/articles/technical/iso3dfd-code-walkthrough.html) to find the best parameters (compilation parameters, space parameters, cache block, number of threads, etc...) in order to optimize the speed of calculation (given in Flops). 

# Run the project and reproduce the experiments

- First, run the python code "make_all.py" to create different execution files with all combination of compilation parameters

- To run a colony on a single node, use the main.py file on which you can add parse arguments :
  
  - Colony's hyper parameters : alpha (default=1), beta , rho , Q , nb_ant, method, nb-to-update, tau_min, tau_max
  - Addition on local search method or not : local-search (default="identity"), time_local, kmax, t0, t-decay, t_min, tabu-size
  - Maximum time of execution of the ant colony : max_time (default=1800)
  - distribution set to "independant"
 
  Example : <code> python main.py --alpha 2 --local-seach simulated_annealing --max_time 600 </code>
  
  The results will be saved in the "Results" directory as "Result_final_pickle" and the vizualisation of the best cost over time can be plot by running the view_result.py file 
  
- To create a colony running on several nodes, you can change the options in the benchmark_30_01.sh file and run a <code> sbatch benchmark_30_01.sh </code> in the shell. The results will be displayed as "RunX" in the Results directory, to visualize it, 






