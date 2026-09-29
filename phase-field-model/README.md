**Generate the MOOSE simulation input files with script generate_moose_simulation_files.py.**

Run the simulations with MOOSE (you can use "batch.sh" for a batch run). After that, you have 2 ways to output data for machine learning:
1. export data from ".e" files using Paraview (5.9.0) as csv, then transfer to ".npy" files using python script "generative_ai/step0_make_cvae_dataset_from_Paraview.py"
2. **(Recommended)** directly use python script "generative_ai/step0_make_cvae_dataset_from_exodus.py" to process the ".e" output files and get transferred ".npy" files 

Note: You need the source files under MooseProject directory to run the simulations. 
Remember to replace the "newt" app name to your own app name.

Note: Some simulation results may not be physically reasonable (because of the randomly chosen parameters), you need to manually filter the results, or choose other ranges for different parameters.