# Running with Docker
## pulling down the images from github
* docker pull ghcr.io/junqi108/rootsim-container:latest

## list of commands in docker compose
* run docker-compose down to turn off the docker
* run docker-compose up to turn on the docker  
* change the command that you want to run in the docker-compose.yml file, like adding the python argument

## steps to run
* check the parameters in the parameter.py in the folder of root_system_lib 
* prepare the observed data to generate summary statistics using the default "rld_for_locations"
    * use the simulated data to generate a template observed data summary statistics, by changing the   
        following variables based on the experiment data:                
        "x_locations": [0.1,0.1,0.1],   # Example x coordinates        
        "y_locations": [0.5,1,1.5],   # Example y coordinates
        "x_tolerance": 0.2,     # Example tolerance for x
        "depth_interval": 0.1,  # Example depth interval in meters
        "ROOT_GROUP": "1,2"     #structural root or fine root
    * Change the values of the generated stats in the data/summary_stats/root_stats.csv
    * You can change its name to avoid accidentally overwriting, but after changing the name you should  
        change the file name in the root_gen_optimise as well  
        # Input 
        add_argument(parser, "--stats_file", "data/summary_stats/root_stats_1.csv", "The observed root 
        statistics file name", str)
* Check all the arguments in the root_gen_optimise.py especially the n_trials, stats_file
* Copy the root_optimise.csv and save it as root_optimise_res.csv
* call the root_gen.py to generate the architecture data with "--from_config", 1 "--dir", "data/optimise"
