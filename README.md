# NOMA_net
Dynamic wireless cellular network simulator using NOMA as Resource Sharing Mechanism and WDBO as primary power allocation strategy.
For macOS users, due to un-avaiable functions for W-DBO on macOS, run the simulator using @docker.
Modify the relative path of your project folder in "run.sh" file.
To to build the docker image, being insided the "docker_init" folder, type:
cmd docker build -t- ubuntu-wdbo .
thus to generate container for the project run ./run.sh .

Porject:
- run.ipynb provides a toy-notebook to run a single simulation instances, and display the perfromance metrics and network visualization.
# NOMA_net

**NOMA_net** is a dynamic wireless cellular network simulator that utilizes **Non-Orthogonal Multiple Access (NOMA)** for resource sharing and implements the **Dynamic Basin Optimization (W-DBO)** algorithm as the primary power allocation strategy, with alternatives including constant and random power allocation.

For **macOS** users:  
Due to unavailable W-DBO functions, it is recommended to run the simulator using **Docker**. Please modify the relative path of your project folder in the `run.sh` file.

## Building and Running the Docker Image

1. Navigate to the `docker_init` folder.
2. Build the Docker image:
   ```bash
   docker build -t ubuntu-wdbo .
   ```
3. To run the container, execute:
   ```bash
   ./run.sh
   ```

## Table of Contents
- [Project Description](#project-description)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [File Structure](#file-structure)
- [References and Further Reading](#references-and-further-reading)

## Project Description

This project simulates a dynamic cellular network environment where users move based on various mobility models (defined in `mobility_class.py`). The goal is to evaluate the effectiveness of different power allocation strategies:

- **Dynamic Basin Optimization (W-DBO)**: A novel strategy tailored for the network.
- **Constant Power Levels**: Predefined power levels for resource allocation.
- **Random Power Levels**: Random allocation of power resources.

The simulator allows observation of network performance metrics and visualization of network dynamics. For a detailed understanding of the W-DBO algorithm, refer to the accompanying paper (link to be provided). Visualizations in the project report provide further insights into simulation outcomes.

## Installation and Setup

1. Clone this repository:
   ```bash
   git clone <repository-url>
   ```
2. **Docker**: Follow the instructions in the `docker_init` folder to set up and run the simulation in a Docker container.
3. **Notebook**: The `run.ipynb` provides a toy notebook to run a single simulation instance, displaying both performance metrics and network visualizations.

## Usage

The main simulation files are located in the `simulation_files` directory, which contains Python scripts for managing mobility models, base station control, and performance visualization. You can modify the mobility models, power allocation methods, and user movement patterns directly in the provided scripts.

### Example Usage
After building the Docker image or setting up the environment, run the simulation using:
- `run.ipynb` for a guided simulation instance via Jupyter Notebook.

## File Structure

- **General Directory**:
  - `docker_init/`: Contains Docker initialization scripts for macOS users.
  - `project/`: Main project folder (details below).
  - `LICENSE`: License information for the repository.
  - `README.md`: This documentation file.
  - **PDF Documents**: Technical insights papers and presentations:
    - `DBO_for_cellular_networks.pdf`
    - `Dynamic_Bayesian_Optimization_works.pdf`

- **Project Directory**:
  - `pre_processing/`: Prepares and processes datasets for simulation.
  - `city_datasets/`: Contains datasets for different urban environments.
  - `bs_eda.ipynb`: Exploratory data analysis for base stations.
  - `simulation_files/`: Main folder containing the simulation scripts.
  - `libs/`: Libraries used by the simulation, including installation files.
  - `install.py, libraries.py`: Utility functions for library management.
  - `mobility/`: Handles user mobility in the simulation.
    - `mobility_class.py`: Defines various mobility models.
    - `base_station.py`: Manages base station interactions.
    - `controller.py`: Controls the simulation flow and logic.
    - `graphic_visualization.py`: Handles graphical output and network state visualization.
    - `main.py`: Entry point for running the simulation.
    - `performance_visualization_utils.py`: Utilities for visualizing performance metrics.
    - `simulation_env.py`: Sets up the simulation environment.
    - `user.py`: Manages user-related aspects such as movement and connection.
    - `utility.py`: General utility functions for the simulator.
  - `run.ipynb`: Jupyter notebook for hands-on simulation experience.
  - `output/`: Stores simulation results.

## References and Further Reading

- Dynamic Basin Optimization (W-DBO) Algorithm Paper (link to be provided)
- Project Visualizations and Report
