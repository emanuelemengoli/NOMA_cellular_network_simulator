## Table of Contents
- [Project Description](#project-description)
- [Installation and Usage](#installation-and-usage)
- [File Structure](#file-structure)
- [References](#references)

## Project Description

**NOMA_net** is a dynamic wireless cellular network simulator that utilizes **Non-Orthogonal Multiple Access (NOMA)** as a resource-sharing mechanism and provides several motion models to approximate user equipment (UEs) trajectories. The goal of the simulator is to evaluate the effectiveness of a novel Bayesian Optimization algorithm, namely **W-DBO** [1], as the primary power allocation strategy. The performances are benchmarked against two simpler alternatives, i.e., constant and random power allocation.

The simulator allows for the observation of network performance metrics and visualization of network "breathing" dynamics. For a detailed understanding of the W-DBO algorithm [1], as well as the model setup and simulator logic [2,3], refer to the Reference section. Network visualizations used in the project report are available at the following [link](https://drive.google.com/drive/folders/1l24CaTQnXVrh6pgIBsbVXmZD273qOft4?usp=share_link).

## Installation and Usage

For **macOS** users:  
Due to unavailable W-DBO functions, it is recommended to run the simulator using **Docker**. Please modify the relative path of your project folder in the `run.sh` file.

### Building and Running the Docker Image

1. Navigate to the `docker_init` folder.
2. Build the Docker image:
   ```bash
   docker build -t ubuntu-wdbo .
   ```
3. To run the container, execute:
   ```bash
   ./run.sh
   ```
4. Inside **VSCode**, attach to the running container to run and modify the project files.

---

### Usage

1. Clone this repository:
   ```bash
   git clone <https://github.com/emanuelemengoli/NOMA_cellular_network_simulator.git>
   ```

2. **Run Notebook**: The `run.ipynb` provides a toy notebook to run a single simulation instance, displaying both performance metrics and network visualizations. Note that running the `main.py` file automatically installs the required dependencies. For more information, see `/project/simulation_files/libs/libraries.py` and `/project/simulation_files/libs/install.py`.

## File Structure

- **General Directory**:
  - `docker_init/`: Contains Docker initialization scripts for macOS users.
  - `project/`: Main project folder (details below).
  - **PDF Documents**: For technical insights:
    - `DBO_for_cellular_networks.pdf`
    - `Dynamic_Bayesian_Optimization_works.pdf`

The main simulation files are located in the `simulation_files` directory.

- **Files Summary**:
  - `city_datasets/`: Contains original urban datasets to retrieve base station (BS) data.
  - `bs_eda.ipynb`: Exploratory data analysis for BS dataset.
  - `install.py`, `libraries.py`: Library and installation management.
  - `mobility_class.py`: Defines various mobility models.
  - `base_station.py`: Defines BS instance.
  - `controller.py`: Controller instance to manage interactions between UEs and BSs.
  - `graphic_visualization.py`: Handles graphical output and network state visualizations.
  - `main.py`: Contains the simulation logic.
  - `performance_visualization_utils.py`: Utilities for visualizing performance metrics.
  - `simulation_env.py`: Sets up the simulation environment (can be generated from `bs_eda.ipynb`).
  - `user.py`: Defines UE instance.
  - `utility.py`: General utility functions for the simulator.
  - `run.ipynb`: Jupyter notebook for hands-on simulation experience.
  - `output/`: Stores simulation results.


## References
[1] Bardou, Anthony, Patrick Thiran, and Giovanni Ranieri. "This Too Shall Pass: Removing Stale Observations in Dynamic Bayesian Optimization." arXiv preprint arXiv:2405.14540 (2024). ([link](https://arxiv.org/pdf/2405.14540)).

[2]`DBO_for_cellular_networks.pdf (presentation)`**.s

[3]`Dynamic_Bayesian_Optimization_for_Improving_the_Performance_of_Cellular_Networks.pdf (report)`.

[4] [Report visualizations](https://drive.google.com/drive/folders/1l24CaTQnXVrh6pgIBsbVXmZD273qOft4?usp=share_link).


** In order to view gif-based animations within the presentation, use a pdf reader that supports them, e.g. Adobe Reader.