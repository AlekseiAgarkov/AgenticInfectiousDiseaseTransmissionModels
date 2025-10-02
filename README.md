# Agent-based Implementations for Infectious Disease Transmission Models
Agent-based Implementations for Infectious Disease Transmission Models.

# Simulators
## Configuration
Configuration files are located in `/configs` folder. Simulator takes CLI parameters and uses them to override default 
configuration from `/configs/*_base.toml`, i.e. CLI parameters have precedence over configuration parameters.

## SIR Simulator
Simulator for basic SIR model (Susceptible, Infected, Recovered) is implemented in `sir_sim.py`. 
The Simulator takes simulation parameters either through config file or CLI arguments and runs an agent-based simulation 
for a given set of infection characteristics.

For every time unit of the simulation a snapshot of metrics is taken. Upon completion, the simulator outputs a CSV-file
to an output location along with simulation parameters JSON.

### CLI Argument Reference
Usage example:
```shell
sir_sim.py -o simulation_data \
-c configs/sir_base.toml \
-r 42 \
-n 1000 \
-t 365 \
-b 0.025 \
-g 0.05
```

Options:
* `-h`, `--help` - show help message and exit

Paths:
*  `-o`, `--output_path` - Simulation data output folder
*  `-c`, `--config_path` - Config path

Simulation Parameters:
* `-r`, `--random_seed` - Random seed
* `-n`, `--n_agents` - Number of agents
* `-t`, `--sim_duration` - Duration of Simulation, units

Agent Parameters:
* `-b`, `--beta` - Model parameter Beta
* `-g`, `--gamma` - Model parameter Gamma

## SEIR Simulator
Simulator for basic SEIR (Susceptible, Exposed, Infected, Recovered) model is implemented in `seir_sim.py`. 
The Simulator takes simulation parameters either through config file or CLI arguments and runs an agent-based simulation 
for a given set of infection characteristics.

For every time unit of the simulation a snapshot of metrics is taken. Upon completion, the simulator outputs a CSV-file
to an output location along with simulation parameters JSON.

### CLI Argument Reference
Usage example:
```shell
seir_sim.py -o simulation_data \
-c configs/sir_base.toml \
-r 42 \
-n 1000 \
-t 365 \
-b 0.025 \
-g 0.05
```

Options:
* `-h`, `--help` - show help message and exit

Paths:
*  `-o`, `--output_path` - Simulation data output folder
*  `-c`, `--config_path` - Config path

Simulation Parameters:
* `-r`, `--random_seed` - Random seed
* `-n`, `--n_agents` - Number of agents
* `-t`, `--sim_duration` - Duration of Simulation, units

Agent Parameters:
* `-b`, `--beta` - Model parameter Beta
* `-g`, `--gamma` - Model parameter Gamma
* `-s`, `--sigma` - Model parameter Sigma

