# Agent-based Implementations for Infectious Disease Transmission Models
Agent-based Implementations for Infectious Disease Transmission Models.

# Simulators
## Configuration
Configuration files are located in `/configs` folder. Simulator takes CLI parameters and uses them to override default 
configuration from `/configs/*_base.toml`, i.e. CLI parameters have precedence over configuration parameters.

Simulators takes simulation parameters either through config file or CLI arguments and run an agent-based simulation 
for a given set of infection and environment characteristics.

## SIR Simulator
Simulator for basic SIR model (Susceptible, Infected, Recovered) is implemented in `sir_sim.py`. 

### Metrics collection
For every time unit of the simulation a snapshot of metrics is taken. Upon completion, the simulator outputs a CSV-file
to an output location along with simulation parameters JSON.

### Agent FSM Graph
![SIRBasicFSMAgent.png](docs%2Fimg%2FSIRBasicFSMAgent.png)

### CLI Argument Reference
Usage example:
```shell
python sir_sim.py -o simulation_data \
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

### Metrics collection
For every time unit of the simulation a snapshot of metrics is taken. Upon completion, the simulator outputs a CSV-file
to an output location along with simulation parameters JSON.

### Agent FSM Graph
![SEIRClassicFSMAgent.png](docs%2Fimg%2FSEIRClassicFSMAgent.png)

### CLI Argument Reference
Usage example:
```shell
python seir_sim.py -o simulation_data \
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

## SEIR with Neighbors on a 2D Plane Simulator
Simulator for SEIR with Neighbors on a 2D plane model is implemented in `seir_neighbors_sim.py`.
It features neighbor interaction. Susceptible agents can transition to Exposed state only if their neighbors 
are Infectious with a probability of $P=\beta*N_{infected}$, where $N_{infected}$ is a number of infected neighbors.

Neighbors are preconfigured through `src/generators/neighbors.py` util.

### Agent FSM Graph
Agents come in two flavors:
* SEIRNeighborsFSMAgent - Base implementation
* SEIRNeighborsFSMExtended - Extended implementation

SEIRNeighborsFSMAgent:
![SEIRNeighborsFSMAgent.png](docs%2Fimg%2FSEIRNeighborsFSMAgent.png)

SEIRNeighborsFSMExtended:
![SEIRNeighborsFSMExtended.png](docs%2Fimg%2FSEIRNeighborsFSMExtended.png)

### Metrics collection
The simulator logs state transitions for every agent. Upon completion, the simulator outputs a CSV-file
to an output location along with simulation parameters JSON.

### Dynamic Immunity
Dynamic immunity in `SEIRNeighborsFSMExtended` is modeled via [Pascal's Limaçon](https://en.wikipedia.org/wiki/Lima%C3%A7on) function, 
adjusted for the purposes of the agent setup. 

Immunity at a specific day of the year is implemented in [immunity_by_year_day](https://github.com/AlekseiAgarkov/AgenticInfectiousDiseaseTransmissionModels/blob/main/src%2Fmodels%2Fimmunity.py#L27) function.
Basically, it takes day of the year and outputs current immunity value. The lower the immunity the higher the chance
to get infected and vice versa. Immunity is at lowest around beginning of the year and at highest at day 182.

![Immunity Modeling.png](docs%2Fimg%2Fimmunity%2FImmunity%20Modeling.png)

See visualization and experiments in more detail in [ImmunityModeling.ipynb](analytics%2FImmunityModeling.ipynb).

### Age sampling and immunity adjustment
[Extended SEIR Simulation](https://github.com/AlekseiAgarkov/AgenticInfectiousDiseaseTransmissionModels/blob/main/src/models/simulation.py#L92) at 
agent configuration stage [samples age](https://github.com/AlekseiAgarkov/AgenticInfectiousDiseaseTransmissionModels/blob/main/src/models/simulation.py#L92)
and updates [lower](https://github.com/AlekseiAgarkov/AgenticInfectiousDiseaseTransmissionModels/blob/main/configs/seir_neighbors_ext_test.toml#L16) and [upper](https://github.com/AlekseiAgarkov/AgenticInfectiousDiseaseTransmissionModels/blob/main/configs/seir_neighbors_ext_test.toml#L17) 
global immunity boundaries with age-dependent [immunity reduction factor](https://github.com/AlekseiAgarkov/AgenticInfectiousDiseaseTransmissionModels/blob/main/configs/seir_neighbors_ext_test.toml#L29)
with [adjust_immunity_by_mid_proportional](https://github.com/AlekseiAgarkov/AgenticInfectiousDiseaseTransmissionModels/blob/main/src/models/immunity.py#L49) function.

Age is sampled in two steps:
1. Age range is sampled from [age ranges](https://github.com/AlekseiAgarkov/AgenticInfectiousDiseaseTransmissionModels/blob/main/configs/seir_neighbors_ext_test.toml#L27) by [range-assigned probability](https://github.com/AlekseiAgarkov/AgenticInfectiousDiseaseTransmissionModels/blob/main/configs/seir_neighbors_ext_test.toml#L28) as defined in [age section of config](https://github.com/AlekseiAgarkov/AgenticInfectiousDiseaseTransmissionModels/blob/main/configs/seir_neighbors_ext_test.toml#L26).
2. Age is randomly sample from uniform distribution between lower and upper bracket of age range sampled at previous step.

### CLI Argument Reference
#### SEIRNeighborsFSMAgent
Usage example:
```shell
python seir_neighbors_sim.py -o simulation_data \
-c configs/seir_neighbors_base.toml \
-r 42 \
-n 1000 \
-t 365 \
-b 0.025 \
--e1 1 \
--e2 3 \
--t1 5 \
--t2 14
```

Options:
* `-h`, `--help` - show help message and exit

Paths:
*  `-o`, `--output_path` - Simulation data output folder
*  `-c`, `--config_path` - Config path
* `--neighbors_data_path` - Path to Neighbors Data 

Simulation Parameters:
* `-r`, `--random_seed` - Random seed
* `-n`, `--n_agents` - Number of agents
* `-t`, `--sim_duration` - Duration of Simulation, units
* `-i`, `--initially_infected` - Number of initially infected agents

Agent Parameters:
* `-b`, `--beta` - Model parameter Beta
* `e1`, Minimal Exposed State Duration
* `e2`, Maximal Exposed State Duration
* `t1`, Minimal Infected State Duration
* `t2`, Maximal Infected State Duration

#### SEIRNeighborsFSMExtended
Usage example:
```shell
python seir_neighbors_ext_sim.py -o simulation_data \
-c configs/seir_neighbors_ext_test.toml \
-r 42 \
-n 1000 \
-t 365 \
-b 0.025 \
--e1 1 \
--e2 3 \
--t1 5 \
--t2 14
```

Options:
* `-h`, `--help` - show help message and exit

Paths:
*  `-o`, `--output_path` - Simulation data output folder
*  `-c`, `--config_path` - Config path
* `--neighbors_data_path` - Path to Neighbors Data 

Simulation Parameters:
* `-r`, `--random_seed` - Random seed
* `-n`, `--n_agents` - Number of agents
* `-t`, `--sim_duration` - Duration of Simulation, units
* `-i`, `--initially_infected` - Number of initially infected agents

Agent Parameters:
* `-b`, `--beta` - Model parameter Beta
* `e1`, Minimal Exposed State Duration
* `e2`, Maximal Exposed State Duration
* `t1`, Minimal Infected State Duration
* `t2`, Maximal Infected State Duration

### Neighbors generator reference
This utility generates a square plane and populates it with agents. Each agent is randomly assigned coordinates and 
gets its neighbor assigned. Result is saved to a CSV file, with the following columns:
- x: float - X coordinate
- y: float - Y coordinate
- neighbors: str - Neighbor indices list. Data structure is Python List, which is wrapped as a string. 
 
Should be parsed with Pandas `pd.read_csv`:
```python
pd.read_csv(path, converters={'neighbors': ast.literal_eval})
```

Assigns name of the output file according to the `NeighborsMap-<s>by<s>-Agents<a>-Neighbors<n>-<TS>` convention, where:
- `s` - length of 2D plane square side;
- `a` - number of agents to generate;
- `n` - number of neighbors any agent has. Number also includes agent itself, so number of neighbors is effectively `n-`.

Resulting 2D plane with agents distributed and neighbors assigned is as follows 
(with arbitrary agent in red and its neighbors in blue):
![neighbors_map.png](docs%2Fimg%2Fneighbors_map.png)

The Neighbor map visualization tool is implemented in [neighbors.ipynb](analytics%2Fneighbors.ipynb).

#### CLI Argument Reference
Usage example:
```shell
python src/generators/neighbors.py -o simulation_conditions_data/neighbors \
-a 100 \
-n 5 \
-s 100
```

Options:
* `-h`, `--help` - show help message and exit

Paths:
* `-o`, `--output_path` - Data output folder

Parameters:
* `-r`, `--random_seed` - Random seed
* `-a`, `--agents_number` - Number of agents to generate
* `-n`, `--neighbors` - Number of neighbors per agent to assign
* `-s`, `--size` - Length of 2D plane square side
