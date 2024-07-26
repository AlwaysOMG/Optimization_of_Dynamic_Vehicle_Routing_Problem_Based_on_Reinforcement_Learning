# Optimization of Dynamic Vehicle Routing Problem Based on Reinforcement Learning

## Introduction
My research addresses vehicle routing problems with soft time windows and dynamic travel times by adopting an improved dynamic attention model and reinforcement learning methods.

Specifically, my research incorporates feature fusion and time advancement mechanisms into the previously proposed dynamic attention model and simplifies the solution process to effectively address the characteristics of time windows and dynamic changes in the problem. Additionally, reinforcement learning methods are used to train the model, enabling it to gradually improve its solution performance.

## Features
- Customizable parameters for dynamic vehicle routing problems with soft time window.
- Training and tracking of the dynamic attention model.
- Testing with various methods, including ALNS, hybrid BSO and ACO, GA, and RL.
- Generation of test results with details such as total cost, travel cost,  penalty cost, number of violated customers, and computation time.
- Visualization of vehicle routes.

## Requirements
- Python
    - NumPy
    - PyTorch
    - TensorBoard
    - ALNS
    - tqdm
    - Matplotlib

## Usage

1. **Install Dependencies**
    - Install basic dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    - Install `torch` according to your environment:
      - **If you use CUDA 11.7**:
      ```bash
      pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
      ```
      - **If you use CPU or a different CUDA version**:
      Please refer to the [PyTorch official installation guide](https://pytorch.org/get-started/locally/) for the correct installation command.

2. **Configure Vehicle Routing Problem Parameters**  
    Create or modify the `config.cfg` file with the following content:
    ```ini
    [instance]
    map_size = 10
    customer_num = 50
    vehicle_num = 5
    vehicle_capacity = 150
    customer_demand_lower_limit = 1
    customer_demand_upper_limit = 25
    time_window_lower_limit = 0
    time_window_upper_limit = 40
    early_penalty_lower_limit = 0
    early_penalty_upper_limit = 0.2
    late_penalty_lower_limit = 0
    late_penalty_upper_limit = 1
    travel_time_cv = 0.2
    travel_time_update_interval = 5
    ```

    - **Parameter Descriptions**:
      - `map_size`: Size of the map.
      - `customer_num`: Number of customers.
      - `vehicle_num`: Number of vehicles.
      - `vehicle_capacity`: Capacity of each vehicle.
      - `customer_demand_lower_limit` and `customer_demand_upper_limit`: Range of customer demands.
      - `time_window_lower_limit` and `time_window_upper_limit`: Range of time windows.
      - `early_penalty_lower_limit` and `early_penalty_upper_limit`: Range of penalties for early service.
      - `late_penalty_lower_limit` and `late_penalty_upper_limit`: Range of penalties for late service.
      - `travel_time_cv`: Coefficient of variation for travel time.
      - `travel_time_update_interval`: Time interval for updating travel times.

3. **Configure Model and Training Hyperparameters**  
    Modify the `[parameter]` and `[train]` sections of `config.cfg`:
    ```ini
    [parameter]
    embed_dim = 128
    num_layers = 3
    num_heads = 4
    clip_c = 10

    [train]
    learning_rate = 0.0001
    batch_size = 128
    epochs_num = 50
    steps_num = 100
    ```

    - **Parameter Descriptions**:
      - `embed_dim`: Dimension of the embedding layer.
      - `num_layers`: Number of layers in the encoder for the feature extraction.
      - `num_heads`: Number of heads in the attention mechanism.
      - `clip_c`: Value for attention score clipping in the ProbLayer in the decoder.
      - `learning_rate`: Learning rate.
      - `batch_size`: Size of each batch.
      - `epochs_num`: Total number of training epochs.
      - `steps_num`: Number of steps per epoch.

4. **Train the Model**  
    Run the following command to train the model:
    ```bash
    python train.py
    ```
    Use TensorBoard to monitor the training process:
    ```bash
    tensorboard --logdir='log'
    ```

5. **Perform Testing**  
    Modify the `test.py` file with the following parameters:
    ```python
    parameter_dir = 'path/to/your/parameters'  # Replace with the actual path to your parameters
    is_plot = False  # Set to True to enable plotting vehicle routes
    is_instance_analysis = False  # Set to True to show detail performance index
    ```

    - **Parameter Descriptions**:
      - `parameter_dir`: Directory containing the model parameters.
      - `is_plot`: Whether to enable routes plotting.
      - `is_instance_analysis`: Whether to enable show detail performance index.
    
    
    In the `test.py` script, you can switch between different methods by modifying the code.
      ```python
      # RL
      action = agent.get_action(obs, True)
      # ALNS
      action = ALNS_Solver(obs).run()
      # GA
      action = GA(obs).run()
      # Hybrid BSO and ACO
      action = BSO_ACO(obs).run()
      ```
