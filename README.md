---

# Layerless Neural Networks: An Experimental SDE-Based Architecture

This project introduces a proprietary approach to constructing **layerless neural networks**, inspired by the structure of mathematical functionals. The core hypothesis is to create an architecture with a theoretically lower demand for computational resources (GPU, CPU, RAM) during both training and inference, compared to traditional multi-layered models.

## Concept

The core of the system is a predictive model based on **Stochastic Differential Equations (SDEs)**. Instead of building a deep, multi-layered network, the model attempts to learn the process dynamics directly. This architecture is a hybrid of classical estimation methods and machine learning, where the **Kalman Filter** plays a pivotal role.

### The Role of the Kalman Filter

The Kalman Filter is at the heart of the advanced models in this project. It performs several key functions:

1.  **Hidden State Estimation:** The filter assumes that the observed process (e.g., price) is a noisy version of a true, hidden state. Its task is to estimate this hidden, "clean" state from the noisy observations.
2.  **Guiding the SDE Process:** In the most advanced model (`KALMANGuidedAttentionSDE`), the Kalman Filter is not merely a passive observer. Its internal, learned dynamics (the `F matrix`) provide guidance for the main SDE process, informing it about the probable future trend direction.
3.  **Noise Reduction and Stabilization:** Acting as a smoothing component, the filter helps to stabilize predictions and makes the model more resilient to sudden, random fluctuations.

With this approach, instead of relying solely on the "brute-force" of deep networks, the system leverages a proven mathematical framework to model uncertainty and estimate trends.

## Core Components

The system is divided into three main, independent components:

1.  **Main Predictor (`KALMAN_FILTER_SDE_LAYERLESS_MODEL`)**
    *   `NEURAL__sde_pure_gaussian_process.py`: The main process that connects to a data source (e.g., Binance), trains the advanced SDE model with a built-in Kalman Filter, consumes "AI features," and generates final predictions.
    *   `accuracy_monitor.py`: Monitors and records the model's historical accuracy.
    *   `data_replicator.py`: Replicates and prepares data for further analysis.
    *   `autoreflection_strategist.py`: A decision-making process that can modify the predictive strategy.
    *   `visualization_updater.py`: Prepares the output data for visualization in the user interface.

2.  **Analyzer / AI Feature Producer (`ANALYZER_MODEL`)**
    *   `run_analyzer.py`: A background process that analyzes historical data, trains a simpler SDE model, and then extracts its internal weights as "AI features," saving them to a file. These features are then utilized by the Main Predictor.

3.  **User Interface (`FRONTEND`)**
    *   `run_frontend.py`: A FastAPI-based web application that serves an interactive chart, visualizing real-time prices, forecasts, and key metrics such as model accuracy and the performance of a simple autoreflection layer.

## Getting Started

### Prerequisites

*   Python 3.10+
*   An active virtual environment

### Installation Steps

1.  Clone the repository:
    ```bash
    git clone https://github.com/OriginalSolutions/LAYERLESS-KALMAN-SDE-FUNCNET.git
    ```

2.  Navigate to the project directory:
    ```bash
    cd LAYERLESS-KALMAN-SDE-FUNCNET
    ```
    *(Note: Your local parent directory might be named `LAYERLESS_NEURAL_NETWORKS`)*

3.  Create and activate the virtual environment:
    ```bash 
    python3 -m venv venv
    source venv/bin/activate 
    ```

4.  Install all required dependencies:
    ```bash 
    pip install -r requirements.txt 
    ```

### Running the System

All processes should be run from within separate terminal sessions with the virtual environment activated. The following commands will run each process in the background and save its logs to a corresponding file.

**1. Run the Main Predictor and its Helper Modules**

Navigate to the directory and run all processes:
```bash
cd BACKEND/KALMAN_FILTER_SDE_LAYERLESS_MODEL

nohup python3 -u NEURAL__sde_pure_gaussian_process.py > NEURAL__sde_pure_gaussian_process.log 2>&1 &
nohup python3 -u accuracy_monitor.py > accuracy_monitor.log 2>&1 &
nohup python3 -u data_replicator.py > data_replicator.log 2>&1 &
nohup python3 -u autoreflection_strategist.py > autoreflection_strategist.log 2>&1 &
nohup python3 -u visualization_updater.py > visualization_updater.log 2>&1 &
```

**2. Run the Analyzer (AI Feature Producer)**

In a second terminal, navigate to the analyzer directory:
```bash
cd BACKEND/ANALYZER_MODEL

nohup python3 -u run_analyzer.py > run_analyzer.log 2>&1 &
```

**3. Run the User Interface**

In a third terminal, navigate to the frontend directory:```bash
cd FRONTEND

nohup uvicorn run_frontend:app --host 127.0.0.1 --port 8050 > frontend.log 2>&1 &
```

**4. Accessing the Application**

After running all processes, open a web browser and navigate to:
[http://127.0.0.1:8050](http://127.0.0.1:8050)
