# IndicWav2vec Model Inference
This directory contains scripts and instructions to run Wav2Vec2 model by IITM SpringLab.


## Setup

### Method 1: Using Makefile
1. Ensure you have `uv` (Universal Virtualenv) installed. If not, install it using:
    ```bash
    pip install uv
    ```
2. Run the setup command:
    ```bash
    make setup
    ``` 

### Method 2: Manual setup
1. Create a virtual environment and activate it:
    ```bash
    uv venv --python=3.11
    source .venv/bin/activate
    ```
2. Install the dependencies:
    ```bash
    uv pip install -r requirements.txt
    uv pip install torch==2.8.0 --torch-backend=cpu
    ```
2. Download the model
    ```bash
    wget -O hindi.pt https://asr.iitm.ac.in/SPRING_INX/models/fine_tuned/SPRING_INX_ccc_wav2vec2_Hindi.pt
    ```
4. Run the script to download the model and test audio:
    ```bash
    python main.py
    ```

## Run the script

1. Activate the environment
    ```bash
    source .venv/bin/activate
    ```
2. Run the main script
    ```bash
    python main.py
    ```