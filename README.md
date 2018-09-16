# ECG-NN-Classifier

Components to train a neural network to classify ecg beats using Keras.

## Installation

1. Make sure you're using python 3.6+ for this project.
2. Run `pip3 install -r requirements.txt` to install project dependencies.
3. Execute the `main.py` script to train a neural network.

*Note:* This project contains the fully-connected model, the conv1d model and the conv2d model.
Those are provided in the `nn` submodule. When replacing the model used in `main.py` some additional changes must be made
to be trained.