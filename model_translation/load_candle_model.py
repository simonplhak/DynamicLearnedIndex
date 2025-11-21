from pathlib import Path
import torch
from torch import nn
from safetensors.torch import load_model, load_file


##### MODEL DEFINITION #####
N_INPUT= 3
N_HIDDEN = 256
N_OUTPUT = 10
BATCH_SIZE = 4

torch.manual_seed(0)
model = nn.Sequential(
    # First Layer: Input to Hidden
    nn.Linear(N_INPUT, N_HIDDEN), 
    
    # Activation Function
    nn.ReLU(),
    
    # Second Layer: Hidden to Output
    nn.Linear(N_HIDDEN, N_OUTPUT),
)

##### LOAD WIGHTS ######
weights = Path('rust-model.safetensors')
load_model(model, weights)


##### COMPARE MODELS #####
inputs_path = Path('rust-test_data.safetensors')
test_data = load_file(inputs_path)
test_x, test_y = test_data['input'], test_data['output']
model.eval()
with torch.no_grad():
    test_output = model(test_x)

print(f'Total difference: {(test_y - test_output).sum()}')
