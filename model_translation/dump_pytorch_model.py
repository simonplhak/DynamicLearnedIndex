from pathlib import Path
import torch
from torch import nn
from safetensors.torch import save_file


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

for module in model.modules():
    if isinstance(module, nn.Linear):
        # Initialize weights (param) and biases
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

##### SAVE MODEL ######
state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
print(state)
out = Path('model.safetensors')
save_file(state, out)
print(f'Safetensors saved to: {out}')


##### TEST INPUTS ######
test_input = torch.randn(BATCH_SIZE, N_INPUT, device="cpu", dtype=torch.float32)
model.eval()
with torch.no_grad():
    test_output = model(test_input)

test_state = {
    "input": test_input,
    "output": test_output.cpu().detach(),
}
test_out = Path('test_data.safetensors')
save_file(test_state, test_out)
print(f'Test data (input/output) saved to: {test_out}')