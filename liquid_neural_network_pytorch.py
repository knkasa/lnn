import torch
import torch.nn as nn

# Liquid State Machine
class LiquidStateMachine(nn.Module):
    def __init__(self, input_size, liquid_size, output_size, spectral_radius=0.9):
        super(LiquidStateMachine, self).__init__()
        
        # Initialize the liquid layer
        self.liquid = nn.Linear(input_size + liquid_size, liquid_size, bias=False)
        self.liquid.weight.data = spectral_radius * torch.randn(liquid_size, input_size + liquid_size)
        
        # Initialize the readout layer
        self.readout = nn.Linear(liquid_size, output_size)
        
    def forward(self, inputs, liquid_state=None):
        batch_size = inputs.size(0)
        time_steps = inputs.size(1)
        
        outputs = []
        
        # Initialize the liquid state if not provided
        if liquid_state is None:
            liquid_state = torch.zeros(batch_size, self.liquid.out_features)
        
        for t in range(time_steps):
            input_step = inputs[:, t, :]
            liquid_state = torch.tanh(self.liquid(torch.cat([input_step, liquid_state], dim=1)))
            output_step = self.readout(liquid_state)
            outputs.append(output_step)
        
        outputs = torch.stack(outputs, dim=1)
        return outputs, liquid_state