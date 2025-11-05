import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Flexible NN class
def get_activation(activation, num_neurons=None):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'softmax':
        return nn.Softmax(dim=1)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'selu':
        return nn.SELU()
    else:
        raise ValueError(f"Unknown activation: {activation}")

class CustomNeuralNetwork(nn.Module):
    def __init__(self, architecture, input_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for num_neurons, activation in architecture:
            layers.append(nn.Linear(prev_dim, num_neurons))
            layers.append(get_activation(activation, num_neurons))
            prev_dim = num_neurons
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class AdaptiveActivationGumbelSoftmax(nn.Module):
    def __init__(self, num_neurons, activation_pool, temperature=0.5, hard=True):
        super().__init__()
        self.num_neurons = num_neurons
        self.activation_pool = activation_pool
        self.num_activations = len(activation_pool)
        self.temperature = temperature
        self.hard = hard
        logits = torch.zeros(num_neurons, self.num_activations)
        indices = torch.randint(0, self.num_activations, (num_neurons,))
        logits[torch.arange(num_neurons), indices] = 0.001
        self.logits = nn.Parameter(logits)

    def sample_gumbel(self, shape, eps=1e-20, device=None):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_softmax(self, logits, tau=1.0, hard=None):
        if hard is None:
            hard = self.hard
        gumbel_noise = self.sample_gumbel(logits.size(), device=logits.device)
        y = (logits + gumbel_noise) / tau
        y_soft = torch.softmax(y, dim=-1)
        if hard:
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
            y = (y_hard - y_soft).detach() + y_soft
        else:
            y = y_soft
        return y

    def forward(self, x):
        weights = self.gumbel_softmax(self.logits, tau=self.temperature)
        activations = []
        for act_fn in self.activation_pool:
            act_out = act_fn(x)
            act_out = torch.clamp(act_out, -10, 10)
            activations.append(act_out)
        stacked = torch.stack(activations, dim=-1)
        out = (stacked * weights.unsqueeze(0)).sum(dim=-1)
        out = torch.clamp(out, -10, 10)
        return out

class NeuralNetworkWithAdaptiveActivation(nn.Module):
    def __init__(self, architecture, input_dim, activation_pool, temperature=0.5, hard=True):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i, (num_neurons, activation) in enumerate(architecture):
            layers.append(nn.Linear(prev_dim, num_neurons))
            if i < len(architecture) - 1:
                layers.append(AdaptiveActivationGumbelSoftmax(num_neurons, activation_pool, temperature, hard=hard))
            else:
                layers.append(get_activation(activation, num_neurons))
            prev_dim = num_neurons
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class PerNeuronActivation(nn.Module):
    def __init__(self, activation_names):
        super().__init__()
        self.activation_names = activation_names
        
        # Pre-compute activation groups for efficiency
        self.activation_groups = {}
        for i, act_name in enumerate(activation_names):
            if act_name not in self.activation_groups:
                self.activation_groups[act_name] = []
            self.activation_groups[act_name].append(i)
        
        # Convert to tensors for faster indexing during forward pass
        self.activation_indices = {}
        for act_name, indices in self.activation_groups.items():
            self.activation_indices[act_name] = torch.tensor(indices, dtype=torch.long)

    def get_activation_summary(self):
        """Return a summary of activation function distribution for this layer"""
        summary = {}
        for act_name, indices in self.activation_groups.items():
            summary[act_name] = len(indices)
        return summary

    def _get_fn(self, name):
        if name == 'relu':
            return torch.relu
        elif name == 'sigmoid':
            return torch.sigmoid
        elif name == 'tanh':
            return torch.tanh
        elif name == 'leaky_relu':
            return torch.nn.functional.leaky_relu
        elif name == 'elu':
            return torch.nn.functional.elu
        elif name == 'selu':
            return torch.nn.functional.selu
        else:
            raise ValueError(f"Unknown activation: {name}")
    def forward(self, x):
        # x: (batch, num_neurons)
        # Create output tensor
        out = torch.zeros_like(x)
        
        # Process each group of neurons with the same activation
        for act_name, neuron_indices in self.activation_indices.items():
            if len(neuron_indices) > 0:
                fn = self._get_fn(act_name)
                # Move indices to same device as input if needed
                if neuron_indices.device != x.device:
                    neuron_indices = neuron_indices.to(x.device)
                # Apply activation to all neurons of this type at once
                out[:, neuron_indices] = fn(x[:, neuron_indices])
        
        return out

def build_hard_activation_network(architecture, input_dim, activation_choices_per_layer, weights_path):
    """
    Build a network with the same architecture, but each neuron uses the activation function chosen during training.
    """
    layers = []
    prev_dim = input_dim
    for layer_idx, (num_neurons, activation) in enumerate(architecture):
        layers.append(nn.Linear(prev_dim, num_neurons))
        if layer_idx < len(architecture) - 1:
            # Use per-neuron activations
            act_names = activation_choices_per_layer[layer_idx]
            layers.append(PerNeuronActivation(act_names))
        else:
            layers.append(get_activation(activation, num_neurons))
        prev_dim = num_neurons
    model = nn.Sequential(*layers)
    # Load weights from the trained model
    state_dict = torch.load(weights_path, map_location='cpu')
    # Remove 'model.' prefix from keys in state_dict and only keep weights/biases
    stripped_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if k.startswith('model.') and (k.endswith('weight') or k.endswith('bias'))}
    model_state = model.state_dict()
    for k in model_state:
        if k in stripped_state_dict:
            model_state[k] = stripped_state_dict[k]
            print(f"Loaded weight for layer: {k}")
    model.load_state_dict(model_state)
    return model

class MixedNeuralNetwork(nn.Module):
    """
    A neural network where each neuron can have its own activation function.
    This is used to create a fixed version of the adaptive network.
    """
    def __init__(self, architecture, input_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        
        for layer_idx, (num_neurons, activation_spec) in enumerate(architecture):
            layers.append(nn.Linear(prev_dim, num_neurons))
            
            if isinstance(activation_spec, list):
                # This layer has per-neuron activations
                layers.append(PerNeuronActivation(activation_spec))
            else:
                # This layer has a single activation function
                layers.append(get_activation(activation_spec, num_neurons))
            
            prev_dim = num_neurons
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
