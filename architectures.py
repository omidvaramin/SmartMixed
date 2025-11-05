"""
Neural Network Architectures for MNIST experiments.
Each architecture is defined with a number and name for easy reference.
"""

ARCHITECTURES = {
    1: {
        'name': 'simple_2layer',
        'layers': [
            [512, None],
            [10, 'softmax']
        ]
    },
    
    2: {
        'name': 'medium_4layer',
        'layers': [
            [512, None],
            [256, None],
            [128, None],
            [10, 'softmax']
        ]
    },
    
    3: {
        'name': 'medium_5layer',
        'layers': [
            [512, None],
            [256, None],
            [128, None],
            [64, None],
            [10, 'softmax']
        ]
    },
    
    4: {
        'name': 'small_3layer',
        'layers': [
            [256, None],
            [128, None],
            [10, 'softmax']
        ]
    },
    
   
    
    5: {
        'name': 'medium_5layer_alt',
        'layers': [
            [512, None],
            [256, None],
            [256, None],
            [128, None],
            [10, 'softmax']
        ]
    },
    
    6: {
        'name': 'wide_2layer',
        'layers': [
            [768, None],
            [10, 'softmax']
        ]
    },
    
    7: {
        'name': 'wide_3layer',
        'layers': [
            [768, None],
            [512, None],
            [10, 'softmax']
        ]
    },
    
    8: {
        'name': 'wide_4layer',
        'layers': [
            [768, None],
            [512, None],
            [512, None],
            [10, 'softmax']
        ]
    },
    
    11: {
        'name': 'wide_5layer',
        'layers': [
            [768, None],
            [512, None],
            [512, None],
            [256, None],
            [10, 'softmax']
        ]
    },
    
    9: {
        'name': 'wide_6layer',
        'layers': [
            [768, None],
            [512, None],
            [512, None],
            [256, None],
            [256, None],
            [10, 'softmax']
        ]
    },
    
    10: {
        'name': 'wide_7layer',
        'layers': [
            [768, None],
            [512, None],
            [512, None],
            [256, None],
            [256, None],
            [128, None],
            [10, 'softmax']
        ]
    },
    
    11: {
        'name': 'wide_8layer',
        'layers': [
            [768, None],
            [512, None],
            [512, None],
            [256, None],
            [256, None],
            [128, None],
            [128, None],
            [10, 'softmax']
        ]
    },
    
    12: {
        'name': 'wide_9layer',
        'layers': [
            [768, None],
            [512, None],
            [512, None],
            [256, None],
            [256, None],
            [128, None],
            [128, None],
            [64, None],
            [10, 'softmax']
        ]
    },
    
    13: {
        'name': 'wide_10layer',
        'layers': [
            [768, None],
            [512, None],
            [512, None],
            [256, None],
            [256, None],
            [128, None],
            [128, None],
            [64, None],
            [64, None],
            [10, 'softmax']
        ]
    },
    
    14: {
        'name': 'wide_11layer',
        'layers': [
            [768, None],
            [512, None],
            [512, None],
            [256, None],
            [256, None],
            [128, None],
            [128, None],
            [64, None],
            [64, None],
            [32, None],
            [10, 'softmax']
        ]
    },
    
    15: {
        'name': 'wide_12layer',
        'layers': [
            [768, None],
            [512, None],
            [512, None],
            [256, None],
            [256, None],
            [128, None],
            [128, None],
            [64, None],
            [64, None],
            [32, None],
            [32, None],
            [10, 'softmax']
        ]
    },
    
    16: {
        'name': 'wide_13layer',
        'layers': [
            [768, None],
            [512, None],
            [512, None],
            [256, None],
            [256, None],
            [128, None],
            [128, None],
            [64, None],
            [64, None],
            [32, None],
            [32, None],
            [16, None],
            [10, 'softmax']
        ]
    },
    
    17: {
        'name': 'wide_14layer',
        'layers': [
            [768, None],
            [512, None],
            [512, None],
            [256, None],
            [256, None],
            [128, None],
            [128, None],
            [64, None],
            [64, None],
            [32, None],
            [32, None],
            [16, None],
            [16, None],
            [10, 'softmax']
        ]
    }
}

def get_architecture(arch_id):
    """Get architecture by ID."""
    if arch_id not in ARCHITECTURES:
        raise ValueError(f"Architecture {arch_id} not found. Available architectures: {list(ARCHITECTURES.keys())}")
    return ARCHITECTURES[arch_id]

def get_default_architecture():
    """Get a default architecture (first available one)."""
    if not ARCHITECTURES:
        raise ValueError("No architectures available")
    default_id = min(ARCHITECTURES.keys())
    return ARCHITECTURES[default_id]

def list_architectures():
    """List all available architectures."""
    print("Available architectures:")
    for arch_id, arch_info in ARCHITECTURES.items():
        layers_count = len(arch_info['layers'])
        print(f"  {arch_id}: {arch_info['name']} ({layers_count} layers)")

def get_all_architecture_ids():
    """Get list of all architecture IDs."""
    return list(ARCHITECTURES.keys())
