import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import time
from models import build_hard_activation_network, MixedNeuralNetwork
from utils import save_loss_curves, save_confusion_matrix, per_class_accuracy, save_per_class_accuracy, save_total_accuracy, save_best_loss
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import datetime
import csv

def get_mnist_loaders(batch_size=64, small_size=False):
    """
    Load MNIST with proper official split:
    - Use official test set (10k samples) as final test set
    - Split training set (60k samples) into 90% train + 10% validation
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    # Load official training set (60k samples)
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    
    # Load official test set (10k samples)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Handle small_size mode
    if small_size:
        # Use only first 200 samples from training, and proportional test samples
        train_indices = np.arange(200)
        test_indices = np.arange(33)  # Keep roughly same proportion (200/60000 * 10000 ≈ 33)
        
        # Create test subset (33 samples from official test set)
        test_set = torch.utils.data.Subset(test_dataset, test_indices)
        
        # Split the 200 training samples: 180 train + 20 validation
        train_targets = np.array([train_dataset.targets[i] for i in train_indices])
        train_idx, val_idx = train_test_split(
            train_indices, test_size=0.1, stratify=train_targets, random_state=42
        )
        
        train_set = torch.utils.data.Subset(train_dataset, train_idx)
        val_set = torch.utils.data.Subset(train_dataset, val_idx)
    else:
        # Full dataset: Split 60k training into 54k train + 6k validation
        train_targets = np.array(train_dataset.targets)
        train_indices = np.arange(len(train_dataset))
        
        train_idx, val_idx = train_test_split(
            train_indices, test_size=0.1, stratify=train_targets, random_state=42
        )
        
        train_set = torch.utils.data.Subset(train_dataset, train_idx)
        val_set = torch.utils.data.Subset(train_dataset, val_idx)
        test_set = test_dataset  # Use full official test set (10k samples)
    
    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def get_all_mnist_loaders(batch_size=64, small_size=False, use_official_split=True):
    """
    Get all portions of the MNIST dataset.
    
    Args:
        batch_size: Batch size for DataLoaders
        small_size: If True, use only a small subset for quick testing
        use_official_split: If True, use official MNIST train/test split.
                           If False, create custom 3-way split from training data only.
    
    Returns:
        If use_official_split=True:
            train_loader, official_test_loader (using official MNIST split)
        If use_official_split=False:
            train_loader, val_loader, test_loader (custom 3-way split)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    if use_official_split:
        # Load both official training and test sets
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        
        if small_size:
            # Use only first 200 samples from training and 100 from test
            train_indices = list(range(min(200, len(train_dataset))))
            test_indices = list(range(min(100, len(test_dataset))))
            train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
            test_dataset = torch.utils.data.Subset(test_dataset, test_indices)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Official MNIST split loaded:")
        print(f"  - Training set: {len(train_dataset)} samples")
        print(f"  - Test set: {len(test_dataset)} samples")
        
        return train_loader, test_loader
    
    else:
        # Use the existing custom 3-way split logic
        return get_mnist_loaders(batch_size=batch_size, small_size=small_size)

def get_mnist_dataset_info():
    """
    Get information about all available MNIST dataset portions.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    # Load official datasets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Count samples per class in training set
    train_targets = np.array(train_dataset.targets)
    train_class_counts = np.bincount(train_targets)
    
    # Count samples per class in test set
    test_targets = np.array(test_dataset.targets)
    test_class_counts = np.bincount(test_targets)
    
    print("="*60)
    print("MNIST DATASET INFORMATION")
    print("="*60)
    print(f"Official Training Set: {len(train_dataset)} samples")
    print(f"Official Test Set: {len(test_dataset)} samples")
    print(f"Total MNIST Dataset: {len(train_dataset) + len(test_dataset)} samples")
    print()
    print("Class distribution in Training Set:")
    for i, count in enumerate(train_class_counts):
        print(f"  Class {i}: {count} samples")
    print()
    print("Class distribution in Test Set:")
    for i, count in enumerate(test_class_counts):
        print(f"  Class {i}: {count} samples")
    print("="*60)
    
    return {
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'train_class_counts': train_class_counts,
        'test_class_counts': test_class_counts,
        'total_size': len(train_dataset) + len(test_dataset)
    }

def get_output_dir(output_dir=None, model_name=None, force_dir=False):
    """
    Returns the output directory for saving results.
    If force_dir is True and output_dir is provided, use output_dir as the directory (full path expected).
    Otherwise, create a timestamped directory under 'outputs' using model_name.
    """
    import os
    import datetime
    if force_dir and output_dir is not None:
        out_dir = output_dir
    elif model_name is not None:
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join('outputs', f'{model_name}_{now}')
    else:
        raise ValueError('Must provide either output_dir (with force_dir=True) or model_name.')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def save_adaptive_activation_choices(model, output_dir, activation_names=None):
    """
    Save the most likely activation function chosen by each neuron in each adaptive layer to a CSV file.
    Each column is a layer, each row is a neuron, and the value is the activation name.
    Also saves a visualization of the network with colored neurons.
    """
    columns = []
    layer_names = []
    adaptive_layer_idx = 1  # Logical index for adaptive layers
    for layer in model.model:
        if hasattr(layer, 'logits'):
            # Get the argmax activation index for each neuron
            choices = layer.logits.detach().cpu().numpy().argmax(axis=1)
            if activation_names is not None:
                choices = [activation_names[idx] for idx in choices]
            columns.append(choices)
            layer_names.append(f'Layer_{adaptive_layer_idx}')
            adaptive_layer_idx += 1
    # Pad columns to the same length
    max_len = max(len(col) for col in columns)
    padded_cols = [list(col) + [None]*(max_len - len(col)) for col in columns]
    # Write to CSV
    import csv
    with open(os.path.join(output_dir, 'adaptive_activation_choices.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(layer_names)
        for row in zip(*padded_cols):
            writer.writerow(row)
    # Visualization
    if columns:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        color_map = {'relu': 'green', 'sigmoid': 'blue', 'tanh': 'orange', 'softmax': 'red'}
        # Build legend
        legend_handles = []
        for act in set(sum([list(set(col)) for col in columns], [])):
            if act in color_map:
                legend_handles.append(mpatches.Patch(color=color_map[act], label=act))
        fig, ax = plt.subplots(figsize=(2*len(columns), 8))
        for lidx, col in enumerate(columns):
            for nidx, act in enumerate(col):
                color = color_map.get(act, 'gray')
                ax.scatter(lidx, -nidx, s=5, color=color, edgecolor='k')
        ax.set_xticks(range(len(columns)))
        ax.set_xticklabels(layer_names)
        ax.set_yticks([])
        ax.set_title('Neural Network Neuron Activation Choices')
        ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'adaptive_activation_choices.png'))
        plt.close()

def plot_activation_counts_bar_chart(output_dir, activation_names=None):
    """
    Reads adaptive_activation_choices.csv and plots a bar chart for each layer showing the count of each activation function.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import os
    if activation_names is None:
        activation_names = ['relu', 'sigmoid', 'tanh']
    csv_path = os.path.join(output_dir, 'adaptive_activation_choices.csv')
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found. Skipping activation count bar chart.")
        return
    df = pd.read_csv(csv_path)
    layer_names = df.columns
    counts = {layer: df[layer].value_counts() for layer in layer_names}
    # Plot
    fig, ax = plt.subplots(figsize=(2*len(layer_names), 6))
    width = 0.12  # Make bars thinner
    x = np.arange(len(layer_names))
    for i, act in enumerate(activation_names):
        values = [counts[layer].get(act, 0) for layer in layer_names]
        # Convert to percentage
        totals = [df[layer].notna().sum() for layer in layer_names]
        percentages = [100 * v / t if t > 0 else 0 for v, t in zip(values, totals)]
        ax.bar(x + i*width, percentages, width, label=act)
    ax.set_xticks(x + width * (len(activation_names) / 2))
    ax.set_xticklabels(layer_names)
    ax.set_ylabel('Percentage of Neurons (%)')
    ax.set_title('Activation Function Percentage per Layer')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activation_counts_bar_chart.png'))
    plt.close()

def plot_activation_percentage_per_epoch(all_choices_per_epoch, output_dir, activation_names=None):
    """
    Plots a chart with subplots for each layer. Each subplot shows the percentage of each activation function per epoch (line chart).
    all_choices_per_epoch: list of DataFrames (one per epoch) with columns as layers and values as activation names.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    if activation_names is None:
        activation_names = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
    layer_names = all_choices_per_epoch[0].columns
    num_layers = len(layer_names)
    num_epochs = len(all_choices_per_epoch)
    fig, axes = plt.subplots(num_layers, 1, figsize=(8, 4*num_layers), sharex=True)
    if num_layers == 1:
        axes = [axes]
    for lidx, layer in enumerate(layer_names):
        for act in activation_names:
            percentages = []
            for epoch_df in all_choices_per_epoch:
                total = epoch_df[layer].notna().sum()
                count = (epoch_df[layer] == act).sum()
                percent = 100 * count / total if total > 0 else 0
                percentages.append(percent)
            axes[lidx].plot(range(1, num_epochs+1), percentages, label=act)
        axes[lidx].set_ylabel(f'{layer} (%)')
        axes[lidx].set_title(f'Layer {layer} Activation Percentages')
        axes[lidx].legend()
    axes[-1].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activation_percentage_per_epoch.png'))
    plt.close()

def plot_activation_percentage_per_epoch_separate_layers(all_choices_per_epoch, output_dir, activation_names=None):
    """
    For each layer, saves a separate line chart showing the percentage of each activation function per epoch.
    Legends are placed at the top right of each figure.
    Includes epoch 0 (initial state) if provided in all_choices_per_epoch[0].
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    if activation_names is None:
        activation_names = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
    layer_names = all_choices_per_epoch[0].columns
    num_layers = len(layer_names)
    num_epochs = len(all_choices_per_epoch) - 1  # Exclude epoch 0 for labeling
    for lidx, layer in enumerate(layer_names):
        plt.figure(figsize=(8, 4))
        for act in activation_names:
            percentages = []
            for epoch_df in all_choices_per_epoch:
                total = epoch_df[layer].notna().sum()
                count = (epoch_df[layer] == act).sum()
                percent = 100 * count / total if total > 0 else 0
                percentages.append(percent)
            plt.plot(range(0, num_epochs+1), percentages, label=act)  # x-axis: 0, 1, ..., num_epochs
        plt.ylabel(f'{layer} (%)')
        plt.title(f'Layer {layer} Activation Percentages')
        plt.xlabel('Epoch')
        plt.legend(loc='upper right')
        plt.tight_layout()
        fname = os.path.join(output_dir, f'activation_percentage_per_epoch_{layer}.png')
        plt.savefig(fname)
        plt.close()

def record_activation_choices_per_epoch(model, activation_names):
    """
    Returns a DataFrame with columns as layers and values as activation names for each neuron in adaptive layers.
    """
    import pandas as pd
    columns = []
    layer_names = []
    adaptive_layer_idx = 1  # Logical index for adaptive layers
    for layer in model.model:
        if hasattr(layer, 'logits'):
            choices = layer.logits.detach().cpu().numpy().argmax(axis=1)
            choices = [activation_names[idx] for idx in choices]
            columns.append(choices)
            layer_names.append(f'Layer_{adaptive_layer_idx}')
            adaptive_layer_idx += 1
    max_len = max(len(col) for col in columns) if columns else 0
    padded_cols = [list(col) + [None]*(max_len - len(col)) for col in columns]
    df = pd.DataFrame({layer: col for layer, col in zip(layer_names, padded_cols)})
    return df

def train_and_evaluate(model, model_name, device, epochs=10, lr=1e-3, batch_size=64, architecture=None, small_size=False, verbose=False, force_dir=False, output_dir=None):
    print(f"[Training] Starting training for {model_name} model")
    print(f"[Training] Device: {device}")
    print(f"[Training] Parameters: epochs={epochs}, lr={lr}, batch_size={batch_size}")
    
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size, small_size=small_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    
    print(f"[Training] Model moved to {device}")
    print(f"[Training] Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses, val_losses = [], []
    output_dir = get_output_dir(output_dir=output_dir, model_name=model_name, force_dir=force_dir)
    
    print(f"[Training] Output directory: {output_dir}")
    
    # Save architecture info
    with open(os.path.join(output_dir, 'architecture.txt'), 'w') as f:
        f.write(f'Model: {model_name}\n')
        f.write(f'Architecture: {architecture}\n')
        f.write(f'Args: epochs={epochs}, lr={lr}, batch_size={batch_size}\n')
        f.write(f'Small size: {small_size}\n')
        f.write(f'Train set size: {len(train_loader.dataset)}\n')
        f.write(f'Val set size: {len(val_loader.dataset)}\n')
        f.write(f'Test set size: {len(test_loader.dataset)}\n')
        
        # For mixed models, save the source selective model path
        if model_name == 'mixed' and hasattr(model, 'source_selective_model_path'):
            f.write(f'Source selective model: {model.source_selective_model_path}\n')
    
    # For mixed models, also create a dedicated file with the source model path
    if model_name == 'mixed' and hasattr(model, 'source_selective_model_path'):
        with open(os.path.join(output_dir, 'source_selective_model.txt'), 'w') as f:
            f.write(f'{model.source_selective_model_path}\n')
        print(f"[Training] Source selective model path saved: {model.source_selective_model_path}")
    best_epoch = None
    all_choices_per_epoch = []
    best_val_acc = None
    
    # Print model architecture details for mixed model
    if model_name == 'mixed':
        print(f"[Training] Mixed model architecture details:")
        total_neurons = 0
        for i, (num_neurons, activation_spec) in enumerate(architecture):
            if isinstance(activation_spec, list):
                # Count different activations in this layer
                act_counts = {}
                for act in activation_spec:
                    act_counts[act] = act_counts.get(act, 0) + 1
                print(f"   Layer {i+1}: {num_neurons} neurons - {act_counts}")
                total_neurons += num_neurons
            else:
                print(f"   Layer {i+1}: {num_neurons} neurons - {activation_spec}")
                total_neurons += num_neurons
        print(f"   Total neurons: {total_neurons}")
    
    # Before training loop, record initial activation choices (epoch 0)
    if model_name == 'selective':
        activation_names = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
        df = record_activation_choices_per_epoch(model, activation_names)
        all_choices_per_epoch.append(df)
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        nan_detected = False
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if verbose:
                for i, layer in enumerate(model.model):
                    if hasattr(layer, 'logits'):
                        logits = layer.logits.detach().cpu().numpy()
                        print(f"[Epoch {epoch+1}] Layer {i+1} logits: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}. Stopping training.")
                nan_detected = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        if nan_detected:
            break
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1).cpu().numpy()
                all_val_preds.append(preds)
                all_val_labels.append(y.cpu().numpy())
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        all_val_preds = np.concatenate(all_val_preds)
        all_val_labels = np.concatenate(all_val_labels)
        val_acc = np.mean(all_val_preds == all_val_labels)
        epoch_time = time.time() - epoch_start_time
        
        if model_name == 'mixed':
            print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(output_dir, 'best_model.pth'))
            save_best_loss(best_val_loss, output_dir)
            best_epoch = epoch + 1
            best_val_acc = val_acc
            # Save best epoch number
            with open(os.path.join(output_dir, 'best_epoch.txt'), 'w') as f:
                f.write(str(best_epoch) + '\n')
            # Save validation accuracy for best model
            with open(os.path.join(output_dir, 'best_val_accuracy.txt'), 'w') as f:
                f.write(f"{best_val_acc:.4f}\n")
            # Save adaptive activation choices and plots if applicable
            if model_name == 'selective':
                model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
                activation_names = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
                save_adaptive_activation_choices(model, output_dir, activation_names=activation_names)
                plot_activation_counts_bar_chart(output_dir, activation_names=activation_names)
                if all_choices_per_epoch:
                    plot_activation_percentage_per_epoch(all_choices_per_epoch, output_dir, activation_names=activation_names)
        # After each epoch, record activation choices for plotting
        if model_name == 'selective':
            activation_names = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
            df = record_activation_choices_per_epoch(model, activation_names)
            all_choices_per_epoch.append(df)
    # Determine activation name for loss curve title
    if model_name == 'selective':
        activation_name = 'selective'
    elif architecture and isinstance(architecture, list) and len(architecture) > 0 and isinstance(architecture[0], list):
        activation_name = architecture[0][1] if architecture[0][1] is not None else model_name
    else:
        activation_name = model_name
    save_loss_curves(train_losses, val_losses, output_dir, activation_name)
    # After training, evaluate best model on test set and save metrics
    print(f"[Training] Training completed. Evaluating best model on test set...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
    model.eval()
    all_test_preds, all_test_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_test_preds.append(preds)
            all_test_labels.append(y.cpu().numpy())
    all_test_preds = np.concatenate(all_test_preds)
    all_test_labels = np.concatenate(all_test_labels)
    
    print(f"[Training] Saving evaluation results...")
    cm = save_confusion_matrix(all_test_labels, all_test_preds, output_dir, class_names=[str(i) for i in range(10)])
    accs = per_class_accuracy(all_test_labels, all_test_preds, 10)
    save_per_class_accuracy(accs, output_dir)
    total_acc = np.mean(all_test_preds == all_test_labels)
    save_total_accuracy(total_acc, output_dir)
    
    print(f"[Training] Final Results:")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best validation accuracy: {best_val_acc:.4f}")
    print(f"   Final test accuracy: {total_acc:.4f}")
    
    if model_name == 'mixed':
        print(f"[Training] Mixed model successfully trained with fixed per-neuron activations")
        print(f"   Model saved to: {output_dir}")
        if hasattr(model, 'source_selective_model_path'):
            print(f"   Source selective model: {model.source_selective_model_path}")
            print(f"   Source model path saved in: {os.path.join(output_dir, 'source_selective_model.txt')}")
    
    # Ensure all adaptive activation plots and csvs are saved for selective model
    if model_name == 'selective':
        activation_names = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
        save_adaptive_activation_choices(model, output_dir, activation_names=activation_names)
        plot_activation_counts_bar_chart(output_dir, activation_names=activation_names)
        if all_choices_per_epoch:
            plot_activation_percentage_per_epoch(all_choices_per_epoch, output_dir, activation_names=activation_names)
            plot_activation_percentage_per_epoch_separate_layers(all_choices_per_epoch, output_dir, activation_names=activation_names)
    print(f"Results saved in {output_dir}")
    return output_dir

def run_inference_from_folder(output_folder):
    # Detect device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Load architecture
    with open(os.path.join(output_folder, 'architecture.txt')) as f:
        lines = f.readlines()
    architecture = []
    for line in lines:
        if line.startswith('Architecture:'):
            arch_str = line.split('Architecture:')[1].strip()
            import ast
            architecture = ast.literal_eval(arch_str)
    input_dim = 28*28
    # Load activation choices
    csv_path = os.path.join(output_folder, 'adaptive_activation_choices.csv')
    df = pd.read_csv(csv_path)
    activation_choices_per_layer = []
    for col in df.columns:
        activation_choices_per_layer.append(df[col].dropna().tolist())
    # Build model
    weights_path = os.path.join(output_folder, 'best_model.pth')
    model = build_hard_activation_network(architecture, input_dim, activation_choices_per_layer, weights_path)
    model = model.to(device)
    model.eval()
    # Save inference network architecture
    inference_dir = os.path.join(output_folder, 'inference')
    os.makedirs(inference_dir, exist_ok=True)
    with open(os.path.join(inference_dir, 'inf_arch.txt'), 'w') as f:
        f.write(str(model))
    # Save per-neuron activation choices for the inference network
    # This is the same as activation_choices_per_layer, but as a CSV like adaptive_activation_choices.csv
    
    max_len = max(len(col) for col in activation_choices_per_layer)
    padded_cols = [list(col) + [None]*(max_len - len(col)) for col in activation_choices_per_layer]
    with open(os.path.join(inference_dir, 'adaptive_activation_choices.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(df.columns)
        for row in zip(*padded_cols):
            writer.writerow(row)
    # Inference
    _, test_loader = get_mnist_loaders(batch_size=128, small_size=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    # Save results
    cm = save_confusion_matrix(all_labels, all_preds, inference_dir, class_names=[str(i) for i in range(10)])
    accs = per_class_accuracy(all_labels, all_preds, 10)
    save_per_class_accuracy(accs, inference_dir)
    total_acc = np.mean(all_preds == all_labels)
    save_total_accuracy(total_acc, inference_dir)
    print(f"Inference results saved in {inference_dir}")

def create_mixed_model_from_selective(selective_folder_path, input_dim):
    """
    Load a trained selective model and extract the activation choices for each neuron,
    then build a new CustomNeuralNetwork with fixed activations based on those choices.
    
    Args:
        selective_folder_path: Path to the folder containing the trained selective model
        input_dim: Input dimension for the network
        
    Returns:
        tuple: (model, architecture) where model is the CustomNeuralNetwork and 
               architecture is the list of [neurons, activation] pairs
    """
    import ast
    import pandas as pd
    
    print(f"[Mixed Model Creation] Step 1/5: Loading architecture from {selective_folder_path}")
    
    # Load architecture from the selective model
    arch_file_path = os.path.join(selective_folder_path, 'architecture.txt')
    if not os.path.exists(arch_file_path):
        raise ValueError(f"Architecture file not found at {arch_file_path}")
    
    with open(arch_file_path) as f:
        lines = f.readlines()
    
    original_architecture = []
    for line in lines:
        if line.startswith('Architecture:'):
            arch_str = line.split('Architecture:')[1].strip()
            original_architecture = ast.literal_eval(arch_str)
            break
    
    if not original_architecture:
        raise ValueError("Could not find architecture in the selective model folder")
    
    print(f"[Mixed Model Creation] Original architecture loaded: {len(original_architecture)} layers")
    
    print("[Mixed Model Creation] Step 2/5: Loading activation choices from CSV")
    
    # Load activation choices from CSV
    csv_path = os.path.join(selective_folder_path, 'adaptive_activation_choices.csv')
    if not os.path.exists(csv_path):
        raise ValueError(f"Activation choices CSV not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    activation_choices_per_layer = []
    
    for col in df.columns:
        activation_choices_per_layer.append(df[col].dropna().tolist())
    
    print(f"[Mixed Model Creation] Activation choices loaded for {len(activation_choices_per_layer)} layers")
    
    print("[Mixed Model Creation] Step 3/5: Analyzing activation distribution per layer")
    
    # Build mixed architecture - convert per-neuron activations to layer-wise
    mixed_architecture = []
    
    # Process each layer that had selective activations
    layer_idx = 0
    for arch_layer_idx, (num_neurons, activation) in enumerate(original_architecture):
        if activation == 'selective' and layer_idx < len(activation_choices_per_layer):
            # This was a selective layer, so we have per-neuron activation choices
            neuron_activations = activation_choices_per_layer[layer_idx]
            
            # Count activation distribution for progress reporting
            activation_counts = {}
            for act_name in neuron_activations:
                activation_counts[act_name] = activation_counts.get(act_name, 0) + 1
            
            print(f"   Layer {layer_idx + 1}: {num_neurons} neurons with activations: {activation_counts}")
            
            mixed_architecture.append([num_neurons, neuron_activations])
            layer_idx += 1
        else:
            # This was not a selective layer (e.g., output layer with softmax)
            print(f"   Output layer: {num_neurons} neurons with {activation} activation")
            mixed_architecture.append([num_neurons, activation])
    
    print("[Mixed Model Creation] Step 4/5: Building optimized mixed neural network model")
    
    # Create a specialized mixed model class that handles per-neuron activations
    model = MixedNeuralNetwork(mixed_architecture, input_dim)
    
    print(f"[Mixed Model Creation] Step 5/5: Mixed model created successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Model architecture: {len(mixed_architecture)} layers")
    
    # Store the source selective model path for reference
    model.source_selective_model_path = selective_folder_path
    
    # Show optimization details for per-neuron activation layers
    print("[Mixed Model Optimization] Per-neuron activation layer details:")
    layer_idx = 0
    for i, layer in enumerate(model.model):
        if hasattr(layer, 'get_activation_summary'):
            summary = layer.get_activation_summary()
            print(f"   Layer {layer_idx + 1}: Optimized groups - {summary}")
            total_neurons = sum(summary.values())
            print(f"   Layer {layer_idx + 1}: {len(summary)} activation types across {total_neurons} neurons")
            layer_idx += 1
    
    print("[Mixed Model Optimization] Model uses optimized vectorized operations for faster training")
    
    return model, mixed_architecture

def train_and_evaluate_smartmixed_phase1(model, model_name, device, epochs=10, lr=1e-4, batch_size=64, architecture=None, small_size=False, verbose=False, main_output_dir=None):
    """
    Phase 1 of smartmixed training: Train a selective model for a specified number of epochs.
    This is essentially the same as train_and_evaluate but specifically for phase 1 of smartmixed.
    
    Args:
        model: The selective neural network model
        model_name: Name of the model (should be 'selective')
        device: PyTorch device to use
        epochs: Number of epochs to train
        lr: Learning rate
        batch_size: Batch size
        architecture: Model architecture
        small_size: Whether to use small dataset
        verbose: Whether to print verbose output
        main_output_dir: Main output directory for smartmixed training
        
    Returns:
        str: Path to the output directory containing the trained selective model
    """
    print(f"[SmartMixed Phase 1] Starting selective model training")
    print(f"[SmartMixed Phase 1] Device: {device}")
    print(f"[SmartMixed Phase 1] Parameters: epochs={epochs}, lr={lr}, batch_size={batch_size}")
    
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size, small_size=small_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    
    print(f"[SmartMixed Phase 1] Model moved to {device}")
    print(f"[SmartMixed Phase 1] Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses, val_losses = [], []
    
    # Create phase1 subdirectory within main output directory
    if main_output_dir:
        output_dir = os.path.join(main_output_dir, 'phase1')
    else:
        # Fallback to timestamped directory if main_output_dir not provided
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('outputs', f'smartmixed_phase1_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[SmartMixed Phase 1] Output directory: {output_dir}")
    
    # Save architecture info
    with open(os.path.join(output_dir, 'architecture.txt'), 'w') as f:
        f.write(f'Model: {model_name}\n')
        f.write(f'Architecture: {architecture}\n')
        f.write(f'Args: epochs={epochs}, lr={lr}, batch_size={batch_size}\n')
        f.write(f'Small size: {small_size}\n')
        f.write(f'Train set size: {len(train_loader.dataset)}\n')
        f.write(f'Val set size: {len(val_loader.dataset)}\n')
        f.write(f'Test set size: {len(test_loader.dataset)}\n')
        f.write(f'Phase: 1 (Selective)\n')
    
    best_epoch = None
    all_choices_per_epoch = []
    best_val_acc = None
    
    # Before training loop, record initial activation choices (epoch 0)
    activation_names = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
    df = record_activation_choices_per_epoch(model, activation_names)
    all_choices_per_epoch.append(df)
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        nan_detected = False
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if verbose:
                for i, layer in enumerate(model.model):
                    if hasattr(layer, 'logits'):
                        logits = layer.logits.detach().cpu().numpy()
                        print(f"[Epoch {epoch+1}] Layer {i+1} logits: min={logits.min():.4f}, max={logits.max():.4f}, mean={logits.mean():.4f}")
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}. Stopping training.")
                nan_detected = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        if nan_detected:
            break
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1).cpu().numpy()
                all_val_preds.append(preds)
                all_val_labels.append(y.cpu().numpy())
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        all_val_preds = np.concatenate(all_val_preds)
        all_val_labels = np.concatenate(all_val_labels)
        val_acc = np.mean(all_val_preds == all_val_labels)
        epoch_time = time.time() - epoch_start_time
        
        print(f"Epoch {epoch+1}/{epochs} | Time: {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(output_dir, 'best_model.pth'))
            save_best_loss(best_val_loss, output_dir)
            best_epoch = epoch + 1
            best_val_acc = val_acc
            # Save best epoch number
            with open(os.path.join(output_dir, 'best_epoch.txt'), 'w') as f:
                f.write(str(best_epoch) + '\n')
            # Save validation accuracy for best model
            with open(os.path.join(output_dir, 'best_val_accuracy.txt'), 'w') as f:
                f.write(f"{best_val_acc:.4f}\n")
            # Save adaptive activation choices and plots for best model
            model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
            activation_names = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
            save_adaptive_activation_choices(model, output_dir, activation_names=activation_names)
            plot_activation_counts_bar_chart(output_dir, activation_names=activation_names)
            if all_choices_per_epoch:
                plot_activation_percentage_per_epoch(all_choices_per_epoch, output_dir, activation_names=activation_names)
        
        # After each epoch, record activation choices for plotting
        activation_names = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
        df = record_activation_choices_per_epoch(model, activation_names)
        all_choices_per_epoch.append(df)
    
    # Save loss curves
    save_loss_curves(train_losses, val_losses, output_dir, 'selective_phase1')
    
    # After training, evaluate best model on test set and save metrics
    print(f"[SmartMixed Phase 1] Training completed. Evaluating best model on test set...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
    model.eval()
    all_test_preds, all_test_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_test_preds.append(preds)
            all_test_labels.append(y.cpu().numpy())
    all_test_preds = np.concatenate(all_test_preds)
    all_test_labels = np.concatenate(all_test_labels)
    
    print(f"[SmartMixed Phase 1] Saving evaluation results...")
    cm = save_confusion_matrix(all_test_labels, all_test_preds, output_dir, class_names=[str(i) for i in range(10)])
    accs = per_class_accuracy(all_test_labels, all_test_preds, 10)
    save_per_class_accuracy(accs, output_dir)
    total_acc = np.mean(all_test_preds == all_test_labels)
    save_total_accuracy(total_acc, output_dir)
    
    print(f"[SmartMixed Phase 1] Final Results:")
    print(f"   Best epoch: {best_epoch}")
    print(f"   Best validation accuracy: {best_val_acc:.4f}")
    print(f"   Final test accuracy: {total_acc:.4f}")
    
    # Ensure all adaptive activation plots and csvs are saved for selective model
    activation_names = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
    save_adaptive_activation_choices(model, output_dir, activation_names=activation_names)
    plot_activation_counts_bar_chart(output_dir, activation_names=activation_names)
    if all_choices_per_epoch:
        plot_activation_percentage_per_epoch(all_choices_per_epoch, output_dir, activation_names=activation_names)
        plot_activation_percentage_per_epoch_separate_layers(all_choices_per_epoch, output_dir, activation_names=activation_names)
    
    print(f"[SmartMixed Phase 1] Results saved in {output_dir}")
    return output_dir


def train_and_evaluate_smartmixed_phase2(model, model_name, device, epochs=10, lr=1e-3, batch_size=64, architecture=None, small_size=False, verbose=False, selective_output_dir=None, transition_epoch=None, total_epochs=None, main_output_dir=None):
    """
    Phase 2 of smartmixed training: Train a mixed model (created from selective) for remaining epochs.
    
    Args:
        model: The mixed neural network model (created from selective)
        model_name: Name of the model (should be 'smartmixed')
        device: PyTorch device to use
        epochs: Number of epochs to train (remaining epochs)
        lr: Learning rate
        batch_size: Batch size
        architecture: Model architecture
        small_size: Whether to use small dataset
        verbose: Whether to print verbose output
        selective_output_dir: Path to phase 1 output directory
        transition_epoch: Epoch number where transition happened
        total_epochs: Total epochs for the entire smartmixed training
        main_output_dir: Main output directory for smartmixed training
        
    Returns:
        str: Path to the output directory containing the trained smartmixed model
    """
    print(f"[SmartMixed Phase 2] Starting mixed model training")
    print(f"[SmartMixed Phase 2] Device: {device}")
    print(f"[SmartMixed Phase 2] Parameters: epochs={epochs}, lr={lr}, batch_size={batch_size}")
    
    train_loader, val_loader, test_loader = get_mnist_loaders(batch_size, small_size=small_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model = model.to(device)
    
    print(f"[SmartMixed Phase 2] Model moved to {device}")
    print(f"[SmartMixed Phase 2] Dataset sizes - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    best_val_loss = float('inf')
    best_model_state = None
    train_losses, val_losses = [], []
    
    # Create phase2 subdirectory within main output directory
    if main_output_dir:
        output_dir = os.path.join(main_output_dir, 'phase2')
    else:
        # Fallback to timestamped directory if main_output_dir not provided
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('outputs', f'smartmixed_phase2_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"[SmartMixed Phase 2] Output directory: {output_dir}")
    
    # Save architecture info
    with open(os.path.join(output_dir, 'architecture.txt'), 'w') as f:
        f.write(f'Model: {model_name}\n')
        f.write(f'Architecture: {architecture}\n')
        f.write(f'Args: epochs={epochs}, lr={lr}, batch_size={batch_size}\n')
        f.write(f'Small size: {small_size}\n')
        f.write(f'Train set size: {len(train_loader.dataset)}\n')
        f.write(f'Val set size: {len(val_loader.dataset)}\n')
        f.write(f'Test set size: {len(test_loader.dataset)}\n')
        f.write(f'Phase: 2 (Mixed)\n')
        f.write(f'Transition epoch: {transition_epoch}\n')
        f.write(f'Total epochs: {total_epochs}\n')
        if selective_output_dir:
            f.write(f'Phase 1 output: {selective_output_dir}\n')
    
    # For mixed models, also create a dedicated file with the source model path
    if hasattr(model, 'source_selective_model_path'):
        with open(os.path.join(output_dir, 'source_selective_model.txt'), 'w') as f:
            f.write(f'{model.source_selective_model_path}\n')
        print(f"[SmartMixed Phase 2] Source selective model path saved: {model.source_selective_model_path}")
    
    # Also save reference to phase 1
    if selective_output_dir:
        with open(os.path.join(output_dir, 'phase1_selective_model.txt'), 'w') as f:
            f.write(f'{selective_output_dir}\n')
        print(f"[SmartMixed Phase 2] Phase 1 selective model path saved: {selective_output_dir}")
    
    best_epoch = None
    best_val_acc = None
    
    # Print model architecture details for mixed model
    print(f"[SmartMixed Phase 2] Mixed model architecture details:")
    total_neurons = 0
    for i, (num_neurons, activation_spec) in enumerate(architecture):
        if isinstance(activation_spec, list):
            # Count different activations in this layer
            act_counts = {}
            for act in activation_spec:
                act_counts[act] = act_counts.get(act, 0) + 1
            print(f"   Layer {i+1}: {num_neurons} neurons - {act_counts}")
            total_neurons += num_neurons
        else:
            print(f"   Layer {i+1}: {num_neurons} neurons - {activation_spec}")
            total_neurons += num_neurons
    print(f"   Total neurons: {total_neurons}")
    
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        nan_detected = False
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch+1}. Stopping training.")
                nan_detected = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        if nan_detected:
            break
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1).cpu().numpy()
                all_val_preds.append(preds)
                all_val_labels.append(y.cpu().numpy())
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        all_val_preds = np.concatenate(all_val_preds)
        all_val_labels = np.concatenate(all_val_labels)
        val_acc = np.mean(all_val_preds == all_val_labels)
        
        epoch_time = time.time() - epoch_start_time
        
        # Show epoch numbers as continuation from phase 1
        actual_epoch = transition_epoch + epoch + 1 if transition_epoch else epoch + 1
        print(f"Epoch {actual_epoch}/{total_epochs} (Phase 2: {epoch+1}/{epochs}) | Time: {epoch_time:.2f} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, os.path.join(output_dir, 'best_model.pth'))
            save_best_loss(best_val_loss, output_dir)
            best_epoch = epoch + 1
            best_val_acc = val_acc
            # Save best epoch number (relative to phase 2)
            with open(os.path.join(output_dir, 'best_epoch.txt'), 'w') as f:
                f.write(str(best_epoch) + '\n')
            # Save absolute best epoch number (relative to total training)
            with open(os.path.join(output_dir, 'best_epoch_absolute.txt'), 'w') as f:
                absolute_epoch = transition_epoch + best_epoch if transition_epoch else best_epoch
                f.write(str(absolute_epoch) + '\n')
            # Save validation accuracy for best model
            with open(os.path.join(output_dir, 'best_val_accuracy.txt'), 'w') as f:
                f.write(f"{best_val_acc:.4f}\n")
    
    # Save loss curves
    save_loss_curves(train_losses, val_losses, output_dir, 'smartmixed_phase2')
    
    # After training, evaluate best model on test set and save metrics
    print(f"[SmartMixed Phase 2] Training completed. Evaluating best model on test set...")
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth'), map_location=device))
    model.eval()
    all_test_preds, all_test_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_test_preds.append(preds)
            all_test_labels.append(y.cpu().numpy())
    all_test_preds = np.concatenate(all_test_preds)
    all_test_labels = np.concatenate(all_test_labels)
    
    print(f"[SmartMixed Phase 2] Saving evaluation results...")
    cm = save_confusion_matrix(all_test_labels, all_test_preds, output_dir, class_names=[str(i) for i in range(10)])
    accs = per_class_accuracy(all_test_labels, all_test_preds, 10)
    save_per_class_accuracy(accs, output_dir)
    total_acc = np.mean(all_test_preds == all_test_labels)
    save_total_accuracy(total_acc, output_dir)
    
    print(f"[SmartMixed Phase 2] Final Results:")
    print(f"   Best epoch (Phase 2): {best_epoch}")
    if transition_epoch:
        absolute_best_epoch = transition_epoch + best_epoch
        print(f"   Best epoch (Absolute): {absolute_best_epoch}")
    print(f"   Best validation accuracy: {best_val_acc:.4f}")
    print(f"   Final test accuracy: {total_acc:.4f}")
    
    print(f"[SmartMixed Phase 2] SmartMixed model successfully trained with two-phase approach")
    print(f"   Model saved to: {output_dir}")
    if hasattr(model, 'source_selective_model_path'):
        print(f"   Source selective model: {model.source_selective_model_path}")
    if selective_output_dir:
        print(f"   Phase 1 output: {selective_output_dir}")
    
    # Copy activation choices from phase 1 to show what activations were learned
    if selective_output_dir:
        import shutil
        phase1_csv = os.path.join(selective_output_dir, 'adaptive_activation_choices.csv')
        if os.path.exists(phase1_csv):
            shutil.copy2(phase1_csv, os.path.join(output_dir, 'adaptive_activation_choices_from_phase1.csv'))
            print(f"   Phase 1 activation choices copied to output directory")
        
        # Also copy phase 1 plots for reference
        phase1_plots = [
            'adaptive_activation_choices.png',
            'activation_counts_bar_chart.png'
        ]
        for plot_file in phase1_plots:
            src_path = os.path.join(selective_output_dir, plot_file)
            if os.path.exists(src_path):
                dst_path = os.path.join(output_dir, f'phase1_{plot_file}')
                shutil.copy2(src_path, dst_path)
    
    print(f"[SmartMixed Phase 2] Results saved in {output_dir}")
    return output_dir
