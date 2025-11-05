#!/usr/bin/env python3
"""
Weight Finder Script for SmartMixed Models

This script extracts weight information from SmartMixed models, analyzing connections
between neurons with different activation functions. For each SmartMixed model, it:
1. Loads the best saved architecture
2. Extracts weight matrices between layers
3. Maps activation functions for each neuron
4. Generates output showing connections between neurons

Output columns:
- Architecture_ID: ID of the architecture
- Neurons_Per_Layer: Number of neurons in each layer
- Architecture_Name: Name of the architecture (e.g., smartmixed)
- Number_of_Layers: Total number of layers
- layer_number: Current layer number (0-indexed)
- activation1_name: Activation function of source neuron
- activation2_name: Activation function of target neuron
- weight: Weight value connecting the neurons
"""

import os
import sys
import pandas as pd
import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to Python path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models import MixedNeuralNetwork, PerNeuronActivation
except ImportError as e:
    print(f"Error importing models: {e}")
    sys.exit(1)


def load_architecture_info(arch_dir):
    """Load architecture information from the architecture directory."""
    arch_file = os.path.join(arch_dir, 'architecture_info.txt')
    if os.path.exists(arch_file):
        architecture_info = {}
        with open(arch_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Architecture ID:'):
                    architecture_info['id'] = line.split(':')[1].strip()
                elif line.startswith('Architecture Name:'):
                    architecture_info['name'] = line.split(':')[1].strip()
                elif line.startswith('Architecture Layers:'):
                    # Parse the architecture layers string
                    arch_str = line.split('Architecture Layers:')[1].strip()
                    try:
                        import ast
                        architecture_info['architecture'] = ast.literal_eval(arch_str)
                    except:
                        print(f"Warning: Could not parse architecture string: {arch_str}")
                        return None
                elif line.startswith('Number of Layers:'):
                    architecture_info['num_layers'] = int(line.split(':')[1].strip())
        return architecture_info
    return None


def load_smartmixed_model(model_path, mixed_architecture, device='cpu'):
    """Load a SmartMixed model from the saved checkpoint."""
    try:
        # Create the model using the mixed architecture (MNIST input dimension is 28*28 = 784)
        input_dim = 28 * 28
        model = MixedNeuralNetwork(mixed_architecture, input_dim)
        
        # Load the saved state dict
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def load_activation_choices(phase2_dir):
    """Load activation choices from the SmartMixed phase2 directory."""
    activation_file = os.path.join(phase2_dir, 'adaptive_activation_choices_from_phase1.csv')
    if os.path.exists(activation_file):
        return pd.read_csv(activation_file)
    return None


def extract_layer_activations(activation_df, layer_idx):
    """Extract activation functions for neurons in a specific layer.
    
    Args:
        activation_df: DataFrame with activation choices
        layer_idx: 0-based layer index (0 = first hidden layer)
    
    Returns:
        List of activation function names for the layer, or None if not found
    """
    if activation_df is None:
        return None
    
    # Column names are like 'Layer_1', 'Layer_2', etc. (1-based indexing)
    # layer_idx=0 corresponds to 'Layer_1', layer_idx=1 to 'Layer_2', etc.
    layer_col = f'Layer_{layer_idx + 1}'
    if layer_col in activation_df.columns:
        return activation_df[layer_col].dropna().tolist()
    return None


def build_mixed_architecture(original_architecture, activation_df):
    """Build mixed architecture with per-neuron activations from the CSV."""
    mixed_architecture = []
    
    for layer_idx, (num_neurons, activation) in enumerate(original_architecture):
        if activation == 'softmax':
            # Output layer - keep as is
            mixed_architecture.append([num_neurons, activation])
        else:
            # Hidden layer - use per-neuron activations from CSV
            layer_activations = extract_layer_activations(activation_df, layer_idx)
            if (layer_activations and len(layer_activations) == num_neurons):
                mixed_architecture.append([num_neurons, layer_activations])
            elif layer_activations:
                # Truncate or pad if mismatch
                if len(layer_activations) > num_neurons:
                    layer_activations = layer_activations[:num_neurons]
                else:
                    layer_activations.extend(['relu'] * (num_neurons - len(layer_activations)))
                mixed_architecture.append([num_neurons, layer_activations])
            else:
                print(f"Warning: No activation choices found for layer {layer_idx}, using default ReLU")
                mixed_architecture.append([num_neurons, ['relu'] * num_neurons])
    
    return mixed_architecture


def extract_weights_and_activations(model, architecture, activation_df, arch_id, arch_name):
    """Extract weights and activation mappings from a SmartMixed model."""
    results = []
    
    # Get architecture info
    neurons_per_layer = [num_neurons for num_neurons, _ in architecture]
    num_layers = len(architecture)
    
    # Convert architecture to string representation
    arch_str = str(architecture)
    
    try:
        # Extract weights from each layer (skip input->hidden layer, only process hidden->hidden and hidden->output)
        layer_idx = 0
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() == 2:  # Weight matrices
                weight_matrix = param.detach().cpu().numpy()
                
                # Skip the first layer (input to first hidden layer) since input has no activation
                if layer_idx == 0:
                    layer_idx += 1
                    continue
                
                # For connections between hidden layers or hidden to output
                # Source layer is the previous hidden layer
                source_layer_csv_idx = layer_idx - 1  # CSV Layer_1, Layer_2, etc.
                source_activations = extract_layer_activations(activation_df, source_layer_csv_idx)
                
                # Target layer
                if layer_idx == num_layers - 1:
                    # This is the output layer (softmax) - skip this layer too
                    layer_idx += 1
                    continue
                else:
                    # Another hidden layer
                    target_layer_csv_idx = layer_idx
                    target_activations = extract_layer_activations(activation_df, target_layer_csv_idx)
                
                # Handle missing activations
                if source_activations is None:
                    print(f"Warning: Could not find source activations for layer {layer_idx-1}")
                    source_activations = ['unknown'] * weight_matrix.shape[1]
                
                if target_activations is None:
                    print(f"Warning: Could not find target activations for layer {layer_idx}")
                    target_activations = ['unknown'] * weight_matrix.shape[0]
                
                # Ensure we have the right number of activations
                if len(source_activations) != weight_matrix.shape[1]:
                    print(f"Warning: Mismatch in source activations for layer {layer_idx-1} - expected {weight_matrix.shape[1]}, got {len(source_activations)}")
                    # Pad or truncate to match
                    if len(source_activations) > weight_matrix.shape[1]:
                        source_activations = source_activations[:weight_matrix.shape[1]]
                    else:
                        source_activations.extend(['unknown'] * (weight_matrix.shape[1] - len(source_activations)))
                
                if len(target_activations) != weight_matrix.shape[0]:
                    print(f"Warning: Mismatch in target activations for layer {layer_idx} - expected {weight_matrix.shape[0]}, got {len(target_activations)}")
                    # Pad or truncate to match
                    if len(target_activations) > weight_matrix.shape[0]:
                        target_activations = target_activations[:weight_matrix.shape[0]]
                    else:
                        target_activations.extend(['unknown'] * (weight_matrix.shape[0] - len(target_activations)))
                
                # Extract each weight connection
                for target_neuron in range(weight_matrix.shape[0]):
                    for source_neuron in range(weight_matrix.shape[1]):
                        weight_value = weight_matrix[target_neuron, source_neuron]
                        
                        results.append({
                            'Architecture_ID': arch_id,
                            'Neurons_Per_Layer': str(neurons_per_layer),
                            'Architecture_Name': arch_name,
                            'Number_of_Layers': num_layers,
                            'layer_number': layer_idx,
                            'activation1_name': source_activations[source_neuron],
                            'activation2_name': target_activations[target_neuron],
                            'weight': weight_value
                        })
                
                layer_idx += 1
                
                # Break if we've processed all layers
                if layer_idx >= num_layers:
                    break
                    
    except Exception as e:
        print(f"Error extracting weights for architecture {arch_id}: {e}")
    
    return results


def find_smartmixed_models(base_dir):
    """Find all SmartMixed models in the experiment directory."""
    smartmixed_models = []
    
    for arch_dir in os.listdir(base_dir):
        arch_path = os.path.join(base_dir, arch_dir)
        if not os.path.isdir(arch_path) or not arch_dir.startswith('arch_'):
            continue
        
        # Look for SmartMixed models in all_models_* subdirectories
        for subdir in os.listdir(arch_path):
            subdir_path = os.path.join(arch_path, subdir)
            if not os.path.isdir(subdir_path) or not subdir.startswith('all_models_'):
                continue
            
            # Check for smartmixed/phase2/best_model.pth
            smartmixed_path = os.path.join(subdir_path, 'smartmixed', 'phase2')
            model_path = os.path.join(smartmixed_path, 'best_model.pth')
            
            if os.path.exists(model_path):
                # Extract architecture ID
                arch_id = arch_dir.replace('arch_', '')
                
                # Load architecture info
                arch_info = load_architecture_info(arch_path)
                architecture = arch_info.get('architecture') if arch_info else None
                
                if architecture is None:
                    print(f"Warning: Could not load architecture info for {arch_dir}")
                    continue
                
                smartmixed_models.append({
                    'arch_id': arch_id,
                    'arch_dir': arch_path,
                    'model_path': model_path,
                    'phase2_dir': smartmixed_path,
                    'architecture': architecture
                })
    
    return smartmixed_models




def create_aggregated_weights(df):
    """Create aggregated weight statistics grouped by architecture, layer, and activation pairs."""
    print("  Computing weight statistics for each group...")
    
    # Group by the specified columns and calculate aggregate statistics
    grouping_cols = [
        'Architecture_ID', 
        'Neurons_Per_Layer', 
        'Architecture_Name', 
        'Number_of_Layers', 
        'layer_number', 
        'activation1_name', 
        'activation2_name'
    ]
    
    # Aggregate statistics
    aggregated = df.groupby(grouping_cols)['weight'].agg([
        ('average_weight', 'mean'),
        ('min_weight', 'min'),
        ('max_weight', 'max'),
        ('Q1', lambda x: x.quantile(0.25)),
        ('Q3', lambda x: x.quantile(0.75)),
        ('median', 'median'),
        ('std', 'std'),
        ('count', 'count')  # Also include count for reference
    ]).round(6).reset_index()
    
    # Fill NaN std values with 0 (happens when there's only one value in a group)
    aggregated['std'] = aggregated['std'].fillna(0)
    
    return aggregated


def create_activation_heatmap(df, output_dir, statistic='mean', suffix=''):
    """Create a heatmap showing weights between activation function pairs."""
    stat_name = 'Average' if statistic == 'mean' else 'Median'
    suffix_text = ' (excluding sigmoid)' if suffix == '_no_sigmoid' else ''
    print(f"Creating activation function heatmap ({stat_name.lower()}{suffix_text})...")
    
    # Calculate statistics for each activation pair
    if statistic == 'mean':
        activation_weights = df.groupby(['activation1_name', 'activation2_name'])['weight'].mean().reset_index()
    else:  # median
        activation_weights = df.groupby(['activation1_name', 'activation2_name'])['weight'].median().reset_index()
    
    # Create pivot table for heatmap
    heatmap_data = activation_weights.pivot(index='activation1_name', 
                                          columns='activation2_name', 
                                          values='weight')
    
    # Get all unique activation functions for consistent ordering
    all_activations = sorted(set(df['activation1_name'].unique()) | set(df['activation2_name'].unique()))
    
    # Reindex to ensure all activation functions are included
    heatmap_data = heatmap_data.reindex(index=all_activations, columns=all_activations)
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))
    
    # Use a diverging colormap centered at 0
    sns.heatmap(heatmap_data, 
                annot=False,  # Don't show values as requested
                cmap='RdBu_r',  # Red-Blue colormap (reversed so positive weights are red)
                center=0,  # Center the colormap at 0
                square=True,  # Make cells square
                linewidths=0.5,  # Add thin lines between cells
                cbar_kws={'label': f'{stat_name} Weight'})
    
    plt.title(f'{stat_name} Weights Between Activation Function Pairs{suffix_text}\n(Source → Target)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Target Activation Function', fontsize=12, fontweight='bold')
    plt.ylabel('Source Activation Function', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_path = os.path.join(output_dir, f'activation_weights_heatmap_{statistic}{suffix}.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✓ {stat_name} heatmap saved as: {heatmap_path}")
    
    # Also save as PDF for better quality
    heatmap_pdf_path = os.path.join(output_dir, f'activation_weights_heatmap_{statistic}{suffix}.pdf')
    plt.savefig(heatmap_pdf_path, bbox_inches='tight')
    print(f"✓ {stat_name} heatmap also saved as PDF: {heatmap_pdf_path}")
    
    plt.close()  # Close the figure to free memory
    
    # Print some statistics about the heatmap
    print(f"\n{stat_name} Heatmap Statistics:")
    print(f"  Activation functions included: {len(all_activations)}")
    print(f"  Activation functions: {', '.join(all_activations)}")
    print(f"  Weight range: {heatmap_data.min().min():.6f} to {heatmap_data.max().max():.6f}")
    
    return heatmap_data


def print_summary_statistics(df):
    """Print comprehensive summary statistics to console."""
    print("\n" + "="*60)
    print("WEIGHT ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"Total weight connections: {len(df):,}")
    print(f"Unique architectures: {df['Architecture_ID'].nunique()}")
    print(f"Architecture IDs: {sorted(df['Architecture_ID'].unique())}")
    
    print(f"\nWeight Statistics:")
    print(f"  Mean: {df['weight'].mean():.6f}")
    print(f"  Std:  {df['weight'].std():.6f}")
    print(f"  Min:  {df['weight'].min():.6f}")
    print(f"  Max:  {df['weight'].max():.6f}")
    
    print(f"\nTop 10 Activation Function Pairs:")
    activation_pairs = df.groupby(['activation1_name', 'activation2_name']).size().sort_values(ascending=False)
    for (act1, act2), count in activation_pairs.head(10).items():
        print(f"  {act1} -> {act2}: {count:,} connections")
    
    print(f"\nArchitecture Summary:")
    arch_summary = df.groupby('Architecture_ID').agg({
        'weight': ['count', 'mean', 'std'],
        'layer_number': 'nunique'
    }).round(4)
    arch_summary.columns = ['connections', 'weight_mean', 'weight_std', 'unique_layers']
    for arch_id, row in arch_summary.iterrows():
        print(f"  Arch {arch_id}: {row['connections']:,} connections, {row['unique_layers']} layers, "
              f"weight_mean={row['weight_mean']:.4f}")

def main():
    # ==================== CONFIGURATION ====================
    # Set the experiment folder name here
    EXPERIMENT_FOLDER = "iterate_20251005_180118_done"
    # ======================================================
    
    # Base directory containing experiment results
    base_dir = f"/Users/aminomidvar/Documents/My Research/adaptive activation function repo/adaptive activation function/outputs/{EXPERIMENT_FOLDER}"
    
    if not os.path.exists(base_dir):
        print(f"Error: Base directory does not exist: {base_dir}")
        sys.exit(1)
    
    # Create weight_finder output directory
    output_dir = os.path.join(base_dir, 'weight_finder')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    print(f"Searching for SmartMixed models in: {base_dir}")
    
    # Find all SmartMixed models
    smartmixed_models = find_smartmixed_models(base_dir)
    
    if not smartmixed_models:
        print("No SmartMixed models found!")
        sys.exit(1)
    
    print(f"Found {len(smartmixed_models)} SmartMixed models")
    
    # Extract weights from all models
    all_results = []
    device = torch.device('cpu')  # Use CPU for inference
    
    for model_info in smartmixed_models:
        arch_id = model_info['arch_id']
        model_path = model_info['model_path']
        phase2_dir = model_info['phase2_dir']
        architecture = model_info['architecture']
        
        print(f"\nProcessing Architecture {arch_id}...")
        print(f"  Model path: {model_path}")
        print(f"  Architecture: {architecture}")
        
        # Load activation choices
        activation_df = load_activation_choices(phase2_dir)
        if activation_df is None:
            print(f"  Warning: Could not load activation choices for {arch_id}")
            continue
        else:
            print(f"  Loaded activation choices: {len(activation_df)} entries")
        
        # Build mixed architecture with per-neuron activations
        mixed_architecture = build_mixed_architecture(architecture, activation_df)
        print(f"  Mixed architecture: {mixed_architecture}")
        
        # Load the model
        model = load_smartmixed_model(model_path, mixed_architecture, device)
        if model is None:
            print(f"  Skipping due to model loading error")
            continue
        
        # Extract weights and activations
        results = extract_weights_and_activations(
            model, mixed_architecture, activation_df, arch_id, 'smartmixed'
        )
        
        print(f"  Extracted {len(results)} weight connections")
        all_results.extend(results)
    
    # Convert to DataFrame and save
    if all_results:
        print(f"\nTotal weight connections extracted: {len(all_results):,}")
        print(f"Number of architectures processed: {len(smartmixed_models)}")
        
        # Save as CSV file with memory-efficient approach
        csv_path = os.path.join(output_dir, 'smartmixed_weights.csv')
        
        print(f"\nSaving complete dataset as CSV: {csv_path}")
        
        # Create DataFrame and save immediately to minimize memory usage
        df = pd.DataFrame(all_results)
        
        # Use chunked writing for very large datasets
        try:
            print("Writing CSV file... (this may take a few minutes for large datasets)")
            df.to_csv(csv_path, index=False, chunksize=50000)
            print(f"✓ CSV file saved successfully with {len(df):,} rows")
            
            # Create aggregated dataset
            print("\nCreating aggregated weight statistics...")
            aggregated_df = create_aggregated_weights(df)
            
            # Save aggregated CSV
            aggregated_csv_path = os.path.join(output_dir, 'smartmixed_weights_aggregated.csv')
            print(f"Saving aggregated dataset as CSV: {aggregated_csv_path}")
            aggregated_df.to_csv(aggregated_csv_path, index=False)
            print(f"✓ Aggregated CSV file saved successfully with {len(aggregated_df):,} rows")
            
            # Create activation heatmaps
            print("\nCreating activation function heatmaps...")
            heatmap_mean_data = create_activation_heatmap(df, output_dir, 'mean')
            heatmap_median_data = create_activation_heatmap(df, output_dir, 'median')
            
            # Create heatmaps excluding sigmoid
            print("\nCreating heatmaps excluding sigmoid...")
            df_no_sigmoid = df[~((df['activation1_name'] == 'sigmoid') | (df['activation2_name'] == 'sigmoid'))]
            if len(df_no_sigmoid) > 0:
                heatmap_mean_no_sigmoid = create_activation_heatmap(df_no_sigmoid, output_dir, 'mean', suffix='_no_sigmoid')
                heatmap_median_no_sigmoid = create_activation_heatmap(df_no_sigmoid, output_dir, 'median', suffix='_no_sigmoid')
                print(f"✓ Excluded sigmoid heatmaps created with {len(df_no_sigmoid):,} connections (removed {len(df) - len(df_no_sigmoid):,})")
            else:
                print("Warning: No connections remain after excluding sigmoid")
            
            # Clear memory before summary
            del all_results
            
            # Print summary statistics
            print_summary_statistics(df)
            
        except Exception as e:
            print(f"Error saving CSV file: {e}")
            print("Attempting alternative save method...")
            try:
                # Alternative: save without chunking
                df.to_csv(csv_path, index=False)
                print(f"✓ CSV file saved successfully with {len(df):,} rows")
                
                # Create aggregated dataset
                print("\nCreating aggregated weight statistics...")
                aggregated_df = create_aggregated_weights(df)
                
                # Save aggregated CSV
                aggregated_csv_path = os.path.join(output_dir, 'smartmixed_weights_aggregated.csv')
                print(f"Saving aggregated dataset as CSV: {aggregated_csv_path}")
                aggregated_df.to_csv(aggregated_csv_path, index=False)
                print(f"✓ Aggregated CSV file saved successfully with {len(aggregated_df):,} rows")
                
                # Create activation heatmaps
                print("\nCreating activation function heatmaps...")
                heatmap_mean_data = create_activation_heatmap(df, output_dir, 'mean')
                heatmap_median_data = create_activation_heatmap(df, output_dir, 'median')
                
                # Create heatmaps excluding sigmoid
                print("\nCreating heatmaps excluding sigmoid...")
                df_no_sigmoid = df[~((df['activation1_name'] == 'sigmoid') | (df['activation2_name'] == 'sigmoid'))]
                if len(df_no_sigmoid) > 0:
                    heatmap_mean_no_sigmoid = create_activation_heatmap(df_no_sigmoid, output_dir, 'mean', suffix='_no_sigmoid')
                    heatmap_median_no_sigmoid = create_activation_heatmap(df_no_sigmoid, output_dir, 'median', suffix='_no_sigmoid')
                    print(f"✓ Excluded sigmoid heatmaps created with {len(df_no_sigmoid):,} connections (removed {len(df) - len(df_no_sigmoid):,})")
                else:
                    print("Warning: No connections remain after excluding sigmoid")
                
                print_summary_statistics(df)
            except Exception as e2:
                print(f"Failed to save CSV file: {e2}")
                # Last resort: save as pickle file
                pickle_path = csv_path.replace('.csv', '_backup.pkl')
                try:
                    df.to_pickle(pickle_path)
                    print(f"✓ Data saved as backup pickle file: {pickle_path}")
                    print("You can load this later with: pd.read_pickle('smartmixed_weights_backup.pkl')")
                except Exception as e3:
                    print(f"Even pickle save failed: {e3}")
                return
                
    else:
        print("No weight connections were extracted!")


if __name__ == "__main__":
    main()