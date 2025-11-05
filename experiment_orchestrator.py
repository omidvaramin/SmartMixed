import os
import torch
import pandas as pd
from models import CustomNeuralNetwork, NeuralNetworkWithAdaptiveActivation
from train import train_and_evaluate

def run_all_experiments(input_dim, device, args, architecture_template, iterate_choices=None):
    """
    Runs experiments for each activation function and selective, organizes outputs, and summarizes results in a CSV.
    architecture_template: list of [size, None] (activation will be filled in)
    iterate_choices: list of model types to run ('custom', 'selective', 'smartmixed'). If None, runs all.
    """
    activation_names = ['relu', 'sigmoid', 'tanh', 'leaky_relu', 'elu', 'selu']
    selective_pool = [torch.relu, torch.sigmoid, torch.tanh, torch.nn.functional.leaky_relu, torch.nn.functional.elu, torch.nn.functional.selu]
    import datetime
    
    # Check if a custom output base is provided (for architecture iterator)
    if hasattr(args, 'output_base') and args.output_base:
        parent_folder = os.path.join(args.output_base, f"all_models_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    else:
        # Use only the existing outputs directory, do not nest outputs/outputs
        parent_folder = os.path.join(os.path.dirname(__file__), 'outputs', f"all_models_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(parent_folder, exist_ok=True)
    
    # Set default iterate_choices if none provided
    if iterate_choices is None:
        iterate_choices = ['custom', 'selective', 'smartmixed']
    
    print(f"Running experiments for model types: {iterate_choices}")
    
    results = []
    
    # Run custom models with different activation functions
    if 'custom' in iterate_choices:
        # Fix: Only create one folder per experiment (relu, sigmoid, etc.)
        for idx, act_name in enumerate(activation_names):
            print(f"[Custom Experiment {idx+1}/{len(activation_names)}] Training with activation: {act_name}")
            subfolder = os.path.join(parent_folder, act_name)
            if not os.path.exists(subfolder):
                os.makedirs(subfolder)
            # Fill in activation for each layer except the last (which should be softmax)
            architecture = [[size, act_name] for size, _ in architecture_template[:-1]] + [architecture_template[-1]]
            model = CustomNeuralNetwork(architecture, input_dim)
            # Pass model_name and output_dir explicitly
            output_dir = train_and_evaluate(model, act_name, device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, architecture=architecture, small_size=args.small_size, verbose=args.verbose, force_dir=True, output_dir=subfolder)
            # Read total_accuracy and best_epoch from output_dir
            try:
                with open(os.path.join(output_dir, 'total_accuracy.txt')) as f:
                    line = f.read().strip()
                    if 'Total accuracy:' in line:
                        total_accuracy = float(line.split('Total accuracy:')[1].strip())
                    else:
                        total_accuracy = float(line)
            except Exception:
                total_accuracy = None
            try:
                with open(os.path.join(output_dir, 'best_epoch.txt')) as f:
                    best_epoch = int(f.read().strip())
            except Exception:
                best_epoch = None
            try:
                with open(os.path.join(output_dir, 'best_val_accuracy.txt')) as f:
                    best_val_accuracy = float(f.read().strip())
            except Exception:
                best_val_accuracy = None
            metrics = {
                'activation': act_name,
                'total_accuracy': total_accuracy,
                'best_epoch': best_epoch,
                'best_val_accuracy': best_val_accuracy,
                'model': f'CustomNeuralNetwork_{act_name}',
                'output_dir': output_dir
            }
            results.append(metrics)
    
    # Run selective model
    if 'selective' in iterate_choices:
        print(f"[Selective Experiment] Training with activation: selective")
        subfolder = os.path.join(parent_folder, 'selective')
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        architecture = [[size, 'selective'] for size, _ in architecture_template[:-1]] + [architecture_template[-1]]
        model = NeuralNetworkWithAdaptiveActivation(architecture, input_dim, selective_pool, temperature=0.3, hard=args.gumbel_hard)
        output_dir = train_and_evaluate(model, 'selective', device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, architecture=architecture, small_size=args.small_size, verbose=args.verbose, force_dir=True, output_dir=subfolder)
        
        try:
            with open(os.path.join(output_dir, 'total_accuracy.txt')) as f:
                line = f.read().strip()
                if 'Total accuracy:' in line:
                    total_accuracy = float(line.split('Total accuracy:')[1].strip())
                else:
                    total_accuracy = float(line)
        except Exception:
            total_accuracy = None
        
        try:
            with open(os.path.join(output_dir, 'best_epoch.txt')) as f:
                best_epoch = int(f.read().strip())
        except Exception:
            best_epoch = None
        
        try:
            with open(os.path.join(output_dir, 'best_val_accuracy.txt')) as f:
                best_val_accuracy = float(f.read().strip())
        except Exception:
            best_val_accuracy = None
        metrics = {
            'activation': 'selective',
            'total_accuracy': total_accuracy,
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_accuracy,
            'model': 'NeuralNetworkWithAdaptiveActivation',
            'output_dir': output_dir
        }
        results.append(metrics)
    
    # Run smartmixed model
    if 'smartmixed' in iterate_choices:
        print(f"[SmartMixed Experiment] Training with smartmixed approach")
        subfolder = os.path.join(parent_folder, 'smartmixed')
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        
        # Create smartmixed architecture (selective layers for phase 1)
        architecture = [[size, 'selective'] for size, _ in architecture_template[:-1]] + [architecture_template[-1]]
        
        # Run smartmixed training (two-phase) directly in the subfolder
        from train import train_and_evaluate_smartmixed_phase1, train_and_evaluate_smartmixed_phase2, create_mixed_model_from_selective
        
        # Phase 1: Train selective model
        print(f"[SmartMixed Phase 1] Training selective model for {args.transition_epoch} epochs")
        activation_pool = [torch.relu, torch.sigmoid, torch.tanh, torch.nn.functional.leaky_relu, torch.nn.functional.elu, torch.nn.functional.selu]
        selective_model = NeuralNetworkWithAdaptiveActivation(architecture, input_dim, activation_pool, temperature=0.3, hard=args.gumbel_hard)
        
        selective_output_dir = train_and_evaluate_smartmixed_phase1(
            selective_model, 'selective', device, 
            epochs=args.transition_epoch, lr=args.lr, batch_size=args.batch_size, 
            architecture=architecture, small_size=args.small_size, verbose=args.verbose,
            main_output_dir=subfolder
        )
        
        # Phase 2: Create mixed model from selective and train for remaining epochs
        remaining_epochs = args.epochs - args.transition_epoch
        print(f"[SmartMixed Phase 2] Creating mixed model and training for {remaining_epochs} epochs")
        
        mixed_model, mixed_architecture = create_mixed_model_from_selective(selective_output_dir, input_dim)
        
        phase2_output_dir = train_and_evaluate_smartmixed_phase2(
            mixed_model, 'smartmixed', device,
            epochs=remaining_epochs, lr=args.lr, batch_size=args.batch_size,
            architecture=mixed_architecture, small_size=args.small_size, verbose=args.verbose,
            selective_output_dir=selective_output_dir, transition_epoch=args.transition_epoch,
            total_epochs=args.epochs, main_output_dir=subfolder
        )
        
        # Read results from phase2 output directory
        if phase2_output_dir and os.path.exists(phase2_output_dir):
            try:
                with open(os.path.join(phase2_output_dir, 'total_accuracy.txt')) as f:
                    line = f.read().strip()
                    if 'Total accuracy:' in line:
                        total_accuracy = float(line.split('Total accuracy:')[1].strip())
                    else:
                        total_accuracy = float(line)
            except Exception:
                total_accuracy = None
            
            try:
                with open(os.path.join(phase2_output_dir, 'best_epoch_absolute.txt')) as f:
                    best_epoch = int(f.read().strip())
            except Exception:
                try:
                    with open(os.path.join(phase2_output_dir, 'best_epoch.txt')) as f:
                        best_epoch = int(f.read().strip())
                        # Add transition epoch if available
                        if args.transition_epoch:
                            best_epoch += args.transition_epoch
                except Exception:
                    best_epoch = None
            
            try:
                with open(os.path.join(phase2_output_dir, 'best_val_accuracy.txt')) as f:
                    best_val_accuracy = float(f.read().strip())
            except Exception:
                best_val_accuracy = None
                
            output_dir = phase2_output_dir
        else:
            total_accuracy = None
            best_epoch = None
            best_val_accuracy = None
            output_dir = subfolder
        
        metrics = {
            'activation': 'smartmixed',
            'total_accuracy': total_accuracy,
            'best_epoch': best_epoch,
            'best_val_accuracy': best_val_accuracy,
            'model': 'SmartMixed',
            'output_dir': output_dir
        }
        results.append(metrics)
    
    # Save summary CSV
    print("[Summary] Writing summary.csv with all results...")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(parent_folder, 'summary.csv'), index=False)
    print(f"All experiments complete. Summary saved to {os.path.join(parent_folder, 'summary.csv')}")
