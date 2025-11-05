import os
import torch
import numpy as np
import pandas as pd
import argparse
import torch.nn as nn
from models import CustomNeuralNetwork, set_seed, NeuralNetworkWithAdaptiveActivation, get_activation, MixedNeuralNetwork
from train import train_and_evaluate, get_mnist_loaders, run_inference_from_folder, create_mixed_model_from_selective, get_all_mnist_loaders, get_mnist_dataset_info
from experiment_orchestrator import run_all_experiments
from architectures import get_architecture, list_architectures, get_all_architecture_ids, get_default_architecture
from architecture_iterator import run_architecture_iterator

def run_smartmixed_training(input_dim, device, args, architecture, output_dir=None):
    """
    Run smartmixed training: two-phase training with selective then mixed models.
    
    Args:
        input_dim: Input dimension for the network
        device: PyTorch device to use
        args: Command line arguments
        architecture: Network architecture definition
        output_dir: Optional custom output directory. If None, creates timestamped directory.
    """
    # Create main smartmixed output directory
    if output_dir is None:
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        main_output_dir = os.path.join('outputs', f'smartmixed_{timestamp}')
    else:
        main_output_dir = output_dir
    os.makedirs(main_output_dir, exist_ok=True)
    
    print("="*60)
    print("SMART MIXED MODEL MODE - Two-phase training: Selective then Mixed")
    print("="*60)
    print(f"Main output directory: {main_output_dir}")
    
    # Phase 1: Train selective model until transition epoch
    print(f"Phase 1: Training selective model for {args.transition_epoch} epochs")
    print("="*60)
    activation_pool = [torch.relu, torch.sigmoid, torch.tanh, torch.nn.functional.leaky_relu, torch.nn.functional.elu, torch.nn.functional.selu]
    selective_model = NeuralNetworkWithAdaptiveActivation(architecture, input_dim, activation_pool, temperature=0.3, hard=args.gumbel_hard)
    
    print(f"Training Parameters for Phase 1:")
    print(f"  - Epochs: {args.transition_epoch}")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Small Size Mode: {args.small_size}")
    print(f"  - Verbose Mode: {args.verbose}")
    print(f"  - Device: {device}")
    print("="*60)
    
    # Train selective model for transition_epoch epochs
    from train import train_and_evaluate_smartmixed_phase1
    selective_output_dir = train_and_evaluate_smartmixed_phase1(
        selective_model, 'selective', device, 
        epochs=args.transition_epoch, lr=args.lr, batch_size=args.batch_size, 
        architecture=architecture, small_size=args.small_size, verbose=args.verbose,
        main_output_dir=main_output_dir
    )
    
    print("\n" + "="*60)
    print("Phase 2: Creating mixed model from selective and training remaining epochs")
    print("="*60)
    
    # Phase 2: Create mixed model from selective and train for remaining epochs
    remaining_epochs = args.epochs - args.transition_epoch
    print(f"Creating mixed model from selective model at: {selective_output_dir}")
    
    from train import create_mixed_model_from_selective
    mixed_model, mixed_architecture = create_mixed_model_from_selective(selective_output_dir, input_dim)
    
    print(f"Training Parameters for Phase 2:")
    print(f"  - Remaining Epochs: {remaining_epochs}")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Small Size Mode: {args.small_size}")
    print(f"  - Verbose Mode: {args.verbose}")
    print(f"  - Device: {device}")
    print("="*60)
    
    # Train mixed model for remaining epochs
    from train import train_and_evaluate_smartmixed_phase2
    final_output_dir = train_and_evaluate_smartmixed_phase2(
        mixed_model, 'smartmixed', device,
        epochs=remaining_epochs, lr=args.lr, batch_size=args.batch_size,
        architecture=mixed_architecture, small_size=args.small_size, verbose=args.verbose,
        selective_output_dir=selective_output_dir, transition_epoch=args.transition_epoch,
        total_epochs=args.epochs, main_output_dir=main_output_dir
    )
    
    print("\n" + "="*60)
    print("SMART MIXED MODEL TRAINING COMPLETED")
    print("="*60)
    print(f"Phase 1 (Selective) results: {selective_output_dir}")
    print(f"Phase 2 (Mixed) results: {final_output_dir}")
    print("="*60)

def main():
    set_seed(42)
    parser = argparse.ArgumentParser(description='Train neural networks on MNIST')
    parser.add_argument('--model', type=str, choices=['custom', 'selective', 'all', 'iterate', 'mixed', 'smartmixed'], default='iterate', help='Model type')
    parser.add_argument('--epochs', type=int, default=400, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--small_size', action='store_true', help='Use only 200 samples for quick run (180 train + 20 val + 33 test)')
    parser.add_argument('--verbose', action='store_true', help='Print detailed logits statistics during training')
    parser.add_argument('--inference_folder', type=str, help='Path to output folder of trained model')
    parser.add_argument('--use_full_mnist', action='store_true', help='Use official MNIST split (60k train + 10k test) instead of custom 3-way split')
    parser.add_argument('--show_mnist_info', action='store_true', help='Show MNIST dataset information and exit')
    parser.add_argument('--gumbel_hard', action=argparse.BooleanOptionalAction, default=True, help='Use hard (one-hot) Gumbel-Softmax for adaptive activations (default: True)')
    parser.add_argument('--architectures', type=str, help='Comma-separated list of architecture IDs to run (for iterate mode). If not specified, runs all architectures.')
    parser.add_argument('--iterate_choice', type=str, default="custom,smartmixed" ,help='Comma-separated list of model types to run in iterate mode. Options: custom, selective, smartmixed. Default is all.')
    parser.add_argument('--selective_model_path', type=str, default="outputs/iterate_20250911_095346_done/arch_12_wide_6layer/all_models_20250912_233558/selective", help='Path to trained selective model folder (required for mixed mode)')
    parser.add_argument('--transition_epoch', type=int, default=50, help='Epoch to transition from selective to mixed in smartmixed mode (default: 50)')
    parser.add_argument('--list_architectures', action='store_true', help='List all available architectures and exit')
    args = parser.parse_args()

    if args.list_architectures:
        list_architectures()
        return

    if args.show_mnist_info:
        get_mnist_dataset_info()
        return

    if args.inference_folder:
        run_inference_from_folder(args.inference_folder)
        return

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    input_dim = 28*28
    
    # Use default architecture for 'all' mode
    default_architecture_template = get_default_architecture()['layers']

    if args.model == 'all':
        run_all_experiments(input_dim, device, args, default_architecture_template)
        return

    if args.model == 'iterate':
        # Parse architecture IDs if provided
        architecture_ids = None
        if args.architectures:
            try:
                architecture_ids = [int(x.strip()) for x in args.architectures.split(',')]
                # Validate that all architecture IDs exist
                for arch_id in architecture_ids:
                    get_architecture(arch_id)  # This will raise an error if ID doesn't exist
            except ValueError as e:
                print(f"Error parsing architecture IDs: {e}")
                return
            except Exception as e:
                print(f"Error: {e}")
                return
        
        # Parse iterate_choice if provided
        iterate_choices = None
        if args.iterate_choice:
            try:
                iterate_choices = [x.strip() for x in args.iterate_choice.split(',')]
                # Validate that all choices are valid
                valid_choices = ['custom', 'selective', 'smartmixed']
                for choice in iterate_choices:
                    if choice not in valid_choices:
                        print(f"Error: Invalid iterate_choice '{choice}'. Valid options are: {valid_choices}")
                        return
            except Exception as e:
                print(f"Error parsing iterate_choice: {e}")
                return
        
        run_architecture_iterator(input_dim, device, args, architecture_ids, iterate_choices)
        return

    if args.model == 'custom':
        architecture = [
                # # (19)
       
            [768, 'leaky_relu'],
            [512, 'relu'],
            [512, 'relu'],
            [256, 'selu'],
            [256, 'elu'],
            [128, 'elu'],
            [128, 'elu'],
            [64, 'elu'],
            [64, 'elu'],
            [32, 'elu'],
            [32, 'selu'],
            [16, 'selu'],
            [10, 'softmax']
        ]
        model = CustomNeuralNetwork(architecture, input_dim)
        model_name = 'CustomNeuralNetwork'
    elif args.model == 'selective':
        architecture = [
            [512, 'selective'],
            [256, 'selective'],
            [128, 'selective'],
            [64, 'selective'],
            [10, 'softmax']
        ]
        # Pool of activation functions for the selective layer
        activation_pool = [torch.relu, torch.sigmoid, torch.tanh, torch.nn.functional.leaky_relu, torch.nn.functional.elu, torch.nn.functional.selu]
        model = NeuralNetworkWithAdaptiveActivation(architecture, input_dim, activation_pool, temperature=0.3, hard=args.gumbel_hard)
        model_name = 'selective'
    elif args.model == 'mixed':
        print("="*60)
        print("MIXED MODEL MODE - Creating fixed network from selective model")
        print("="*60)
        
        if not args.selective_model_path:
            print("Error: --selective_model_path is required for mixed mode")
            return
        
        if not os.path.exists(args.selective_model_path):
            print(f"Error: Selective model path does not exist: {args.selective_model_path}")
            return
        
        print(f"Source selective model: {args.selective_model_path}")
        model, architecture = create_mixed_model_from_selective(args.selective_model_path, input_dim)
        model_name = 'mixed'
        
        print("="*60)
        print("STARTING MIXED MODEL TRAINING")
        print("="*60)
        print(f"Training Parameters:")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Learning Rate: {args.lr}")
        print(f"  - Batch Size: {args.batch_size}")
        print(f"  - Small Size Mode: {args.small_size}")
        print(f"  - Verbose Mode: {args.verbose}")
        print(f"  - Device: {device}")
        print("="*60)
    elif args.model == 'smartmixed':
        architecture = [
            [512, 'selective'],
            [256, 'selective'],
            [128, 'selective'],
            [64, 'selective'],
            [10, 'softmax']
        ]
        run_smartmixed_training(input_dim, device, args, architecture)
        return

    train_and_evaluate(model, model_name, device, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, architecture=architecture, small_size=args.small_size, verbose=args.verbose, force_dir=False, output_dir=None)

if __name__ == '__main__':
    main()
