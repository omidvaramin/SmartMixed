"""
Architecture iterator for running experiments across multiple architectures.
"""

import os
import datetime
from architectures import get_architecture, get_all_architecture_ids
from experiment_orchestrator import run_all_experiments

def run_architecture_iterator(input_dim, device, args, architecture_ids=None, iterate_choices=None):
    """
    Run experiments for multiple architectures.
    
    Args:
        input_dim: Input dimension for the network
        device: PyTorch device to use
        args: Command line arguments
        architecture_ids: List of architecture IDs to run. If None, runs all architectures.
        iterate_choices: List of model types to run ('custom', 'selective', 'smartmixed'). If None, runs all.
    """
    if architecture_ids is None:
        architecture_ids = get_all_architecture_ids()
    
    # Create main iterator folder
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    iterator_folder = os.path.join('outputs', f'iterate_{timestamp}')
    os.makedirs(iterator_folder, exist_ok=True)
    
    print(f"Starting architecture iterator experiments in: {iterator_folder}")
    print(f"Running experiments for {len(architecture_ids)} architectures: {architecture_ids}")
    if iterate_choices:
        print(f"Model types to run: {iterate_choices}")
    else:
        print("Model types to run: all (custom, selective, smartmixed)")
    
    results_summary = []
    
    for arch_id in architecture_ids:
        try:
            arch_info = get_architecture(arch_id)
            arch_name = arch_info['name']
            arch_layers = arch_info['layers']
            
            print(f"\n{'='*80}")
            print(f"Running experiments for Architecture {arch_id}: {arch_name}")
            print(f"Layers: {arch_layers}")
            print(f"{'='*80}")
            
            # Create architecture-specific folder
            arch_folder = os.path.join(iterator_folder, f"arch_{arch_id}_{arch_name}")
            os.makedirs(arch_folder, exist_ok=True)
            
            # Save architecture info
            with open(os.path.join(arch_folder, 'architecture_info.txt'), 'w') as f:
                f.write(f'Architecture ID: {arch_id}\n')
                f.write(f'Architecture Name: {arch_name}\n')
                f.write(f'Architecture Layers: {arch_layers}\n')
                f.write(f'Number of Layers: {len(arch_layers)}\n')
            
            # Temporarily modify args to use this specific output directory
            original_output_base = getattr(args, 'output_base', None)
            args.output_base = arch_folder
            
            # Run all experiments for this architecture
            run_all_experiments(input_dim, device, args, arch_layers, iterate_choices)
            
            # Restore original output base
            if original_output_base is not None:
                args.output_base = original_output_base
            elif hasattr(args, 'output_base'):
                delattr(args, 'output_base')
            
            results_summary.append({
                'arch_id': arch_id,
                'arch_name': arch_name,
                'layers_count': len(arch_layers),
                'status': 'completed',
                'folder': arch_folder
            })
            
            print(f"✅ Completed experiments for Architecture {arch_id}: {arch_name}")
            
        except Exception as e:
            print(f"❌ Error running experiments for Architecture {arch_id}: {e}")
            results_summary.append({
                'arch_id': arch_id,
                'arch_name': arch_info.get('name', 'unknown') if 'arch_info' in locals() else 'unknown',
                'layers_count': len(arch_info.get('layers', [])) if 'arch_info' in locals() else 0,
                'status': f'error: {str(e)}',
                'folder': arch_folder if 'arch_folder' in locals() else 'not_created'
            })
    
    # Save overall summary
    summary_file = os.path.join(iterator_folder, 'iterator_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Architecture Iterator Summary\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Total architectures: {len(architecture_ids)}\n")
        f.write(f"Model types: {iterate_choices if iterate_choices else 'all (custom, selective, smartmixed)'}\n")
        f.write(f"Arguments: epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}\n")
        f.write(f"Small size: {args.small_size}\n")
        f.write(f"\nResults:\n")
        f.write("-" * 80 + "\n")
        
        for result in results_summary:
            f.write(f"Architecture {result['arch_id']}: {result['arch_name']}\n")
            f.write(f"  Layers: {result['layers_count']}\n")
            f.write(f"  Status: {result['status']}\n")
            f.write(f"  Folder: {result['folder']}\n")
            f.write("-" * 40 + "\n")
    
    print(f"\n{'='*80}")
    print(f"Architecture iterator experiments completed!")
    print(f"Results saved in: {iterator_folder}")
    print(f"Summary saved in: {summary_file}")
    print(f"{'='*80}")
    
    return iterator_folder
