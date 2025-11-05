#!/usr/bin/env python3
"""
Performance Aggregator for Adaptive Activation Function Experiments

This script aggregates performance data from the iterate_20250911_095346_done experiment
and generates an Excel file with accuracy metrics for each architecture and model.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def collect_performance_data(base_path):
    """
    Collect performance data from all architecture folders.
    
    Args:
        base_path (str): Path to the iterate_20250911_095346_done folder
        
    Returns:
        pd.DataFrame: Aggregated performance data
    """
    all_data = []
    
    # Get all architecture folders
    arch_folders = [f for f in os.listdir(base_path) if f.startswith('arch_') and os.path.isdir(os.path.join(base_path, f))]
    arch_folders.sort(key=lambda x: int(x.split('_')[1]))  # Sort by architecture number
    
    print(f"Found {len(arch_folders)} architecture folders")
    
    for arch_folder in arch_folders:
        arch_path = os.path.join(base_path, arch_folder)
        
        # Read architecture info
        arch_info_path = os.path.join(arch_path, 'architecture_info.txt')
        arch_id = None
        arch_name = None
        arch_layers = None
        num_layers = None
        
        if os.path.exists(arch_info_path):
            with open(arch_info_path, 'r') as f:
                for line in f:
                    if line.startswith('Architecture ID:'):
                        arch_id = int(line.split(':')[1].strip())
                    elif line.startswith('Architecture Name:'):
                        arch_name = line.split(':')[1].strip()
                    elif line.startswith('Architecture Layers:'):
                        arch_layers = line.split(':')[1].strip()
                    elif line.startswith('Number of Layers:'):
                        num_layers = int(line.split(':')[1].strip())
        
        # Find the models folder (should be all_models_*)
        models_folders = [f for f in os.listdir(arch_path) if f.startswith('all_models_') and os.path.isdir(os.path.join(arch_path, f))]
        
        if not models_folders:
            print(f"Warning: No models folder found in {arch_folder}")
            continue
            
        models_path = os.path.join(arch_path, models_folders[0])
        summary_csv_path = os.path.join(models_path, 'summary.csv')
        
        if not os.path.exists(summary_csv_path):
            print(f"Warning: No summary.csv found in {models_path}")
            continue
            
        # Read the summary CSV
        try:
            df = pd.read_csv(summary_csv_path)
            
            for _, row in df.iterrows():
                data_row = {
                    'Architecture_ID': arch_id,
                    'Architecture_Name': arch_name,
                    'Architecture_Layers': arch_layers,
                    'Number_of_Layers': num_layers,
                    'Activation_Function': row['activation'],
                    'Model_Type': row['model'],
                    'Test_Accuracy': row['total_accuracy'],
                    'Validation_Accuracy': row['best_val_accuracy'], 
                    'Best_Epoch': row['best_epoch'],
                    'Output_Directory': row['output_dir']
                }
                all_data.append(data_row)
                
        except Exception as e:
            print(f"Error reading {summary_csv_path}: {e}")
            continue
            
        print(f"Processed {arch_folder}: {len(df)} models")
    
    return pd.DataFrame(all_data)

def generate_excel_report(df, output_path):
    """
    Generate an Excel file with multiple sheets for different views of the data.
    
    Args:
        df (pd.DataFrame): Performance data
        output_path (str): Path where to save the Excel file
    """
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Sheet 1: Complete data
        df_complete = df.copy()
        df_complete = df_complete.round(4)
        df_complete.to_excel(writer, sheet_name='Complete_Data', index=False)
        
        # Sheet 2: Summary by Architecture
        df_arch_summary = df.groupby(['Architecture_ID', 'Architecture_Name', 'Number_of_Layers']).agg({
            'Test_Accuracy': ['mean', 'max', 'min', 'std'],
            'Validation_Accuracy': ['mean', 'max', 'min', 'std'],
            'Best_Epoch': ['mean', 'max', 'min', 'std']
        }).round(4)
        
        # Flatten column names
        df_arch_summary.columns = ['_'.join(col).strip() for col in df_arch_summary.columns.values]
        df_arch_summary = df_arch_summary.reset_index()
        df_arch_summary.to_excel(writer, sheet_name='Architecture_Summary', index=False)
        
        # Sheet 3: Best performing model per architecture
        df_best = df.loc[df.groupby('Architecture_ID')['Test_Accuracy'].idxmax()]
        df_best = df_best[['Architecture_ID', 'Architecture_Name', 'Number_of_Layers', 
                          'Activation_Function', 'Model_Type', 'Test_Accuracy', 
                          'Validation_Accuracy', 'Best_Epoch']].round(4)
        df_best.to_excel(writer, sheet_name='Best_Per_Architecture', index=False)
        
        # Sheet 4: Activation function comparison
        df_activation = df.groupby('Activation_Function').agg({
            'Test_Accuracy': ['mean', 'max', 'min', 'std', 'count'],
            'Validation_Accuracy': ['mean', 'max', 'min', 'std'],
            'Best_Epoch': ['mean', 'max', 'min', 'std']
        }).round(4)
        
        # Flatten column names
        df_activation.columns = ['_'.join(col).strip() for col in df_activation.columns.values]
        df_activation = df_activation.reset_index()
        df_activation.to_excel(writer, sheet_name='Activation_Comparison', index=False)
        
        # Sheet 5: Top 10 performing models overall
        df_top10 = df.nlargest(10, 'Test_Accuracy')[['Architecture_ID', 'Architecture_Name', 
                                                     'Number_of_Layers', 'Activation_Function', 
                                                     'Model_Type', 'Test_Accuracy', 
                                                     'Validation_Accuracy', 'Best_Epoch']].round(4)
        df_top10.to_excel(writer, sheet_name='Top_10_Models', index=False)

def create_first_rank_chart(df, output_dir):
    """
    Create a bar chart showing how many times each activation function achieved first rank.
    
    Args:
        df (pd.DataFrame): Performance data
        output_dir (str): Directory to save the chart
    """
    # Find the best performing activation function for each architecture
    best_per_arch = df.loc[df.groupby('Architecture_ID')['Test_Accuracy'].idxmax()]
    
    # Count how many times each activation function was the best
    first_rank_counts = best_per_arch['Activation_Function'].value_counts()
    
    # Create the bar chart
    plt.figure(figsize=(12, 8))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create bar plot
    bars = plt.bar(first_rank_counts.index, first_rank_counts.values, 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(first_rank_counts)])
    
    # Customize the chart
    plt.title('Number of Times Each Activation Function Achieved First Rank\n(Best Test Accuracy per Architecture)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Activation Function', fontsize=14, fontweight='bold')
    plt.ylabel('Number of First Place Finishes', fontsize=14, fontweight='bold')
    
    # Add value labels on top of bars
    for i, (bar, count) in enumerate(zip(bars, first_rank_counts.values)):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                f'{count}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the chart
    chart_path = os.path.join(output_dir, 'first_rank_frequency_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"First rank frequency chart saved to: {chart_path}")
    
    # Print the ranking summary
    print("\nFirst Rank Summary:")
    print("=" * 50)
    total_architectures = df['Architecture_ID'].nunique()
    for activation, count in first_rank_counts.items():
        percentage = (count / total_architectures) * 100
        print(f"{activation:12}: {count:2d} times ({percentage:5.1f}%)")
    
    return first_rank_counts

def create_average_accuracy_chart(df, output_dir):
    """
    Create a bar chart showing the average test accuracy for each activation function.
    
    Args:
        df (pd.DataFrame): Performance data
        output_dir (str): Directory to save the chart
    """
    # Calculate average accuracy for each activation function
    avg_accuracy = df.groupby('Activation_Function')['Test_Accuracy'].agg(['mean', 'std']).reset_index()
    avg_accuracy = avg_accuracy.sort_values('mean', ascending=False)
    
    # Create the bar chart
    plt.figure(figsize=(12, 8))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create bar plot with error bars
    bars = plt.bar(avg_accuracy['Activation_Function'], avg_accuracy['mean'], 
                   yerr=avg_accuracy['std'], capsize=5,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(avg_accuracy)])
    
    # Customize the chart
    plt.title('Average Test Accuracy by Activation Function\n(Across All Architectures)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Activation Function', fontsize=14, fontweight='bold')
    plt.ylabel('Average Test Accuracy', fontsize=14, fontweight='bold')
    
    # Add value labels on top of bars
    for i, (bar, mean_acc, std_acc) in enumerate(zip(bars, avg_accuracy['mean'], avg_accuracy['std'])):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std_acc + 0.001,
                f'{mean_acc:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Set y-axis to show a reasonable range
    y_min = avg_accuracy['mean'].min() - avg_accuracy['std'].max() - 0.005
    y_max = avg_accuracy['mean'].max() + avg_accuracy['std'].max() + 0.005
    plt.ylim(y_min, y_max)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the chart
    chart_path = os.path.join(output_dir, 'average_accuracy_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Average accuracy chart saved to: {chart_path}")
    
    # Print the accuracy summary
    print("\nAverage Accuracy Summary:")
    print("=" * 60)
    for _, row in avg_accuracy.iterrows():
        activation = row['Activation_Function']
        mean_acc = row['mean']
        std_acc = row['std']
        print(f"{activation:12}: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return avg_accuracy

def create_ranking_distribution_chart(df, output_dir):
    """
    Create a grouped bar chart showing how many times each activation function 
    achieved first, second, and third rank across all architectures.
    
    Args:
        df (pd.DataFrame): Performance data
        output_dir (str): Directory to save the chart
    """
    # Initialize ranking data structure
    ranking_data = {}
    
    # For each architecture, rank the activation functions by test accuracy
    for arch_id in df['Architecture_ID'].unique():
        arch_data = df[df['Architecture_ID'] == arch_id].copy()
        
        # Sort by test accuracy in descending order
        arch_data = arch_data.sort_values('Test_Accuracy', ascending=False)
        
        # Assign ranks (1st, 2nd, 3rd, 4th, 5th, 6th, 7th)
        for rank, (_, row) in enumerate(arch_data.iterrows(), 1):
            activation = row['Activation_Function']
            
            if activation not in ranking_data:
                ranking_data[activation] = {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0, '5th': 0, '6th': 0, '7th': 0}
            
            if rank == 1:
                ranking_data[activation]['1st'] += 1
            elif rank == 2:
                ranking_data[activation]['2nd'] += 1
            elif rank == 3:
                ranking_data[activation]['3rd'] += 1
            elif rank == 4:
                ranking_data[activation]['4th'] += 1
            elif rank == 5:
                ranking_data[activation]['5th'] += 1
            elif rank == 6:
                ranking_data[activation]['6th'] += 1
            elif rank == 7:
                ranking_data[activation]['7th'] += 1
    
    # Convert to DataFrame for easier plotting
    ranking_df = pd.DataFrame(ranking_data).T.fillna(0)
    
    # Sort by total first place finishes
    ranking_df = ranking_df.sort_values('1st', ascending=False)
    
    # Create the grouped bar chart
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Set the width of bars and positions (thinner bars)
    bar_width = 0.10
    positions = np.arange(len(ranking_df))
    
    # Colors for each rank
    colors = ['#2E8B57', '#FF6347', '#4682B4', '#DAA520', '#9467BD', '#8C564B', '#FF69B4']  # Green, Red, Blue, Gold, Purple, Brown, Pink
    
    # Create bars for each rank
    bars1 = ax.bar(positions - 3*bar_width, ranking_df['1st'], bar_width, 
                   label='1st Place', color=colors[0], alpha=0.8)
    bars2 = ax.bar(positions - 2*bar_width, ranking_df['2nd'], bar_width, 
                   label='2nd Place', color=colors[1], alpha=0.8)
    bars3 = ax.bar(positions - 1*bar_width, ranking_df['3rd'], bar_width, 
                   label='3rd Place', color=colors[2], alpha=0.8)
    bars4 = ax.bar(positions, ranking_df['4th'], bar_width, 
                   label='4th Place', color=colors[3], alpha=0.8)
    bars5 = ax.bar(positions + 1*bar_width, ranking_df['5th'], bar_width, 
                   label='5th Place', color=colors[4], alpha=0.8)
    bars6 = ax.bar(positions + 2*bar_width, ranking_df['6th'], bar_width, 
                   label='6th Place', color=colors[5], alpha=0.8)
    bars7 = ax.bar(positions + 3*bar_width, ranking_df['7th'], bar_width, 
                   label='7th Place', color=colors[6], alpha=0.8)
    
    # Add value labels on top of bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if bar has height
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    add_value_labels(bars4)
    add_value_labels(bars5)
    add_value_labels(bars6)
    add_value_labels(bars7)
    
    # Customize the chart
    ax.set_title('Ranking Distribution by Activation Function\n(Across All Architectures)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Activation Function', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Times Achieved Rank', fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(positions)
    ax.set_xticklabels(ranking_df.index, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis to start from 0
    ax.set_ylim(0, max(ranking_df.max()) + 1)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the chart
    chart_path = os.path.join(output_dir, 'ranking_distribution_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Ranking distribution chart saved to: {chart_path}")
    
    # Print the ranking summary
    print("\nRanking Distribution Summary:")
    print("=" * 90)
    print(f"{'Activation':<12} {'1st':<4} {'2nd':<4} {'3rd':<4} {'4th':<4} {'5th':<4} {'6th':<4} {'7th':<4} {'Total':<6}")
    print("-" * 90)
    
    for activation in ranking_df.index:
        first = int(ranking_df.loc[activation, '1st'])
        second = int(ranking_df.loc[activation, '2nd'])
        third = int(ranking_df.loc[activation, '3rd'])
        fourth = int(ranking_df.loc[activation, '4th'])
        fifth = int(ranking_df.loc[activation, '5th'])
        sixth = int(ranking_df.loc[activation, '6th'])
        seventh = int(ranking_df.loc[activation, '7th'])
        total = first + second + third + fourth + fifth + sixth + seventh
        
        print(f"{activation:<12} {first:<4} {second:<4} {third:<4} {fourth:<4} {fifth:<4} {sixth:<4} {seventh:<4} {total:<6}")
    
    return ranking_df

def create_architecture_ranking_excel(df, output_dir):
    """
    Create an Excel file where each row represents an architecture and columns show 
    which activation function achieved each rank (1st through 7th) for that architecture.
    
    Args:
        df (pd.DataFrame): Performance data
        output_dir (str): Directory to save the Excel file
    """
    # Initialize the results list
    architecture_rankings = []
    
    # For each architecture, rank the activation functions by test accuracy
    for arch_id in sorted(df['Architecture_ID'].unique()):
        arch_data = df[df['Architecture_ID'] == arch_id].copy()
        
        # Get architecture info
        arch_name = arch_data['Architecture_Name'].iloc[0]
        num_layers = arch_data['Number_of_Layers'].iloc[0]
        arch_layers_raw = arch_data['Architecture_Layers'].iloc[0]
        
        # Extract just the neuron counts from the architecture layers
        # Convert from "[[768, None], [512, None], [256, None], [10, 'softmax']]" 
        # to "[768,512,256,10]"
        try:
            import ast
            layers_list = ast.literal_eval(arch_layers_raw)
            neuron_counts = [layer[0] for layer in layers_list]
            neurons_str = str(neuron_counts)
        except:
            # Fallback if parsing fails
            neurons_str = arch_layers_raw
        
        # Filter out 'selective' activation function
        arch_data = arch_data[arch_data['Activation_Function'] != 'selective'].copy()
        
        # Sort by test accuracy in descending order
        arch_data = arch_data.sort_values('Test_Accuracy', ascending=False)
        
        # Create a dictionary for this architecture's rankings
        ranking_row = {
            'Architecture_ID': arch_id,
            'Neurons_Per_Layer': neurons_str,
            'Architecture_Name': arch_name,
            'Number_of_Layers': num_layers,
            '1st_Rank': None,
            '2nd_Rank': None,
            '3rd_Rank': None,
            '4th_Rank': None,
            '5th_Rank': None,
            '6th_Rank': None,
            '7th_Rank': None
        }
        
        # Fill in the rankings
        rank_columns = ['1st_Rank', '2nd_Rank', '3rd_Rank', '4th_Rank', '5th_Rank', '6th_Rank', '7th_Rank']
        
        for rank, (_, row) in enumerate(arch_data.iterrows()):
            if rank < len(rank_columns):  # Only fill up to 7th rank
                activation_function = row['Activation_Function']
                # Store only activation function name (no accuracy)
                ranking_row[rank_columns[rank]] = activation_function
        
        architecture_rankings.append(ranking_row)
    
    # Convert to DataFrame
    ranking_df = pd.DataFrame(architecture_rankings)
    
    # Save to Excel
    excel_path = os.path.join(output_dir, 'architecture_rankings.xlsx')
    ranking_df.to_excel(excel_path, index=False, engine='openpyxl')
    
    print(f"Architecture rankings Excel file saved to: {excel_path}")
    
    # Print a summary
    print("\nArchitecture Rankings Summary:")
    print("=" * 100)
    print(f"{'Arch_ID':<8} {'Neurons':<20} {'Arch_Name':<15} {'Layers':<7} {'1st Place':<15} {'2nd Place':<15} {'3rd Place':<15}")
    print("-" * 100)
    
    for _, row in ranking_df.iterrows():
        arch_id = row['Architecture_ID']
        neurons = row['Neurons_Per_Layer'][:19] if row['Neurons_Per_Layer'] else 'N/A'  # Truncate for display
        arch_name = row['Architecture_Name'][:14] if row['Architecture_Name'] else 'N/A'  # Truncate for display
        layers = row['Number_of_Layers']
        
        # Get activation function names
        first = row['1st_Rank'] if row['1st_Rank'] else 'N/A'
        second = row['2nd_Rank'] if row['2nd_Rank'] else 'N/A'
        third = row['3rd_Rank'] if row['3rd_Rank'] else 'N/A'
        
        print(f"{arch_id:<8} {neurons:<20} {arch_name:<15} {layers:<7} {first:<15} {second:<15} {third:<15}")
    
    return ranking_df

def create_mean_reciprocal_rank_chart(df, output_dir):
    """
    Calculate Mean Reciprocal Rank (MRR) for each activation function and create a bar chart.
    MRR = average of (1/rank) across all architectures for each activation function.
    
    Args:
        df (pd.DataFrame): Performance data
        output_dir (str): Directory to save the chart
    """
    # Initialize MRR data structure
    mrr_data = {}
    
    # For each architecture, rank the activation functions by test accuracy
    for arch_id in df['Architecture_ID'].unique():
        arch_data = df[df['Architecture_ID'] == arch_id].copy()
        
        # Sort by test accuracy in descending order to get ranks
        arch_data = arch_data.sort_values('Test_Accuracy', ascending=False)
        
        # Assign ranks and calculate reciprocal ranks
        for rank, (_, row) in enumerate(arch_data.iterrows(), 1):
            activation = row['Activation_Function']
            reciprocal_rank = 1.0 / rank
            
            if activation not in mrr_data:
                mrr_data[activation] = []
            
            mrr_data[activation].append(reciprocal_rank)
    
    # Calculate MRR for each activation function
    mrr_results = []
    for activation, reciprocal_ranks in mrr_data.items():
        mrr = np.mean(reciprocal_ranks)
        mrr_results.append({
            'Activation_Function': activation,
            'MRR': mrr,
            'Count': len(reciprocal_ranks)
        })
    
    # Convert to DataFrame and sort by MRR
    mrr_df = pd.DataFrame(mrr_results)
    mrr_df = mrr_df.sort_values('MRR', ascending=False)
    
    # Create the bar chart
    plt.figure(figsize=(12, 8))
    
    # Set style
    sns.set_style("whitegrid")
    
    # Create bar plot
    bars = plt.bar(mrr_df['Activation_Function'], mrr_df['MRR'], 
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(mrr_df)])
    
    # Customize the chart
    plt.title('Mean Reciprocal Rank (MRR) by Activation Function\n(Higher values indicate better average ranking)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Activation Function', fontsize=14, fontweight='bold')
    plt.ylabel('Mean Reciprocal Rank (MRR)', fontsize=14, fontweight='bold')
    
    # Add value labels on top of bars
    for i, (bar, mrr_value) in enumerate(zip(bars, mrr_df['MRR'])):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{mrr_value:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    
    # Set y-axis to show a reasonable range (MRR is between 0 and 1)
    plt.ylim(0, max(mrr_df['MRR']) * 1.1)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the chart
    chart_path = os.path.join(output_dir, 'mean_reciprocal_rank_chart.png')
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Mean Reciprocal Rank chart saved to: {chart_path}")
    
    # Print the MRR summary
    print("\nMean Reciprocal Rank Summary:")
    print("=" * 60)
    print(f"{'Activation':<12} {'MRR':<8} {'Count':<6} {'Interpretation'}")
    print("-" * 60)
    
    for _, row in mrr_df.iterrows():
        activation = row['Activation_Function']
        mrr = row['MRR']
        count = row['Count']
        
        # Interpretation based on MRR value
        if mrr >= 0.8:
            interpretation = "Excellent"
        elif mrr >= 0.6:
            interpretation = "Very Good"
        elif mrr >= 0.4:
            interpretation = "Good"
        elif mrr >= 0.2:
            interpretation = "Fair"
        else:
            interpretation = "Poor"
        
        print(f"{activation:<12} {mrr:<7.4f} {count:<6} {interpretation}")
    
    return mrr_df

def main():
    """Main function to aggregate data and generate Excel report."""
    
    # ==================== CONFIGURATION ====================
    # Set the experiment folder name here
    EXPERIMENT_FOLDER = "iterate_20251005_180118_done"
    
    # Set the Excel filename here
    EXCEL_FILENAME = "performance_analysis_iterate_20251005_180118_done.xlsx"
    # ======================================================
    
    # Set the base path to the experiment results
    base_path = f"/Users/aminomidvar/Documents/My Research/adaptive activation function repo/adaptive activation function/outputs/{EXPERIMENT_FOLDER}"
    
    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return
    
    # Create aggregator output directory
    aggregator_dir = os.path.join(base_path, 'aggregator')
    os.makedirs(aggregator_dir, exist_ok=True)
    print(f"Aggregator output directory: {aggregator_dir}")
    
    print("Starting performance data aggregation...")
    print(f"Base path: {base_path}")
    
    # Collect all performance data
    df = collect_performance_data(base_path)
    
    if df.empty:
        print("No data collected. Please check the folder structure.")
        return
        
    print(f"\nCollected data for {len(df)} model configurations")
    print(f"Architectures: {df['Architecture_ID'].nunique()}")
    print(f"Activation functions: {df['Activation_Function'].nunique()}")
    
    # Generate Excel report
    output_path = os.path.join(aggregator_dir, EXCEL_FILENAME)
    print(f"\nGenerating Excel report: {output_path}")
    
    generate_excel_report(df, output_path)
    
    print("Excel report generated successfully!")
    print(f"Report saved to: {output_path}")
    
    # Create first rank frequency chart
    print("\nGenerating first rank frequency chart...")
    first_rank_counts = create_first_rank_chart(df, aggregator_dir)
    
    # Create average accuracy chart
    print("\nGenerating average accuracy chart...")
    avg_accuracy = create_average_accuracy_chart(df, aggregator_dir)
    
    # Create ranking distribution chart
    print("\nGenerating ranking distribution chart...")
    ranking_distribution = create_ranking_distribution_chart(df, aggregator_dir)
    
    # Create architecture ranking Excel file
    print("\nGenerating architecture ranking Excel file...")
    architecture_rankings = create_architecture_ranking_excel(df, aggregator_dir)
    
    # Create Mean Reciprocal Rank chart
    print("\nGenerating Mean Reciprocal Rank chart...")
    mrr_results = create_mean_reciprocal_rank_chart(df, aggregator_dir)
    
    # Print some quick statistics
    print("\nQuick Statistics:")
    print(f"Best overall test accuracy: {df['Test_Accuracy'].max():.4f}")
    best_model = df.loc[df['Test_Accuracy'].idxmax()]
    print(f"Best model: {best_model['Activation_Function']} on {best_model['Architecture_Name']} (Architecture {best_model['Architecture_ID']})")
    print(f"Average test accuracy: {df['Test_Accuracy'].mean():.4f}")
    print(f"Standard deviation: {df['Test_Accuracy'].std():.4f}")

if __name__ == "__main__":
    main()