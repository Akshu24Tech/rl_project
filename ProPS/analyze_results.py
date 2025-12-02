"""
Analyze and compare results from multiple ProPS runs
"""

import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_results(results_dir="PrPoS/results"):
    """Load all result JSON files"""
    pattern = os.path.join(results_dir, "results_*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No result files found in {results_dir}")
        return []
    
    all_results = []
    for file in files:
        with open(file, 'r') as f:
            data = json.load(f)
            all_results.append({
                'file': os.path.basename(file),
                'data': data
            })
    
    return all_results

def aggregate_by_model(all_results):
    """Aggregate results by model name"""
    model_data = {}
    
    for result in all_results:
        for run in result['data']:
            model = run['model']
            if model not in model_data:
                model_data[model] = {
                    'final_rewards': [],
                    'max_rewards': [],
                    'avg_rewards': [],
                    'times': [],
                    'all_rewards': []
                }
            
            model_data[model]['final_rewards'].append(run['final_reward'])
            model_data[model]['max_rewards'].append(run['max_reward'])
            model_data[model]['avg_rewards'].append(run['avg_reward'])
            model_data[model]['times'].append(run['time_seconds'])
            model_data[model]['all_rewards'].append(run['rewards'])
    
    return model_data

def print_summary(model_data):
    """Print statistical summary"""
    print("\n" + "="*70)
    print("STATISTICAL SUMMARY")
    print("="*70)
    
    for model, data in model_data.items():
        print(f"\n{model}:")
        print(f"  Runs: {len(data['final_rewards'])}")
        
        if len(data['final_rewards']) > 0:
            print(f"  Final Reward:")
            print(f"    Mean: {np.mean(data['final_rewards']):.2f}")
            print(f"    Std:  {np.std(data['final_rewards']):.2f}")
            print(f"    Min:  {np.min(data['final_rewards']):.2f}")
            print(f"    Max:  {np.max(data['final_rewards']):.2f}")
            
            print(f"  Max Reward:")
            print(f"    Mean: {np.mean(data['max_rewards']):.2f}")
            print(f"    Std:  {np.std(data['max_rewards']):.2f}")
            
            print(f"  Avg Time: {np.mean(data['times']):.2f}s")

def plot_comparison(model_data, save_path="PrPoS/results"):
    """Create comprehensive comparison plots"""
    
    # Figure 1: Learning curves with confidence intervals
    plt.figure(figsize=(14, 10))
    
    # Subplot 1: Learning curves
    plt.subplot(2, 2, 1)
    for model, data in model_data.items():
        if len(data['all_rewards']) > 0:
            # Calculate mean and std across runs
            max_len = max(len(r) for r in data['all_rewards'])
            padded = [r + [r[-1]]*(max_len-len(r)) for r in data['all_rewards']]
            rewards_array = np.array(padded)
            
            mean_rewards = np.mean(rewards_array, axis=0)
            std_rewards = np.std(rewards_array, axis=0)
            episodes = range(len(mean_rewards))
            
            plt.plot(episodes, mean_rewards, label=model, linewidth=2)
            plt.fill_between(episodes, 
                           mean_rewards - std_rewards,
                           mean_rewards + std_rewards,
                           alpha=0.2)
    
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Learning Curves (Mean ± Std)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Final reward comparison
    plt.subplot(2, 2, 2)
    models = list(model_data.keys())
    final_means = [np.mean(model_data[m]['final_rewards']) for m in models]
    final_stds = [np.std(model_data[m]['final_rewards']) for m in models]
    
    x_pos = range(len(models))
    plt.bar(x_pos, final_means, yerr=final_stds, capsize=5, alpha=0.7)
    plt.xticks(x_pos, [m.split('-')[-1] for m in models], rotation=45)
    plt.ylabel("Final Reward")
    plt.title("Final Reward Comparison")
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 3: Max reward comparison
    plt.subplot(2, 2, 3)
    max_means = [np.mean(model_data[m]['max_rewards']) for m in models]
    max_stds = [np.std(model_data[m]['max_rewards']) for m in models]
    
    plt.bar(x_pos, max_means, yerr=max_stds, capsize=5, alpha=0.7, color='orange')
    plt.xticks(x_pos, [m.split('-')[-1] for m in models], rotation=45)
    plt.ylabel("Max Reward")
    plt.title("Maximum Reward Achieved")
    plt.grid(True, alpha=0.3, axis='y')
    
    # Subplot 4: Execution time
    plt.subplot(2, 2, 4)
    time_means = [np.mean(model_data[m]['times']) for m in models]
    time_stds = [np.std(model_data[m]['times']) for m in models]
    
    plt.bar(x_pos, time_means, yerr=time_stds, capsize=5, alpha=0.7, color='green')
    plt.xticks(x_pos, [m.split('-')[-1] for m in models], rotation=45)
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time")
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    filename = os.path.join(save_path, f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nAnalysis plot saved: {filename}")
    plt.show()

def find_best_model(model_data):
    """Determine which model performed best"""
    print("\n" + "="*70)
    print("BEST MODEL ANALYSIS")
    print("="*70)
    
    rankings = {}
    
    for model in model_data.keys():
        rankings[model] = {
            'final_reward': np.mean(model_data[model]['final_rewards']),
            'max_reward': np.mean(model_data[model]['max_rewards']),
            'consistency': -np.std(model_data[model]['final_rewards']),  # Lower std is better
            'speed': -np.mean(model_data[model]['times'])  # Faster is better
        }
    
    # Rank by each metric
    for metric in ['final_reward', 'max_reward', 'consistency', 'speed']:
        sorted_models = sorted(rankings.items(), key=lambda x: x[1][metric], reverse=True)
        print(f"\nBest by {metric.replace('_', ' ').title()}:")
        for i, (model, scores) in enumerate(sorted_models[:3], 1):
            value = scores[metric]
            if metric in ['consistency', 'speed']:
                value = -value  # Convert back to original scale
            print(f"  {i}. {model}: {value:.2f}")
    
    # Overall winner (by final reward)
    best_model = max(rankings.items(), key=lambda x: x[1]['final_reward'])[0]
    print(f"\n🏆 Overall Winner: {best_model}")
    print(f"   Final Reward: {rankings[best_model]['final_reward']:.2f}")

def export_latex_table(model_data, save_path="PrPoS/results"):
    """Export results as LaTeX table"""
    filename = os.path.join(save_path, "results_table.tex")
    
    with open(filename, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Model & Final Reward & Max Reward & Avg Reward & Time (s) \\\\\n")
        f.write("\\hline\n")
        
        for model, data in model_data.items():
            final = f"{np.mean(data['final_rewards']):.2f} ± {np.std(data['final_rewards']):.2f}"
            max_r = f"{np.mean(data['max_rewards']):.2f} ± {np.std(data['max_rewards']):.2f}"
            avg = f"{np.mean(data['avg_rewards']):.2f} ± {np.std(data['avg_rewards']):.2f}"
            time = f"{np.mean(data['times']):.1f}"
            
            f.write(f"{model} & {final} & {max_r} & {avg} & {time} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{ProPS Performance Comparison}\n")
        f.write("\\label{tab:props_comparison}\n")
        f.write("\\end{table}\n")
    
    print(f"\nLaTeX table saved: {filename}")

def main():
    print("ProPS Results Analysis")
    print("="*70)
    
    # Load results
    all_results = load_results()
    
    if not all_results:
        print("\nNo results found. Run experiments first:")
        print("  python props_gemini_comparison.py")
        return
    
    print(f"\nFound {len(all_results)} result file(s)")
    
    # Aggregate by model
    model_data = aggregate_by_model(all_results)
    
    if not model_data:
        print("No valid data found in result files")
        return
    
    # Print summary
    print_summary(model_data)
    
    # Find best model
    find_best_model(model_data)
    
    # Create plots
    print("\nGenerating plots...")
    plot_comparison(model_data)
    
    # Export LaTeX table
    export_latex_table(model_data)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

if __name__ == "__main__":
    main()
