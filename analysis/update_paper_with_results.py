#!/usr/bin/env python3
"""
Script to update the paper with real experimental results.
Reads the latest experiment results JSON and updates the LaTeX table.
"""

import json
import glob
import os
from datetime import datetime


def find_latest_results():
    """Find the most recent experiment results file."""
    results_files = glob.glob('results/combined/experiment_results_*.json')
    if not results_files:
        print("No experiment results found in results/combined/")
        return None

    # Sort by modification time, most recent first
    latest = max(results_files, key=os.path.getmtime)
    print(f"Found latest results: {latest}")
    return latest


def load_results(filepath):
    """Load and parse experiment results."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def format_improvement(improvement):
    """Format improvement percentage for LaTeX."""
    if improvement is None:
        return "N/A"
    return f"\\textbf{{{improvement:.1f}\\%}}"


def format_episodes(episodes):
    """Format episodes for LaTeX."""
    if episodes is None:
        return "N/A"
    return str(episodes)


def generate_table_rows(results):
    """Generate LaTeX table rows from results."""
    rows = []

    # GridWorld
    if 'GridWorld' in results:
        gw = results['GridWorld']
        baseline = format_episodes(gw.get('baseline_episodes'))
        qbound = format_episodes(gw.get('qbound_episodes'))
        improvement = format_improvement(gw.get('improvement_percent'))
        rows.append(f"GridWorld ($10 \\times 10$) & 80\\% success & {baseline} & {qbound} & {improvement} \\\\")

    # FrozenLake
    if 'FrozenLake' in results:
        fl = results['FrozenLake']
        baseline = format_episodes(fl.get('baseline_episodes'))
        qbound = format_episodes(fl.get('qbound_episodes'))
        improvement = format_improvement(fl.get('improvement_percent'))
        rows.append(f"FrozenLake ($4 \\times 4$) & 70\\% success & {baseline} & {qbound} & {improvement} \\\\")

    # CartPole
    if 'CartPole' in results:
        cp = results['CartPole']
        baseline = format_episodes(cp.get('baseline_episodes'))
        qbound = format_episodes(cp.get('qbound_episodes'))
        improvement = format_improvement(cp.get('improvement_percent'))
        rows.append(f"CartPole & 475 avg reward & {baseline} & {qbound} & {improvement} \\\\")

    return rows


def update_paper(results):
    """Update the main.tex file with real results."""
    paper_path = 'LatexDocs/main.tex'

    if not os.path.exists(paper_path):
        print(f"Paper not found at {paper_path}")
        return False

    # Read the paper
    with open(paper_path, 'r') as f:
        content = f.read()

    # Generate new table rows
    rows = generate_table_rows(results)
    table_content = "\n".join(rows)

    # Find and replace the table
    start_marker = "\\midrule\n"
    end_marker = "\\bottomrule\n\\end{tabular}"

    start_idx = content.find(start_marker, content.find("\\caption{Sample Efficiency Results"))
    end_idx = content.find(end_marker, start_idx)

    if start_idx == -1 or end_idx == -1:
        print("Could not find table markers in the paper")
        return False

    # Replace the table content
    new_content = (
        content[:start_idx + len(start_marker)] +
        table_content + "\n" +
        content[end_idx:]
    )

    # Write back to file
    backup_path = f'LatexDocs/main.tex.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    print(f"Creating backup at {backup_path}")
    with open(backup_path, 'w') as f:
        f.write(content)

    print(f"Updating {paper_path}")
    with open(paper_path, 'w') as f:
        f.write(new_content)

    print("Paper updated successfully!")
    return True


def print_summary(results):
    """Print a summary of the results."""
    print("\n" + "="*70)
    print("EXPERIMENTAL RESULTS SUMMARY")
    print("="*70)

    for env_name, data in results.items():
        print(f"\n{env_name}:")
        print(f"  Baseline:    {data.get('baseline_episodes', 'N/A')} episodes")
        print(f"  QBound:      {data.get('qbound_episodes', 'N/A')} episodes")

        improvement = data.get('improvement_percent')
        if improvement is not None:
            print(f"  Improvement: {improvement:.1f}%")
        else:
            print(f"  Improvement: N/A")

        print(f"  Total Reward (Baseline): {data.get('baseline_total_reward', 'N/A'):.1f}")
        print(f"  Total Reward (QBound):   {data.get('qbound_total_reward', 'N/A'):.1f}")


def main():
    """Main function."""
    print("QBound Paper Update Script")
    print("="*70)

    # Find latest results
    results_file = find_latest_results()
    if not results_file:
        print("\nERROR: No experiment results found!")
        print("Please run: python run_all_experiments.py")
        return

    # Load results
    results = load_results(results_file)

    # Print summary
    print_summary(results)

    # Update paper
    print("\n" + "="*70)
    if update_paper(results):
        print("\n✓ Paper successfully updated with real experimental results!")
        print("\nNext steps:")
        print("1. Review the updated table in LatexDocs/main.tex")
        print("2. Compile the LaTeX document to check formatting")
        print("3. Add any additional analysis or discussion based on results")
    else:
        print("\n✗ Failed to update paper")
        return

    print("="*70)


if __name__ == "__main__":
    main()
