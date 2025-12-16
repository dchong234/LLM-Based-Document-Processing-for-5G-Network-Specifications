"""
Visualization Script for Benchmark Results
Creates visualizations and HTML report from benchmark evaluation results.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib is not installed. Install with: pip install matplotlib")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly is not installed. Install with: pip install plotly")

try:
    import config
    OUTPUT_DIR = config.OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "checkpoints")


def load_benchmark_results(json_path: str) -> Dict[str, Any]:
    """
    Load benchmark results from JSON file.
    
    Args:
        json_path: Path to benchmark results JSON file
    
    Returns:
        Dictionary with benchmark results
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


def plot_accuracy_by_difficulty(results: Dict[str, Any], output_dir: Path, use_plotly: bool = False) -> str:
    """
    Create bar chart showing accuracy by difficulty level.
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plot
        use_plotly: Whether to use plotly instead of matplotlib (default: False)
    
    Returns:
        Path to saved plot file
    """
    perf_by_diff = results.get('performance_by_difficulty', {})
    
    if not perf_by_diff:
        print("Warning: No performance by difficulty data found")
        return None
    
    difficulties = ['Easy', 'Medium', 'Hard']
    accuracies = [perf_by_diff.get(d, {}).get('accuracy', 0.0) for d in difficulties]
    totals = [perf_by_diff.get(d, {}).get('total', 0) for d in difficulties]
    accurates = [perf_by_diff.get(d, {}).get('accurate', 0) for d in difficulties]
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Bar(
                x=difficulties,
                y=accuracies,
                text=[f'{a:.1%}' for a in accuracies],
                textposition='auto',
                marker=dict(
                    color=['#2ecc71', '#f39c12', '#e74c3c'],
                    line=dict(color='#000000', width=1)
                )
            )
        ])
        
        fig.update_layout(
            title='Accuracy by Difficulty Level',
            xaxis_title='Difficulty',
            yaxis_title='Accuracy',
            yaxis=dict(range=[0, 1.1], tickformat='.0%'),
            template='plotly_white',
            height=500,
            width=800
        )
        
        output_path = output_dir / 'accuracy_by_difficulty.png'
        fig.write_image(str(output_path))
        return str(output_path)
    
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        bars = ax.bar(difficulties, accuracies, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, acc, total, accurate) in enumerate(zip(bars, accuracies, totals, accurates)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{acc:.1%}\n({accurate}/{total})',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_xlabel('Difficulty Level', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy by Difficulty Level', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = output_dir / 'accuracy_by_difficulty.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    else:
        print("Error: Neither matplotlib nor plotly is available")
        return None


def plot_accuracy_by_category(results: Dict[str, Any], output_dir: Path, use_plotly: bool = False) -> str:
    """
    Create bar chart showing accuracy by category.
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plot
        use_plotly: Whether to use plotly instead of matplotlib (default: False)
    
    Returns:
        Path to saved plot file
    """
    perf_by_cat = results.get('performance_by_category', {})
    
    if not perf_by_cat:
        print("Warning: No performance by category data found")
        return None
    
    # Sort categories by accuracy
    categories = sorted(perf_by_cat.keys(), key=lambda x: perf_by_cat[x].get('accuracy', 0), reverse=True)
    accuracies = [perf_by_cat[cat].get('accuracy', 0.0) for cat in categories]
    totals = [perf_by_cat[cat].get('total', 0) for cat in categories]
    accurates = [perf_by_cat[cat].get('accurate', 0) for cat in categories]
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=accuracies,
                text=[f'{a:.1%}' for a in accuracies],
                textposition='auto',
                marker=dict(
                    color=accuracies,
                    colorscale='Viridis',
                    line=dict(color='#000000', width=1)
                )
            )
        ])
        
        fig.update_layout(
            title='Accuracy by Category',
            xaxis_title='Category',
            yaxis_title='Accuracy',
            yaxis=dict(range=[0, 1.1], tickformat='.0%'),
            template='plotly_white',
            height=600,
            width=1200,
            xaxis=dict(tickangle=-45)
        )
        
        output_path = output_dir / 'accuracy_by_category.png'
        fig.write_image(str(output_path))
        return str(output_path)
    
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars = ax.barh(categories, accuracies, edgecolor='black', linewidth=1)
        
        # Color bars based on accuracy
        for bar, acc in zip(bars, accuracies):
            if acc >= 0.8:
                bar.set_color('#2ecc71')  # Green
            elif acc >= 0.6:
                bar.set_color('#f39c12')  # Orange
            else:
                bar.set_color('#e74c3c')  # Red
        
        # Add value labels
        for i, (bar, acc, total, accurate) in enumerate(zip(bars, accuracies, totals, accurates)):
            width = bar.get_width()
            ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{acc:.1%} ({accurate}/{total})',
                   ha='left', va='center', fontsize=10)
        
        ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_ylabel('Category', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy by Category', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.1)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.0%}'.format(x)))
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        output_path = output_dir / 'accuracy_by_category.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    else:
        print("Error: Neither matplotlib nor plotly is available")
        return None


def plot_semantic_similarity_distribution(results: Dict[str, Any], output_dir: Path, use_plotly: bool = False) -> str:
    """
    Create histogram showing distribution of semantic similarity scores.
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plot
        use_plotly: Whether to use plotly instead of matplotlib (default: False)
    
    Returns:
        Path to saved plot file
    """
    detailed_results = results.get('detailed_results', [])
    
    if not detailed_results:
        print("Warning: No detailed results found")
        return None
    
    # Extract semantic similarity scores
    similarities = []
    for result in detailed_results:
        if result.get('metrics') and 'semantic_similarity' in result['metrics']:
            similarities.append(result['metrics']['semantic_similarity'])
    
    if not similarities:
        print("Warning: No semantic similarity scores found")
        return None
    
    accuracy_threshold = results.get('accuracy_threshold', 0.7)
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=similarities,
            nbinsx=20,
            marker=dict(
                color='#3498db',
                line=dict(color='#000000', width=1)
            ),
            name='Semantic Similarity'
        ))
        
        # Add threshold line
        fig.add_vline(
            x=accuracy_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({accuracy_threshold})",
            annotation_position="top"
        )
        
        fig.update_layout(
            title='Distribution of Semantic Similarity Scores',
            xaxis_title='Semantic Similarity Score',
            yaxis_title='Frequency',
            template='plotly_white',
            height=500,
            width=800
        )
        
        output_path = output_dir / 'semantic_similarity_distribution.png'
        fig.write_image(str(output_path))
        return str(output_path)
    
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n, bins, patches = ax.hist(similarities, bins=20, edgecolor='black', linewidth=1.5, alpha=0.7)
        
        # Color bars based on threshold
        for i, (patch, bin_left) in enumerate(zip(patches, bins[:-1])):
            if bin_left >= accuracy_threshold:
                patch.set_facecolor('#2ecc71')  # Green
            else:
                patch.set_facecolor('#e74c3c')  # Red
        
        # Add threshold line
        ax.axvline(accuracy_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Accuracy Threshold ({accuracy_threshold})')
        
        ax.set_xlabel('Semantic Similarity Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Distribution of Semantic Similarity Scores', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.0)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.legend()
        
        # Add statistics text
        mean_sim = sum(similarities) / len(similarities)
        ax.text(0.02, 0.98, f'Mean: {mean_sim:.3f}\nCount: {len(similarities)}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        output_path = output_dir / 'semantic_similarity_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    else:
        print("Error: Neither matplotlib nor plotly is available")
        return None


def plot_similarity_per_question(results: Dict[str, Any], output_dir: Path, use_plotly: bool = False) -> str:
    """
    Create line plot showing semantic similarity scores for each question.
    
    Args:
        results: Benchmark results dictionary
        output_dir: Directory to save plot
        use_plotly: Whether to use plotly instead of matplotlib (default: False)
    
    Returns:
        Path to saved plot file
    """
    detailed_results = results.get('detailed_results', [])
    
    if not detailed_results:
        print("Warning: No detailed results found")
        return None
    
    # Extract data
    question_ids = []
    similarities = []
    difficulties = []
    is_accurate = []
    
    for result in detailed_results:
        if result.get('metrics') and 'semantic_similarity' in result['metrics']:
            question_ids.append(result.get('question_id', 'Unknown'))
            similarities.append(result['metrics']['semantic_similarity'])
            difficulties.append(result.get('difficulty', 'Unknown'))
            is_accurate.append(result.get('is_accurate', False))
    
    if not similarities:
        print("Warning: No semantic similarity scores found")
        return None
    
    accuracy_threshold = results.get('accuracy_threshold', 0.7)
    
    if use_plotly and PLOTLY_AVAILABLE:
        fig = go.Figure()
        
        # Color points based on accuracy
        colors = ['#2ecc71' if acc else '#e74c3c' for acc in is_accurate]
        
        fig.add_trace(go.Scatter(
            x=list(range(len(question_ids))),
            y=similarities,
            mode='lines+markers',
            marker=dict(
                size=8,
                color=colors,
                line=dict(color='black', width=1)
            ),
            line=dict(color='#3498db', width=2),
            name='Semantic Similarity',
            text=question_ids,
            hovertemplate='<b>%{text}</b><br>Similarity: %{y:.3f}<extra></extra>'
        ))
        
        # Add threshold line
        fig.add_hline(
            y=accuracy_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold ({accuracy_threshold})",
            annotation_position="right"
        )
        
        fig.update_layout(
            title='Semantic Similarity Scores per Question',
            xaxis_title='Question Index',
            yaxis_title='Semantic Similarity Score',
            template='plotly_white',
            height=500,
            width=1200,
            hovermode='x unified'
        )
        
        output_path = output_dir / 'similarity_per_question.png'
        fig.write_image(str(output_path))
        return str(output_path)
    
    elif MATPLOTLIB_AVAILABLE:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x_positions = list(range(len(question_ids)))
        
        # Plot line
        ax.plot(x_positions, similarities, color='#3498db', linewidth=2, alpha=0.7, label='Semantic Similarity')
        
        # Plot points with colors based on accuracy
        for i, (x, y, acc) in enumerate(zip(x_positions, similarities, is_accurate)):
            color = '#2ecc71' if acc else '#e74c3c'
            ax.scatter(x, y, color=color, s=100, edgecolors='black', linewidths=1, zorder=5)
        
        # Add threshold line
        ax.axhline(accuracy_threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Accuracy Threshold ({accuracy_threshold})')
        
        ax.set_xlabel('Question Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Semantic Similarity Score', fontsize=12, fontweight='bold')
        ax.set_title('Semantic Similarity Scores per Question', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 1.0)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend()
        
        # Rotate x-axis labels if many questions
        if len(question_ids) > 10:
            ax.set_xticks(x_positions[::max(1, len(question_ids)//10)])
            ax.set_xticklabels([question_ids[i] for i in x_positions[::max(1, len(question_ids)//10)]],
                              rotation=45, ha='right')
        else:
            ax.set_xticks(x_positions)
            ax.set_xticklabels(question_ids, rotation=45, ha='right')
        
        plt.tight_layout()
        output_path = output_dir / 'similarity_per_question.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return str(output_path)
    
    else:
        print("Error: Neither matplotlib nor plotly is available")
        return None


def create_html_report(results: Dict[str, Any], plot_paths: Dict[str, str], output_path: Path, output_dir: Path):
    """
    Create HTML report with all visualizations and summary statistics.
    
    Args:
        results: Benchmark results dictionary
        plot_paths: Dictionary mapping plot names to file paths
        output_path: Path to save HTML report
        output_dir: Directory where plots are saved (for relative paths)
    """
    timestamp = results.get('timestamp', 'Unknown')
    model_path = results.get('model_path', 'Unknown')
    base_model = results.get('base_model', 'Unknown')
    total_questions = results.get('total_questions', 0)
    valid_results = results.get('valid_results', 0)
    overall_accuracy = results.get('overall_accuracy', 0.0)
    avg_metrics = results.get('average_metrics', {})
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Benchmark Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
            padding: 10px;
            background-color: white;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .plot-container {{
            margin: 30px 0;
            text-align: center;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #7f8c8d;
            font-size: 12px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Benchmark Evaluation Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p><strong>Timestamp:</strong> {timestamp}</p>
            <p><strong>Model:</strong> {base_model}</p>
            <p><strong>Model Path:</strong> {model_path}</p>
            <p><strong>Total Questions:</strong> {total_questions}</p>
            <p><strong>Valid Results:</strong> {valid_results}</p>
            
            <div style="margin-top: 20px;">
                <div class="metric">
                    <div class="metric-label">Overall Accuracy</div>
                    <div class="metric-value">{overall_accuracy:.1%}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Semantic Similarity</div>
                    <div class="metric-value">{avg_metrics.get('semantic_similarity', 0):.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Keyword Match</div>
                    <div class="metric-value">{avg_metrics.get('keyword_match', 0):.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">BLEU Score</div>
                    <div class="metric-value">{avg_metrics.get('bleu_score', 0):.3f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">F1 Score</div>
                    <div class="metric-value">{avg_metrics.get('f1_score', 0):.3f}</div>
                </div>
            </div>
        </div>
"""
    
    # Add visualizations (use relative paths)
    if plot_paths.get('accuracy_by_difficulty'):
        plot_name = plot_paths['accuracy_by_difficulty']
        html_content += f"""
        <div class="plot-container">
            <h2>Accuracy by Difficulty</h2>
            <img src="{plot_name}" alt="Accuracy by Difficulty">
        </div>
"""
    
    if plot_paths.get('accuracy_by_category'):
        plot_name = plot_paths['accuracy_by_category']
        html_content += f"""
        <div class="plot-container">
            <h2>Accuracy by Category</h2>
            <img src="{plot_name}" alt="Accuracy by Category">
        </div>
"""
    
    if plot_paths.get('semantic_similarity_distribution'):
        plot_name = plot_paths['semantic_similarity_distribution']
        html_content += f"""
        <div class="plot-container">
            <h2>Semantic Similarity Distribution</h2>
            <img src="{plot_name}" alt="Semantic Similarity Distribution">
        </div>
"""
    
    if plot_paths.get('similarity_per_question'):
        plot_name = plot_paths['similarity_per_question']
        html_content += f"""
        <div class="plot-container">
            <h2>Semantic Similarity per Question</h2>
            <img src="{plot_name}" alt="Semantic Similarity per Question">
        </div>
"""
    
    # Add performance tables
    perf_by_diff = results.get('performance_by_difficulty', {})
    if perf_by_diff:
        html_content += """
        <h2>Performance by Difficulty</h2>
        <table>
            <tr>
                <th>Difficulty</th>
                <th>Total</th>
                <th>Accurate</th>
                <th>Accuracy</th>
                <th>Semantic Similarity</th>
                <th>Keyword Match</th>
                <th>BLEU Score</th>
                <th>F1 Score</th>
            </tr>
"""
        for difficulty in ['Easy', 'Medium', 'Hard']:
            if difficulty in perf_by_diff:
                stats = perf_by_diff[difficulty]
                metrics = stats.get('average_metrics', {})
                html_content += f"""
            <tr>
                <td>{difficulty}</td>
                <td>{stats.get('total', 0)}</td>
                <td>{stats.get('accurate', 0)}</td>
                <td>{stats.get('accuracy', 0):.1%}</td>
                <td>{metrics.get('semantic_similarity', 0):.3f}</td>
                <td>{metrics.get('keyword_match', 0):.3f}</td>
                <td>{metrics.get('bleu_score', 0):.3f}</td>
                <td>{metrics.get('f1_score', 0):.3f}</td>
            </tr>
"""
        html_content += """
        </table>
"""
    
    html_content += """
        <div class="footer">
            <p>Generated by visualize_results.py</p>
            <p>Report generated on: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
        </div>
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ HTML report saved to: {output_path}")


def visualize_benchmark_results(
    json_path: str,
    output_dir: Optional[str] = None,
    use_plotly: bool = False
) -> Dict[str, str]:
    """
    Create all visualizations from benchmark results.
    
    Args:
        json_path: Path to benchmark results JSON file
        output_dir: Directory to save plots (default: same as JSON file directory)
        use_plotly: Whether to use plotly instead of matplotlib (default: False)
    
    Returns:
        Dictionary mapping plot names to file paths
    """
    if not MATPLOTLIB_AVAILABLE and not PLOTLY_AVAILABLE:
        raise ImportError("Either matplotlib or plotly is required. Install with: pip install matplotlib or pip install plotly")
    
    # Load results
    print(f"Loading benchmark results from: {json_path}")
    results = load_benchmark_results(json_path)
    
    # Determine output directory
    if output_dir is None:
        output_dir = Path(json_path).parent
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving plots to: {output_dir}\n")
    
    # Create visualizations
    plot_paths = {}
    
    print("Creating visualizations...")
    
    # Accuracy by difficulty
    print("  1. Accuracy by difficulty...")
    path = plot_accuracy_by_difficulty(results, output_dir, use_plotly)
    if path:
        plot_paths['accuracy_by_difficulty'] = os.path.basename(path)
        print(f"     ✓ Saved: {path}")
    
    # Accuracy by category
    print("  2. Accuracy by category...")
    path = plot_accuracy_by_category(results, output_dir, use_plotly)
    if path:
        plot_paths['accuracy_by_category'] = os.path.basename(path)
        print(f"     ✓ Saved: {path}")
    
    # Semantic similarity distribution
    print("  3. Semantic similarity distribution...")
    path = plot_semantic_similarity_distribution(results, output_dir, use_plotly)
    if path:
        plot_paths['semantic_similarity_distribution'] = os.path.basename(path)
        print(f"     ✓ Saved: {path}")
    
    # Similarity per question
    print("  4. Similarity per question...")
    path = plot_similarity_per_question(results, output_dir, use_plotly)
    if path:
        plot_paths['similarity_per_question'] = os.path.basename(path)
        print(f"     ✓ Saved: {path}")
    
    # Create HTML report
    print("\nCreating HTML report...")
    html_path = output_dir / 'benchmark_report.html'
    create_html_report(results, plot_paths, html_path, output_dir)
    
    print(f"\n✓ All visualizations created successfully!")
    print(f"  Output directory: {output_dir}")
    print(f"  HTML report: {html_path}")
    
    return plot_paths


def main():
    """Main function to run visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create visualizations from benchmark results"
    )
    parser.add_argument(
        '--json',
        type=str,
        default=None,
        help='Path to benchmark results JSON file (default: OUTPUT_DIR/benchmark_results.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save plots (default: same as JSON file directory)'
    )
    parser.add_argument(
        '--use-plotly',
        action='store_true',
        default=False,
        help='Use plotly instead of matplotlib (default: False)'
    )
    
    args = parser.parse_args()
    
    # Determine JSON path
    if args.json:
        json_path = args.json
    else:
        json_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        print("Please run run_benchmark.py first to generate benchmark results.")
        return 1
    
    try:
        # Create visualizations
        plot_paths = visualize_benchmark_results(
            json_path=json_path,
            output_dir=args.output_dir,
            use_plotly=args.use_plotly
        )
        
        print("\n✓ Visualization completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

