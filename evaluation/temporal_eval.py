import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.font_manager as fm
import os, argparse


font_path = "./Times.ttf"
fm.fontManager.addfont(font_path)
custom_font = fm.FontProperties(fname=font_path, size=15)
font_name = custom_font.get_name()

# ==========================================
# Drawing Style Configuration
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [font_name],
    "pdf.fonttype": 42,               
    "ps.fonttype": 42,            
    "axes.linewidth": 1.5,
    "axes.labelsize": 16,          
    "axes.titlesize": 18,             
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "grid.alpha": 0.3,
    "figure.dpi": 600                
})

def process_single_csv(file_path, target='tench', n_f=17):
    """Core processing logic: read -> clean -> frame alignment mean"""
    try:
        df = pd.read_csv(file_path, encoding="latin-1")
        cat_col = df.columns[6]
        score_col = df.columns[8]
        clean_scores = df.apply(
            lambda x: x[score_col] if str(x[cat_col]).strip().lower() == target.lower() else 0, 
            axis=1
        )
        df['temp_scores'] = clean_scores
        df['frame_id'] = df.index % n_f
        return df.groupby('frame_id')['temp_scores'].mean()
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None

def process_single_csv_nudity(file_path, n_f=17):
    try:
        df = pd.read_csv(file_path, encoding="latin-1")
        df['temp_scores'] = df['Unsafe_rate']
        df['frame_id'] = df.index % n_f
        return df.groupby('frame_id')['temp_scores'].mean()
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None

def process_single_csv_identity(file_path, n_f=17):
    try:
        df = pd.read_csv(file_path, encoding="latin-1")
        df['temp_scores'] = df['similarity']
        df['frame_id'] = df.index % n_f
        return df.groupby('frame_id')['temp_scores'].mean()
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None

def get_style(label):
    label_lower = label.lower()
    if 'origin' in label_lower:
        return style_mapping['Origin']
    elif 'probe' in label_lower or 'ci' in label_lower:
        return style_mapping['PROBE']
    else:
        return style_mapping['Erased'] 

def plot_subplot(ax, file_map, type_input, n_frames, target_word, title, show_legend=False):
    print(f"\n{'='*50}")
    print(f"Statistics Analysis - {title}")
    print(f"{'='*50}")
    
    for label, path in file_map.items():
        if path.exists():
            if type_input == 'nudity':
                means = process_single_csv_nudity(path, n_f=n_frames)
            else:
                means = process_single_csv(path, target=target_word, n_f=n_frames)
                
            if means is not None:
                style = get_style(label)
                ax.plot(x_frames, means, label=label, 
                        linewidth=2.5, markersize=8,
                        markerfacecolor='white', markeredgewidth=1.5,
                        **style)
                
                arr = means.to_numpy()
                mean_val = np.mean(arr)      
                var_val = np.var(arr)          
                sec_moment = np.mean(arr ** 2)   
                
                print(f"[{label.ljust(15)}] | Mean: {mean_val:.4f} | Var: {var_val:.4f} | 2nd Moment: {sec_moment:.4f}")
                
        else:
            print(f"Skipping: {path.name} (File not found)")

    ax.set_title(title, pad=15, fontproperties=custom_font)
    ax.set_xlabel('Frame Sequence', fontweight='bold', fontproperties=custom_font)
    ax.set_ylabel('Mean Detection Score', fontproperties=custom_font) 
    
    ax.set_xticks(x_frames)
    ax.set_ylim(-0.05, 0.40)
    ax.grid(True, linestyle=':', alpha=0.5)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_legend:
        ax.legend(frameon=True, loc='upper right', edgecolor='black', fancybox=False, fontsize=12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='temporal_eval', description='Evaluate temporal detection performance')
    parser.add_argument('--base_path', type=str, default='./objects/tench', help='Base path for input files')
    parser.add_argument('--type_input', type=str, default='objects', help='Type of evaluation: objects, nudity, identity')
    parser.add_argument('--target_word', type=str, default='tench', help='Target word for object detection')
    parser.add_argument('--n_frames', type=int, default=17, help='Number of frames in each sequence')
    args = parser.parse_args()
    base_path = Path(args.base_path)

    file_map_np = {
        'Origin': base_path / 'cogx2b_top5_tench_origin.csv',
        'NegPrompt': base_path / 'tench_negprompt.csv', 
        'PROBE': base_path / 'tench_negprompt_CI.csv',
    }

    file_map_safree = {
        'Origin': base_path / 'cogx2b_top5_tench_origin.csv',
        'SAFREE': base_path / 'tench_safree.csv',
        'PROBE': base_path / 'tench_safree_CI.csv'
    }

    file_map_t2v = {
        'Origin': base_path / 'cogx2b_top5_tench_origin.csv',
        'T2VUnlearning': base_path / 'tench_T2VUnlearning.csv',
        'PROBE': base_path / 'tench_T2VUnlearning_CI.csv'
    }

    style_mapping = {
        'Origin': {'color': '#009E73', 'marker': 'o', 'ls': '-'},
        'Erased': {'color': '#1F3A5F', 'marker': 's', 'ls': '--'},
        'PROBE':  {'color': '#9E3A2B', 'marker': '^', 'ls': '-.'}
    }


    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    x_frames = range(1, args.n_frames + 1)


    plot_subplot(axes[0], file_map_np, args.type_input, args.n_frames, args.target_word, "(a) Nudity NegPrompt Erasure", show_legend=True)
    plot_subplot(axes[1], file_map_safree, args.type_input, args.n_frames, args.target_word, "(b) Nudity SAFREE Erasure", show_legend=True)
    plot_subplot(axes[2], file_map_t2v, args.type_input, args.n_frames, args.target_word, "(c) Nudity T2VUnlearning Erasure", show_legend=True)

    plt.tight_layout()

    # ======================================================================
    save_pdf = base_path / f'eval_{args.target_word}_comparison.pdf'
    plt.savefig(save_pdf, format='pdf', dpi=600, bbox_inches='tight')
    # ======================================================================
    save_png = base_path / f'eval_{args.target_word}_comparison.png'
    plt.savefig(save_png, format='png', dpi=600, bbox_inches='tight')

    print(f"\nProcessing complete. High-res PDF saved to: {save_pdf}")