import pandas as pd
import numpy as np
import glob
import os
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa

BASE_VALIDATION_DIR = "./human_validation/" 
GT_DIR = "./ground_truth/" 

def load_all_ground_truths(gt_dir):
    gt_files = glob.glob(os.path.join(gt_dir, "**/*ground_truth*.csv"), recursive=True)
    if not gt_files:
        raise ValueError(f"not found any ground truth csv files in {gt_dir} !")
    
    gt_list = [pd.read_csv(f) for f in gt_files]
    master_gt = pd.concat(gt_list, ignore_index=True)
    master_gt = master_gt.drop_duplicates(subset=['blind_id'])
    
    def extract_model(path):
        path_lower = str(path).lower()
        if 'cogx-2b' in path_lower: return 'CogX-2b'
        elif 'cogx-5b' in path_lower: return 'CogX-5b'
        elif 'wan2.2-5b' in path_lower or 'wan-5b' in path_lower: return 'Wan2.2-5b'
        else: return 'Unknown'
            
    def extract_category(path):
        path_lower = str(path).lower()
        path_str = str(path)
        if '/ID/' in path_str or 'identity' in path_lower: return 'Identity'
        elif 'nudity' in path_lower: return 'Nudity'
        elif 'object' in path_lower: return 'Objects'
        else: return 'Unknown'

    path_col = 'original_file' if 'original_file' in master_gt.columns else 'original_path'
    
    if path_col in master_gt.columns:
        master_gt['base_model'] = master_gt[path_col].apply(extract_model)
        master_gt['category'] = master_gt[path_col].apply(extract_category)
    else:
        print("⚠️ Error: Ground Truth not contain 'original_file' or 'original_path' column, can not parse model and category!")
        
    return master_gt

def load_and_merge_human_scores(base_dir):
    """
    Merge all human scores from P1 to P10 folders
    """
    merged_df = None
    for i in range(1, 11):
        rater_folder = os.path.join(base_dir, f"P{i}")
        if not os.path.exists(rater_folder): continue
            
        csv_files = glob.glob(os.path.join(rater_folder, "*.csv"))
        if not csv_files: continue
            
        rater_all_tasks = []
        for file in csv_files:
            try:
                df = pd.read_csv(file)
                temp_df = df.iloc[:, [0, 2]].copy()
                temp_df.columns = ['Video ID', f'Rater_{i}']
                temp_df[f'Rater_{i}'] = pd.to_numeric(temp_df[f'Rater_{i}'], errors='coerce')
                rater_all_tasks.append(temp_df)
            except Exception as e:
                continue
            
        if not rater_all_tasks: continue
        rater_df = pd.concat(rater_all_tasks, ignore_index=True)
        
        if merged_df is None: merged_df = rater_df
        else: merged_df = pd.merge(merged_df, rater_df, on='Video ID', how='outer')
            
    final_df = merged_df.dropna()
    print(f"✅ Success merge human scores! total {len(final_df)} videos been evaluated by all raters.")
    return final_df

def calculate_fleiss_kappa(merged_df):
    rater_cols = [col for col in merged_df.columns if col.startswith('Rater_')]
    ratings_matrix = merged_df[rater_cols].values
    ratings_matrix = np.clip(ratings_matrix, 0, 2).astype(int)
    agg_ratings, _ = aggregate_raters(ratings_matrix)
    return fleiss_kappa(agg_ratings, method='fleiss')

def calculate_method_averages(merged_df, master_gt_df):
    rater_cols = [col for col in merged_df.columns if col.startswith('Rater_')]
    merged_df['Human_Mean_Score'] = merged_df[rater_cols].mean(axis=1)
    
    eval_df = pd.merge(merged_df[['Video ID', 'Human_Mean_Score']], 
                       master_gt_df, left_on='Video ID', right_on='blind_id')
    
    summary_df = eval_df.groupby('method_state')['Human_Mean_Score'].agg(['mean', 'std', 'count']).reset_index()
    summary_df = summary_df.rename(columns={'mean': 'Average_Score', 'std': 'Std_Dev', 'count': 'Sample_Size'})
    return summary_df.sort_values(by='Average_Score', ascending=False)

if __name__ == "__main__":
    print("="*80)
    print("🚀 Start Human Evaluation (Category -> Base Model)")
    print("="*80)
    
    try:
        master_gt = load_all_ground_truths(GT_DIR)
        human_df = load_and_merge_human_scores(BASE_VALIDATION_DIR)
        
        target_categories = ['Objects', 'Nudity', 'Identity']
        target_models = ['CogX-2b', 'CogX-5b', 'Wan2.2-5b']
        
        for cat in target_categories:
            print(f"\n\n{'='*80}")
            print(f"🌟🌟🌟 Analysis for category: {cat.upper()} 🌟🌟🌟")
            print(f"{'='*80}")
            
            cat_gt = master_gt[master_gt['category'] == cat]
            if cat_gt.empty:
                print(f"⚠️ Skipped {cat} category: No corresponding records found in Ground Truth.")
                continue
            
            for model in target_models:
                print(f"\n--- For model: {model} ---")
                
                sub_gt = cat_gt[cat_gt['base_model'] == model]
                if sub_gt.empty:
                    print(f"  No evaluation data for {model} in {cat} category.")
                    continue
                    
                valid_ids = sub_gt['blind_id'].tolist()
                sub_human_df = human_df[human_df['Video ID'].isin(valid_ids)].copy()
                
                if sub_human_df.empty:
                    print(f"  No valid human scores found for {model} in {cat} category.")
                    continue
                    
                print(f"  ✅ Success load {len(sub_human_df)} human scores for {model} in {cat} category.")
                
                kappa = calculate_fleiss_kappa(sub_human_df)
                print(f"  📊 Fleiss' Kappa: {kappa:.4f}")
                
                method_summary = calculate_method_averages(sub_human_df, sub_gt)
                print("  📈 Different methods human perception score summary (0=Erased, 2=Clearly Present):")
                print(method_summary.to_string(index=False, float_format="%.3f"))
                
    except Exception as e:
        print(f"Error: {e}")