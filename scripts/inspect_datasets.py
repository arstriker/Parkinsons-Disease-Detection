import os
import glob
import pandas as pd

def build_metadata(dataset_dir="dataset", output_csv="data/metadata.csv"):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    records = []

    # 1. Parse Kaggle dataset
    kaggle_dir = os.path.join(dataset_dir, "kaggle")
    if os.path.exists(kaggle_dir):
        for root, dirs, files in os.walk(kaggle_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, dataset_dir)
                    path_parts = rel_path.lower().split(os.sep)
                    
                    # Class determination
                    if 'healthy' in path_parts or 'control' in path_parts:
                        label = 'healthy'
                    elif 'parkinson' in path_parts or 'patient' in path_parts or 'patients' in path_parts:
                        label = 'parkinson'
                    else:
                        continue
                    
                    # Modality determination
                    if 'spiral' in path_parts:
                        modality = 'spiral'
                    elif 'wave' in path_parts:
                        modality = 'wave'
                    elif 'meander' in path_parts:
                        modality = 'meander'
                    else:
                        modality = 'drawing'

                    # Split determination
                    if 'training' in path_parts or 'train' in path_parts:
                        split = 'train'
                    elif 'testing' in path_parts or 'test' in path_parts:
                        split = 'test'
                    else:
                        split = 'train'

                    records.append({
                        'filepath': full_path.replace('\\', '/'),
                        'label': label,
                        'source_dataset': 'kaggle',
                        'modality': modality,
                        'split': split
                    })

    # 2. Parse HandPD (Spiral & Meander)
    for handpd_folder, modality in [("Spiral_HandPD", "spiral"), ("Meander_HandPD", "meander")]:
        handpd_path = os.path.join(dataset_dir, handpd_folder)
        if os.path.exists(handpd_path):
            for root, dirs, files in os.walk(handpd_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, handpd_path).lower()
                        
                        if 'control' in rel_path:
                            label = 'healthy'
                        elif 'patient' in rel_path or 'patients' in rel_path:
                            label = 'parkinson'
                        else:
                            continue

                        records.append({
                            'filepath': full_path.replace('\\', '/'),
                            'label': label,
                            'source_dataset': 'handpd',
                            'modality': modality,
                            'split': 'train'
                        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Generated metadata CSV at '{output_csv}' with {len(df)} records.")
    if not df.empty:
        print("\nDistribution by Dataset & Label:")
        print(df.groupby(['source_dataset', 'modality', 'label']).size())
    return df

if __name__ == "__main__":
    build_metadata()
