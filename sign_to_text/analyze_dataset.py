"""Dataset Analysis - Sequence Length vs Target Length"""
import numpy as np
import pandas as pd
from pathlib import Path

print('='*60)
print('Dataset Analysis - Sequence Length vs Target Length')
print('='*60)

# Load CSV
csv_path = Path('../data/iSign_v1.1.csv')
data_dir = Path('../extracted_landmarks_v2')

df = pd.read_csv(csv_path)
print(f'Total entries in CSV: {len(df)}')

# Analyze all available .npy files
stats = []
missing = 0
corrupt = 0

for idx, row in df.iterrows():
    uid = str(row['uid'])
    npy_path = data_dir / f'{uid}.npy'
    
    if not npy_path.exists():
        missing += 1
        continue
    
    try:
        landmarks = np.load(npy_path)
        if not np.isfinite(landmarks).all():
            corrupt += 1
            continue
        
        frames = len(landmarks)
        label = str(row['text']).lower().strip()
        target_len = len(label)
        enc_len_after_subsample = (frames + 1) // 2  # subsample_factor=2
        ctc_valid = enc_len_after_subsample >= target_len
        
        stats.append({
            'uid': uid,
            'frames': frames,
            'target_len': target_len,
            'enc_len': enc_len_after_subsample,
            'ctc_valid': ctc_valid,
            'label': label[:30]  # First 30 chars
        })
    except Exception as e:
        corrupt += 1

print(f'Missing .npy files: {missing}')
print(f'Corrupt .npy files: {corrupt}')
print(f'Valid samples: {len(stats)}')

# Convert to DataFrame for analysis
df_stats = pd.DataFrame(stats)

print(f'\n' + '='*60)
print('Frame Statistics')
print('='*60)
print(f'Min frames: {df_stats["frames"].min()}')
print(f'Max frames: {df_stats["frames"].max()}')
print(f'Mean frames: {df_stats["frames"].mean():.1f}')
print(f'Median frames: {df_stats["frames"].median():.1f}')

print(f'\n' + '='*60)
print('Target Length Statistics')
print('='*60)
print(f'Min target len: {df_stats["target_len"].min()}')
print(f'Max target len: {df_stats["target_len"].max()}')
print(f'Mean target len: {df_stats["target_len"].mean():.1f}')

print(f'\n' + '='*60)
print('CTC Validity (enc_len >= target_len)')
print('='*60)
valid_count = df_stats['ctc_valid'].sum()
invalid_count = len(df_stats) - valid_count
print(f'Valid for CTC: {valid_count} ({100*valid_count/len(df_stats):.1f}%)')
print(f'Invalid for CTC: {invalid_count} ({100*invalid_count/len(df_stats):.1f}%)')

print(f'\n' + '='*60)
print('Samples with Shortest Sequences (Bottom 20)')
print('='*60)
shortest = df_stats.nsmallest(20, 'frames')
for _, row in shortest.iterrows():
    status = 'OK' if row['ctc_valid'] else 'FAIL'
    print(f'  {row["frames"]:3d} frames -> {row["enc_len"]:3d} enc | target: {row["target_len"]:2d} chars | [{status}] | {row["label"]}')

print(f'\n' + '='*60)
print('Frame Distribution (Histogram)')
print('='*60)
bins = [0, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 10000]
hist, _ = np.histogram(df_stats['frames'], bins=bins)
for i in range(len(bins)-1):
    pct = 100 * hist[i] / len(df_stats)
    bar = '#' * int(pct/2)
    print(f'  {bins[i]:4d}-{bins[i+1]:4d}: {hist[i]:6d} ({pct:5.1f}%) {bar}')

print(f'\n' + '='*60)
print('CTC Failure Analysis')
print('='*60)
failures = df_stats[~df_stats['ctc_valid']]
if len(failures) > 0:
    print(f'Average frames in failed samples: {failures["frames"].mean():.1f}')
    print(f'Average target len in failed samples: {failures["target_len"].mean():.1f}')
    print(f'Most common failure patterns (frames -> target):')
    failure_patterns = failures.groupby(['frames', 'target_len']).size().nlargest(10)
    for (frames, target), count in failure_patterns.items():
        enc = (frames + 1) // 2
        deficit = target - enc
        print(f'    {frames} frames -> {enc} enc, target {target} chars (deficit: {deficit}) - {count} samples')
else:
    print('No CTC failures detected!')

print(f'\n' + '='*60)
print('Recommendation')
print('='*60)
# Calculate what percentage would be valid with different min_frames
print('If we filter samples where frames >= 2*target_len:')
df_stats['would_pass'] = df_stats['frames'] >= 2 * df_stats['target_len']
would_pass = df_stats['would_pass'].sum()
print(f'  Would keep: {would_pass} ({100*would_pass/len(df_stats):.1f}%)')
print(f'  Would lose: {len(df_stats) - would_pass} samples')
