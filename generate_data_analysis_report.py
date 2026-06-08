#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import statistics


def parse_metadata(filename: str):
    name = Path(filename).name
    parts = name.replace('.npz', '').split('__')
    if len(parts) >= 5:
        word = parts[0]
        signer = parts[1]
        # emotion usually last
        emotion = parts[-1]
        return {
            'word': word,
            'signer': signer,
            'emotion': emotion,
            'name': name,
        }
    # fallback: attempt to infer signer with pattern _Sdd_
    signer = None
    for p in parts:
        if p.startswith('S') and len(p) == 3 and p[1:].isdigit():
            signer = p
    word = parts[0] if parts else name
    emotion = parts[-1] if parts else ''
    return {'word': word, 'signer': signer, 'emotion': emotion, 'name': name}


def try_sequence_length(npz):
    # prefer explicit keys
    for key in ('raw_length', 'length', 'seq_length', 'sequence_length'):
        if key in npz:
            try:
                return int(npz[key].tolist())
            except Exception:
                pass
    # else find any array with 2+ dims and take first dim
    for k, v in npz.items():
        if hasattr(v, 'shape') and len(v.shape) >= 1:
            if v.shape[0] > 0 and v.shape[0] < 10000:
                return int(v.shape[0])
    return None


def main():
    if len(sys.argv) < 4:
        print('Usage: generate_data_analysis_report.py <data_dir> <label_mapping.json> <output_report.txt>')
        sys.exit(2)
    data_dir = Path(sys.argv[1])
    mapping_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    if not data_dir.exists():
        print('Data dir not found:', data_dir)
        sys.exit(1)
    if not mapping_path.exists():
        print('Label mapping not found:', mapping_path)
        sys.exit(1)

    mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
    label_to_word = mapping.get('label_to_word') or {}
    # label_to_emotion may be str keys
    label_to_emotion = mapping.get('label_to_emotion') or {}
    # convert to int->emotion mapping
    label2emo = {}
    for k, v in label_to_emotion.items():
        try:
            label2emo[int(k)] = v
        except Exception:
            # if keys are emotions -> labels
            pass

    # also create mapping emotion->label if present
    emotion_to_label = mapping.get('emotion_to_label') or {}

    npz_files = sorted([p for p in data_dir.rglob('*.npz')])

    total_files = len(npz_files)
    words = Counter()
    emotions = Counter()
    signer_counts = Counter()
    signer_word_sets = defaultdict(set)
    signer_emotion_counts = defaultdict(Counter)
    word_counts = Counter()
    seq_lengths = []
    per_signer_seq = defaultdict(list)

    for p in npz_files:
        meta = parse_metadata(p.name)
        word = meta['word']
        signer = meta['signer'] or 'UNK'
        emotion_raw = meta['emotion']
        # map emotion via mapping if possible
        emotion_label = None
        emotion = emotion_raw
        try:
            # if emotion is integer label string
            if emotion_raw.isdigit() and int(emotion_raw) in label2emo:
                emotion = label2emo[int(emotion_raw)]
        except Exception:
            pass
        # if mapping exists from emotion_to_label invert it
        if emotion_to_label and emotion_raw in emotion_to_label:
            emotion = emotion_raw
        words[word] += 1
        word_counts[word] += 1
        emotions[emotion] += 1
        signer_counts[signer] += 1
        signer_word_sets[signer].add(word)
        signer_emotion_counts[signer][emotion] += 1
        # open npz to read length if available
        try:
            with np.load(p, allow_pickle=True) as npz:
                L = try_sequence_length(npz)
                if L is None:
                    # fallback: try to get body array
                    for key in ('body','pose','pose_landmarks'):
                        if key in npz:
                            arr = npz[key]
                            if hasattr(arr, 'shape') and len(arr.shape) >= 1:
                                L = int(arr.shape[0])
                                break
                if L is not None:
                    seq_lengths.append(L)
                    per_signer_seq[signer].append(L)
        except Exception:
            pass

    unique_word_classes = len(word_counts)
    unique_emotion_classes = len(emotions)

    # Splits based on signer
    test_signer = 'S11'
    train_val_signers = ['S02','S03','S06','S07','S10','S12']
    split_counts = {
        'train_val': sum(count for s, count in signer_counts.items() if s in train_val_signers),
        'test': signer_counts.get(test_signer, 0),
    }

    # Per-word stats
    counts_list = list(word_counts.values())
    min_c = min(counts_list) if counts_list else 0
    max_c = max(counts_list) if counts_list else 0
    mean_c = statistics.mean(counts_list) if counts_list else 0
    median_c = statistics.median(counts_list) if counts_list else 0
    fewest = [w for w,c in word_counts.items() if c == min_c]

    # Sequence stats
    avg_seq = statistics.mean(seq_lengths) if seq_lengths else 0
    min_seq = min(seq_lengths) if seq_lengths else 0
    max_seq = max(seq_lengths) if seq_lengths else 0

    # Build report
    lines = []
    lines.append('DATA ANALYSIS REPORT')
    lines.append('Data dir: ' + str(data_dir))
    lines.append('Label mapping: ' + str(mapping_path))
    lines.append('')
    lines.append('1. OVERALL STATISTICS')
    lines.append(f'   Total files: {total_files}')
    lines.append(f'   Total word classes: {unique_word_classes}')
    lines.append(f'   Total emotion classes: {unique_emotion_classes}')
    lines.append(f'   Files per split: train+val={split_counts["train_val"]}, test={split_counts["test"]}')
    lines.append(f'   Test signer: {test_signer}')
    lines.append(f'   Train+val signers: {", ".join(train_val_signers)}')
    lines.append('')
    lines.append('2. PER-SIGNER STATISTICS')
    # Canonical emotion order
    canonical_emotions = ['neutral', 'happy', 'sad', 'negation', 'question']
    lines.append('Signer | Files | Word classes covered | Neutral | Happy | Sad | Negation | Question')
    for s in ['S02','S03','S06','S07','S10','S11','S12']:
        files = signer_counts.get(s, 0)
        words_cov = len(signer_word_sets.get(s, set()))
        sc = signer_emotion_counts.get(s, Counter())
        # case-insensitive mapping of counts
        sc_lower = {k.lower(): v for k, v in sc.items()}
        counts = [sc_lower.get(e, 0) for e in canonical_emotions]
        lines.append(f'{s} | {files} | {words_cov} | ' + ' | '.join(str(c) for c in counts))
    lines.append('')
    lines.append('3. PER-EMOTION STATISTICS')
    total_em = sum(emotions.values())
    # Print in canonical order when possible
    for emo in ['happy', 'negation', 'neutral', 'sad', 'question']:
        cnt = emotions.get(emo, emotions.get(emo.capitalize(), 0))
        pct = (cnt/total_em*100) if total_em>0 else 0
        lines.append(f'   {emo}: {cnt} files ({pct:.2f}%)')
    lines.append('')
    lines.append('Per-signer per-emotion breakdown:')
    for s in ['S02','S03','S06','S07','S10','S11','S12']:
        sc = signer_emotion_counts.get(s, Counter())
        sc_lower = {k.lower(): v for k, v in sc.items()}
        lines.append(
            f'   {s}: ' + ', '.join(f'{emo}:{sc_lower.get(emo,0)}' for emo in canonical_emotions)
        )
    lines.append('')
    lines.append('4. PER-WORD-CLASS STATISTICS')
    lines.append(f'   Unique word classes: {unique_word_classes}')
    lines.append('   Files per word class (sample):')
    for w, c in word_counts.most_common(20):
        lines.append(f'      {w}: {c}')
    lines.append(f'   Min files/class: {min_c}')
    lines.append(f'   Max files/class: {max_c}')
    lines.append(f'   Mean files/class: {mean_c:.2f}')
    lines.append(f'   Median files/class: {median_c}')
    lines.append(f'   Classes with fewest samples ({min_c} samples): ' + ', '.join(fewest))
    lines.append('')
    lines.append('5. SEQUENCE STATISTICS')
    lines.append(f'   Average sequence length: {avg_seq:.2f}')
    lines.append(f'   Min sequence length: {min_seq}')
    lines.append(f'   Max sequence length: {max_seq}')
    lines.append('   Distribution across signers:')
    for s in ['S02','S03','S06','S07','S10','S11','S12']:
        arr = per_signer_seq.get(s, [])
        if arr:
            lines.append(f'      {s}: count={len(arr)}, mean={statistics.mean(arr):.2f}, min={min(arr)}, max={max(arr)}')
        else:
            lines.append(f'      {s}: count=0')

    report = '\n'.join(lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding='utf-8')
    print(report)


if __name__ == '__main__':
    main()
