#!/usr/bin/env python3
"""
Extract situations from EmpatheticDialogues and prepare for distillation.

EmpatheticDialogues has good situations but responses are too "friend-like".
We extract the situations and will use Claude to generate coaching responses.

Output format (for distillation):
{
    "situation": "...",  # Original context
    "emotion": "...",    # Emotion label
    "original_response": "...",  # Original empathetic response (for reference)
    "messages": [{"role": "user", "content": "..."}],  # Starting point
}
"""

import json
import csv
from collections import defaultdict
from pathlib import Path

def load_empathetic_dialogues():
    """Load EmpatheticDialogues CSV files."""
    data = {'train': [], 'valid': [], 'test': []}
    
    for split in ['train', 'valid', 'test']:
        filepath = f'/home/laihenyi/autoresearch/external_data/empatheticdialogues/{split}.csv'
        with open(filepath) as f:
            reader = csv.DictReader(f)
            for row in reader:
                data[split].append(row)
    
    return data

def extract_unique_situations(data):
    """Extract unique situations with their emotion labels."""
    situations = {}
    
    for split, rows in data.items():
        for row in rows:
            conv_id = row['conv_id']
            context = row['context']  # Emotion label like "sentimental"
            prompt = row['prompt']    # Situation description
            
            if conv_id not in situations:
                situations[conv_id] = {
                    'conv_id': conv_id,
                    'emotion': context,
                    'situation': prompt,
                    'utterances': [],
                }
            
            situations[conv_id]['utterances'].append({
                'speaker_idx': row['speaker_idx'],
                'utterance': row['utterance'],
                'utterance_idx': int(row['utterance_idx']),
            })
    
    return list(situations.values())

def prepare_distillation_input(situations, max_samples=500):
    """
    Prepare situations for distillation.
    We'll use the situation as user's opening, and ask Claude to generate
    coaching-style responses.
    """
    distill_data = []
    
    # Filter for diverse emotions
    emotion_counts = defaultdict(int)
    max_per_emotion = max_samples // 10  # Distribute across emotions
    
    for sit in situations:
        emotion = sit['emotion']
        if emotion_counts[emotion] >= max_per_emotion:
            continue
        
        # Get the first user utterance (situation description)
        user_utterances = [u for u in sit['utterances'] 
                          if u['speaker_idx'] == '1' and u['utterance_idx'] == 1]
        
        if user_utterances:
            user_msg = user_utterances[0]['utterance']
            
            # Get the original empathetic response (for reference)
            resp_utterances = [u for u in sit['utterances'] 
                              if u['speaker_idx'] == '0' and u['utterance_idx'] == 2]
            original_resp = resp_utterances[0]['utterance'] if resp_utterances else ''
            
            distill_data.append({
                'messages': [{'role': 'user', 'content': user_msg}],
                'metadata': {
                    'source': 'empathetic_dialogues',
                    'emotion': emotion,
                    'situation': sit['situation'],
                    'original_response': original_resp,
                }
            })
            emotion_counts[emotion] += 1
    
    return distill_data

def main():
    print("=" * 60)
    print("Processing EmpatheticDialogues...")
    print("=" * 60)
    
    data = load_empathetic_dialogues()
    print(f"Loaded splits: train={len(data['train'])}, valid={len(data['valid'])}, test={len(data['test'])}")
    
    situations = extract_unique_situations(data)
    print(f"Unique conversations: {len(situations)}")
    
    # Emotion distribution
    from collections import Counter
    emotions = Counter(s['emotion'] for s in situations)
    print(f"\nEmotion distribution (top 10):")
    for e, c in emotions.most_common(10):
        print(f"  {e}: {c}")
    
    # Prepare distillation input
    distill_data = prepare_distillation_input(situations, max_samples=500)
    print(f"\nPrepared {len(distill_data)} situations for distillation")
    
    # Save
    output_path = '/home/laihenyi/autoresearch/external_data/converted/empathetic_distill_input.jsonl'
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in distill_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved to: {output_path}")
    
    # Show samples
    print("\n--- Samples for distillation ---")
    for i, item in enumerate(distill_data[:3]):
        print(f"\n[{i}] Emotion: {item['metadata']['emotion']}")
        print(f"  User: {item['messages'][0]['content'][:100]}...")
        print(f"  Original (friend-like): {item['metadata']['original_response'][:80]}...")

if __name__ == '__main__':
    main()
