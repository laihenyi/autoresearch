#!/usr/bin/env python3
"""
Convert AnnoMI to DPO (Direct Preference Optimization) format.

AnnoMI has high/low MI quality labels, perfect for preference pairs.

Output format:
{
    "prompt": [{"role": "user", "content": "..."}],
    "chosen": [{"role": "assistant", "content": "..."}],  # high MI quality
    "rejected": [{"role": "assistant", "content": "..."}],  # low MI quality
}
"""

import json
import csv
from collections import defaultdict
from pathlib import Path

def load_annomi():
    """Load AnnoMI full CSV."""
    rows = []
    with open('/home/laihenyi/autoresearch/external_data/AnnoMI/AnnoMI-full.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows

def group_by_transcript(rows):
    """Group utterances by transcript_id."""
    transcripts = defaultdict(list)
    for row in rows:
        transcripts[row['transcript_id']].append(row)
    return transcripts

def extract_therapist_turns(transcript_rows):
    """Extract therapist turns with their MI quality and behavior labels."""
    turns = []
    for row in transcript_rows:
        if row['interlocutor'] == 'therapist':
            turns.append({
                'utterance_id': int(row['utterance_id']),
                'text': row['utterance_text'],
                'mi_quality': row['mi_quality'],
                'behavior': row['main_therapist_behaviour'],
                'question_type': row.get('question_subtype', ''),
                'reflection_type': row.get('reflection_subtype', ''),
            })
    return turns

def get_context_before(transcript_rows, utterance_id):
    """Get conversation context before a given utterance."""
    context = []
    for row in transcript_rows:
        if int(row['utterance_id']) >= utterance_id:
            break
        role = 'user' if row['interlocutor'] == 'client' else 'assistant'
        context.append({
            'role': role,
            'content': row['utterance_text']
        })
    return context

def build_dpo_pairs(transcripts):
    """Build DPO preference pairs from high vs low MI quality."""
    dpo_pairs = []
    
    # Group by topic for matching
    by_topic = defaultdict(list)
    for tid, rows in transcripts.items():
        topic = rows[0].get('topic', 'unknown')
        mi_quality = rows[0].get('mi_quality', 'unknown')
        by_topic[(topic, mi_quality)].append((tid, rows))
    
    # Find matching high/low pairs by topic
    topics = set(t for t, q in by_topic.keys())
    
    for topic in topics:
        high_transcripts = by_topic.get((topic, 'high'), [])
        low_transcripts = by_topic.get((topic, 'low'), [])
        
        # Match high quality responses with low quality on similar context
        for high_tid, high_rows in high_transcripts:
            high_turns = extract_therapist_turns(high_rows)
            
            for low_tid, low_rows in low_transcripts:
                low_turns = extract_therapist_turns(low_rows)
                
                # Find matching positions (same utterance position in dialog)
                for h_turn in high_turns[:5]:  # Focus on early turns
                    for l_turn in low_turns[:5]:
                        # Same behavior type, different quality
                        if h_turn['behavior'] == l_turn['behavior'] and h_turn['behavior'] in ['question', 'reflection']:
                            # Get context from high quality transcript
                            context = get_context_before(high_rows, h_turn['utterance_id'])
                            
                            if len(context) >= 2:  # At least one exchange
                                # Use last user message as prompt
                                user_msgs = [m for m in context if m['role'] == 'user']
                                if user_msgs:
                                    prompt = [user_msgs[-1]]
                                    dpo_pairs.append({
                                        'prompt': prompt,
                                        'chosen': [{'role': 'assistant', 'content': h_turn['text']}],
                                        'rejected': [{'role': 'assistant', 'content': l_turn['text']}],
                                        'metadata': {
                                            'topic': topic,
                                            'behavior': h_turn['behavior'],
                                            'high_transcript': high_tid,
                                            'low_transcript': low_tid,
                                        }
                                    })
    
    return dpo_pairs

def build_dpo_pairs_v2(rows):
    """
    Alternative approach: Use same transcript, compare high-quality behaviors vs low-quality.
    For AnnoMI, high/low is at transcript level, so we pair similar contexts across transcripts.
    """
    transcripts = group_by_transcript(rows)
    
    # Separate high and low quality transcripts
    high_quality = {tid: rows for tid, rows in transcripts.items() 
                    if rows[0]['mi_quality'] == 'high'}
    low_quality = {tid: rows for tid, rows in transcripts.items() 
                   if rows[0]['mi_quality'] == 'low'}
    
    print(f"High quality transcripts: {len(high_quality)}")
    print(f"Low quality transcripts: {len(low_quality)}")
    
    dpo_pairs = []
    
    # Build context-response pairs from each transcript
    high_pairs = []
    for tid, trows in high_quality.items():
        topic = trows[0].get('topic', '')
        for i, row in enumerate(trows):
            if row['interlocutor'] == 'therapist' and row['main_therapist_behaviour'] in ['question', 'reflection']:
                # Get context (previous utterances)
                context = []
                for j in range(i):
                    role = 'user' if trows[j]['interlocutor'] == 'client' else 'assistant'
                    context.append({'role': role, 'content': trows[j]['utterance_text']})
                
                if context:
                    user_msgs = [m for m in context if m['role'] == 'user']
                    if user_msgs:
                        high_pairs.append({
                            'context': context,
                            'prompt': [user_msgs[-1]],
                            'response': row['utterance_text'],
                            'behavior': row['main_therapist_behaviour'],
                            'topic': topic,
                            'transcript_id': tid,
                        })
    
    low_pairs = []
    for tid, trows in low_quality.items():
        topic = trows[0].get('topic', '')
        for i, row in enumerate(trows):
            if row['interlocutor'] == 'therapist':
                context = []
                for j in range(i):
                    role = 'user' if trows[j]['interlocutor'] == 'client' else 'assistant'
                    context.append({'role': role, 'content': trows[j]['utterance_text']})
                
                if context:
                    user_msgs = [m for m in context if m['role'] == 'user']
                    if user_msgs:
                        low_pairs.append({
                            'context': context,
                            'prompt': [user_msgs[-1]],
                            'response': row['utterance_text'],
                            'behavior': row['main_therapist_behaviour'],
                            'topic': topic,
                            'transcript_id': tid,
                        })
    
    # Match by topic and behavior
    from collections import Counter
    topic_behavior_counts = Counter((p['topic'], p['behavior']) for p in high_pairs)
    print(f"\nHigh quality pairs by (topic, behavior):")
    for (t, b), c in topic_behavior_counts.most_common(10):
        print(f"  {t[:30]:30} | {b:15} | {c}")
    
    # Build DPO pairs
    for hp in high_pairs:
        # Find matching low quality pair
        for lp in low_pairs:
            if (hp['topic'] == lp['topic'] and 
                hp['behavior'] == lp['behavior'] and
                len(hp['context']) >= 2):
                
                dpo_pairs.append({
                    'prompt': hp['prompt'],
                    'chosen': [{'role': 'assistant', 'content': hp['response']}],
                    'rejected': [{'role': 'assistant', 'content': lp['response']}],
                    'metadata': {
                        'topic': hp['topic'],
                        'behavior': hp['behavior'],
                        'high_tid': hp['transcript_id'],
                        'low_tid': lp['transcript_id'],
                    }
                })
                break  # One match per high pair
    
    return dpo_pairs

def build_sft_from_high_mi(rows):
    """
    Extract high-quality MI turns for SFT training.
    Focus on question and reflection behaviors.
    """
    transcripts = group_by_transcript(rows)
    
    sft_data = []
    
    for tid, trows in transcripts.items():
        if trows[0]['mi_quality'] != 'high':
            continue
        
        # Build full conversation
        messages = []
        for row in trows:
            role = 'user' if row['interlocutor'] == 'client' else 'assistant'
            content = row['utterance_text']
            
            # Merge consecutive same-role messages
            if messages and messages[-1]['role'] == role:
                messages[-1]['content'] += ' ' + content
            else:
                messages.append({'role': role, 'content': content})
        
        if len(messages) >= 4:
            sft_data.append({
                'messages': messages,
                'metadata': {
                    'source': 'annomi',
                    'topic': trows[0].get('topic', ''),
                    'mi_quality': 'high',
                }
            })
    
    return sft_data

def main():
    print("=" * 60)
    print("Converting AnnoMI...")
    print("=" * 60)
    
    rows = load_annomi()
    print(f"Loaded {len(rows)} utterances")
    
    # Build DPO pairs
    dpo_pairs = build_dpo_pairs_v2(rows)
    print(f"\nGenerated {len(dpo_pairs)} DPO pairs")
    
    # Build SFT from high MI
    sft_data = build_sft_from_high_mi(rows)
    print(f"Generated {len(sft_data)} SFT dialogs from high-quality MI")
    
    # Save DPO data
    dpo_path = '/home/laihenyi/autoresearch/external_data/converted/annomi_dpo.jsonl'
    with open(dpo_path, 'w', encoding='utf-8') as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\n')
    print(f"\nDPO saved to: {dpo_path}")
    
    # Save SFT data
    sft_path = '/home/laihenyi/autoresearch/external_data/converted/annomi_sft_high.jsonl'
    with open(sft_path, 'w', encoding='utf-8') as f:
        for dialog in sft_data:
            f.write(json.dumps(dialog, ensure_ascii=False) + '\n')
    print(f"SFT saved to: {sft_path}")
    
    # Show samples
    print("\n--- DPO Sample ---")
    if dpo_pairs:
        sample = dpo_pairs[0]
        print(f"Topic: {sample['metadata']['topic']}")
        print(f"Behavior: {sample['metadata']['behavior']}")
        print(f"Prompt: {sample['prompt'][0]['content'][:80]}...")
        print(f"Chosen (high MI): {sample['chosen'][0]['content'][:80]}...")
        print(f"Rejected (low MI): {sample['rejected'][0]['content'][:80]}...")
    
    print("\n--- SFT Sample ---")
    if sft_data:
        sample = sft_data[0]
        print(f"Topic: {sample['metadata']['topic']}")
        for m in sample['messages'][:6]:
            print(f"  [{m['role']}]: {m['content'][:60]}...")

if __name__ == '__main__':
    main()
