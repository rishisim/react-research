
import json
import os
import collections

def analyze_distribution(split='dev'):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, f'../data/musique_v1.0/musique_ans_v1.0_{split}.jsonl')
    
    print(f"Analyzing {data_path}...")
    
    hop_counts = collections.Counter()
    id_prefixes = collections.Counter()
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                
                # Analyze by decomposition length
                decomp = item.get('question_decomposition', [])
                hops = len(decomp)
                hop_counts[hops] += 1
                
                # Analyze by ID prefix (often indicates type)
                qid = item.get('id', '')
                if '__' in qid:
                    prefix = qid.split('__')[0]
                    id_prefixes[prefix] += 1
                    
        print(f"\nDistribution for '{split}' set:")
        print("-" * 30)
        print("By Decomposition Length:")
        for hops, count in sorted(hop_counts.items()):
            print(f"  {hops}-hop: {count} ({count/sum(hop_counts.values())*100:.1f}%)")
            
        print("\nBy ID Prefix:")
        for prefix, count in sorted(id_prefixes.items()):
            print(f"  {prefix}: {count} ({count/sum(id_prefixes.values())*100:.1f}%)")
            
    except FileNotFoundError:
        print(f"File not found: {data_path}")

if __name__ == "__main__":
    analyze_distribution('dev')
