import json
import csv

def parse_baseline(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    rewards = []
    for item in data:
        rewards.append(int(item['reward']))
    return rewards

def parse_multi_trace(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    rewards = []
    for item in data:
        rewards.append(int(item['reward']))
    return rewards

def main():
    fever_baseline_rewards = parse_baseline('FEVER_Experiment/react_baseline_results.json')
    fever_multi_trace_rewards = parse_multi_trace('FEVER_Experiment/react_multi_trace_results.json')
    hotpotqa_baseline_rewards = parse_baseline('HotPotQA_Experiment/react_baseline_results.json')
    hotpotqa_multi_trace_rewards = parse_multi_trace('HotPotQA_Experiment/react_cot_synth_results.json')

    with open('analysis_results.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Dataset', 'Q', 'react_baseline', 'react_multi_trace'])

        for i in range(len(hotpotqa_baseline_rewards)):
            writer.writerow(['HotPotQA', i + 1, hotpotqa_baseline_rewards[i], hotpotqa_multi_trace_rewards[i]])

        for i in range(len(fever_baseline_rewards)):
            writer.writerow(['FEVER', i + 1, fever_baseline_rewards[i], fever_multi_trace_rewards[i]])

    print("FEVER Dataset:")
    total_fever = len(fever_baseline_rewards)
    baseline_correct_fever = sum(fever_baseline_rewards)
    multi_trace_correct_fever = sum(fever_multi_trace_rewards)
    baseline_accuracy_fever = (baseline_correct_fever / total_fever) * 100
    multi_trace_accuracy_fever = (multi_trace_correct_fever / total_fever) * 100

    print(f"  Baseline Correct: {baseline_correct_fever} / {total_fever}")
    print(f"  Baseline Accuracy: {baseline_accuracy_fever:.2f}%")
    print(f"  Multi-Trace Correct: {multi_trace_correct_fever} / {total_fever}")
    print(f"  Multi-Trace Accuracy: {multi_trace_accuracy_fever:.2f}%")
    print("  Difference (Multi-Trace vs. Baseline):")
    print(f"    Correct Questions: {multi_trace_correct_fever - baseline_correct_fever}")
    print(f"    Accuracy: {multi_trace_accuracy_fever - baseline_accuracy_fever:.2f}%")

    print("\nHotPotQA Dataset:")
    total_hotpotqa = len(hotpotqa_baseline_rewards)
    baseline_correct_hotpotqa = sum(hotpotqa_baseline_rewards)
    multi_trace_correct_hotpotqa = sum(hotpotqa_multi_trace_rewards)
    baseline_accuracy_hotpotqa = (baseline_correct_hotpotqa / total_hotpotqa) * 100
    multi_trace_accuracy_hotpotqa = (multi_trace_correct_hotpotqa / total_hotpotqa) * 100

    print(f"  Baseline Correct: {baseline_correct_hotpotqa} / {total_hotpotqa}")
    print(f"  Baseline Accuracy: {baseline_accuracy_hotpotqa:.2f}%")
    print(f"  Multi-Trace Correct: {multi_trace_correct_hotpotqa} / {total_hotpotqa}")
    print(f"  Multi-Trace Accuracy: {multi_trace_accuracy_hotpotqa:.2f}%")
    print("  Difference (Multi-Trace vs. Baseline):")
    print(f"    Correct Questions: {multi_trace_correct_hotpotqa - baseline_correct_hotpotqa}")
    print(f"    Accuracy: {multi_trace_accuracy_hotpotqa - baseline_accuracy_hotpotqa:.2f}%")

if __name__ == '__main__':
    main()
