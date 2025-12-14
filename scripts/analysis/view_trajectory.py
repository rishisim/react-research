import json

data = json.load(open('results/fever/20251125_203516_n15_gemini-2.5-flash/baseline.json'))

print('=== TRAJECTORY EXAMPLE 1 ===\n')
print(data[0]['traj'])
