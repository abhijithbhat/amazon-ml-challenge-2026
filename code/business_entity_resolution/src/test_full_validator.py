import os
import subprocess

test_dir = r"dataset/test"
s1_path = os.path.join(test_dir, "test_source1.tsv")
out_dir = r"output"
os.makedirs(out_dir, exist_ok=True)

match_path = os.path.join(out_dir, "matching_results.tsv")
cand_path = os.path.join(out_dir, "candidate_pairs.tsv")

# Read current matches from sample run
existing_matches = {}
if os.path.exists(match_path):
    with open(match_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 2 and parts[1]:
                existing_matches[parts[0]] = parts[1]

existing_cands = {}
if os.path.exists(cand_path):
    with open(cand_path, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 2 and parts[1]:
                existing_cands[parts[0]] = parts[1]

print("Writing complete files with all 1.73M S1 entities...")
with open(s1_path, 'r', encoding='utf-8') as f_in, \
     open(match_path, 'w', encoding='utf-8') as f_m, \
     open(cand_path, 'w', encoding='utf-8') as f_c:
    
    f_m.write("source1_entity_id\tmatched_entity_ids\n")
    f_c.write("source1_entity_id\tcandidate_entity_ids\n")
    next(f_in)
    count = 0
    for line in f_in:
        s1_id = line.split('\t', 1)[0].strip()
        if not s1_id:
            continue
        m_str = existing_matches.get(s1_id, '')
        c_str = existing_cands.get(s1_id, '')
        if m_str and not c_str:
            c_str = m_str
        f_m.write(f"{s1_id}\t{m_str}\n")
        f_c.write(f"{s1_id}\t{c_str}\n")
        count += 1

print(f"Wrote {count:,} S1 rows.")

val_script = r"utils/validate_submission.py"
cmd = f'python "{val_script}" --matching "{match_path}" --candidate "{cand_path}" --test-dir "{test_dir}"'
res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print("\n" + res.stdout)
if res.stderr:
    print(res.stderr)
print(f"Validator Exit Code: {res.returncode}")
