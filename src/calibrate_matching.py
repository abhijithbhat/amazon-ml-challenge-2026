import pickle
from collections import defaultdict
import os

print("=" * 70)
print("  HIGH-PRECISION CALIBRATED MATCHER (FROM CACHE)")
print("=" * 70)

print("Step 1: Loading cached pairs...")
with open("output/scored_pairs_cache.pkl", "rb") as f:
    pairs = pickle.load(f)
print(f"  Loaded {len(pairs):,} pairs.")

# Priority function: break prob=1.0 ties using confirmed street digits
def get_priority(p):
    prob, s1, v, d_stat, country = p
    bonus = 0.05 if d_stat == "confirm" else (0.0 if d_stat == "absent" else -0.50)
    return prob + bonus

print("Step 2: Sorting pairs by priority descending...")
pairs.sort(key=get_priority, reverse=True)

print("Step 3: Running globally disjoint assignment with calibrated caps...")
assigned_vendors = set()
s1_matches = defaultdict(list)
s1_s2_count = defaultdict(int)
s1_s3_count = defaultdict(int)

# Policy:
# First match per catalog: prob >= 0.85 (confirm) or 0.90 (absent)
# Second match per catalog: STRICT confirm digits required + prob >= 0.92
for prob, s1, v, d_stat, country in pairs:
    if v in assigned_vendors:
        continue
        
    is_s2 = v.startswith("S2-")
    count = s1_s2_count[s1] if is_s2 else s1_s3_count[s1]
    
    if count == 0:
        # First match for this vendor catalog
        thresh = 0.85 if d_stat == "confirm" else 0.90
        if prob < thresh:
            continue
    elif count == 1:
        # Second match: require confirmed street numbers and high probability
        if d_stat != "confirm" or prob < 0.92:
            continue
    else:
        # Cap at 2 per catalog to prevent precision degradation
        continue
        
    assigned_vendors.add(v)
    s1_matches[s1].append(v)
    if is_s2:
        s1_s2_count[s1] += 1
    else:
        s1_s3_count[s1] += 1

# Step 4: Write matching_results_calibrated.tsv
out_path = "output/matching_results_calibrated.tsv"
print(f"Step 4: Writing results to {out_path}...")

singletons = 0
matched_entities = 0
total_vendor_records = 0

with open("dataset/test/test_source1.tsv", "r") as f_in, open(out_path, "w") as f_out:
    header = f_in.readline()
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    
    for line in f_in:
        s1_id = line.split("\t")[0].strip()
        v_list = s1_matches.get(s1_id, [])
        if v_list:
            matched_entities += 1
            total_vendor_records += len(v_list)
            match_str = ",".join(sorted(v_list))
            f_out.write(f"{s1_id}\t{match_str}\n")
        else:
            singletons += 1
            f_out.write(f"{s1_id}\t\n")

print("\n--- Calibration Results ---")
print(f"  Total S1 Entities:             1,732,544")
print(f"  Entities with matches:         {matched_entities:,} ({matched_entities/1732544*100:.2f}%)")
print(f"  Singletons (empty):            {singletons:,} ({singletons/1732544*100:.2f}%)")
print(f"  Total Vendor Records Assigned: {total_vendor_records:,}")
if matched_entities > 0:
    print(f"  Average matches per entity:    {total_vendor_records/matched_entities:.2f}")
print("=" * 70)
