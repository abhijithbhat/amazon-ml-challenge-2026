"""
=============================================================================
STEP 1-3: Load Data -> Inspect Columns -> Identify the Real Mess
=============================================================================
Your role: Data & Features (Address normalization, Name cleaning, Landmarks)

This script does:
  1. Load S1, S2, S3 from TRAIN and TEST into DataFrames
  2. Inspect columns: shape, dtypes, nulls, sample rows
  3. Identify the REAL MESS: name variations, address abbreviations,
     landmarks, special characters, transliterations, etc.
=============================================================================
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import re
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: LOAD ALL SOURCE FILES INTO DATAFRAMES
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 80)
print("STEP 1: LOADING DATA INTO DATAFRAMES")
print("=" * 80)

TRAIN_DIR = "dataset/train"
TEST_DIR  = "dataset/test"

# Train DataFrames
train_s1 = pd.read_csv(f"{TRAIN_DIR}/train_source1.tsv", sep="\t")
# NOTE: train_source2.tsv is MISSING from your dataset folder!
# We'll handle this gracefully
try:
    train_s2 = pd.read_csv(f"{TRAIN_DIR}/train_source2.tsv", sep="\t")
    has_train_s2 = True
except FileNotFoundError:
    train_s2 = None
    has_train_s2 = False
    print("⚠️  WARNING: train_source2.tsv is MISSING from train folder!")
    print("   You need to download it. The ground truth references S2 IDs,")
    print("   so you definitely need this file.\n")

train_s3 = pd.read_csv(f"{TRAIN_DIR}/train_source3.tsv", sep="\t")
train_gt = pd.read_csv(f"{TRAIN_DIR}/train_ground_truth.tsv", sep="\t")

# Test DataFrames
test_s1 = pd.read_csv(f"{TEST_DIR}/test_source1.tsv", sep="\t")
test_s2 = pd.read_csv(f"{TEST_DIR}/test_source2.tsv", sep="\t")
test_s3 = pd.read_csv(f"{TEST_DIR}/test_source3.tsv", sep="\t")

datasets = {
    "TRAIN S1": train_s1,
    "TRAIN S3": train_s3,
    "TRAIN Ground Truth": train_gt,
    "TEST S1": test_s1,
    "TEST S2": test_s2,
    "TEST S3": test_s3,
}
if has_train_s2:
    datasets["TRAIN S2"] = train_s2

for name, df in datasets.items():
    print(f"\n✅ {name}: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: INSPECT THE COLUMNS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("STEP 2: INSPECTING COLUMNS & DATA QUALITY")
print("=" * 80)

# We'll inspect the source files (not ground truth)
source_dfs = {
    "TRAIN S1": train_s1,
    "TRAIN S3": train_s3,
    "TEST S1": test_s1,
    "TEST S2": test_s2,
    "TEST S3": test_s3,
}
if has_train_s2:
    source_dfs["TRAIN S2"] = train_s2

for name, df in source_dfs.items():
    print(f"\n{'─' * 60}")
    print(f"📋 {name}")
    print(f"{'─' * 60}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Shape:   {df.shape}")
    print(f"  Dtypes:")
    for col in df.columns:
        print(f"    {col}: {df[col].dtype}")
    print(f"  Null counts:")
    for col in df.columns:
        null_count = df[col].isna().sum()
        pct = null_count / len(df) * 100
        print(f"    {col}: {null_count:,} ({pct:.2f}%)")

    print(f"\n  📝 First 5 rows:")
    for i, row in df.head(5).iterrows():
        print(f"    [{row['entity_id']}]")
        if 'business_name' in df.columns:
            print(f"      Name:    {row['business_name']}")
        if 'business_address' in df.columns:
            print(f"      Address: {row['business_address']}")
        if 'country' in df.columns:
            print(f"      Country: {row['country']}")

# Ground truth inspection
print(f"\n{'─' * 60}")
print(f"📋 GROUND TRUTH")
print(f"{'─' * 60}")
print(f"  Shape: {train_gt.shape}")
print(f"  Columns: {list(train_gt.columns)}")
# Count singletons (empty matched_entity_ids)
singletons = train_gt['matched_entity_ids'].isna().sum()
print(f"  Singletons (no matches): {singletons:,} ({singletons/len(train_gt)*100:.1f}%)")
print(f"  Entities with matches:   {len(train_gt) - singletons:,}")

# Count how many have S2 matches, S3 matches, or both
has_s2 = 0
has_s3 = 0
has_both = 0
match_counts = []
for _, row in train_gt.iterrows():
    if pd.isna(row['matched_entity_ids']) or row['matched_entity_ids'] == '':
        match_counts.append(0)
        continue
    ids = str(row['matched_entity_ids']).split(',')
    match_counts.append(len(ids))
    s2_ids = [x for x in ids if x.startswith('S2')]
    s3_ids = [x for x in ids if x.startswith('S3')]
    if s2_ids:
        has_s2 += 1
    if s3_ids:
        has_s3 += 1
    if s2_ids and s3_ids:
        has_both += 1

print(f"  Entities with S2 matches: {has_s2:,}")
print(f"  Entities with S3 matches: {has_s3:,}")
print(f"  Entities with BOTH:       {has_both:,}")
print(f"  Avg matches per entity:   {sum(match_counts)/len(match_counts):.2f}")
print(f"  Max matches per entity:   {max(match_counts)}")

# Country distribution
print(f"\n{'─' * 60}")
print(f"📊 COUNTRY DISTRIBUTION")
print(f"{'─' * 60}")
for name, df in source_dfs.items():
    if 'country' in df.columns:
        print(f"\n  {name}:")
        for country, count in df['country'].value_counts().items():
            print(f"    {country}: {count:,} ({count/len(df)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: IDENTIFY THE REAL MESS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("STEP 3: IDENTIFYING THE REAL MESS")
print("=" * 80)

# We'll combine all source data for pattern analysis
# (just for INSPECTION — we're not merging train+test for model purposes)
all_names = []
all_addresses = []
all_countries = []

for df in [train_s1, train_s3, test_s1, test_s2, test_s3]:
    if df is not None and 'business_name' in df.columns:
        all_names.extend(df['business_name'].dropna().tolist())
    if df is not None and 'business_address' in df.columns:
        all_addresses.extend(df['business_address'].dropna().tolist())
    if df is not None and 'country' in df.columns:
        all_countries.extend(df['country'].dropna().tolist())

if has_train_s2:
    all_names.extend(train_s2['business_name'].dropna().tolist())
    all_addresses.extend(train_s2['business_address'].dropna().tolist())

# Sample for faster analysis (full data is millions of rows)
import random
random.seed(42)
sample_size = min(200000, len(all_names))
name_sample = random.sample(all_names, sample_size)
addr_sample = random.sample(all_addresses, min(sample_size, len(all_addresses)))

# ── MESS #1: Business Name Legal Suffixes ──
print("\n" + "─" * 60)
print("🔍 MESS #1: BUSINESS NAME — Legal Suffix Variations")
print("─" * 60)

suffix_patterns = {
    'Private Limited':  r'\bprivate\s+limited\b',
    'Pvt Ltd':          r'\bpvt\.?\s*ltd\.?\b',
    'Pvt. Limited':     r'\bpvt\.?\s+limited\b',
    'Private Ltd':      r'\bprivate\s+ltd\.?\b',
    'Limited':          r'\blimited\b',
    'Ltd':              r'\bltd\.?\b',
    'LLC':              r'\bllc\b',
    'L.L.C':            r'\bl\.l\.c\.?\b',
    'Inc':              r'\binc\.?\b',
    'Incorporated':     r'\bincorporated\b',
    'Corp':             r'\bcorp\.?\b',
    'Corporation':      r'\bcorporation\b',
    'Co':               r'\bco\.?\b',
    'Company':          r'\bcompany\b',
    'LLP':              r'\bllp\b',
    'SARL':             r'\bsarl\b',
    'SAS':              r'\bsas\b',
    'SASU':             r'\bsasu\b',
    'SCI':              r'\bsci\b',
    'SA':               r'\bsa\b',
    'S.A.':             r'\bs\.a\.?\b',
    'S.A.S':            r'\bs\.a\.s\.?\b',
    'PC':               r'\bpc\b',
    'PLLC':             r'\bpllc\b',
    'DBA / dba':        r'\bdba\b',
}

print("  Pattern matches in name sample:")
for label, pat in suffix_patterns.items():
    count = sum(1 for n in name_sample if re.search(pat, str(n), re.IGNORECASE))
    if count > 0:
        pct = count / len(name_sample) * 100
        print(f"    {label:25s} → {count:>7,} hits ({pct:.2f}%)")

# ── MESS #2: Address Abbreviations ──
print("\n" + "─" * 60)
print("🔍 MESS #2: ADDRESS — Street/Road/Location Abbreviations")
print("─" * 60)

address_abbrevs = {
    'Street / St':      (r'\bstreet\b', r'\bst\.?\b'),
    'Road / Rd':        (r'\broad\b', r'\brd\.?\b'),
    'Avenue / Ave':     (r'\bavenue\b', r'\bave\.?\b'),
    'Boulevard / Blvd': (r'\bboulevard\b', r'\bblvd\.?\b'),
    'Drive / Dr':       (r'\bdrive\b', r'\bdr\.?\b'),
    'Lane / Ln':        (r'\blane\b', r'\bln\.?\b'),
    'Court / Ct':       (r'\bcourt\b', r'\bct\.?\b'),
    'Circle / Cir':     (r'\bcircle\b', r'\bcir\.?\b'),
    'Highway / Hwy':    (r'\bhighway\b', r'\bhwy\.?\b'),
    'Place / Pl':       (r'\bplace\b', r'\bpl\.?\b'),
    'Floor / Fl':       (r'\bfloor\b', r'\bfl\.?\b'),
    'Apartment / Apt':  (r'\bapartment\b', r'\bapt\.?\b'),
    'Building / Bldg':  (r'\bbuilding\b', r'\bbldg\.?\b'),
    'Suite / Ste':      (r'\bsuite\b', r'\bste\.?\b'),
    'Opposite / Opp':   (r'\bopposite\b', r'\bopp\.?\b'),
    'Near':             (r'\bnear\b', None),
    'Behind':           (r'\bbehind\b', None),
    'Nagar':            (r'\bnagar\b', None),
    'Colony':           (r'\bcolony\b', None),
    'PO Box':           (r'\bpo\s*box\b', None),
}

print("  Full vs Abbreviated forms in address sample:")
for label, pats in address_abbrevs.items():
    full_pat, abbr_pat = pats
    full_count = sum(1 for a in addr_sample if re.search(full_pat, str(a), re.IGNORECASE))
    if abbr_pat:
        abbr_count = sum(1 for a in addr_sample if re.search(abbr_pat, str(a), re.IGNORECASE))
        print(f"    {label:25s} → Full: {full_count:>7,}  |  Abbrev: {abbr_count:>7,}")
    else:
        print(f"    {label:25s} → Hits: {full_count:>7,}")


# ── MESS #3: Capitalization Chaos ──
print("\n" + "─" * 60)
print("🔍 MESS #3: CAPITALIZATION CHAOS")
print("─" * 60)

all_upper = sum(1 for n in name_sample if str(n) == str(n).upper() and len(str(n)) > 3)
all_lower = sum(1 for n in name_sample if str(n) == str(n).lower() and len(str(n)) > 3)
title_case = sum(1 for n in name_sample if str(n) == str(n).title() and len(str(n)) > 3)
mixed = len(name_sample) - all_upper - all_lower - title_case

print(f"  Business names (sample of {len(name_sample):,}):")
print(f"    ALL UPPER:    {all_upper:>7,}")
print(f"    all lower:    {all_lower:>7,}")
print(f"    Title Case:   {title_case:>7,}")
print(f"    MiXeD cAsE:   {mixed:>7,}")

addr_upper = sum(1 for a in addr_sample if str(a) == str(a).upper() and len(str(a)) > 3)
addr_lower = sum(1 for a in addr_sample if str(a) == str(a).lower() and len(str(a)) > 3)
addr_title = sum(1 for a in addr_sample if str(a) == str(a).title() and len(str(a)) > 3)
addr_mixed = len(addr_sample) - addr_upper - addr_lower - addr_title

print(f"\n  Addresses (sample of {len(addr_sample):,}):")
print(f"    ALL UPPER:    {addr_upper:>7,}")
print(f"    all lower:    {addr_lower:>7,}")
print(f"    Title Case:   {addr_title:>7,}")
print(f"    MiXeD cAsE:   {addr_mixed:>7,}")


# ── MESS #4: Special Characters & Non-ASCII ──
print("\n" + "─" * 60)
print("🔍 MESS #4: SPECIAL CHARACTERS & NON-ASCII")
print("─" * 60)

special_patterns = {
    '& (ampersand)':     r'&',
    '"and" word':        r'\band\b',
    '# (hash/number)':   r'#',
    'No. / No ':         r'\bno\.?\s',
    '. (dots in names)': r'\.',
    '- (hyphens)':       r'-',
    '/ (slashes)':       r'/',
    '( ) parentheses':   r'[()]',
    '[ ] brackets':      r'[\[\]]',
    '<< >> angle':       r'[<>]',
    'Accented chars':    r'[àáâãäåèéêëìíîïòóôõöùúûüýÿñçÀÁÂÃÄÅÈÉÊËÌÍÎÏÒÓÔÕÖÙÚÛÜÝŸÑÇ]',
    'Hindi/Devanagari':  r'[\u0900-\u097F]',
    'Tamil':             r'[\u0B80-\u0BFF]',
    'Kannada':           r'[\u0C80-\u0CFF]',
    'Telugu':            r'[\u0C00-\u0C7F]',
    'Bengali':           r'[\u0980-\u09FF]',
    'Gujarati':          r'[\u0A80-\u0AFF]',
    'Malayalam':         r'[\u0D00-\u0D7F]',
}

combined_sample = name_sample[:50000] + addr_sample[:50000]
print(f"  Checking in combined name+address sample ({len(combined_sample):,}):")
for label, pat in special_patterns.items():
    count = sum(1 for s in combined_sample if re.search(pat, str(s)))
    if count > 0:
        print(f"    {label:25s} → {count:>7,} occurrences")


# ── MESS #5: Indian Landmarks & Location References ──
print("\n" + "─" * 60)
print("🔍 MESS #5: INDIAN LANDMARKS & LOCATION REFERENCES")
print("─" * 60)

landmark_patterns = {
    'Near ...':          r'\bnear\b',
    'Opp / Opposite':    r'\b(opp\.?|opposite)\b',
    'Behind':            r'\bbehind\b',
    'Adjacent / Adj':    r'\b(adjacent|adj\.?)\b',
    'Next to':           r'\bnext\s+to\b',
    'In front of':       r'\bin\s+front\s+of\b',
    'Beside':            r'\bbeside\b',
    'ATM':               r'\batm\b',
    'Hospital':          r'\bhospital\b',
    'Temple / Mandir':   r'\b(temple|mandir|mandiar)\b',
    'School':            r'\bschool\b',
    'Church':            r'\bchurch\b',
    'Mosque / Masjid':   r'\b(mosque|masjid)\b',
    'Post Office':       r'\bpost\s*office\b',
    'Police Station':    r'\bpolice\s*station\b',
    'Bus Stand/Stop':    r'\bbus\s*(stand|stop)\b',
    'Railway Station':   r'\brailway\s*station\b',
}

# Only check addresses
print(f"  Landmark patterns in address sample ({len(addr_sample):,}):")
for label, pat in landmark_patterns.items():
    count = sum(1 for a in addr_sample if re.search(pat, str(a), re.IGNORECASE))
    if count > 0:
        print(f"    {label:25s} → {count:>7,} occurrences")


# ── MESS #6: State Name Variations (India) ──
print("\n" + "─" * 60)
print("🔍 MESS #6: INDIAN STATE NAME VARIATIONS")
print("─" * 60)

state_patterns = {
    'Maharashtra / MH':      (r'\bmaharashtra\b', r'\bMH\b'),
    'Karnataka / KA':        (r'\bkarnataka\b', r'\bKA\b'),
    'Tamil Nadu / TN':       (r'\btamil\s*nadu\b', r'\bTN\b'),
    'Delhi / DL':            (r'\bdelhi\b', r'\bDL\b'),
    'Gujarat / GJ':          (r'\bgujarat\b', r'\bGJ\b'),
    'Rajasthan / RJ':        (r'\brajasthan\b', r'\bRJ\b'),
    'Uttar Pradesh / UP':    (r'\buttar\s*pradesh\b', r'\bUP\b'),
    'West Bengal / WB':      (r'\bwest\s*bengal\b', r'\bWB\b'),
    'Telangana / TG':        (r'\btelangana\b', r'\bTG\b'),
    'Andhra Pradesh / AP':   (r'\bandhra\s*pradesh\b', r'\bAP\b'),
    'Madhya Pradesh / MP':   (r'\bmadhya\s*pradesh\b', r'\bMP\b'),
    'Kerala / KL':           (r'\bkerala\b', r'\bKL\b'),
    'Haryana / HR':          (r'\bharyana\b', r'\bHR\b'),
    'Orissa/Odisha':         (r'\b(orissa|odisha)\b', None),
}

print(f"  State name forms in address sample:")
for label, pats in state_patterns.items():
    full_pat, abbr_pat = pats
    full_count = sum(1 for a in addr_sample if re.search(full_pat, str(a), re.IGNORECASE))
    if abbr_pat:
        abbr_count = sum(1 for a in addr_sample if re.search(abbr_pat, str(a)))
        print(f"    {label:30s} → Full: {full_count:>6,}  |  Code: {abbr_count:>6,}")
    else:
        print(f"    {label:30s} → Hits: {full_count:>6,}")


# ── MESS #7: US State Variations ──
print("\n" + "─" * 60)
print("🔍 MESS #7: US STATE NAME VARIATIONS")
print("─" * 60)

us_state_patterns = {
    'California / CA':   (r'\bcalifornia\b', r'\bCA\b'),
    'Texas / TX':        (r'\btexas\b', r'\bTX\b'),
    'New York / NY':     (r'\bnew\s*york\b', r'\bNY\b'),
    'Florida / FL':      (r'\bflorida\b', r'\bFL\b'),
    'North Carolina/NC': (r'\bnorth\s*carolina\b', r'\bNC\b'),
    'Virginia / VA':     (r'\bvirginia\b', r'\bVA\b'),
    'Ohio / OH':         (r'\bohio\b', r'\bOH\b'),
    'Illinois / IL':     (r'\billinois\b', r'\bIL\b'),
    'Tennessee / TN':    (r'\btennessee\b', r'\bTN\b'),
    'Georgia / GA':      (r'\bgeorgia\b', r'\bGA\b'),
}

print(f"  US state forms in address sample:")
for label, pats in us_state_patterns.items():
    full_pat, abbr_pat = pats
    full_count = sum(1 for a in addr_sample if re.search(full_pat, str(a), re.IGNORECASE))
    abbr_count = sum(1 for a in addr_sample if re.search(abbr_pat, str(a)))
    if full_count + abbr_count > 0:
        print(f"    {label:25s} → Full: {full_count:>6,}  |  Code: {abbr_count:>6,}")


# ── MESS #8: French Patterns (TEST SET SURPRISE!) ──
print("\n" + "─" * 60)
print("🔍 MESS #8: FRENCH PATTERNS (Test set has France!)")
print("─" * 60)

french_patterns = {
    'Rue':               r'\brue\b',
    'Boulevard / Bd':    r'\b(boulevard|bd\.?)\b',
    'Avenue / Av':       r'\b(avenue|av\.?)\b',
    'Allée':             r'\ball[ée]e?\b',
    'Place':             r'\bplace\b',
    'Impasse':           r'\bimpasse\b',
    'Chemin':            r'\bchemin\b',
    'Hauts-de-France':   r'hauts.de.france',
    'Nouvelle-Aquitaine':r'nouvelle.aquitaine',
    'SARL':              r'\bsarl\b',
    'SAS / S.A.S':       r'\bs\.?a\.?s\.?\b',
    'SASU':              r'\bsasu\b',
    'SCI':               r'\bsci\b',
    'Accented chars':    r'[àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ]',
}

# Check in test data (where France appears)
test_all = list(test_s1['business_name'].dropna()) + list(test_s2['business_name'].dropna()) + list(test_s3['business_name'].dropna())
test_addr = list(test_s1['business_address'].dropna()) + list(test_s2['business_address'].dropna()) + list(test_s3['business_address'].dropna())
test_combined = test_all[:50000] + test_addr[:50000]

print(f"  French patterns in test data sample ({len(test_combined):,}):")
for label, pat in french_patterns.items():
    count = sum(1 for s in test_combined if re.search(pat, str(s), re.IGNORECASE))
    if count > 0:
        print(f"    {label:25s} → {count:>7,} occurrences")


# ── MESS #9: Address Component Ordering Chaos ──
print("\n" + "─" * 60)
print("🔍 MESS #9: ADDRESS COMPONENT ORDERING ISSUES")
print("─" * 60)

print("  Examples of DIFFERENT orderings (from actual data):")
print()

# Show some S1 examples with weird ordering
for _, row in test_s1.head(100).iterrows():
    addr = str(row.get('business_address', ''))
    name = str(row.get('business_name', ''))
    country = str(row.get('country', ''))

    # Find ones where state/city comes BEFORE the street (US pattern)
    if country == 'US' and re.match(r'^[A-Z]{2},', addr):
        print(f"    [{row['entity_id']}] State-first: \"{addr}\"")
    # Find ones where city comes before street (India pattern)
    if country == 'India' and ',' in addr:
        parts = addr.split(',')
        if len(parts) >= 3:
            # Just show a few interesting ones
            if any(keyword in parts[0].lower() for keyword in ['mumbai', 'delhi', 'kolkata', 'chennai', 'bangalore']):
                print(f"    [{row['entity_id']}] City-first: \"{addr[:80]}...\"")

# Show specific examples from the data we already saw
print()
print("  Concrete examples from the data:")
print("    S1: 'OH, Columbus, 5559 Orville Avenue'  ← State first!")
print("    S1: 'Charlotte, NC, 833 Reliance Street'  ← City, State, then Street!")
print("    S1: 'Unit BUILDING 3030, MD, 2701 Eastern Boulevard, Middle River'")
print("    Normal: '1795 Westchester Drive, High Point, NC'")


# ── MESS #10: Typos & Misspellings ──
print("\n" + "─" * 60)
print("🔍 MESS #10: TYPOS & MISSPELLINGS (from actual test data)")
print("─" * 60)

# Show some clear typos from S2/S3 we already spotted
print("  Examples spotted in the data:")
print("    S2: 'VIRGINIA BBEACH'    → should be 'Virginia Beach'")
print("    S2: 'AMMARILLO'          → should be 'Amarillo'")
print("    S3: 'Hurricanne'         → should be 'Hurricane'")
print("    S3: 'Haltom City'        → could be misspelling")
print("    S2: 'Dmaigesostcis'      → scrambled/typo")
print("    S2: 'Synaighsoue'        → scrambled/typo")
print("    S3: 'Connmre'            → typo")
print()
print("  ⚠️  NOTE: You can't fix ALL typos with rules.")
print("     Your similarity matching model handles most of these.")
print("     YOUR job is to fix the SYSTEMATIC patterns (abbreviations, suffixes, etc.)")


# ── MESS #11: Mixed Script Names ──
print("\n" + "─" * 60)
print("🔍 MESS #11: MIXED SCRIPT NAMES (Hindi/Tamil/etc + English)")
print("─" * 60)

mixed_script_count = 0
mixed_examples = []
for s in name_sample[:50000]:
    s_str = str(s)
    has_ascii = bool(re.search(r'[a-zA-Z]', s_str))
    has_devanagari = bool(re.search(r'[\u0900-\u097F]', s_str))
    has_tamil = bool(re.search(r'[\u0B80-\u0BFF]', s_str))
    has_kannada = bool(re.search(r'[\u0C80-\u0CFF]', s_str))
    has_indic = has_devanagari or has_tamil or has_kannada
    if has_ascii and has_indic:
        mixed_script_count += 1
        if len(mixed_examples) < 5:
            mixed_examples.append(s_str)

print(f"  Mixed-script business names: {mixed_script_count:,} (out of {min(50000, len(name_sample)):,} sampled)")
if mixed_examples:
    print("  Examples:")
    for ex in mixed_examples:
        print(f"    \"{ex}\"")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("📊 SUMMARY: THE REAL MESS YOU NEED TO HANDLE")
print("=" * 80)
print("""
YOUR NORMALIZATION TARGETS (in priority order):

1. CAPITALIZATION
   → Lowercase everything for comparison
   → "HALOCAST" = "Halocast" = "halocast"

2. LEGAL SUFFIXES (Business Names)
   → Private Limited / Pvt Ltd / Pvt. Ltd. / P. Ltd → "private limited"
   → LLC / L.L.C. → "llc"  
   → Inc / Incorporated / Inc. → "inc"
   → Corp / Corporation → "corp"
   → Ltd / Limited → "limited"
   → SARL / SAS / SASU / SCI / S.A. (French) → standardize

3. ADDRESS ABBREVIATIONS
   → Street/St/St. → "street"
   → Road/Rd/Rd. → "road"
   → Avenue/Ave → "avenue"
   → Boulevard/Blvd → "boulevard"
   → Drive/Dr → "drive"
   → Floor/Fl → "floor"
   → Building/Bldg → "building"

4. STATE NAMES
   → India: Maharashtra/MH, Karnataka/KA, Tamil Nadu/TN, etc.
   → US: California/CA, Texas/TX, New York/NY, etc.
   → France: Hauts-de-France, Nouvelle-Aquitaine, etc.
   → Pick ONE canonical form and map everything to it

5. INDIAN LANDMARK HANDLING
   → "Near SBI ATM" / "Opp. Railway Station" / "Behind Temple"
   → Don't DELETE them (they help matching!)
   → But standardize: "opp." → "opposite", "nr." → "near"

6. PUNCTUATION & SPECIAL CHARS
   → Remove: . , # << >> ( ) [ ]
   → Normalize: & → "and"
   → Collapse extra whitespace

7. FRENCH ACCENTED CHARACTERS
   → é/è/ê/ë → e
   → à/â → a  
   → ù/û/ü → u
   → ç → c
   → î/ï → i

8. NON-LATIN SCRIPTS
   → Keep them (they might help matching within same language)
   → But also keep the Latin part for cross-source matching

9. ADDRESS ORDERING
   → This is HARD to fix with regex
   → Best handled by your similarity/blocking teammates
   → Your job: normalize the individual TOKENS, not reorder them
""")

print("Script complete! ✅")
print("Next step: Run this, review the output, then we build normalization.py")
