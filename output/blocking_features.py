"""
=============================================================================
blocking_features.py  -  FEATURE EXTRACTION & BLOCKING KEYS
=============================================================================
Role:  Data & Features  (Member 3, Pod 2)
Task:  Extract blocking keys & features from normalized records to enable
       high-speed, high-precision candidate generation for Pod 2.

Public API:
    extract_postal_code(address, country)   -> string (5-digit ZIP / 6-digit PIN)
    extract_state_code(address_clean, country) -> 2-letter code (e.g. 'ca', 'mh', 'wb')
    extract_significant_tokens(name_clean) -> list of core business words
    get_blocking_keys(row)                 -> set of blocking keys for indexing
    generate_candidate_pairs(df_s1, df_s2, df_s3) -> DataFrame of candidate pairs

Strategy (F_0.5 Metric Alignment):
    - Hard country isolation: never cross US, India, or France.
    - Multi-pass blocking: captures candidates even if postal code or state is missing.
    - Precision protection: ignores ubiquitous legal terms (llc, inc, pvt, ltd)
      when creating name-based buckets to prevent massive oversized buckets.
=============================================================================
"""

import re
import pandas as pd
from normalization import (
    INDIAN_STATE_MAP, US_STATE_MAP, FRENCH_REGION_MAP,
    normalize_name, normalize_address
)

# Stopwords, generic business words, and honorifics that should NOT serve as blocking keys
NAME_STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'of', 'for', 'in', 'at', 'by', 'with', 'to',
    'private', 'limited', 'pvt', 'ltd', 'inc', 'incorporated', 'corp', 'corporation',
    'llc', 'llp', 'co', 'company', 'sarl', 'sas', 'sasu', 'sci', 'sa', 'pc', 'pllc',
    'dba', 'group', 'holdings', 'services', 'solutions', 'enterprises', 'associates',
    'shri', 'smt', 'm/s', 'mr', 'mrs', 'ms', 'dr', 'prof'
}


def extract_postal_code(address, country):
    """
    Extract postal code from address with disambiguation.

    US:     5-digit ZIP code (near end or preceded by state/comma).
    India:  6-digit PIN code (ignores Plot/Khasra/Survey numbers).
    France: 5-digit postal code (01xxx to 98xxx).
    """
    if not address or not isinstance(address, str) or not country:
        return ''

    addr = address.strip()

    if country == 'US':
        # 5 digits followed optionally by 4 digits, at end or preceded by 2-letter state
        m = re.search(r'(?:,\s*[A-Za-z]{2}\s+|\b[A-Za-z]{2}\s+)(\d{5})(?:-\d{4})?\b', addr)
        if m:
            return m.group(1)
        # End of address fallback
        m = re.search(r'\b(\d{5})(?:-\d{4})?\s*$', addr)
        if m:
            return m.group(1)
        return ''

    elif country == 'India':
        # Priority 1: explicitly labelled pin code
        m = re.search(r'\b(?:pin|pincode|pin\s*code)\s*[-:]?\s*([1-9]\d{5})\b', addr, re.IGNORECASE)
        if m:
            return m.group(1)

        # Priority 2: find all 6-digit numbers not preceded by plot/khasra/survey/phone
        matches = list(re.finditer(r'\b([1-9]\d{5})\b', addr))
        if not matches:
            return ''

        valid = []
        for m_obj in matches:
            before = addr[:m_obj.start()].lower().strip()
            if re.search(r'\b(?:plot\s*no\.?|h\.?\s*no\.?|kh\.?\s*no\.?|khasra|flat\s*no\.?|survey|sr\s*no\.?|ph\s*no\.?|mobile|tel|phone|contact)\s*[-:]?\s*$', before):
                continue
            valid.append(m_obj.group(1))

        if valid:
            return valid[-1]  # Return the last valid one (PIN is near end)
        return ''

    elif country == 'France':
        m = re.search(r'\b(0[1-9]\d{3}|[1-8]\d{4}|9[0-8]\d{3})\b', addr)
        if m:
            return m.group(1)
        return ''

    return ''


def extract_state_code(address_clean, country):
    """
    Extract canonical 2-letter state code or region name from normalized address.
    """
    if not address_clean or not country:
        return ''

    text = address_clean.lower().strip()

    if country == 'US':
        valid_codes = set(US_STATE_MAP.values())
        # Check tokens from end of address backwards
        segments = [s.strip() for s in text.split(',')]
        for seg in reversed(segments):
            tokens = seg.split()
            for tok in reversed(tokens):
                if tok in valid_codes:
                    return tok
        return ''

    elif country == 'India':
        valid_codes = set(INDIAN_STATE_MAP.values())
        segments = [s.strip() for s in text.split(',')]
        for seg in reversed(segments):
            tokens = seg.split()
            for tok in reversed(tokens):
                if tok in valid_codes:
                    return tok
        return ''

    elif country == 'France':
        for region in sorted(FRENCH_REGION_MAP.values(), key=len, reverse=True):
            if region in text:
                return region
        return ''

    return ''


def extract_significant_tokens(name_clean):
    """
    Extract non-generic, significant business name tokens.
    Filters out legal forms and stop words.
    """
    if not name_clean or not isinstance(name_clean, str):
        return []

    words = re.findall(r'[\w]+', name_clean.lower())
    significant = [w for w in words if len(w) > 1 and w not in NAME_STOPWORDS]
    return significant


def get_blocking_keys(row):
    """
    Generate multiple blocking keys for a single record row.
    Row should contain: country, name_clean, address_clean.

    Returns
    -------
    list[str]
        Unique blocking key strings for this record.
    """
    country = row.get('country', '')
    name_clean = row.get('name_clean', '')
    addr_clean = row.get('address_clean', '')

    if not country:
        return []

    state = extract_state_code(addr_clean, country)
    pin = extract_postal_code(addr_clean, country)
    tokens = extract_significant_tokens(name_clean)

    keys = []
    first_tok = tokens[0] if tokens else ''
    first_3 = first_tok[:3] if len(first_tok) >= 3 else first_tok

    # Pass 1: State + First Significant Token (High Precision)
    if state and first_tok:
        keys.append(f"{country}_ST_{state}_{first_tok}")

    # Pass 2: Postal Code + First Significant Token Prefix (High Precision)
    if pin and first_3:
        keys.append(f"{country}_PIN_{pin}_{first_3}")

    # Pass 3: State + First 3 Chars of Name (Handles minor name variations in same state)
    if state and first_3:
        keys.append(f"{country}_ST_PRE3_{state}_{first_3}")

    # Pass 4: State + Second Significant Token (Handles prefix additions like new/north/south/city)
    if state and len(tokens) >= 2:
        second_tok = tokens[1]
        if len(second_tok) >= 3:
            keys.append(f"{country}_ST_{state}_{second_tok}")

    # Pass 5: State + Distinct Address Token (Recovers cross-script or heavily altered names)
    if state:
        addr_tokens = [
            t for t in re.findall(r'[a-zA-Z0-9]+', addr_clean.lower())
            if len(t) >= 5 and t not in {
                'road', 'street', 'floor', 'lane', 'avenue', 'near',
                'opposite', 'behind', 'cross', 'main', 'block', 'phase',
                'nagar', 'colony', 'complex', 'building', 'house', 'first',
                'second', 'third', 'ground'
            }
        ]
        for t in addr_tokens[:2]:
            keys.append(f"{country}_ADDR_{state}_{t}")

    # Pass 6: Full First Token (Fallback when address/state is completely missing)
    if first_tok and len(first_tok) >= 4:
        keys.append(f"{country}_TOK_{first_tok}")

    return keys


# ═══════════════════════════════════════════════════════════════════════════════
#  SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 72)
    print("  BLOCKING FEATURES SELF-TEST")
    print("=" * 72)

    sample_cases = [
        {
            'country': 'India',
            'business_name': 'Halocast Private Limited',
            'business_address': 'Plot No 45, Lake Road, Kolkata, 700029, West Bengal'
        },
        {
            'country': 'US',
            'business_name': 'Custom Wealth Services LLC',
            'business_address': '500 Market St, San Jose, CA 95113'
        },
        {
            'country': 'France',
            'business_name': 'ZNB Club SARL',
            'business_address': '18 Rue Jen Zay, 59140 Dunkerque, Hauts-de-France'
        }
    ]

    for item in sample_cases:
        c = item['country']
        n_clean = normalize_name(item['business_name'], c)
        a_clean = normalize_address(item['business_address'], c)
        row = {'country': c, 'name_clean': n_clean, 'address_clean': a_clean}

        pin = extract_postal_code(a_clean, c)
        state = extract_state_code(a_clean, c)
        tokens = extract_significant_tokens(n_clean)
        keys = get_blocking_keys(row)

        print(f"\nCountry: {c}")
        print(f"  Raw Name:      {item['business_name']}")
        print(f"  Clean Name:    {n_clean}")
        print(f"  Tokens:        {tokens}")
        print(f"  Clean Address: {a_clean}")
        print(f"  Extracted State: {state}")
        print(f"  Extracted Postal: {pin}")
        print(f"  Blocking Keys: {keys}")

    print("\n" + "=" * 72)
    print("  SELF-TEST COMPLETE")
    print("=" * 72)
