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

# Generic address words that create oversized, low-entropy buckets
ADDR_STOPWORDS = {
    'road', 'street', 'floor', 'lane', 'avenue', 'near', 'opposite', 'behind', 'cross', 'main', 'block', 'phase',
    'nagar', 'colony', 'complex', 'building', 'house', 'first', 'second', 'third', 'ground', 'center', 'centre', 'plaza',
    'tower', 'towers', 'market', 'sector', 'industrial', 'area', 'estate', 'park', 'plot', 'point', 'circle', 'highway',
    'hwy', 'expressway', 'bldg', 'flr', 'suite', 'ste', 'room', 'door', 'dept', 'office', 'unit', 'bazaar', 'chowk', 'rasta',
    'gali', 'marg', 'puram', 'city', 'town', 'village', 'state', 'india', 'delhi', 'mumbai', 'bangalore', 'pune', 'chennai',
    'hyderabad', 'kolkata', 'ahmedabad', 'jaipur', 'surat', 'kanpur'
}

# Fast Indic character phonetic translation table (Devanagari, Telugu, Tamil, Bengali, Gujarati)
INDIC_MAP = {
    # Devanagari (\u0900-\u097F)
    ord('अ'): 'a', ord('आ'): 'a', ord('इ'): 'i', ord('ई'): 'i', ord('उ'): 'u', ord('ऊ'): 'u', ord('ए'): 'e', ord('ऐ'): 'ai', ord('ओ'): 'o', ord('औ'): 'au',
    ord('क'): 'k', ord('ख'): 'kh', ord('ग'): 'g', ord('घ'): 'gh', ord('च'): 'ch', ord('छ'): 'ch', ord('ज'): 'j', ord('झ'): 'jh',
    ord('ट'): 't', ord('ठ'): 'th', ord('ड'): 'd', ord('ढ'): 'dh', ord('ण'): 'n', ord('त'): 't', ord('थ'): 'th', ord('द'): 'd', ord('ध'): 'dh', ord('न'): 'n',
    ord('प'): 'p', ord('फ'): 'ph', ord('ब'): 'b', ord('भ'): 'bh', ord('म'): 'm', ord('य'): 'y', ord('र'): 'r', ord('ल'): 'l', ord('व'): 'v', ord('श'): 'sh', ord('ष'): 'sh', ord('स'): 's', ord('ह'): 'h',
    ord('ा'): 'a', ord('ि'): 'i', ord('ी'): 'i', ord('ु'): 'u', ord('ू'): 'u', ord('े'): 'e', ord('ै'): 'ai', ord('ो'): 'o', ord('ौ'): 'au', ord('्'): '', ord('ं'): 'n',
    # Telugu (\u0C00-\u0C7F)
    ord('అ'): 'a', ord('ఆ'): 'a', ord('ఇ'): 'i', ord('ఈ'): 'i', ord('ఉ'): 'u', ord('ఊ'): 'u', ord('ఎ'): 'e', ord('ఏ'): 'e', ord('ఐ'): 'ai', ord('ఒ'): 'o', ord('ఓ'): 'o', ord('ఔ'): 'au',
    ord('క'): 'k', ord('ఖ'): 'kh', ord('గ'): 'g', ord('ఘ'): 'gh', ord('చ'): 'ch', ord('ఛ'): 'ch', ord('జ'): 'j', ord('ఝ'): 'jh',
    ord('ట'): 't', ord('ఠ'): 'th', ord('డ'): 'd', ord('ఢ'): 'dh', ord('ణ'): 'n', ord('త'): 't', ord('థ'): 'th', ord('ద'): 'd', ord('ధ'): 'dh', ord('న'): 'n',
    ord('ప'): 'p', ord('ఫ'): 'ph', ord('బ'): 'b', ord('భ'): 'bh', ord('మ'): 'm', ord('య'): 'y', ord('ర'): 'r', ord('ల'): 'l', ord('వ'): 'v', ord('శ'): 'sh', ord('ష'): 'sh', ord('స'): 's', ord('హ'): 'h',
    ord('ా'): 'a', ord('ి'): 'i', ord('ీ'): 'i', ord('ు'): 'u', ord('ూ'): 'u', ord('ె'): 'e', ord('ే'): 'e', ord('ై'): 'ai', ord('ొ'): 'o', ord('ో'): 'o', ord('ౌ'): 'au', ord('్'): '', ord('ం'): 'n',
    # Tamil (\u0B80-\u0BFF)
    ord('அ'): 'a', ord('ஆ'): 'a', ord('இ'): 'i', ord('ஈ'): 'i', ord('உ'): 'u', ord('ஊ'): 'u', ord('எ'): 'e', ord('ஏ'): 'e', ord('ஐ'): 'ai', ord('ஒ'): 'o', ord('ஓ'): 'o', ord('ஔ'): 'au',
    ord('க'): 'k', ord('ங'): 'ng', ord('ச'): 's', ord('ஞ'): 'ny', ord('ட'): 't', ord('ண'): 'n', ord('த'): 't', ord('ந'): 'n', ord('ப'): 'p', ord('ம'): 'm',
    ord('ய'): 'y', ord('ர'): 'r', ord('ல'): 'l', ord('வ'): 'v', ord('ழ'): 'zh', ord('ள'): 'l', ord('ற'): 'r', ord('ன'): 'n',
    ord('ா'): 'a', ord('ி'): 'i', ord('ீ'): 'i', ord('ு'): 'u', ord('ூ'): 'u', ord('ெ'): 'e', ord('ே'): 'e', ord('ை'): 'ai', ord('ொ'): 'o', ord('ோ'): 'o', ord('ௌ'): 'au', ord('்'): '',
    # Bengali (\u0980-\u09FF)
    ord('অ'): 'a', ord('আ'): 'a', ord('ই'): 'i', ord('ঈ'): 'i', ord('উ'): 'u', ord('ঊ'): 'u', ord('এ'): 'e', ord('ঐ'): 'ai', ord('ও'): 'o', ord('ঔ'): 'au',
    ord('ক'): 'k', ord('খ'): 'kh', ord('গ'): 'g', ord('ঘ'): 'gh', ord('চ'): 'ch', ord('ছ'): 'ch', ord('জ'): 'j', ord('ঝ'): 'jh',
    ord('ট'): 't', ord('ঠ'): 'th', ord('ড'): 'd', ord('ঢ'): 'dh', ord('ণ'): 'n', ord('ত'): 't', ord('থ'): 'th', ord('দ'): 'd', ord('ধ'): 'dh', ord('ন'): 'n',
    ord('প'): 'p', ord('ফ'): 'ph', ord('ব'): 'b', ord('ভ'): 'bh', ord('ম'): 'm', ord('য'): 'y', ord('র'): 'r', ord('ল'): 'l', ord('শ'): 'sh', ord('ষ'): 'sh', ord('স'): 's', ord('হ'): 'h',
    ord('া'): 'a', ord('ి'): 'i', ord('ী'): 'i', ord('ু'): 'u', ord('ূ'): 'u', ord('ে'): 'e', ord('ৈ'): 'ai', ord('ো'): 'o', ord('ৌ'): 'au', ord('্'): '', ord('ং'): 'n',
    # Gujarati (\u0A80-\u0AFF)
    ord('અ'): 'a', ord('આ'): 'a', ord('ઇ'): 'i', ord('ઈ'): 'i', ord('ઉ'): 'u', ord('ઊ'): 'u', ord('એ'): 'e', ord('ઐ'): 'ai', ord('ઓ'): 'o', ord('ઔ'): 'au',
    ord('ક'): 'k', ord('ખ'): 'kh', ord('ગ'): 'g', ord('ઘ'): 'gh', ord('ચ'): 'ch', ord('છ'): 'ch', ord('જ'): 'j', ord('ઝ'): 'jh',
    ord('ટ'): 't', ord('ઠ'): 'th', ord('ડ'): 'd', ord('ઢ'): 'dh', ord('ણ'): 'n', ord('ત'): 't', ord('થ'): 'th', ord('દ'): 'd', ord('ધ'): 'dh', ord('ન'): 'n',
    ord('પ'): 'p', ord('ફ'): 'ph', ord('બ'): 'b', ord('ભ'): 'bh', ord('મ'): 'm', ord('ય'): 'y', ord('ર'): 'r', ord('લ'): 'l', ord('વ'): 'v', ord('શ'): 'sh', ord('ષ'): 'sh', ord('સ'): 's', ord('હ'): 'h',
    ord('ા'): 'a', ord('િ'): 'i', ord('ી'): 'i', ord('ુ'): 'u', ord('ૂ'): 'u', ord('ે'): 'e', ord('ૈ'): 'ai', ord('ો'): 'o', ord('ૌ'): 'au', ord('્'): '', ord('ં'): 'n',
}

def transliterate_indic(text):
    """Fast Indic script to Latin transliteration via Unicode mapping."""
    if not text:
        return ''
    return text.translate(INDIC_MAP)


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
    Filters out legal forms, stop words, and transliterates Indic characters.
    """
    if not name_clean or not isinstance(name_clean, str):
        return []

    # Transliterate Indic characters if present
    trans = transliterate_indic(name_clean)
    words = re.findall(r'[\w]+', trans.lower())
    significant = [w for w in words if len(w) > 1 and w not in NAME_STOPWORDS]
    return significant


def get_blocking_keys(row):
    """
    Generate multiple independent blocking keys for a single record row.
    Row should contain: country, name_clean (or business_name), address_clean.

    Returns
    -------
    list[str]
        Unique blocking key strings for this record.
    """
    country = row.get('country', '')
    if not country:
        return []

    name_clean = row.get('name_clean', '')
    raw_name = row.get('business_name', '')
    addr_clean = row.get('address_clean', '')

    target_name = name_clean if name_clean else raw_name
    tokens = extract_significant_tokens(target_name)
    state = extract_state_code(addr_clean, country)
    pin = extract_postal_code(addr_clean, country)

    keys = []
    first_tok = tokens[0] if tokens else ''
    first_3 = first_tok[:3] if len(first_tok) >= 3 else first_tok

    # Pass 1: State + First Significant Token (High Precision)
    if state and first_tok:
        keys.append(f"{country}_ST_{state}_{first_tok}")

    # Pass 2: State + Second Significant Token (Handles prefix additions)
    if state and len(tokens) >= 2 and len(tokens[1]) >= 3:
        keys.append(f"{country}_ST_{state}_{tokens[1]}")

    # Pass 3: Two-Token Conjunction (Handles missing state & reordered tokens!)
    if len(tokens) >= 2:
        pair = sorted([tokens[0], tokens[1]])
        keys.append(f"{country}_2TOK_{pair[0]}_{pair[1]}")

    # Pass 4: State + First 3 Chars of Name (Pure Selective PRE3)
    if state and len(first_tok) >= 3:
        keys.append(f"{country}_ST_PRE3_{state}_{first_tok[:3]}")

    # Pass 5: State + Distinct Address Token (Selective filtering with ADDR_STOPWORDS)
    if state:
        addr_tokens = [
            t for t in re.findall(r'[a-zA-Z0-9]+', addr_clean.lower())
            if len(t) >= 5 and t not in ADDR_STOPWORDS
        ]
        for t in addr_tokens[:2]:
            keys.append(f"{country}_ADDR_{state}_{t}")

    # Pass 6: Postal Code + First Token Prefix
    if pin and first_3:
        keys.append(f"{country}_PIN_{pin}_{first_3}")

    # Pass 7: Fallback when state is missing: single significant token (len >= 4)
    if not state and first_tok and len(first_tok) >= 4:
        keys.append(f"{country}_TOK_{first_tok}")

    return keys


def generate_candidate_pairs(
    df_s1,
    df_s2,
    df_s3,
    bucket_cap=25,
    max_candidates=12,
    selective_large_bucket=True,
    large_bucket_cap=200
):
    """
    Generate candidate pairs between S1 reference entities and (S2, S3) targets
    using multi-branch inverted index, selective large-bucket recovery, and
    internal candidate priority ranking before truncation.

    Priority score components:
    - Number of independent blocking branches that retrieved the candidate
    - Learned branch reliability weights (state+token, 2tok_conjunction, state+addr_token)
    - Multi-branch consensus bonus
    - Bucket specificity (inverse bucket frequency)
    - Cheap token-level name similarity & overlap
    - Cheap token-level address similarity

    Parameters
    ----------
    df_s1 : pd.DataFrame
        Source 1 records (with entity_id, country, and normalized/raw fields).
    df_s2 : pd.DataFrame
        Source 2 records.
    df_s3 : pd.DataFrame
        Source 3 records.
    bucket_cap : int
        Maximum size of an index bucket to expand unconditionally. Default 25.
    max_candidates : int, optional
        Maximum candidates to retain per S1 entity (default 12).
    selective_large_bucket : bool
        If True, retrieves top candidates from buckets up to large_bucket_cap
        using cheap token pre-filtering instead of discarding the bucket.
    large_bucket_cap : int
        Upper cap for selective large-bucket retrieval. Default 200.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['source1_entity_id', 'candidate_entity_ids']
        where candidate_entity_ids is a comma-separated string of IDs.
    """
    import math
    from collections import defaultdict

    BRANCH_WEIGHTS = {
        'state+token': 4.0,
        '2tok_conjunction': 4.0,
        'state+addr_token': 3.5,
        'state+second_token': 2.5,
        'pin+prefix': 2.0,
        'country+token_fallback': 2.0,
        'state+prefix3': 1.5,
        'state+prefix4': 1.5,
        'other': 1.0,
    }

    # Helper to extract lightweight token features for rapid scoring
    def _extract_record_meta(row):
        country = row.get('country', '')
        name_clean = row.get('name_clean', '')
        raw_name = row.get('business_name', '')
        addr_clean = row.get('address_clean', '')

        target_name = name_clean if name_clean else raw_name
        name_toks = extract_significant_tokens(target_name)
        addr_toks = [
            t for t in re.findall(r'[a-zA-Z0-9]+', addr_clean.lower())
            if len(t) >= 4 and t not in ADDR_STOPWORDS
        ]
        return {
            'entity_id': row['entity_id'],
            'country': country,
            'name_tok_set': set(name_toks),
            'addr_tok_set': set(addr_toks),
        }

    # 1. Index targets (S2 and S3)
    block_index = defaultdict(list)
    target_meta_map = {}

    for df_target in [df_s2, df_s3]:
        for row in df_target.to_dict('records'):
            eid = row['entity_id']
            target_meta_map[eid] = _extract_record_meta(row)
            for k in get_blocking_keys(row):
                block_index[k].append(eid)

    bucket_sizes = {k: len(v) for k, v in block_index.items()}

    # 2. Query candidates for each S1 entity with branch tracking & scoring
    s1_ids = []
    cand_lists = []

    s1_records = df_s1.to_dict('records')
    for row in s1_records:
        s1_id = row['entity_id']
        s1_meta = _extract_record_meta(row)
        s1_name_toks = s1_meta['name_tok_set']
        s1_addr_toks = s1_meta['addr_tok_set']
        keys = get_blocking_keys(row)

        cand_meta = defaultdict(lambda: {'branches': set(), 'spec': 0.0})

        for k in keys:
            if k not in block_index:
                continue
            bsize = bucket_sizes[k]

            # Pure Selective PRE3: admission cap 8, large bucket threshold 100, top 1
            if '_ST_PRE3_' in k:
                btype = 'state+prefix3'
                eff_cap = 8
                eff_large_cap = 100
                top_take = 1
            else:
                eff_cap = bucket_cap
                eff_large_cap = large_bucket_cap
                top_take = 2
                if '_ST_' in k:
                    btype = 'state+token'
                elif '_2TOK_' in k:
                    btype = '2tok_conjunction'
                elif '_ADDR_' in k:
                    btype = 'state+addr_token'
                elif '_PIN_' in k:
                    btype = 'pin+prefix'
                elif '_TOK_' in k:
                    btype = 'country+token_fallback'
                else:
                    btype = 'other'

            if bsize <= eff_cap:
                spec = 1.0 / math.sqrt(bsize)
                for cid in block_index[k]:
                    cm = cand_meta[cid]
                    cm['branches'].add(btype)
                    cm['spec'] += spec
            elif selective_large_bucket and bsize <= eff_large_cap:
                # Selective large-bucket retrieval: filter candidates by token overlap
                large_matches = []
                for cid in block_index[k]:
                    t_cand = target_meta_map.get(cid)
                    if not t_cand:
                        continue
                    n_overlap = len(s1_name_toks & t_cand['name_tok_set'])
                    a_overlap = len(s1_addr_toks & t_cand['addr_tok_set'])
                    if n_overlap >= 1 or a_overlap >= 2:
                        large_matches.append((n_overlap * 2.0 + a_overlap, cid))
                large_matches.sort(reverse=True)
                for _, cid in large_matches[:top_take]:
                    cm = cand_meta[cid]
                    cm['branches'].add(btype)
                    cm['spec'] += 0.1

        if not cand_meta:
            s1_ids.append(s1_id)
            cand_lists.append("")
            continue

        # Score all collected candidates
        scored_cands = []
        for cid, meta in cand_meta.items():
            t_cand = target_meta_map.get(cid)
            if not t_cand:
                continue

            # Branch reliability weight + consensus bonus
            b_score = sum(BRANCH_WEIGHTS.get(b, 1.0) for b in meta['branches'])
            if len(meta['branches']) >= 2:
                b_score += 3.0 * (len(meta['branches']) - 1)

            # Bucket specificity score
            spec_score = min(3.0, meta['spec'])

            # Cheap Name Similarity (Jaccard + overlap)
            t_name_toks = t_cand['name_tok_set']
            inter_n = len(s1_name_toks & t_name_toks)
            union_n = len(s1_name_toks | t_name_toks)
            name_jaccard = inter_n / max(1, union_n)
            name_score = name_jaccard * 4.0 + inter_n * 1.5

            # Cheap Address Similarity (Jaccard + overlap)
            t_addr_toks = t_cand['addr_tok_set']
            inter_a = len(s1_addr_toks & t_addr_toks)
            union_a = len(s1_addr_toks | t_addr_toks)
            addr_jaccard = inter_a / max(1, union_a)
            addr_score = addr_jaccard * 2.5 + inter_a * 0.8

            total_score = b_score + spec_score + name_score + addr_score
            scored_cands.append((total_score, cid))

        # Rank candidates descending by priority score
        scored_cands.sort(reverse=True)

        if max_candidates is not None and len(scored_cands) > max_candidates:
            final_cands = [cid for _, cid in scored_cands[:max_candidates]]
        else:
            final_cands = [cid for _, cid in scored_cands]

        s1_ids.append(s1_id)
        cand_lists.append(','.join(final_cands))

    return pd.DataFrame({
        'source1_entity_id': s1_ids,
        'candidate_entity_ids': cand_lists
    })


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
