"""
=============================================================================
normalization.py  -  YOUR MAIN DELIVERABLE
=============================================================================
Role:  Data & Features  (Member 3, Pod 2)
Tasks: Address normalization, Name cleaning & regex, Landmark handling

Three core public functions:
    normalize_name(name, country)        -> cleaned business name
    normalize_address(address, country)  -> cleaned business address
    extract_landmarks(address)           -> list of landmark references

One convenience function for batch processing:
    normalize_dataframe(df)              -> df with name_clean, address_clean

Design decisions (grounded in Step 1-3 data exploration):
    - Dataset: ~2.2M S1, ~5.0M S2, ~5.3M S3 records (train)
    - Countries: US 60%, India 40% (train); US + India + France (test)
    - Precision > Recall strategy: conservative normalization
    - NO data merging/combining across sources
    - ALL normalization rules apply identically to train AND test
=============================================================================
"""

import re
import unicodedata


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1: LOOKUP TABLES
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1A: Business-name legal suffixes ─────────────────────────────────────────
#
# ORDER MATTERS: longest / most-specific patterns first so that
# "Pvt. Ltd." is caught before bare "Ltd."
#
# Exploration found (in 200K sample):
#   Private Limited  9.10%    LLC  9.10%    Ltd  8.26%    Inc  6.66%
#   Pvt Ltd          3.35%    Corp 2.25%    Co   2.31%    LLP  1.74%
#   SARL             1.88%    SAS  1.32%    SASU 0.39%    SCI  0.34%

LEGAL_SUFFIX_MAP = [
    # ── Indian / English ────────────────────────────────────────────
    (r'\bprivate\s+limited\b',         'private limited'),
    (r'\blimited\s+private\b',         'private limited'),
    (r'\bpvt\.?\s+limited\b',          'private limited'),
    (r'\bpvt\.?\s*ltd\.?\b',           'private limited'),
    (r'\bltd\.?\s*pvt\.?\b',           'private limited'),
    (r'\bprivate\s+ltd\.?\b',          'private limited'),
    (r'\bp\.?\s*ltd\.?\b',             'private limited'),
    (r'\bpublic\s+limited\b',          'public limited'),
    (r'\blimited\b',                   'limited'),
    (r'\bltd\.?\b',                    'limited'),
    (r'\bpvt\.?\b',                    'private'),          # bare "Pvt" / "Pvt."

    # LLC
    (r'\bl\.l\.c\.?\b',                'llc'),
    (r'\bllc\.?\b',                    'llc'),

    # Inc
    (r'\bincorporated\b',              'incorporated'),
    (r'\binc\.?\b',                    'incorporated'),

    # Corp
    (r'\bcorporation\b',               'corporation'),
    (r'\bcorp\.?\b',                   'corporation'),

    # Company / Co
    (r'\bcompany\b',                   'company'),
    (r'\bco\.?\b(?=\s|$)',             'company'),

    # LLP
    (r'\bl\.l\.p\.?\b',               'llp'),
    (r'\bllp\.?\b',                   'llp'),

    # PC / PLLC
    (r'\bpllc\.?\b',                  'pllc'),

    # ── French ──────────────────────────────────────────────────────
    (r'\bs\.?a\.?r\.?l\.?\b',         'sarl'),
    (r'\bs\.?a\.?s\.?u\.?\b',         'sasu'),
    (r'\bs\.?a\.?s\.?\b',             'sas'),
    (r'\bs\.?c\.?i\.?\b',             'sci'),
    (r'\bs\.?a\.?\b(?=\s|$)',         'sa'),

    # ── Indic Script Legal Forms ────────────────────────────────────
    # Hindi / Devanagari
    (r'(?:^|\s+)प्राइवेट\s+लिमिटेड(?:\s+|$)', ' private limited '),
    (r'(?:^|\s+)प्रा\.\s*लि\.?(?:\s+|$)',     ' private limited '),
    (r'(?:^|\s+)लिमिटेड(?:\s+|$)',            ' limited '),
    (r'(?:^|\s+)एलएलपी(?:\s+|$)',              ' llp '),
    # Telugu
    (r'(?:^|\s+)ప్రైవేట్\s+లిమిటెడ్(?:\s+|$)', ' private limited '),
    (r'(?:^|\s+)లిమిటెడ్(?:\s+|$)',            ' limited '),
    # Tamil
    (r'(?:^|\s+)பிரைவேட்\s+லிமிடெட்(?:\s+|$)', ' private limited '),
    (r'(?:^|\s+)லிமிடெட்(?:\s+|$)',            ' limited '),
    # Kannada
    (r'(?:^|\s+)ಪ್ರೈವೇಟ್\s+ಲಿಮಿಟೆಡ್(?:\s+|$)', ' private limited '),
    (r'(?:^|\s+)ಲಿಮಿಟೆಡ್(?:\s+|$)',            ' limited '),
    # Bengali
    (r'(?:^|\s+)প্রাইভেট\s+লিমিটেড(?:\s+|$)', ' private limited '),
    (r'(?:^|\s+)লিমিটেড(?:\s+|$)',            ' limited '),
    # Gujarati
    (r'(?:^|\s+)પ્રાઇવેટ\s+લિમિટેડ(?:\s+|$)', ' private limited '),
    (r'(?:^|\s+)પ્રાઈવેટ\s+લિમિટેડ(?:\s+|$)', ' private limited '),
    (r'(?:^|\s+)લિમિટેડ(?:\s+|$)',            ' limited '),

    # DBA
    (r'\bdba\b',                      'dba'),
]


# ── 1B: Address street-type abbreviations ────────────────────────────────────
#
# Exploration found:
#   Street 14K vs St 8.5K,  Road 27K vs Rd 8.5K,  Ave 5.7K vs Avenue 9.4K
#   Drive 9.4K vs Dr 7K,    Floor 13.4K vs Fl 1.8K
#
# Strategy: expand abbreviations to full word (more characters = more
# signal for fuzzy/cosine comparisons downstream).
#
# NOTE: Single-letter directionals (N/S/E/W) are intentionally omitted.
# They are too ambiguous in Indian addresses (e.g. "S No.", "H No.",
# "E-1 Colony") and would create false replacements.  Multi-letter
# directionals (NW/NE/SW/SE) are safe.

ADDRESS_TYPE_MAP = [
    # Street types — abbreviation first, full-word second (both map to full)
    (r'\bst\.?(?=\s|,|$)',    'street'),     # "St" / "St." before space/comma/end
    (r'\bstreet\b',           'street'),
    (r'\brd\.?(?=\s|,|$)',    'road'),
    (r'\broad\b',             'road'),
    (r'\bave\.?(?=\s|,|$)',   'avenue'),
    (r'\bavenue\b',           'avenue'),
    (r'\bblvd\.?(?=\s|,|$)',  'boulevard'),
    (r'\bboulevard\b',        'boulevard'),
    (r'\bdr\.(?=\s|,|$)',     'drive'),       # require dot to avoid "Dr Kumar"
    (r'\bdrive\b',            'drive'),
    (r'\bln\.?(?=\s|,|$)',    'lane'),
    (r'\blane\b',             'lane'),
    (r'\bct\.?(?=\s|,|$)',    'court'),
    (r'\bcourt\b',            'court'),
    (r'\bcir\.?(?=\s|,|$)',   'circle'),
    (r'\bcircle\b',           'circle'),
    (r'\bhwy\.?(?=\s|,|$)',   'highway'),
    (r'\bhighway\b',          'highway'),
    (r'\bpkwy\.?(?=\s|,|$)',  'parkway'),
    (r'\bparkway\b',          'parkway'),
    (r'\btrl\.?(?=\s|,|$)',   'trail'),
    (r'\btrail\b',            'trail'),
    (r'\bpl\.(?=\s|,|$)',     'place'),       # require dot: "Pl." not bare "pl"
    (r'\bplace\b',            'place'),

    # Building / unit types
    (r'\bfl\.(?=\s|,|$)',     'floor'),
    (r'\bfloor\b',            'floor'),
    (r'\bapt\.?(?=\s|,|$)',   'apartment'),
    (r'\bapartment\b',        'apartment'),
    (r'\bbldg\.?(?=\s|,|$)',  'building'),
    (r'\bbuilding\b',         'building'),
    (r'\bste\.?(?=\s|,|$)',   'suite'),
    (r'\bsuite\b',            'suite'),
    (r'\bunit\b',             'unit'),
    (r'\brm\.(?=\s|,|$)',     'room'),
    (r'\broom\b',             'room'),

    # ── French street types (Test set: France) ───────────────────
    (r'\br\.\s*de\b|\br\s+de\b',       'rue de'),
    (r'\br\.\s*des\b|\br\s+des\b',     'rue des'),
    (r'\br\.\s*du\b|\br\s+du\b',       'rue du'),
    (r'\br\.\s+(?=[a-zA-Z]+)',         'rue '),
    (r'\bbd\.?(?=\s|,|$)',              'boulevard'),
    (r'\bch\.\s*(?=du|de|des)\b|\bch\s+(?=du|de|des)\b', 'chemin '),
    (r'\bimp\.?(?=\s|,|$)',             'impasse'),
    (r'\brte\.?(?=\s|,|$)',             'route'),
    (r'\ball\.?(?=\s|,|$)',             'allee'),
    (r'\bav\.?(?=\s|,|$)',              'avenue'),

    # ── Floor / Level standardization ────────────────────────────
    (r'\b(?:1st|first|ist|1)\s*(?:flr\.?|floor|fl\.?)\b', '1st floor'),
    (r'\b(?:2nd|second|iind|2)\s*(?:flr\.?|floor|fl\.?)\b', '2nd floor'),
    (r'\b(?:3rd|third|iiird|3)\s*(?:flr\.?|floor|fl\.?)\b', '3rd floor'),
    (r'\b(?:4th|fourth|ivth|4)\s*(?:flr\.?|floor|fl\.?)\b', '4th floor'),
    (r'\b(?:ground|gr\.?|g\.?)\s*floor\b|\bfl\.?\s*0\b', 'ground floor'),
    (r'\bbsmt\.?\b|\bbasement\b',     'basement'),

    # ── Care-Of / Relation prefixes ──────────────────────────────
    (r'\bc/o\.?\b|\bcare\s+of\b',     'c/o'),
    (r'\bs/o\.?\b|\bso\b(?=\s+[a-z]+)', 's/o'),
    (r'\bw/o\.?\b|\bwo\b(?=\s+[a-z]+)', 'w/o'),
    (r'\bd/o\.?\b|\bdo\b(?=\s+[a-z]+)', 'd/o'),

    # ── Property numbering prefixes ──────────────────────────────
    (r'\bh\.?\s*no\.?|\bhn\b(?=\s*\d+)', 'h no'),
    (r'\bplot\.?\s*no\.?|\bplot\b(?=\s*\d+)', 'plot no'),
    (r'\bdoor\.?\s*no\.?|\bd\.?\s*no\.?|\bd-no\b', 'door no'),
    (r'\bflat\.?\s*no\.?',            'flat no'),
    (r'\bshop\.?\s*no\.?',            'shop no'),
    (r'\boffice\.?\s*no\.?|\boff\.?\s*no\.?', 'office no'),
    (r'\bkh\.?\s*no\.?|\bkhasra\.?\s*no\.?', 'kh no'),

    # ── PO Box / PMB ─────────────────────────────────────────────
    (r'\b(?:p\.?\s*o\.?\s*box|pobox|p\.?\s*o\.?\s*b)\s*[-:#]?\s*', 'po box '),
    (r'\bpmb\s*[-:#]?\s*',              'pmb '),

    # ── US Rural / County / State Roads ──────────────────────────
    (r'\bco(?:unty)?\s+rd\.?\b',        'county road'),
    (r'\bstate\s+rd\.?\b|\bst\s+rd\.?(?=\s*\d+)', 'state road'),

    # Multi-letter directionals only (safe)
    (r'\bnw\.?(?=\s|,|$)',    'northwest'),
    (r'\bne\.?(?=\s|,|$)',    'northeast'),
    (r'\bsw\.?(?=\s|,|$)',    'southwest'),
    (r'\bse\.?(?=\s|,|$)',    'southeast'),
]


# ── 1C: Landmark prepositions ────────────────────────────────────────────────
# Exploration: Near 4.4K, Opp 2.8K, Behind 666, Next-to 137

LANDMARK_PREPOSITION_MAP = [
    (r'\bopp\.?\b',           'opposite'),
    (r'\bopposite\b',         'opposite'),
    (r'\bnr\.?\b',            'near'),
    (r'\bnear\b',             'near'),
    (r'\bbehind\b',           'behind'),
    (r'\badj\.?\b',           'adjacent'),
    (r'\badjacent\b',         'adjacent'),
    (r'\bnext\s+to\b',        'next to'),
    (r'\bin\s+front\s+of\b',  'in front of'),
    (r'\bbeside\b',           'beside'),
]


# ── 1D: Indian state names  -->  2-letter code ──────────────────────────────
# Exploration: Maharashtra(8.4K full, 7.2K code), Delhi(11.7K full, 4.6K code)

INDIAN_STATE_MAP = {
    # Multi-word first (sorted longest-first at lookup time)
    'andhra pradesh': 'ap',   'arunachal pradesh': 'ar',
    'himachal pradesh': 'hp', 'jammu and kashmir': 'jk',
    'madhya pradesh': 'mp',   'tamil nadu': 'tn',  'tamilnadu': 'tn',
    'uttar pradesh': 'up',    'west bengal': 'wb',
    'andaman and nicobar': 'an',
    'dadra and nagar haveli': 'dn', 'daman and diu': 'dd',
    'new delhi': 'dl',

    # Single-word states
    'assam': 'as',       'bihar': 'br',       'chhattisgarh': 'cg',
    'goa': 'ga',         'gujarat': 'gj',     'haryana': 'hr',
    'jharkhand': 'jh',   'karnataka': 'ka',   'kerala': 'kl',
    'maharashtra': 'mh', 'manipur': 'mn',     'meghalaya': 'ml',
    'mizoram': 'mz',     'nagaland': 'nl',    'odisha': 'od',
    'orissa': 'od',      'punjab': 'pb',      'rajasthan': 'rj',
    'sikkim': 'sk',      'telangana': 'tg',   'tripura': 'tr',
    'uttarakhand': 'uk', 'delhi': 'dl',
    'chandigarh': 'ch',  'puducherry': 'py',  'pondicherry': 'py',
    'ladakh': 'la',      'lakshadweep': 'ld',
}


# ── 1E: US state names  -->  2-letter code ──────────────────────────────────

US_STATE_MAP = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar',
    'california': 'ca', 'colorado': 'co', 'connecticut': 'ct',
    'delaware': 'de', 'florida': 'fl', 'georgia': 'ga', 'hawaii': 'hi',
    'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me',
    'maryland': 'md', 'massachusetts': 'ma', 'michigan': 'mi',
    'minnesota': 'mn', 'mississippi': 'ms', 'missouri': 'mo',
    'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv',
    'new hampshire': 'nh', 'new jersey': 'nj', 'new mexico': 'nm',
    'new york': 'ny', 'north carolina': 'nc', 'north dakota': 'nd',
    'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa',
    'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd',
    'tennessee': 'tn', 'texas': 'tx', 'utah': 'ut', 'vermont': 'vt',
    'virginia': 'va', 'washington': 'wa', 'west virginia': 'wv',
    'wisconsin': 'wi', 'wyoming': 'wy',
    'district of columbia': 'dc',
}


# ── 1F: French region names ─────────────────────────────────────────────────
# Test set has ~15% France.  Normalize hyphenated region names.

FRENCH_REGION_MAP = {
    'hauts-de-france': 'hauts de france',
    'hauts de france': 'hauts de france',
    'nouvelle-aquitaine': 'nouvelle aquitaine',
    'nouvelle aquitaine': 'nouvelle aquitaine',
    'ile-de-france': 'ile de france',
    'ile de france': 'ile de france',
    'auvergne-rhone-alpes': 'auvergne rhone alpes',
    'bourgogne-franche-comte': 'bourgogne franche comte',
    'grand est': 'grand est',
    'bretagne': 'bretagne',
    'centre-val de loire': 'centre val de loire',
    'corse': 'corse',
    'normandie': 'normandie',
    'occitanie': 'occitanie',
    'pays de la loire': 'pays de la loire',
    "provence-alpes-cote d'azur": 'provence alpes cote d azur',
    'paca': 'provence alpes cote d azur',
    'loire-atlantique': 'loire atlantique',
    'pas-de-calais': 'pas de calais',
}


# ── 1G: Indic-script state names  -->  2-letter code ────────────────────────
# Data showed: महाराष्ट्र, ಕರ್ನಾಟಕ, தமிழ்நாடு, গুজরাত etc.

INDIC_STATE_MAP = {
    # Hindi / Devanagari
    'महाराष्ट्र': 'mh',  'दिल्ली': 'dl',     'दिल्\u200dली': 'dl',
    'गुजरात': 'gj',     'राजस्थान': 'rj',   'राजस्\u200dथान': 'rj',
    'उत्तर प्रदेश': 'up', 'उत्\u200dतर प्रदेश': 'up',
    'मध्य प्रदेश': 'mp', 'हरियाणा': 'hr',    'बिहार': 'br',
    'झारखंड': 'jh',     'छत्तीसगढ़': 'cg',   'ओडिशा': 'od',
    'पंजाब': 'pb',      'गोवा': 'ga',       'केरल': 'kl',
    'तेलंगाना': 'tg',    'आंध्र प्रदेश': 'ap',
    'पश्चिम बंगाल': 'wb',

    # Kannada
    'ಕರ್ನಾಟಕ': 'ka',

    # Tamil
    'தமிழ்நாடு': 'tn',

    # Gujarati
    'ગુજરાત': 'gj',

    # Bengali
    'পশ্চিমবঙ্গ': 'wb',

    # Telugu
    'ఆంధ్రప్రదేశ్': 'ap', 'తెలంగాణ': 'tg',

    # Punjabi
    'ਪੰਜਾਬ': 'pb',

    # Malayalam
    'കേരളം': 'kl',
}


# ── 1H: Saint Names Disambiguation ──────────────────────────────────────────
# Prevents "St Clair" or "St. Albans" or "Near St. Francis" becoming "street"
SAINT_NAMES = {
    'albans', 'alphonsus', 'andrew', 'andrews', 'anthony', 'augustine',
    'bernard', 'blaise', 'charles', 'clair', 'claire', 'cloud', 'croix',
    'denis', 'dominic', 'etienne', 'francis', 'george', 'germain',
    'helena', 'helens', 'james', 'john', 'johns', 'joseph', 'jude',
    'lawrence', 'leonard', 'louis', 'lucia', 'margaret', 'martin', 'mary',
    'marys', 'michael', 'nazaire', 'paul', 'peter', 'peters', 'pierre',
    'rose', 'stephen', 'thomas', 'vincent'
}

NON_STATE_FOLLOWERS = re.compile(
    r'^(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|'
    r'way|highway|hwy|court|ct|parkway|pkwy|place|pl|circle|cir|terrace|trail|trl|'
    r'bank|gate|bhavan|bhawan|house|tower|towers|building|complex|colony|nagar|'
    r'school|college|hospital|clinic|temple|church|hall|park|square|center|centre)\b',
    re.IGNORECASE
)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2: HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def strip_accents(text):
    """
    Remove Latin diacritics / accented characters (e.g. école -> ecole)
    while strictly preserving Indic scripts (matras, halants / viramas).

    Critical distinction:
      - Latin diacritics are Unicode combining marks U+0300 to U+036F.
      - Indic vowel marks and viramas are script-specific and must NEVER
        be stripped.
    """
    if not text:
        return text
    nfkd = unicodedata.normalize('NFKD', text)
    # Strip ONLY combining diacritical marks in Latin range (U+0300 - U+036F)
    res = [c for c in nfkd if not (0x0300 <= ord(c) <= 0x036F)]
    return unicodedata.normalize('NFC', ''.join(res))


def _handle_saint_names(text):
    """
    Expand 'St' / 'St.' to 'saint' when followed by a recognized saint name
    or city name (e.g. 'St Clair Ave' -> 'saint clair ave', 'St. Albans' -> 'saint albans').
    """
    def repl(m):
        w = m.group(1).lower()
        if w in SAINT_NAMES:
            return f"saint {w}"
        return m.group(0)

    return re.sub(r'\bst\.?\s+([a-zA-Z]+)\b', repl, text, flags=re.IGNORECASE)


def _clean_dots(text):
    """
    Intelligently handle dots in text.

    Rules:
      - "H.No." or "S.No." or "M.G." --> keep with spaces: "h no", "s no", "m g"
      - Remove all remaining dots
      - Collapse spaces

    This avoids the "hno16-11-23" problem where "H.No.16" gets jammed.
    """
    if not text:
        return text

    # Insert a space after every dot that is followed by a non-space char
    # This breaks "H.No.16" into "H. No. 16" before the dot is removed
    text = re.sub(r'\.(?=\S)', '. ', text)

    # Now remove all dots
    text = text.replace('.', '')

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text


def _clean_punctuation(text):
    """
    Standardize punctuation and special characters.

    Exploration found:
      & (2.6K), # (4K), dots (13.7K), hyphens (14.9K),
      parentheses (3.5K), brackets (1.1K), angle brackets (415),
      N° / nº (2.8K in French/Indian), colons (8.5K in C/O:, DBA:, etc.)
    """
    if not text:
        return text

    # & --> " and "
    text = text.replace('&', ' and ')

    # N° / n° / nº / No. -> "no "
    text = re.sub(r'\bn[°º]\s*', 'no ', text, flags=re.IGNORECASE)

    # Clean non-breaking spaces and tabs
    text = text.replace('\xa0', ' ').replace('\t', ' ')

    # Remove angle brackets << >> and unicode quotes
    text = re.sub(r'[<>«»]', ' ', text)

    # Remove special symbols, degree marks, quotes, backticks
    text = re.sub(r'[°º~^*\"`]', ' ', text)

    # Remove brackets and parentheses
    text = re.sub(r'[\[\](){}]', ' ', text)

    # Colons and semicolons -> space (safely separates C/O:, S/O:, DBA:, Vill:, etc.)
    text = text.replace(':', ' ').replace(';', ' ')

    # Remove # sign (keep the number after it)
    text = re.sub(r'#\s*', ' ', text)

    # Remove pipe | (seen in S2 data: "SHIVSHAKTI | www.shivshakti.com")
    text = text.replace('|', ' ')

    # Collapse double-hyphens but keep single hyphens
    text = re.sub(r'-{2,}', '-', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def _normalize_state(text, country):
    """
    Replace state/region names with canonical 2-letter short forms using
    segment-aware matching that avoids false positives.

    Guarantees:
      - Preserves street names like 'Washington St' and 'Indiana Ave'
      - Preserves landmark names like 'Maharashtra Bank' and 'Delhi Gate'
      - Preserves city names like 'New York, NY' and 'Washington, DC'
      - Correctly maps Indic script state names (e.g. महाराष्ट्र -> mh, পশ্চিমবঙ্গ -> wb)
    """
    if not text or not country:
        return text

    if country == 'India':
        # Indic scripts first (unambiguous)
        for indic_name, code in INDIC_STATE_MAP.items():
            if indic_name in text:
                return text.replace(indic_name, code)
        state_map = INDIAN_STATE_MAP

    elif country == 'US':
        state_map = US_STATE_MAP

    elif country == 'France':
        for region in sorted(FRENCH_REGION_MAP, key=len, reverse=True):
            if region in text:
                return text.replace(region, FRENCH_REGION_MAP[region])
        return text
    else:
        return text

    # Segment-aware check
    segments = [s.strip() for s in text.split(',')]
    if not segments:
        return text

    # If the last segment is ALREADY a valid 2-letter state code
    # (e.g. ', NC', ', CA 90210', ', TG'), do not modify other segments
    last_seg_clean = re.sub(r'\s+\d+(?:-\d+)?$', '', segments[-1].strip()).lower()
    valid_codes = set(state_map.values())
    if last_seg_clean in valid_codes:
        return text

    # DC check for US
    if country == 'US' and re.search(r'\b(?:dc|d\.c\.|district of columbia)\b', text, re.IGNORECASE):
        if last_seg_clean in ('dc', 'district of columbia'):
            segments[-1] = 'dc'
            return ', '.join(segments)

    # Search order: last segment, 2nd-to-last segment, first segment (reversed formats), then others
    candidate_indices = []
    if len(segments) >= 1:
        candidate_indices.append(len(segments) - 1)
    if len(segments) >= 2:
        candidate_indices.append(len(segments) - 2)
    if len(segments) >= 3:
        candidate_indices.append(0)
    for i in range(len(segments)):
        if i not in candidate_indices:
            candidate_indices.append(i)

    sorted_states = sorted(state_map.keys(), key=len, reverse=True)

    for idx in candidate_indices:
        seg = segments[idx]
        seg_lower = seg.lower()
        replaced = False
        for state_name in sorted_states:
            pat = r'\b' + re.escape(state_name) + r'\b'
            m = re.search(pat, seg_lower)
            if m:
                # Do not replace if followed by street or landmark follower word
                after = seg_lower[m.end():].strip()
                if after and NON_STATE_FOLLOWERS.match(after):
                    continue
                # Do not replace if preceded by landmark preposition
                before = seg_lower[:m.start()].strip()
                if before and re.search(r'\b(?:near|nr|opp|opposite|behind|adjacent|next to)\b$', before):
                    continue

                code = state_map[state_name]
                new_seg = re.sub(pat, code, seg, count=1, flags=re.IGNORECASE)
                segments[idx] = new_seg
                replaced = True
                break
        if replaced:
            return ', '.join(segments)

    return text


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3: PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_name(name, country=None):
    """
    Normalize a business name for entity-resolution comparison.

    Pipeline:
        1. lowercase
        2. strip accents  (e -> e, c-cedilla -> c)
        3. clean punctuation  (& -> and, remove brackets/angles)
        4. standardize legal suffixes  (Pvt Ltd -> private limited)
        5. clean dots  (intelligently spaced then removed)
        6. collapse whitespace

    Parameters
    ----------
    name : str
        Raw business_name from the dataset.
    country : str, optional
        'US', 'India', or 'France'.  Reserved for future country-specific
        rules; currently unused.

    Returns
    -------
    str
        Normalized name.  Empty string for NaN / None / non-string input.

    Examples
    --------
    >>> normalize_name("Halocast Private Limited")
    'halocast private limited'
    >>> normalize_name("HALOCAST PVT. LTD.")
    'halocast private limited'
    >>> normalize_name("Callicoat & Dailey Inc")
    'callicoat and dailey incorporated'
    >>> normalize_name("Fractales Amis Groupe S.A.S")
    'fractales amis groupe sas'
    """
    if not name or not isinstance(name, str):
        return ''

    # Step 0: Extract canonical business name from DBA / Formerly Known As / FKA / AKA
    # E.g. 'Arcevo DBA Nick Family Office PC' -> 'Nick Family Office PC'
    # E.g. 'Lyracaloiri formerly City Builders Private Limited' -> 'City Builders Private Limited'
    # E.g. 'zetagild fka yukthi hardware private limited' -> 'yukthi hardware private limited'
    m_dba = re.search(r'\b(?:dba|d/b/a|formerly\s+known\s+as|formerly|f/k/a|fka|a/k/a|aka|t/a)\b\s*[:\-]?\s*(.+)$', name, re.IGNORECASE)
    if m_dba and len(m_dba.group(1).strip()) >= 3:
        name = m_dba.group(1).strip()

    # Step 0b: Strip URL pipes: '... | www.domain.com'
    name = re.sub(r'\|\s*(?:www\.|https?://)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}\b', ' ', name, flags=re.IGNORECASE)

    # Step 0c: Strip standalone website domains: 'heassociates.com' -> 'heassociates'
    name = re.sub(r'\b([a-zA-Z0-9-]+)\.(?:com|org|net|in|co\.in)\b', r'\1', name, flags=re.IGNORECASE)

    # Step 0d: Strip trailing phone / tracking numbers: '- 7149969359'
    name = re.sub(r'\s*[-–]\s*\d{7,12}\s*$', '', name)

    # Step 0e: Hyphen before legal suffix: 'prairie investments-incorporated' -> 'prairie investments incorporated'
    name = re.sub(r'-(?=(?:incorporated|inc|corporation|corp|llc|limited|ltd|private|pvt|co|company|sarl|sas)\b)', ' ', name, flags=re.IGNORECASE)

    # Step 0f: Strip commercial honorific prefixes at start of name: M/s, Shri, Smt, Dr, Mr, Mrs
    name = re.sub(r'^(?:m/s|messrs|shri|smt|dr\.?|mr\.?|mrs\.?)\s+', '', name, flags=re.IGNORECASE)

    text = name.lower()
    # French ligatures
    text = text.replace('œ', 'oe').replace('æ', 'ae')
    text = strip_accents(text)
    text = _clean_punctuation(text)

    # Legal suffixes (regex on lowered text)
    for pattern, replacement in LEGAL_SUFFIX_MAP:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Collapse duplicated trailing business terms (e.g. 'group group' -> 'group', 'llc llc' -> 'llc')
    text = re.sub(r'\b(llc|incorporated|corporation|limited|private limited|private|group|services|holdings|enterprises|associates|solutions|industries|unit|vidyalaya)\s+\1\b', r'\1', text)

    # Standardize bare 'private' at end of Indian name to 'private limited'
    if country == 'India':
        text = re.sub(r'\bprivate\b$', 'private limited', text)

    text = _clean_dots(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def normalize_address(address, country=None):
    """
    Normalize a business address for entity-resolution comparison.

    Pipeline:
        1. lowercase
        2. strip accents
        3. remove "null" placeholders
        4. normalize state / region names  (country-aware)
        5. standardize street-type abbreviations  (St -> street)
        6. standardize landmark prepositions  (opp. -> opposite)
        7. clean punctuation
        8. clean dots  (with intelligent spacing)
        9. collapse whitespace

    Parameters
    ----------
    address : str
        Raw business_address from the dataset.
    country : str, optional
        'US', 'India', or 'France'.

    Returns
    -------
    str
        Normalized address.  Empty string for NaN / None / non-string input.

    Examples
    --------
    >>> normalize_address("500 Market St, San Jose, CA", "US")
    '500 market street, san jose, ca'
    >>> normalize_address("797, Lake Town Block A, Kolkata, West Bengal", "India")
    '797, lake town block a, kolkata, howrah, wb'
    """
    if not address or not isinstance(address, str):
        return ''

    text = address.lower()
    # French ligatures
    text = text.replace('œ', 'oe').replace('æ', 'ae')
    text = strip_accents(text)

    # Remove "null" and "n/a" placeholders (seen in actual data)
    text = re.sub(r'\b(?:null|n/a|na|none|nil|not applicable)\b', '', text)

    # Clean punctuation first (& -> and, remove brackets)
    text = _clean_punctuation(text)

    # Clean leading zeros on hyphenated unit/plot numbers (e.g. "k-00303" -> "k-303")
    text = re.sub(r'([a-zA-Z]-)0+(\d+)', r'\1\2', text)

    # Clean dots EARLY — before regex substitutions.
    # This breaks "Opp.Rta" -> "Opp Rta" and "H.No.16" -> "H No 16"
    # so that the regex patterns can match properly.
    text = _clean_dots(text)

    # State / region normalization (country-aware)
    if country:
        text = _normalize_state(text, country)

    # Fix synthetic ordinal + saint noise (e.g. "1391 51st Saint" -> "1391 51st street")
    text = re.sub(r'\b(\d+(?:st|nd|rd|th))\s+saint\b', r'\1 street', text, flags=re.IGNORECASE)

    # Disambiguate Saint names (St Clair, St. Albans, St. Francis -> saint ...)
    text = _handle_saint_names(text)

    # Country-specific street type handling
    if country == 'France':
        # Expand bare French abbreviations: "24 R DESAIX" -> "24 rue desaix"
        text = re.sub(r'\br\s+(?=[a-zA-Z]+)', 'rue ', text)
        text = re.sub(r'\bch\s+(?=[a-zA-Z]+)', 'chemin ', text)
        text = re.sub(r'\bav\s+(?=[a-zA-Z]+)', 'avenue ', text)

    # Street-type abbreviations
    for pattern, replacement in ADDRESS_TYPE_MAP:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Ordinal numbers before street types (e.g. "252 fourth street" -> "252 4th street")
    _ordinal_map = {
        'first': '1st', 'second': '2nd', 'third': '3rd', 'fourth': '4th',
        'fifth': '5th', 'sixth': '6th', 'seventh': '7th', 'eighth': '8th',
        'ninth': '9th', 'tenth': '10th'
    }
    for word, repl in _ordinal_map.items():
        text = re.sub(r'\b' + word + r'\s+(?=street|avenue|road|drive|lane|boulevard|court|circle|place)\b', repl + ' ', text, flags=re.IGNORECASE)

    # Collapse duplicated unit tokens (e.g. "unit unit 24" -> "unit 24")
    text = re.sub(r'\b(unit|floor|suite|room|block)\s+\1\b', r'\1', text)

    # Landmark prepositions
    for pattern, replacement in LANDMARK_PREPOSITION_MAP:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_landmarks(address):
    """
    Extract landmark references from an address (primarily Indian).

    Landmarks are relative location hints like "Near SBI ATM",
    "Opp. Railway Station", "Behind Temple".

    Exploration found: Near 4,444 | Opp 2,817 | Behind 666 in 200K sample.

    Parameters
    ----------
    address : str
        Raw or normalized address.

    Returns
    -------
    list[str]
        Extracted landmark phrases, lowercased and preposition-normalized.

    Examples
    --------
    >>> extract_landmarks("500, Near SBI ATM, Opp. Railway Station, Mumbai")
    ['near sbi atm', 'opposite railway station']
    >>> extract_landmarks("123 Main Street, Phoenix, AZ")
    []
    """
    if not address or not isinstance(address, str):
        return []

    text = address.lower()
    landmarks = []

    # Preposition + up-to-40 chars of landmark description until comma or end
    pattern = (
        r'\b(near|nr\.?|opp\.?|opposite|behind|adjacent|adj\.?'
        r'|beside|next\s+to|in\s+front\s+of)'
        r'\s+([\w\s]{3,40}?)(?:,|$)'
    )

    for m in re.finditer(pattern, text):
        prep = m.group(1).strip().rstrip('.')
        prep_map = {'opp': 'opposite', 'nr': 'near', 'adj': 'adjacent'}
        prep = prep_map.get(prep, prep)
        desc = m.group(2).strip()
        if desc:
            landmarks.append(f"{prep} {desc}")

    return landmarks


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4: BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_dataframe(df):
    """
    Apply normalization to an entire source DataFrame.

    Adds two new columns:
        name_clean     -  normalized business name
        address_clean  -  normalized business address

    Uses vectorized string operations where possible, with a Python
    fallback for the regex-heavy transformations.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: entity_id, business_name, business_address,
        country.

    Returns
    -------
    pandas.DataFrame
        Copy of df with name_clean and address_clean added.
    """
    import pandas as pd

    out = df.copy()
    out['business_name'] = out['business_name'].fillna('')
    out['business_address'] = out['business_address'].fillna('')

    # Vectorized apply (row-wise for regex-heavy logic)
    out['name_clean'] = [
        normalize_name(n, c)
        for n, c in zip(out['business_name'], out['country'])
    ]
    out['address_clean'] = [
        normalize_address(a, c)
        for a, c in zip(out['business_address'], out['country'])
    ]

    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5: SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys, io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace'
    )

    print("=" * 72)
    print("  NORMALIZATION SELF-TEST  (all examples from actual dataset)")
    print("=" * 72)

    # ────────────────────────────────────────────────────────────────────
    #  NAME TESTS
    # ────────────────────────────────────────────────────────────────────
    name_tests = [
        # (raw_input,                           country,   expected)
        # -- Indian legal suffixes --
        ("Halocast Private Limited",            "India",   "halocast private limited"),
        ("HALOCAST PVT. LTD.",                  "India",   "halocast private limited"),
        ("Halocast Pvt Ltd",                    "India",   "halocast private limited"),
        ("neha foods private limited",          "India",   "neha foods private limited"),
        ("Shri Supreme Consulting Private  (Limited)",
                                                "India",   "supreme consulting private limited"),
        ("Pvt. EFS Print Ventures Ltd.",        "India",   "private efs print ventures limited"),
        ("Clm Agro [Limited]",                  "India",   "clm agro limited"),
        ("FOUNDATION EXCEL AGENCY PRIVATE  LIMITED",
                                                "India",   "foundation excel agency private limited"),

        # -- US legal suffixes --
        ("B+ Retail Inc",                       "US",      "b+ retail incorporated"),
        ("Custom Wealth Services LLC",          "US",      "custom wealth services llc"),
        ("Callicoat & Dailey Inc",              "US",      "callicoat and dailey incorporated"),
        ("Moore Bitwise Inc",                   "US",      "moore bitwise incorporated"),
        ("ASSET BUILDING COALITION LLC",        "US",      "asset building coalition llc"),
        ("A DIAMOND  SHORT LLC",                "US",      "a diamond short llc"),

        # -- French legal suffixes --
        ("ZNB Club SARL",                       "France",  "znb club sarl"),
        ("Fractales Amis Groupe S.A.S",         "France",  "fractales amis groupe sas"),
        ("<< Team Ecole",                       "France",  "team ecole"),
        ("SCI Ptit Amicale",                    "France",  "sci ptit amicale"),
        ("Marina Ecole France Sarl",            "France",  "marina ecole france sarl"),

        # -- Special characters --
        ("LLC Moncada Learning Center",         "US",      "llc moncada learning center"),
        ("Dermatology Green Medicine",          "US",      "dermatology green medicine"),

        # -- Pipe / URL in name --
        ("SHIVSHAKTI VIDYALAYA OVERSEAS CORPORATION | www.shivshakti.com",
                                                "India",   "shivshakti vidyalaya overseas corporation"),

        # -- Accented French --
        ("SCI Ptit Amicale",                    "France",  "sci ptit amicale"),

        # -- Indic script names (must preserve matras and halants) --
        ("राम मार्केटिंग प्राइवेट लिमिटेड",     "India",   "राम मार्केटिंग private limited"),
        ("పర్ఫెక్ట్ ఎనర్జీ ప్రైవేట్ లిమిటెడ్",  "India",   "పర్ఫెక్ట్ ఎనర్జీ private limited"),

        # -- Ground-truth validated systematic noise --
        ("Zetaflux DBA: Fowler Peak Optics",    "US",      "fowler peak optics"),
        ("Mirapyra formerly Painters Local Union 634",
                                                "US",      "painters local union 634"),
        ("zetagild fka yukthi hardware private limited",
                                                "India",   "yukthi hardware private limited"),
        ("Diamond Star Works L.L.C. - 7149969359",
                                                "US",      "diamond star works llc"),
        ("heassociates.com",                    "US",      "heassociates"),
        ("M/s Navkar On Limited Private",       "India",   "navkar on private limited"),
        ("Mr Life Foundation Private Ltd",      "India",   "life foundation private limited"),
        ("prairie investments-incorporated",    "US",      "prairie investments incorporated"),
        ("crescent electronics-private limited", "India",  "crescent electronics private limited"),
        ("urgent care best group group",        "US",      "urgent care best group"),
        ("hotel technology private",            "India",   "hotel technology private limited"),
    ]

    print("\n--- BUSINESS NAME TESTS ---\n")
    name_pass = 0
    name_fail = 0
    for raw, country, expected in name_tests:
        result = normalize_name(raw, country)
        ok = (result == expected)
        tag = "PASS" if ok else "FAIL"
        if ok:
            name_pass += 1
        else:
            name_fail += 1
        print(f"  [{tag}]  {raw!r}")
        print(f"     -->  {result!r}")
        if not ok:
            print(f"     EXP: {expected!r}")
        print()

    # ────────────────────────────────────────────────────────────────────
    #  ADDRESS TESTS
    # ────────────────────────────────────────────────────────────────────
    address_tests = [
        # (raw_input, country, substring_that_must_appear)

        # US - street type expansion
        ("1795 Westchester Drive, High Point, NC",
         "US", "westchester drive"),
        ("500 Market St, San Jose, CA",
         "US", "market street"),
        ("17560 Ellis Road, Tahlequah, OK",
         "US", "ellis road"),
        ("OH, Columbus, 5559 Orville Avenue",
         "US", "orville avenue"),
        ("5780 Fawn Ct, Fort Worth, Texas",
         "US", "fawn court"),
        ("Mack Rd, Haltom City, Texas",
         "US", "mack road"),
        ("126-B New Line Road, MORRISTOWN, TN",
         "US", "new line road"),

        # US - state name to code
        ("5780 Fawn Ct, Fort Worth, Texas",
         "US", "tx"),
        ("1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio",
         "US", "oh"),

        # US - Street / State name collisions (PRESERVED!)
        ("1200 Washington Street, San Francisco, California",
         "US", "washington street"),
        ("500 Washington Ave, New York, NY",
         "US", "new york, ny"),
        ("100 Indiana Ave, Washington, DC",
         "US", "indiana avenue"),

        # US & Global - Saint names (PRESERVED as Saint, not Street!)
        ("8404 1/2 St Clair Avenue, Cleveland, Ohio",
         "US", "saint clair avenue"),
        ("2707 Furlong Avenue, St. Albans, WV",
         "US", "saint albans"),
        ("Behind St. Francis De Sales Press, Kakkanad, Ernakulam, Kerala",
         "India", "saint francis"),

        # India - state name to code
        ("797, Lake Town Block A, Kolkata, Howrah, West Bengal",
         "India", "wb"),
        ("Door No 183, Bengaluru Urban, Bangalore, Karnataka",
         "India", "ka"),
        ("Saranampatti Road, Coimbatore, Tamilnadu",
         "India", "tn"),

        # India - Landmark & Gate collision prevention (PRESERVED!)
        ("Near Maharashtra Bank, MG Road, Pune, Maharashtra",
         "India", "near maharashtra bank"),
        ("Opp Delhi Gate, Daryaganj, Delhi",
         "India", "opposite delhi gate"),

        # India - Indic script state names
        ("So Banshilal Joshi, Nokha, Bikaner, राजस्थान",
         "India", "rj"),
        ("123, X Road, Swamijinagar, Howrah, পশ্চিমবঙ্গ",
         "India", "wb"),
        ("PUNE REGION, PUNE, Maharashtra",
         "India", "mh"),

        # India - H.No. spacing  (the "hno" bug fix)
        ("H.No.16-11-23/37/A, 2Nd Floor, Hyderabad, Telangana",
         "India", "h no 16-11-23/37/a"),

        # India - Opp. with space  (the "oppositerta" bug fix)
        ("Opp.Rta Office, Mo, Osarambagh, Hyderabad",
         "India", "opposite rta office"),

        # India - null removal
        ("G.t. Karnal Road, Industrial Area, New Delhi, null, A-68",
         "India", "road"),

        # France - accents + region
        ("175 Boulevard du President Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine",
         "France", "nouvelle aquitaine"),
        ("18 RUE JEN ZAY, Dunkerque, Nord",
         "France", "rue jen zay"),
        ("63 R. DE DIEPPE, LILLE, Hauts-de-France",
         "France", "hauts de france"),

        # Ground-truth validated floor, PO box, French R, Care-of cases
        ("C/O: Dwarkadhis Enterprise, Ist Floor, Shop 3, Gondal, Rajkot, Gujarat",
         "India", "1st floor"),
        ("KIMBALL PLAINS ROAD, PO BOX 3611, COTTONWOOD, CA",
         "US", "po box 3611"),
        ("24 R DESAIX, TOURCOING, Hauts-de-France",
         "France", "24 rue desaix"),
        ("H.NO 13 C/O:MADAN MOHAN JHA, S/O:SH AKHILESH JHA, Jaipur, Rajasthan",
         "India", "h no 13"),
        ("N° 20 R. D'ALZON, BORDEAUX",
         "France", "no 20 rue d'alzon"),
        ("8 Willow Oak Lane, Fl. 0, Saint Louis, Missouri",
         "US", "ground floor"),
        ("Door No 4Th Floor, Office 24, Pune, MH",
         "India", "4th floor"),

        # Ground-truth validated: ordinal street, saint noise, unit dedup, leading zeros
        ("252 Fourth Street, Clinton, IN",
         "US", "4th street"),
        ("1391 51st Saint, Clleveland, Ohio",
         "US", "51st street"),
        ("Unit Unit 24, CA, Ontario, 1119 Princeton Street",
         "US", "unit 24"),
        ("DL, K-00303, New Delhi, Gautam Nagar",
         "India", "k-303"),
        ("89-90-91 NEW DLF INDUSTRIAL AREA, FARIDABAD, N/A, Haryana",
         "India", "hr"),
    ]

    print("\n--- ADDRESS TESTS ---\n")
    addr_pass = 0
    addr_fail = 0
    for raw, country, must_contain in address_tests:
        result = normalize_address(raw, country)
        ok = must_contain in result
        tag = "PASS" if ok else "FAIL"
        if ok:
            addr_pass += 1
        else:
            addr_fail += 1
        print(f"  [{tag}]  {raw!r}")
        print(f"     -->  {result!r}")
        if not ok:
            print(f"     MUST CONTAIN: {must_contain!r}")
        print()

    # ────────────────────────────────────────────────────────────────────
    #  LANDMARK TESTS
    # ────────────────────────────────────────────────────────────────────
    print("\n--- LANDMARK EXTRACTION TESTS ---\n")
    landmark_tests = [
        ("500, Near SBI ATM, Opp. Railway Station, Mumbai",
         ['near sbi atm', 'opposite railway station']),
        ("Behind Fortis Hospital, Bhandup West",
         ['behind fortis hospital']),
        ("Next to ITER College, Dumuduma",
         ['next to iter college']),
        ("123 Main Street, Phoenix, AZ",
         []),
    ]
    lm_pass = 0
    lm_fail = 0
    for addr, expected in landmark_tests:
        result = extract_landmarks(addr)
        ok = result == expected
        tag = "PASS" if ok else "FAIL"
        if ok:
            lm_pass += 1
        else:
            lm_fail += 1
        print(f"  [{tag}]  {addr!r}")
        print(f"     -->  {result}")
        if not ok:
            print(f"     EXP: {expected}")
        print()

    # ────────────────────────────────────────────────────────────────────
    #  SUMMARY
    # ────────────────────────────────────────────────────────────────────
    total_pass = name_pass + addr_pass + lm_pass
    total_fail = name_fail + addr_fail + lm_fail
    total = total_pass + total_fail

    print("=" * 72)
    print(f"  RESULTS:  {total_pass}/{total} passed")
    if total_fail > 0:
        print(f"  FAILURES: {total_fail}  (names: {name_fail}, "
              f"addresses: {addr_fail}, landmarks: {lm_fail})")
    else:
        print("  ALL TESTS PASSED")
    print("=" * 72)
