"""
available_ptms.py

Maps post-translational modification (PTM) names to their 3-letter codes.

Reads from: ptm_list.json (converted from Excel)
Expected fields:
  ["Natural Amino Acid", "Modifications", "Three letter code", "MAP notation",
   "Type", "User Code", "User Input"]
"""

import pandas as pd
import json
import re
import os

# Global dictionaries
AVAILABLE_PTMS = {}
PTM_LOOKUP = {}
PTM_ALIAS_LOOKUP = {}


def load_ptm_mappings(json_path="ptm_list.json"):
    """Load PTM mapping data from JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find PTM JSON file: {json_path}")

    # Load JSON into DataFrame (for same processing logic)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    required_cols = {"Natural Amino Acid", "Three letter code"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in {json_path}: {required_cols}")

    user_input_cols = [c for c in ["User Input", "User Code"] if c in df.columns]

    for _, row in df.iterrows():
        natural_field = str(row["Natural Amino Acid"]).strip()
        if "/" in natural_field:
            natural = natural_field.split("/")[-1].strip().upper()
        else:
            natural = natural_field.upper()

        mod_name = str(row.get("Modifications", "")).strip()
        mod_code = str(row.get("Three letter code", "")).strip().upper()
        map_notation = str(row.get("MAP notation", "")).strip()
        entry_type = "ptm"

        if not mod_code or not map_notation:
            continue

        tag_match = re.search(r"ptm:(\w+)", map_notation)
        tag = tag_match.group(1).upper() if tag_match else mod_code

        if natural not in AVAILABLE_PTMS:
            AVAILABLE_PTMS[natural] = {}

        AVAILABLE_PTMS[natural][tag] = {
            "mod_code": mod_code,
            "modification": mod_name,
            "notation": map_notation,
            "type": entry_type,
            "user_inputs": [],
        }

        PTM_LOOKUP[(natural, mod_code)] = {
            "tag": tag,
            "modification": mod_name,
            "notation": map_notation,
            "type": entry_type,
        }

        # Add aliases
        alias_list = []
        for col in user_input_cols:
            alias_val = row.get(col)
            if pd.notna(alias_val):
                alias_val = str(alias_val).strip().lower()
                if alias_val:
                    alias_list.append(alias_val)

        alias_list.append(tag.lower())
        alias_list.append(mod_code.lower())

        for alias in set(alias_list):
            PTM_ALIAS_LOOKUP[(natural, alias)] = mod_code
            AVAILABLE_PTMS[natural][tag]["user_inputs"].append(alias)

    return AVAILABLE_PTMS


# Load mappings at import
load_ptm_mappings()


def resolve_ptm(residue, user_input):
    """Resolve a user-friendly PTM name into its correct 3-letter code (residue-specific)."""
    if not user_input or str(user_input).lower() == "none":
        return None

    res = residue.strip().upper()
    inp_lower = user_input.strip().lower()
    inp_upper = user_input.strip().upper()

    # --- 1) Direct match on PTM code for this residue ---
    if (res, inp_upper) in PTM_LOOKUP:
        return inp_upper

    # --- 2) Alias match for this residue only ---
    if (res, inp_lower) in PTM_ALIAS_LOOKUP:
        return PTM_ALIAS_LOOKUP[(res, inp_lower)]

    # --- 3) Check within only this residue's PTM entries ---
    for tag, entry in AVAILABLE_PTMS.get(res, {}).items():
        # Match alias
        if inp_lower in entry.get("user_inputs", []):
            return entry["mod_code"]
        # Match long description (e.g., “hydroxy-L-methionine”)
        if inp_lower == entry.get("modification", "").lower():
            return entry["mod_code"]

    # =============== CRITICAL FIX =======================
    # Remove **all cross-residue alias checking**.
    # No more matching aliases from other amino acids!
    # =====================================================

    # --- 4) Error reporting with valid aliases for this residue ---
    valid_aliases = []
    for tag, entry in AVAILABLE_PTMS.get(res, {}).items():
        valid_aliases.extend(entry.get("user_inputs", []))

    valid_aliases = sorted(set(valid_aliases))

    raise ValueError(
        f"Unknown PTM '{user_input}' for residue {res}. "
        f"Valid aliases for {res}: {', '.join(valid_aliases) if valid_aliases else 'None found'}"
    )


__all__ = ["AVAILABLE_PTMS", "PTM_LOOKUP", "PTM_ALIAS_LOOKUP", "resolve_ptm"]
