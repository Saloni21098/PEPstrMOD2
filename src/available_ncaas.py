"""
available_ncaas.py

Maps non-canonical amino acid (NCAA) names and tags to their 3-letter codes.

Reads from: ncaas.json (converted from Excel)
Expected fields:
  ["Natural Amino Acid", "Modifications", "Three letter code", "MAP notation"]
"""

import pandas as pd
import json
import re
import os

# Global dictionaries
AVAILABLE_NCAAS = {}
MODCODE_LOOKUP = {}  # (residue, actual_code) → full info
TAG_TO_BASE = {}


def load_ncaa_mappings(json_path="ncaa_list.json"):
    """Load NCAA mapping data from JSON file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Cannot find NCAA JSON file: {json_path}")

    # Load JSON → DataFrame for same iteration logic
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    required_cols = {"Natural Amino Acid", "Three letter code", "MAP notation"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns in {json_path}: {required_cols}")

    for _, row in df.iterrows():
        natural_field = str(row["Natural Amino Acid"]).strip()
        natural = natural_field.split("/")[-1].strip().upper()

        mod_name = str(row.get("Modifications", "")).strip()
        code = str(row.get("Three letter code", "")).strip().upper()
        map_notation = str(row.get("MAP notation", "")).strip()

        if not code or not map_notation:
            continue

        tag_match = re.search(r"nn[rm]:(\w+)", map_notation)
        type_match = re.search(r"(nn[rm]):", map_notation)

        if not tag_match or not type_match:
            continue

        tag = tag_match.group(1).upper()
        entry_type = type_match.group(1).lower()

        TAG_TO_BASE[tag] = natural

        if natural not in AVAILABLE_NCAAS:
            AVAILABLE_NCAAS[natural] = {}

        AVAILABLE_NCAAS[natural][tag] = {
            "mod_code": code,
            "modification": mod_name,
            "notation": map_notation,
            "type": entry_type,
        }

        MODCODE_LOOKUP[(natural, code)] = {
            "tag": tag,
            "modification": mod_name,
            "notation": map_notation,
            "type": entry_type,
        }

    return AVAILABLE_NCAAS


# Load mappings automatically on import
load_ncaa_mappings()


__all__ = ["AVAILABLE_NCAAS", "MODCODE_LOOKUP", "TAG_TO_BASE"]
