"""
available_terminal_mods.py

Auto-maps terminal modifications for PEPstrMOD2 using JSON files:
  - n_list.json (N-terminal)
  - c_list.json (C-terminal)

Supports:
  - User-friendly inputs (acet, Coxy, Amid, etc.)
  - User-defined codes (from 'User Code' column)
  - Direct 3-letter PDB codes (5VV, 0NC, NME, etc.)
  - Case-insensitive matching for all user inputs

Expected JSON fields:
  ["Natural Amino Acid", "Three letter code", "User Input", "User Code"]
"""

import os
import re
import json
import pandas as pd


# -------------------------------------------------------------------
# 1. Load JSON mappings dynamically
# -------------------------------------------------------------------

def load_terminal_mods():
    """Load terminal modification mappings from JSON files."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mappings = {"nt": {}, "ct": {}}

    json_files = {
        "nt": os.path.join(base_dir, "n_list.json"),
        "ct": os.path.join(base_dir, "c_list.json"),
    }

    for term, path in json_files.items():
        if not os.path.exists(path):
            print(f"Warning: {os.path.basename(path)} not found — skipping {term}-terminal mods.")
            continue

        # Load JSON → DataFrame (for consistency)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

        required_cols = {"Natural Amino Acid", "Three letter code"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"File {path} must contain at least: {', '.join(required_cols)}")

        # Handle optional columns gracefully
        user_input_cols = [col for col in ["User Input", "User Code"] if col in df.columns]

        for _, row in df.iterrows():
            mod_code = str(row["Three letter code"]).strip().upper()

            residue_field = str(row["Natural Amino Acid"]).strip()
            residue = residue_field.split("/")[-1].strip().upper()  # e.g., "Alanine/Ala/A" → "A"

            for col in user_input_cols:
                alias_value = row.get(col)
                if pd.isna(alias_value):
                    continue

                alias = str(alias_value).strip().lower()
                if not alias:
                    continue

                if alias not in mappings[term]:
                    mappings[term][alias] = {}

                mappings[term][alias][residue] = mod_code

    return mappings


# Load mappings globally (once)
USER_FRIENDLY_TERMINAL_MODS = load_terminal_mods()


# -------------------------------------------------------------------
# 2. Resolver function
# -------------------------------------------------------------------

def resolve_terminal_mod(mod_name, residue_name, terminal_type):
    """
    Resolve a terminal modification name to its 3-letter code.

    Parameters
    ----------
    mod_name : str
        User-provided modification name (e.g., 'coxy', 'Coxy', '5VV', 'nme', 'NME').
    residue_name : str
        The residue name (e.g., 'ALA', 'ASN', or 'A').
    terminal_type : str
        Either 'nt' (N-terminal) or 'ct' (C-terminal).

    Returns
    -------
    str
        Resolved 3-letter modification code (e.g., '5VV', 'NME').

    Raises
    ------
    ValueError
        If the modification or residue is not found.
    """
    if not mod_name or str(mod_name).lower() == "none":
        return None

    mod_name = mod_name.strip()
    residue_name = residue_name.strip().upper() if residue_name else None
    terminal_type = terminal_type.strip().lower()

    # --- Get mappings for this terminal type ---
    mods_for_terminal = USER_FRIENDLY_TERMINAL_MODS.get(terminal_type, {})

    # --- Check if mod_name is a known alias ---
    if mod_name.lower() in mods_for_terminal:
        mod_key = mod_name.lower()
    else:
        # If it's a valid 3-letter PDB code, return directly
        if len(mod_name) == 3 and re.match(r"^[A-Z0-9]{3}$", mod_name.upper()):
            return mod_name.upper()
        mod_key = mod_name.lower()

    if mod_key not in mods_for_terminal:
        valid_aliases = ', '.join(sorted(mods_for_terminal.keys()))
        raise ValueError(f"Unknown {terminal_type}-terminal modification '{mod_name}'. Valid: {valid_aliases}")

    mod_dict = mods_for_terminal[mod_key]

    if not residue_name:
        raise ValueError(f"Cannot resolve terminal modification '{mod_name}' without residue context.")

    # Try exact residue or fallback to 1-letter
    if residue_name not in mod_dict:
        short_res = residue_name[0]
        if short_res in mod_dict:
            residue_name = short_res
        else:
            valid = ", ".join(mod_dict.keys())
            raise ValueError(
                f"'{mod_name}' not valid for residue '{residue_name}'. Valid residues: {valid}"
            )

    mapped_code = mod_dict[residue_name]
    if not mapped_code:
        raise ValueError(f"No valid structure available for '{mod_name}' on residue '{residue_name}'.")

    print(f"{terminal_type.upper()}-terminal '{mod_name}' mapped to {mapped_code} for {residue_name}")
    return mapped_code


__all__ = ["USER_FRIENDLY_TERMINAL_MODS", "resolve_terminal_mod"]
