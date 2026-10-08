"""Canonical feature extraction for TIR spacer sequences.

This is the SINGLE source of truth for extract_complete_features().
All scripts and modules import from here instead of duplicating the function.
"""

import pandas as pd
import numpy as np
import RNA

from .config import (
    UPSTREAM_SEQ, DOWNSTREAM_SEQ, SD_CORE, ANTI_SD,
    MW_MAP, STACK_MAP, FEATURE_NAMES
)


def extract_complete_features(spacer):
    """Extract 14 physiochemical features from a spacer sequence.

    Features (no hard-coded length bias):
      Spacing        - spacer length (6, 7, or 8 nt)
      is_len_6       - one-hot: length == 6
      is_len_7       - one-hot: length == 7
      is_len_8       - one-hot: length == 8
      Total_MW       - sum of nucleotide molecular weights (Da)
      Purine_Ratio   - fraction of purines (A+G) in spacer
      Total_Stacking - sum of base stacking energies (kcal/mol)
      GC_Front       - GC content of 5' half of spacer
      GC_Back        - GC content of 3' half of spacer
      DG_total       - MFE of full mRNA fold (kcal/mol)
      DG_RBS         - SD/anti-SD duplex binding energy (kcal/mol)
      RBS_acc        - SD core accessibility (fraction unpaired in MFE structure)
      ATG_acc        - AUG start codon accessibility (fraction unpaired)
      GC_spacer      - overall GC content of spacer

    Returns:
        pd.Series with named features (FEATURE_NAMES).
    """
    try:
        if pd.isna(spacer) or str(spacer).strip() == "":
            return pd.Series([np.nan] * len(FEATURE_NAMES), index=FEATURE_NAMES)

        spacer = str(spacer).strip().upper().replace("T", "U")
        s_len = len(spacer)
        # Uppercase the whole construct: DOWNSTREAM_SEQ stores the start codon as
        # lowercase "atg", and a case-sensitive find("AUG") never matches it.
        full_rna = (UPSTREAM_SEQ + spacer + DOWNSTREAM_SEQ).upper().replace("T", "U")

        # --- Length encoding (unbiased one-hot, no Spacing_Penalty) ---
        spacing = s_len
        is_len_6 = 1.0 if s_len == 6 else 0.0
        is_len_7 = 1.0 if s_len == 7 else 0.0
        is_len_8 = 1.0 if s_len == 8 else 0.0

        # --- Physicochemical features ---
        total_mw = sum(MW_MAP.get(b, 0) for b in spacer)
        purine_count = spacer.count('A') + spacer.count('G')
        purine_ratio = purine_count / s_len if s_len > 0 else 0.0
        total_stacking = sum(STACK_MAP.get(b, 0) for b in spacer)

        # --- Sequence heterogeneity ---
        mid = s_len // 2
        gc_front = (spacer[:mid].count('G') + spacer[:mid].count('C')) / mid if mid > 0 else 0.0
        gc_back = (spacer[mid:].count('G') + spacer[mid:].count('C')) / (s_len - mid) if (s_len - mid) > 0 else 0.0

        # --- Thermodynamic features ---
        (ss, mfe) = RNA.fold(full_rna)
        duplex = RNA.duplexfold(SD_CORE.replace("T", "U"), ANTI_SD)
        dg_rbs = duplex.energy

        sd_start = full_rna.find(SD_CORE.replace("T", "U"))
        # The spacer sits between the SD core and the start codon, so the start
        # codon index follows from the construct. Searching for "AUG" would find
        # a spacer-internal ATG first whenever the spacer carries one, and would
        # have returned -1 before the construct was uppercased.
        atg_start = len(UPSTREAM_SEQ) + s_len

        if sd_start == -1:
            rbs_acc, atg_acc = 0.0, 0.0
        else:
            rbs_struct = ss[sd_start: sd_start + len(SD_CORE)]
            atg_struct = ss[atg_start: atg_start + 3]
            rbs_acc = rbs_struct.count('.') / len(SD_CORE)
            atg_acc = atg_struct.count('.') / 3

        gc_spacer = (spacer.count('G') + spacer.count('C')) / s_len if s_len > 0 else 0.0

        return pd.Series([
            spacing, is_len_6, is_len_7, is_len_8,
            total_mw, purine_ratio, total_stacking,
            gc_front, gc_back, mfe, dg_rbs,
            rbs_acc, atg_acc, gc_spacer
        ], index=FEATURE_NAMES)

    except Exception as e:
        print(f"Error processing {spacer}: {e}")
        return pd.Series([np.nan] * len(FEATURE_NAMES), index=FEATURE_NAMES)


def extract_features_batch(spacers):
    """Vectorized wrapper: extracts features for a list of spacer sequences."""
    feature_df = pd.DataFrame([extract_complete_features(s) for s in spacers])
    return feature_df
