"""Centralized paths, constants, and configuration for the GAL framework."""

import os

# Project root (GAL/ directory)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

# Data files
SPACER_DATA = os.path.join(DATA_DIR, 'spacer_data.csv')
VIENNA_FEATURES = os.path.join(DATA_DIR, 'spacer_ViennaRNA_features.csv')
ONEHOT_FEATURES = os.path.join(DATA_DIR, 'spacer_onehot_features.csv')
NT_FEATURES = os.path.join(DATA_DIR, 'spacer_NT_features.csv')
ADVANCED_FEATURES = os.path.join(DATA_DIR, 'spacer_advanced_features2.csv')
TOP10_OPTIMIZED = os.path.join(DATA_DIR, 'top10_optimized.csv')
ALL_VIRTUAL_PREDICTIONS = os.path.join(DATA_DIR, 'all_virtual_spacer_predictions.csv')

# Biophysical constants (standardized across all modules)
UPSTREAM_SEQ = "TTTGTTTAACTTTAAGAAGGAGA"
DOWNSTREAM_SEQ = "atgGTTAGCAAAGGTGAAGAACTGTTTAC"
SD_CORE = "AAGAAGGAGA"
ANTI_SD = "ACCUCCUUA"

# Physicochemical parameter maps
MW_MAP = {'A': 135.13, 'G': 151.13, 'C': 111.10, 'U': 112.09}
STACK_MAP = {'A': -7.6, 'G': -8.2, 'C': -6.8, 'U': -6.2}

# Feature names (after removing Spacing_Penalty bias, adding length one-hot)
FEATURE_NAMES = [
    'Spacing', 'is_len_6', 'is_len_7', 'is_len_8',
    'Total_MW', 'Purine_Ratio', 'Total_Stacking',
    'GC_Front', 'GC_Back', 'DG_total', 'DG_RBS',
    'RBS_acc', 'ATG_acc', 'GC_spacer'
]

# Physical constraint thresholds (documented as post-hoc from training data)
DG_MIN = -8.0            # Minimum DG_total (kcal/mol); see docstring in exploration.py
LENGTH_MIN = 6           # Minimum spacer length
LENGTH_MAX = 8           # Maximum spacer length

# Default hyperparameters
DEFAULT_RANDOM_STATE = 42
DEFAULT_KAPPA = 10.0
DEFAULT_N_ENSEMBLE = 5
DEFAULT_OUTER_CV = 5
DEFAULT_INNER_CV = 3

# RFE configuration
RFE_N_FEATURES_MAX = 30
RFE_STEP_FRACTION = 10  # step = n_features / RFE_STEP_FRACTION
