"""
SAR detection constants.

Ported from SAMUDRA NETRA ``config.SAR`` (see docs/MERGE_ARCHITECTURE.md, SAR
detection = selected [SN]). Every value keeps its original citation. Kept as a
plain class - no environment overrides - because these are physical/algorithmic
constants the trained classifier depends on, not deployment knobs.
"""
from __future__ import annotations


class SAR:
    # -- Sentinel-1 IW GRD nominal properties -----------------------------
    NOMINAL_PIXEL_M = 10.0            # 10 m ground range detected
    EQUIV_NUM_LOOKS = 4.4            # ENL of S1 IW GRDH (Sentinel-1 PHB)
    NOMINAL_INCIDENCE_DEG = 33.0     # mid-swath IW incidence angle

    # -- Refined-Lee speckle filter -------------------------------------
    LEE_WINDOW = 7                   # kernel size (odd)
    LEE_DAMPING = 1.0

    # -- dark-spot segmentation ---------------------------------------
    ADAPTIVE_BLOCK = 151            # local-mean window for adaptive threshold
    ADAPTIVE_OFFSET_DB = 2.2        # dark if pix < local_mean - offset (dB)
    MIN_SPILL_AREA_PX = 350         # pixel floor for a candidate
    MIN_SPILL_AREA_KM2 = 0.5        # physical floor (whichever is larger wins)
    MAX_SPILL_AREA_FRAC = 0.35      # a "spill" over 35% of the scene = artefact
    MORPH_KERNEL = 5
    LAND_BACKSCATTER_DB = -6.0      # land is much brighter than calm sea

    # -- classifier decision -----------------------------------------
    # Tuned on SN's simulated radiometry (sea ~ -11 dB). Sits on the recall side
    # of the knee: a missed spill costs more than a checked look-alike.
    OIL_PROBABILITY_THRESHOLD = 0.30
    # Real Sentinel-1 sits near -28 dB; recalibrated on the Zenodo test split.
    REAL_DECISION_THRESHOLD = 0.75
    CONFIDENCE_BANDS = {"HIGH": 0.70, "MEDIUM": 0.45, "LOW": 0.30}

    # -- look-alike context gates (STEP 4) -------------------------------
    # A "slick" in wind this strong, or hard against the surf zone, or with a
    # diffuse edge and round shape, is treated as a look-alike regardless of the
    # classifier probability.
    LOOKALIKE_MAX_WIND_MS = 13.0        # above this Bragg damping is not usable
    LOOKALIKE_MIN_COAST_KM = 0.6        # inside the surf zone radiometry is junk
    LOOKALIKE_MIN_BORDER_GRAD = 0.06    # dB/px; below this the edge is diffuse
    LOOKALIKE_MAX_SPREADING = 82.0      # Solberg spreading; ~100 = round blob
    LOOKALIKE_MIN_CONTRAST_DB = 1.8     # weaker than this is indistinguishable
