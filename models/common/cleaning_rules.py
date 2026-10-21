"""Water-quality short names and expected value ranges used when cleaning the nutrient data."""

# Characteristic name in the raw file (lower-cased) -> short feature name.
NUTRIENT_MAP = {
    "nitrogen": "tn",
    "phosphorus as p": "tp",
    "nitrogen, nitrite (no2) + nitrate (no3) as n": "nox",
    "phosphorus, phosphate (po4) as p": "po4",
    "nitrogen, ammonia as n": "nh3",
    "chlorophyll a, corrected for pheophytin": "chl_insitu",
    "temperature, water": "water_temp_f",
    "specific conductance": "spcond",
    "turbidity": "turbidity",
    "dissolved oxygen (do)": "do",
    # Not in the main feature sets yet; kept so they can be tested (feature group "water_quality_extra").
    "bod, biochemical oxygen demand": "bod",
    "chlorophyll a, uncorrected for pheophytin": "chl_uncorr",
    "dissolved oxygen saturation": "do_sat",
    "fecal coliform": "fecal_coliform",
    "nitrogen, kjeldahl": "tkn",
    "ph": "ph",
    "total suspended solids (tss)": "tss",
    "true color": "color",
}

# Expected ranges (gross range test, U.S. IOOS 2020); values outside are removed
# (e.g. 412 F water temperature, 0 uS/cm conductance in a tidal bay).
NUTRIENT_VALID = {
    "water_temp_f": (40, 100), "spcond": (100, 70000), "do": (0, 25),
    "chl_insitu": (0, 1000), "tn": (0, 20), "tp": (0, 5),
}
