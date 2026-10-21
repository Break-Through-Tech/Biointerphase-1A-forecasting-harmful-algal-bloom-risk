"""Feature groups (columns of the weekly table, grouped by source) and the feature sets the models compare."""

# Every other bay/creek characteristic, as 8-week medians. Not part of any feature set yet.
NUTRIENT_EXTRA = ["bay_bod", "bay_ph", "bay_tkn", "bay_do_sat", "bay_tss", "bay_color",
                  "creek_po4", "creek_nh3", "creek_chl_insitu", "creek_chl_uncorr", "creek_spcond", "creek_turbidity",
                  "creek_do", "creek_do_sat", "creek_fecal_coliform", "creek_tkn", "creek_bod", "creek_ph",
                  "creek_tss", "creek_color"]

FEATURE_GROUPS = {
    "season": ["woy_sin", "woy_cos"],
    "kb_lags": ["kb_max_log", "kb_p95_log", "kb_mean_log", "kb_frac_1e4", "kb_frac_1e5", "kb_n_samples",
                "rkb_max_log", "rkb_frac_1e5", "rkb_n_samples", "last_class", "weeks_since_sample",
                "weeks_since_bloom", "kb_max_log_L1", "kb_max_log_L2", "kb_frac_1e5_L1", "kb_frac_1e5_L2",
                "rkb_max_log_L1", "rkb_max_log_L2", "kb_max_log_4w", "kb_max_log_8w", "rkb_max_log_4w"],
    "satellite_chl": ["chl_p90", "chl_frac_gt1", "chl_frac_gt5", "chl_coastal_mean", "chl_mean_log",
                      "chl_anomaly", "chl_coastal_anomaly", "chl_mean_log_change_1w"],
    "water_quality": ["bay_tn_8w", "bay_tp_8w", "bay_nox_8w", "bay_po4_8w", "bay_chl_insitu_8w",
                      "bay_spcond_8w", "bay_turbidity_8w", "bay_do_8w", "creek_tn_8w", "creek_tp_8w", "creek_nox_8w"],
    # Left-out water-quality characteristics: available for feature selection, not in any FEATURE_SET yet.
    "water_quality_extra": [f"{c}_8w" for c in NUTRIENT_EXTRA],
    "water_temp": ["water_temp_c", "water_temp_change_2w", "warm_weeks_8w", "habsos_salinity"],
    "air_rain": ["air_temp_f", "air_temp_max_f", "rain_in_1w", "rain_in_4w", "rain_in_8w", "rain_in_12w"],
    # Short-coverage sources (2019+/2020+/2022+): mostly NaN in early years.
    "wind_short": ["wind_u", "wind_v", "wind_speed", "wind_v_4w", "wind_u_4w", "grid_uwnd", "grid_uwnd_4w",
                   "ksrq_air_temp_c", "pressure_mb", "mote_water_temp_c"],
}

FEATURE_SETS = {
    # Does the history of K. brevis alone carry the forecast? (Medina: top features were lagged Kb)
    "kb_only": ["season", "kb_lags"],
    # Can the environment alone predict blooms, without being told about current cell counts?
    "env_only": ["season", "satellite_chl", "water_quality", "water_temp", "air_rain"],
    # Everything with 2010+ coverage.
    "long": ["season", "kb_lags", "satellite_chl", "water_quality", "water_temp", "air_rain"],
    # Long + the short-coverage wind/Mote sources (NaN before they start).
    "full": ["season", "kb_lags", "satellite_chl", "water_quality", "water_temp", "air_rain", "wind_short"],
}


def list_feature_columns(feature_set: str) -> list[str]:
    """List the feature columns in a feature set, group by group.

    Args:
        feature_set (str): A key of FEATURE_SETS: "kb_only", "env_only", "long" or "full".

    Returns:
        list[str]: Column names, in FEATURE_GROUPS order.

    Example:
        >>> len(list_feature_columns("kb_only")), len(list_feature_columns("full"))
        (23, 62)
    """
    return [c for g in FEATURE_SETS[feature_set] for c in FEATURE_GROUPS[g]]
