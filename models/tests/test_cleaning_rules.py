"""Unit tests for common/cleaning_rules.py."""
from common.cleaning_rules import NUTRIENT_MAP, NUTRIENT_VALID


def test_nutrient_map_keys_are_lower_case():
    assert all(k == k.lower() for k in NUTRIENT_MAP)


def test_nutrient_map_short_names_are_unique():
    assert len(set(NUTRIENT_MAP.values())) == len(NUTRIENT_MAP)


def test_every_valid_range_belongs_to_a_mapped_short_name():
    assert set(NUTRIENT_VALID) <= set(NUTRIENT_MAP.values())


def test_valid_ranges_have_low_below_high():
    assert all(lo < hi for lo, hi in NUTRIENT_VALID.values())
