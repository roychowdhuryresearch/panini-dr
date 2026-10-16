"""Task 1: seed date functions + library registry (gsw_codegen_compute).

TDD red->green. Seeds compute day-granular date diffs and birthday-aware
ages; the registry seeds those under stable operation keys.
"""

from panini_dr.adapters.ours.codegen_library import registry, seeds


def test_date_diff_days_day_granular():
    assert seeds.date_diff_days("2015-03-15", "2015-03-20") == 5
    assert seeds.date_diff_days("March 15, 2015", "2015-03-20") == 5  # mixed formats


def test_date_diff_days_absolute_and_reversed():
    # Absolute count irrespective of argument order.
    assert seeds.date_diff_days("2015-03-20", "2015-03-15") == 5


def test_age_years_on_birthday_aware():
    assert seeds.age_years_on("1965-06-30", "2020-02-22") == 54  # birthday not yet reached
    assert seeds.age_years_on("1965-06-30", "2020-07-01") == 55


def test_age_years_on_birthday_exact_day():
    # On the birthday itself the completed-year count ticks over.
    assert seeds.age_years_on("1965-06-30", "2020-06-30") == 55


def test_registry_seeded_lookup():
    lib = registry.Library.load()
    fn = lib.lookup(("diff", ("date", "date"), "days"))
    assert fn is not None
    assert callable(fn)
    assert fn("2015-03-15", "2015-03-20") == 5


def test_registry_seeded_age_lookup():
    lib = registry.Library.load()
    fn = lib.lookup(("diff", ("date", "date"), "age_years"))
    assert fn is not None
    assert fn("1965-06-30", "2020-02-22") == 54


def test_registry_miss_returns_none():
    lib = registry.Library.load()
    assert lib.lookup(("diff", ("number", "number"), "no_such_intent")) is None
