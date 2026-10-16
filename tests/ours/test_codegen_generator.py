"""generator.resolve: library hits only (the frozen library the paper ran).

No network, no LLM: a hit runs a seeded function, a miss falls back.
"""

from panini_dr.adapters.ours.codegen_library import generator, registry


def test_library_hit():
    lib = registry.Library.load()
    value, prov = generator.resolve(
        lib,
        key=("diff", ("date", "date"), "days"),
        inputs=["2015-03-15", "2015-03-20"],
        expected_type="number",
    )
    assert value == 5
    assert prov == "library_hit"


def test_library_miss_falls_back():
    lib = registry.Library.load()
    value, prov = generator.resolve(
        lib, key=("mul", ("number", "number"), "product"),
        inputs=[6, 7], expected_type="number",
    )
    assert value is None
    assert prov == "fallback"


def test_wrong_typed_hit_falls_back():
    lib = registry.Library.load()
    value, prov = generator.resolve(
        lib,
        key=("diff", ("date", "date"), "days"),
        inputs=["2015-03-15", "2015-03-20"],
        expected_type="bool",
    )
    assert value is None
    assert prov == "fallback"
