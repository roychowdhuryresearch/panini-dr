from panini_dr.adapters.ours import _gsw_temporal as gt


def test_finish_value_copied_semantics():
    assert gt.finish_value(12.0, value_type="number", question="years only") == "12"
    assert gt.finish_value(12.15, value_type="number", question="Represent in years only.") == "12"
    assert gt.finish_value(8140.0, value_type="number",
                           question="How old, rounded to the nearest year?") == "22"
    assert gt.finish_value("Shane Gillis", value_type="entity", question="who") == "Shane Gillis"


def test_finish_value_does_not_mangle_year_answers():
    # Review finding 1: the >150 days-defense must NOT fire on year/duration
    # questions whose answer is a legitimate large number. It fires ONLY under
    # genuine age semantics, and never when the value is itself a plain 4-digit
    # calendar year or the target is date-typed. Otherwise "In what year..."
    # (answer 1994) and "how many years..." (answer 220) get mangled into
    # nonsense, regressing the exact year class L1 is meant to improve.
    assert gt.finish_value("1994", value_type="number",
                           question="In what year was the company founded?") == "1994"
    assert gt.finish_value("2008", value_type="date",
                           question="In which year did it launch?") == "2008"
    assert gt.finish_value("1985", value_type="number",
                           question="In what year was it built?") == "1985"
    assert gt.finish_value("220", value_type="number",
                           question="How many years between 1800 and 2020?") == "220"
    # date-typed target is never treated as days even under an age cue
    assert gt.finish_value("2001", value_type="date",
                           question="how old — in what year?") == "2001"
    # float-shaped 4-digit year is likewise protected
    assert gt.finish_value(1994.0, value_type="number",
                           question="how old, in what year did it start?") == "1994"
    # q630 age class STILL converts a mis-routed day count -> years
    assert gt.finish_value(8140.0, value_type="number",
                           question="How old, rounded to the nearest year?") == "22"


def test_finish_value_percentage_fraction_is_scaled():
    """A 'what percentage' question answered with a bare fraction reads as a percent.

    monaco_448 asked "What percentage of Emmy Award winning shows ... were won by
    shows about royal families?" (gold 38.46) and the pipeline emitted 0.2857142857
    -- the fraction, not the percent. Scored as-is that is 0.0; as 28.57 it is the
    closer of the two arms. Only a bare fraction strictly inside (0, 1) is scaled:
    1.0 is left alone because "1" is a legitimate percentage answer and cannot be
    told apart from 100%.
    """
    pct_q = "What percentage of Emmy winning shows were about royal families?"
    assert gt.finish_value(0.2857142857, value_type="number", question=pct_q) == "28.57142857"
    assert gt.finish_value("0.5", value_type="number", question=pct_q) == "50"
    # already a percentage -> untouched
    assert gt.finish_value(38.46, value_type="number", question=pct_q) == "38.46"
    # boundaries are left alone: 0 carries no fraction/percent ambiguity to fix,
    # and 1.0 is ambiguous between 1% and 100%
    assert gt.finish_value(1.0, value_type="number", question=pct_q) == "1"
    assert gt.finish_value(0.0, value_type="number", question=pct_q) == "0"
    # no percent cue -> untouched (a bare ratio answer stays a ratio)
    assert gt.finish_value(0.2857142857, value_type="number",
                           question="What fraction of shows were about royalty?") == "0.2857142857"
    # non-numeric still passes through
    assert gt.finish_value("The Crown", value_type="entity", question=pct_q) == "The Crown"


