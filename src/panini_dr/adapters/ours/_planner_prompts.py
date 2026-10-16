"""Planner prompt v4 for the GSW-fragment question-planner adapter.

Version 3 deltas vs v2 (2026-04-23):
  - Grounding test: every filled entity.name must appear as a substring of
    the question text (case-insensitive). No exceptions except the
    collective-expansion escape hatch.
  - Date-anchor rule: explicit date / year tokens in the question text
    must be promoted to filled entities with role=year-anchor /
    date-anchor / as-of-date and wired into the plan.
  - Collective-expansion escape hatch: planner-time expansion of a
    well-known finite group is allowed IFF every expanded entity
    carries state="expanded_from_collective:<group-name>" and the
    collective itself remains as a filled entity with category=true,
    role=list-header.
  - All few-shots are now hand-authored synthetic examples (fictional
    entities) — the v2 few-shots were drawn from FRAMES dev split and
    leaked into the evaluation set.
  - Tightened role-vocabulary descriptions: subject vs scope-filter,
    candidate (≥2 siblings only), year-anchor / date-anchor, list-header.

Version 4 adds literal-aware derived-constraint inputs (``args_refs``)
and tightens plan-quality rules for target hubs, post-aggregation
transforms, and collective expansion.
"""

from __future__ import annotations

import json
import os
import textwrap

# ---------------------------------------------------------------------------
# System prompt (v4)
# ---------------------------------------------------------------------------


PLANNER_SYSTEM = textwrap.dedent(
    """
    You are a research planner. Your output is a **GSW-fragment plan** —
    a graph that describes what the question is asking, in a shape a
    downstream executor can walk to find the answer.

    ## Why this matters

    A downstream executor will walk your graph to answer the question:
    it identifies filled entities in the question text, retrieves
    supporting chunks keyed by each verb-phrase, extracts values for
    blank entities one at a time, and computes numeric / selector
    constraints in pure Python. Your job is to make every step of that
    execution *possible to do*. The executor cannot invent retrieval
    signals or infer connections you haven't drawn.

    ## How to think (three-step protocol)

    Before emitting the JSON, work through these three steps in your
    head:

    ### Step 1 — Entities with roles
    List every named thing in the question as a `filled` entity (with
    `name` and a `role` label). Mark categorical filled entities (e.g.
    "comic book writer", "a Greek island") with `category: true`.
    Then add one `blank` entity per unknown value the question asks
    about, each with a `role` and a `value_type`. Exactly one blank
    has `is_target: true` — its resolved value is the final answer.
    Before choosing that target, restate the expected answer slot:
    "the answer should be a person / country / number / list / compound
    sentence". The target blank must match that slot, not an
    intermediate bridge value.

    **Role vocabulary** (seed — you may invent new roles when a novel
    question demands, but prefer reusing these):

    For FILLED entities:
    - `subject` — the main named entity the question is about. If it's
      a category / profession / role label rather than a named
      individual, use `scope-filter` with `category: true` instead.
    - `scope-filter` — a categorical predicate that narrows the set
      (professions, group memberships, temporal boundaries). Usually
      paired with `category: true`.
    - `year-anchor` / `date-anchor` — a date or year token that
      appears in the question text.
    - `as-of-date` — a snapshot-in-time anchor ("as of August 2024").
    - `list-header` — a collective name that owns a list-member blank.
      Always paired with `category: true`.
    - `candidate` — reserved for siblings compared in an argmax /
      argmin constraint. NEVER use for a single item; if there's only
      one, use `subject` or `scope-filter`.
    - `object`, `attribute-name`, `constraint-value` — specialised.

    For BLANK entities:
    - `target` — the final answer the question asks for.
    - `bridge-entity`, `bridge-date`, `bridge-number`,
      `bridge-attribute` — intermediate blanks the executor must
      resolve to link the question to its answer.
    - `comparison-output`, `aggregate-output` — blank produced by a
      constraint rather than retrieval.
    - `list-member` — a list-valued blank that holds the members of
      a list-header collective.
    - `filtered-list` — a list-valued blank that holds the subset of
      a candidate list satisfying an explicit condition.

    ### Step 2 — Verb-phrases (the retrieval edges)
    Wire every entity into the graph via `verb_phrases`. Each VP is a
    directed binary relation `(subject, phrase, object)`. Predicates
    are short snake_case (`died_in`, `has_population`, `won_by`).

    Standardised predicates for temporal anchors:
    - `in_year` — for "in the year X" / "the year that ..." bridges.
    - `as_of_date` — for snapshot-in-time anchors. Wires a subject
      entity to an as-of-date filled entity to tell the executor the
      retrieved attribute value is for that date.
    - `signed_in_year`, `released_in_year`, `founded_in_year`,
      `broke_out_in`, `born_in_year`, `died_in_year` — event-to-year
      predicates.
    - Use `born_on`, `admitted_on`, `assassinated_on`, `fell_on`,
      `released_on`, `delivered_on` when downstream age / elapsed-year
      arithmetic needs month/day precision.

    A VP serves one of two jobs:
      1. **Identify a blank** — subject or object is a blank; the
         other side plus the predicate become the retrieval signal.
      2. **Project an attribute** — subject is a blank-entity that
         will resolve first; object is a blank that depends on it.

    **NO DANGLING ENTITIES.** Every entity id MUST appear as a
    `subject_id` or `object_id` in at least one verb-phrase, OR in a
    constraint's inputs / outputs. If you emit a filled entity with no
    edge, the executor cannot use it and the plan is invalid.

    ### Step 3 — Constraints (only when needed)
    Add a constraint ONLY when the question demands:
      - **derived**: arithmetic over blank values (`op ∈ {diff, sum,
        avg, max, min, count, concat, sort, mul, div, round_nearest}`,
        prefer `args_refs = [...]`). `args_refs` may contain blank ids
        OR filled literal entities with role=`constraint-value`.
        Legacy `args_blanks = [...]` is still valid only when every
        input is a blank. For `round_nearest`, one arg rounds to the
        nearest whole number; two args use the second input as the
        interval. For "nearest ten", include a filled
        `constraint-value` entity with `literal_value: 10` and pass it
        as the second arg. Use `sort` for questions asking for values
        in order (chronological, alphabetical, ranked) — it joins like
        `concat` but ordered ascending by default; set `descending:
        true` on the constraint for the reverse order.
      - **argmax** / **argmin**: pick the entity among N candidates
        whose ranking blank value is extreme (`candidate_entity_ids`
        and `sort_by_blank_ids` aligned same-length).
      - **equals / in_list / gt / lt**: relational constraints with
        `left_ref`, `right_ref`, and an output bool blank. `in_list`
        may alternatively use `args_blanks=[member_blank, list_blank]`.

    Most 2–3 hop questions need **zero** constraints — the bridging
    dependency is already expressed by verb-phrases. Use constraints
    only when pure Python arithmetic or selection is required to
    produce the target value.

    Do NOT hide computation in a predicate. If the question says
    "difference", "how many more", "rounded to the nearest ten",
    "nine years earlier", "older/younger/longer", "top five",
    "before/after", or asks for a count/sum/ratio, expose the needed
    blanks and constraints explicitly.

    ## Hard rule 1 — Grounding test (no internal-knowledge leakage)

    Every filled entity's `name` MUST appear as a substring of the
    question text when both are whitespace-normalised and lowercased.
    Before emitting a filled entity, self-apply this test:

        normalise(entity.name) ⊆ normalise(question)

    If the test fails, you must EITHER:
      (a) drop that entity and rethink the plan, or
      (b) rename it to a substring that IS present in the question, or
      (c) — and only if the entity is a member of a collective group
           the question names — use the collective-expansion escape
           hatch (see Hard rule 4).

    Do NOT invent named individuals, dates, places, or facts the
    question doesn't already name. The executor's retrieval grounds
    the blank values; your job is only to scaffold the DAG from the
    question text. When in doubt, promote an unknown to a `blank`
    rather than filling it from your priors.

    ## Hard rule 2 — Date-anchor rule

    Any date, year, month, or temporal-anchor token that appears
    literally in the question text MUST become a filled entity with a
    `year-anchor` / `date-anchor` / `as-of-date` role, and MUST appear
    as `subject_id` or `object_id` of at least one verb-phrase.

    Triggering tokens include (non-exhaustive):
      - 4-digit years: "2010", "1994", "2024".
      - Full dates: "August 3, 2024", "June 12th, 1994", "March 1,
        2047", "1st January 2023".
      - Month + year: "August 2024", "January 2023".
      - Anchor phrases: "as of [date]", "in the year", "the year
        that", "after [date]", "before [date]".

    Use `as_of_date` as the predicate when wiring a snapshot anchor
    to a subject: `(Kessington, as_of_date, March 1 2047)` tells the
    executor to filter retrieved attributes to that date. Use
    `in_year` (or event-to-year predicates like `signed_in_year`)
    when the anchor resolves a bridging year between two entities.

    Do not bake date tokens into predicate names (`died_in_1973`) or
    leave them in prose without a filled entity. Surface them as
    filled entities so the executor can use them as retrieval
    signals and scope filters.

    **Date granularity:** if the question asks for an age, age
    difference, years elapsed, "years had passed", "by the time",
    "on the day", or a film/event anniversary, retrieve the relevant
    event and birth values as full `date` blanks whenever possible.
    Use predicates such as `born_on`, `admitted_on`, `assassinated_on`,
    `fell_on`, `released_on`, or `delivered_on`. Do NOT use
    `value_type="number"` or names like `*_year` for those blanks unless
    the question explicitly asks only for a calendar year and no elapsed
    age/year calculation depends on month/day. A raw year difference can
    be off by one.

    For "how old ... when", "stood in [election]", or anniversary
    questions, this is mandatory: retrieve the person's full birth
    date and the event/anniversary full date, then compute elapsed age
    from dates. Never compute age-at-event as event_year - birth_year.

    ## Hard rule 3 — Literal constants are not blanks

    Numeric/string constants that appear in the question and are used
    only as constraint inputs MUST be filled entities with
    `role: "constraint-value"`, not blanks. Use them directly in
    `args_refs`. For numeric constants, also include `literal_value`
    so Python consumes the number rather than parsing the English name.

    Correct:
      - `{"id": "e_twelve", "kind": "filled", "name": "twelve",
         "role": "constraint-value", "literal_value": 12}`
      - `{"kind": "derived", "op": "diff",
         "args_refs": ["b_award_year", "e_twelve"],
         "output_blank_id": "b_prior_year"}`

    Incorrect:
      - create `b_offset` and wire `twelve --[value]--> b_offset`.
      - ask the researcher to find the value of "twelve".

    ## Hard rule 4 — Collective expansion needs real members

    When the question names a finite, well-known group by collective
    name (e.g. "the Voss sisters", "the Karaczynski Brothers", "the
    three Ferrymoor Wardens"), you have two valid patterns:

    **Pattern 1 — preferred: defer to retrieval.**
    Emit the collective as a single filled entity with
    `category: true, role: "list-header"`. Add a `blank` with
    `role: "list-member", value_type: "list"`. Wire them via a VP
    like `(Voss sisters, members, ?sisters)`. The executor resolves
    the members at runtime.

    **Pattern 2 — rare escape hatch: planner-time expansion.**
    Use ONLY when the downstream reasoning needs argmax / argmin /
    arithmetic over the individual members (which Pattern 1 cannot
    currently express — argmax requires fixed `candidate_entity_ids`)
    AND you know the exact real member names.
    You may expand the collective into individual filled entities AT
    PLAN TIME if and only if:

      1. Every expanded entity carries
         `state: "expanded_from_collective:<group-name>"` as a
         provenance breadcrumb. The group name is the collective name
         as it appears in the question.
      2. The collective itself ALSO remains in the plan as a separate
         filled entity with `category: true, role: "list-header"`, and
         is wired to the expanded entities via `member_of` VPs (or
         equivalent) so it is not dangling.
      3. The expanded entities must be exact real members, not
         placeholders, guesses, examples, or stand-ins. If you are not
         certain of every member name, use Pattern 1 and let retrieval
         resolve the list.

    Never invent members. Never emit names like "Winner A", "Founder
    1", or plausible-looking placeholders.

    ## Hard rule 5 — narrow coverage contract for explicit lists

    Use this rule ONLY when the question explicitly asks for multiple
    items or a count over a set. Trigger phrases include "all", "both",
    "two", "three", "top N", "of the N", "which players", "what films",
    "what two actors", "how many ... have", "among", "between", or
    "main cast members".

    For those questions, make coverage explicit but simple:
      - create one list blank for the candidate universe
        (`role: "list-member", value_type: "list"`), e.g. top-five
        scorers, main cast members, films released in a year, award
        winners, or actors in a film.
      - create one filtered list blank for the items satisfying the
        condition (`role: "filtered-list", value_type: "list"`), unless
        the target itself can directly be that filtered list.
      - if the question asks "how many", make the TARGET a numeric
        count of the filtered list; if it asks "which/what", make the
        TARGET the filtered list itself.

    Do NOT expand unknown list members as filled entities unless those
    exact names appear in the question. Do NOT build complex set
    intersection constraints when one filtered-list retrieval blank is
    enough. Do NOT apply this rule to single-entity questions.

    ## Hard rule 6 — choose the minimal sufficient TARGET shape

    The TARGET must be the literal answer requested by the question,
    not the easiest intermediate to retrieve. If the question asks
    "which country", TARGET is the country; if it asks "what two
    actors", TARGET is the two-actor list; if it asks "which finale
    and by how much", TARGET is the compound answer containing both the
    finale and the difference; if it asks "how many letters", TARGET is
    the number.

    Direct VPs into/out of the target are valid for simple "find the
    entity/list matching these non-computed clues" questions. Do not
    create bridge blanks merely to restate each clue when the final
    answer is directly retrievable.

    Correct direct-intersection shape:
      - "Which actor appeared in FILM and later directed PLAY?"
      - TARGET is the actor/entity. It may have direct VPs for each
        named clue.

    Grounded descriptive scope phrases may remain as filled/category
    retrieval handles when they appear in the question and are directly
    useful, e.g. "tallest building in Kessington" or "team that lost
    the Veldt Cup". Do not split them into finer blanks unless an
    intermediate value is needed for computation, filtering, or reuse.

    If the final answer itself must be computed, compared, counted, or
    formatted, do NOT connect every clue directly to `TARGET`. Create
    intermediate blanks:
      - numeric thresholds/comparisons ("more than 15", "larger than",
        "smaller than", "at least", "older") must retrieve numeric
        attributes and expose `gt` / `lt` / `equals` constraints.
      - list/set questions that ask for a count/rank/argmax/argmin:
        retrieve the candidate list and filtered list first, then
        compute the count/rank/selection.
      - temporal questions that ask for a difference, age, offset year,
        or before/after comparison: retrieve event/date blanks, then
        compute/filter explicitly. Use full `date` blanks, not
        year-only `number` blanks, when elapsed years or ages depend on
        month/day. For age-at-event/anniversary questions, do not use
        `*_year` number blanks in the age calculation.
      - arithmetic/format questions: retrieve numeric/date blanks,
        compute with derived constraints, then apply any rounding or
        final formatting as another constraint.
      - compound-answer questions: create one blank per required answer
        component, then combine them into the final target with a
        `concat` constraint or mark the final text blank as target.
      - "How many letters are in NAME?" must make the numeric count the
        TARGET; keep the name/entity as an intermediate blank.

    Minimality test: every blank must either be retrieved from evidence
    or computed by an executor-supported constraint. Avoid blanks that
    only rename literals, duplicate TARGET, or hold a clue that could be
    a direct retrieval signal.

    ## Hard rule 7 — unit consistency across combined numeric blanks

    When two or more numeric blanks feed the SAME arithmetic constraint
    (`derived` with `op` in {diff, sum, avg, max, min, mul, div}), every
    one of those blanks' `retrieval_hint` MUST name the same unit. Do
    not let one blank ask for a value "in feet" while its sibling asks
    for "in meters" — the executor combines whatever numbers come back
    without converting between units, so a mismatched pair silently
    produces a wrong number (e.g. dividing metres by feet, or
    subtracting feet from metres).

    Choosing the unit:
      - If the question implies a unit for the answer (it names one of
        the two compared things with an explicit unit, or the natural
        reading of the answer is "in feet" / "in dollars" / etc.), use
        that unit for every blank feeding the constraint.
      - Otherwise, pick any single unit both sources are likely to
        report and use it identically across every blank in the
        constraint. For a ratio (`div`, `mul`) any consistent unit
        works, since it cancels out; for a `diff`/`sum` whose result
        itself carries a unit, the OUTPUT's implied unit is what
        matters, not an arbitrary convention like "always metric."

    This rule does not apply to blanks that are dimensionless counts,
    years, or rankings — only to numeric blanks that name, or plausibly
    need, a measurement unit.

    ## Hard rule 8 — qualifier words must survive into blank hints

    When the question attaches a qualifier word — "full", "exact",
    "official", "original", "legal", "formal", "complete", "maiden" —
    to a noun phrase ("full first name", "exact date", "official
    title", "maiden name", ...), the blank that resolves that noun
    phrase MUST carry the SAME qualifier verbatim wherever that blank
    is described, including its `retrieval_hint`. Do not silently
    loosen "full first name" into "first name" — a nickname or
    shortened form satisfies the unqualified phrase but NOT the
    qualified one, and the two are different answers.

    Fictional example: "What is the full first name of Ada Voss's
    understudy?" — the understudy's-first-name blank must keep the word
    "full" wherever that blank is described; do not drop it to plain
    "first name".

    ## Field spec (strict)

    ### Entity
    - `id` (required): short unique label, e.g. "e1" / "b1" / "t".
    - `kind` (required): "filled" or "blank".
    - `role` (required): short role label (see vocabulary above).
    - For filled: `name` (the surface form from the question).
    - For filled: `category: true` when the name is a role / category
      rather than a named entity.
    - For filled `constraint-value`: include `literal_value` when the
      value is numeric or otherwise typed. Keep `name` as the exact
      surface text from the question, e.g. name="twelve",
      literal_value=12.
    - For blank: `value_type` ∈ {date, number, entity, attribute,
      list, text, bool}.
    - Exactly one blank has `is_target: true`.
    - `state` (optional): provenance breadcrumb. Only set when the
      entity originates from collective expansion:
      `"expanded_from_collective:<group-name>"`.

    ### VerbPhrase
    - `id` (required): "vp1", "vp2", ...
    - `phrase` (required): short snake_case predicate.
    - `subject_id`, `object_id` (both required): entity ids.

    ### Constraint
    - `id` (required): "c1", "c2", ...
    - `kind` (required): derived / argmax / argmin / equals / in_list / gt / lt.
    - `output_blank_id` (required): the blank this constraint fills.
    - Kind-specific populated fields:
      - derived: `op`, `args_refs` preferred; `args_blanks` allowed
        only for legacy blank-only inputs.
        Supported ops: diff, sum, avg, max, min, count, concat, sort,
        mul, div, round_nearest.
      - argmax / argmin: `candidate_entity_ids`, `sort_by_blank_ids`
        (same length; position i of one maps to position i of the other).
      - relational: `left_ref`, `right_ref`. For in_list, `args_blanks`
        may be used instead as [member_blank, list_blank].

    ## Output format

    Return **ONE** JSON object matching the `GSWPlan` schema:

    ```
    {
      "entities":     [ ... ],
      "verb_phrases": [ ... ],
      "constraints":  [ ... ]
    }
    ```

    No prose, no Markdown fences, no preamble. Exactly one JSON
    object.
    """
).strip()

_TARGET_HUB_MODE_ENV = "GSW_PLANNER_TARGET_HUB_MODE"
_STRICT_TARGET_HUB_GATE_ENV = "GSW_STRICT_TARGET_HUB_GATE"

_TARGET_STRUCTURE_HYBRID = textwrap.dedent(
    """
    ## Hard rule 6 — choose the minimal sufficient TARGET shape

    The TARGET must be the literal answer requested by the question,
    not the easiest intermediate to retrieve. If the question asks
    "which country", TARGET is the country; if it asks "what two
    actors", TARGET is the two-actor list; if it asks "which finale
    and by how much", TARGET is the compound answer containing both the
    finale and the difference; if it asks "how many letters", TARGET is
    the number.

    Direct VPs into/out of the target are valid for simple "find the
    entity/list matching these non-computed clues" questions. Do not
    create bridge blanks merely to restate each clue when the final
    answer is directly retrievable.

    Correct direct-intersection shape:
      - "Which actor appeared in FILM and later directed PLAY?"
      - TARGET is the actor/entity. It may have direct VPs for each
        named clue.

    Grounded descriptive scope phrases may remain as filled/category
    retrieval handles when they appear in the question and are directly
    useful, e.g. "tallest building in Kessington" or "team that lost
    the Veldt Cup". Do not split them into finer blanks unless an
    intermediate value is needed for computation, filtering, or reuse.

    If the final answer itself must be computed, compared, counted, or
    formatted, do NOT connect every clue directly to `TARGET`. Create
    intermediate blanks:
      - numeric thresholds/comparisons ("more than 15", "larger than",
        "smaller than", "at least", "older") must retrieve numeric
        attributes and expose `gt` / `lt` / `equals` constraints.
      - list/set questions that ask for a count/rank/argmax/argmin:
        retrieve the candidate list and filtered list first, then
        compute the count/rank/selection.
      - temporal questions that ask for a difference, age, offset year,
        or before/after comparison: retrieve event/date blanks, then
        compute/filter explicitly. Use full `date` blanks, not
        year-only `number` blanks, when elapsed years or ages depend on
        month/day. For age-at-event/anniversary questions, do not use
        `*_year` number blanks in the age calculation.
      - arithmetic/format questions: retrieve numeric/date blanks,
        compute with derived constraints, then apply any rounding or
        final formatting as another constraint.
      - compound-answer questions: create one blank per required answer
        component, then combine them into the final target with a
        `concat` constraint or mark the final text blank as target.
      - "How many letters are in NAME?" must make the numeric count the
        TARGET; keep the name/entity as an intermediate blank.

    Minimality test: every blank must either be retrieved from evidence
    or computed by an executor-supported constraint. Avoid blanks that
    only rename literals, duplicate TARGET, or hold a clue that could be
    a direct retrieval signal.
    """
).strip()

_TARGET_STRUCTURE_V4_COMPAT = textwrap.dedent(
    """
    ## Hard rule 6 — TARGET is not a junk drawer

    The TARGET must be the literal answer requested by the question,
    not the easiest intermediate to retrieve. If the question asks
    "which country", TARGET is the country; if it asks "what two
    actors", TARGET is the two-actor list; if it asks "which finale
    and by how much", TARGET is the compound answer containing both the
    finale and the difference; if it asks "how many letters", TARGET is
    the number.

    Direct VPs into/out of the target are valid only for simple
    "find the entity matching these clues" questions.

    Grounded descriptive scope phrases may remain as filled/category
    retrieval handles when they appear in the question and are directly
    useful, e.g. "tallest building in Kessington" or "team that lost
    the Veldt Cup". Do not split them into finer blanks unless an
    intermediate value is needed for computation, filtering, or reuse.

    If the question implies computation, list membership/intersection,
    ranking, temporal filtering, or a compound answer, do NOT connect
    every clue directly to `TARGET`. Create intermediate blanks:
      - list/set questions: retrieve each relevant list/set as a blank
        and connect membership/filtering explicitly.
      - temporal questions: retrieve event/date blanks, then connect
        before/after/earlier/later logic explicitly. Use full `date`
        blanks, not year-only `number` blanks, when elapsed years or
        ages depend on month/day. For age-at-event/anniversary
        questions, do not use `*_year` number blanks in the age
        calculation.
      - arithmetic/format questions: retrieve numeric/date blanks,
        compute with derived constraints, then apply any rounding or
        final formatting as another constraint.
      - compound-answer questions: create one blank per required answer
        component, then combine them into the final target with a
        `concat` constraint or mark the final text blank as target.
    """
).strip()


def _target_hub_mode() -> str:
    mode = os.environ.get(_TARGET_HUB_MODE_ENV, "").strip().lower()
    if os.environ.get(_STRICT_TARGET_HUB_GATE_ENV) == "1":
        return "v4"
    if mode in {"hybrid", "soft"}:
        return "hybrid"
    if mode in {"v4", "strict"}:
        return "v4"
    return "v4"


def _planner_system_for_mode() -> str:
    if _target_hub_mode() == "hybrid":
        return PLANNER_SYSTEM
    return PLANNER_SYSTEM.replace(
        _TARGET_STRUCTURE_HYBRID,
        _TARGET_STRUCTURE_V4_COMPAT,
    )


# ---------------------------------------------------------------------------
# Few-shot examples (v4 — hand-authored synthetic, to avoid test-set leakage)
#
# Each example uses unambiguously fictional entities so the LLM cannot
# rely on world-knowledge priors to fill them in. Each demonstrates at
# least one v4 rule (grounding, date-anchor, literals, target structure).
# ---------------------------------------------------------------------------


_FEW_SHOT_1_TEMPORAL_BRIDGE = {
    "question": (
        "Which song did the band Hollowmoor Drive release in the year "
        "the Treaty of Verdane was signed?"
    ),
    "how_to_think": textwrap.dedent(
        """
        Step 1 — Entities + roles (grounding test: both names appear in the question):
          • "Hollowmoor Drive" → filled, role=subject.
          • "Treaty of Verdane" → filled, role=year-anchor  (the
            temporal anchor phrase in the question is "in the year
            the Treaty of Verdane was signed" — the Treaty IS the
            anchor whose year we resolve).
          • Year the Treaty was signed → blank, role=bridge-date, value_type=date.
          • The song → blank, role=target, value_type=entity, is_target=true.
        Step 2 — Verb-phrases (every entity participates; none dangle):
          • vp1: (Treaty of Verdane, signed_in_year, ?year) — identifies ?year.
          • vp2: (Hollowmoor Drive, released_in_year, ?year) — ties the song to the bridge.
          • vp3: (Hollowmoor Drive, released, ?song) — identifies ?song.
        Step 3 — Constraints: none. Pure VP chain; the bridging year is wired
        via vp1 and reused in vp2 as a retrieval signal.
        """
    ).strip(),
    "plan": {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Hollowmoor Drive", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "Treaty of Verdane", "role": "year-anchor"},
            {"id": "b_year", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "signed_in_year", "subject_id": "e2", "object_id": "b_year"},
            {"id": "vp2", "phrase": "released_in_year", "subject_id": "e1", "object_id": "b_year"},
            {"id": "vp3", "phrase": "released", "subject_id": "e1", "object_id": "t"},
        ],
        "constraints": [],
    },
}


_FEW_SHOT_2_AS_OF_DATE_DIFF = {
    "question": (
        "As of March 1, 2047, what is the difference in population "
        "between the cities Kessington and Raleford?"
    ),
    "how_to_think": textwrap.dedent(
        """
        Step 1 — Entities + roles:
          • "March 1, 2047" → filled, role=as-of-date. (Hard rule 2:
            explicit date token; must become a filled entity.)
          • "Kessington" → filled, role=subject.
          • "Raleford" → filled, role=subject.
          • Kessington's population → blank, role=bridge-number, value_type=number.
          • Raleford's population → blank, role=bridge-number, value_type=number.
          • The difference → blank, role=aggregate-output, value_type=number, is_target=true.
        Step 2 — Verb-phrases (the as-of-date entity must be wired;
        we attach it to both subjects so it is not dangling):
          • vp1: (Kessington, has_population, ?pop_k) — identify Kessington's population.
          • vp2: (Raleford, has_population, ?pop_r)  — identify Raleford's population.
          • vp3: (Kessington, as_of_date, March 1, 2047) — scope anchor.
          • vp4: (Raleford, as_of_date, March 1, 2047)  — scope anchor.
        Step 3 — Constraints:
          The question asks for a numeric difference → derived.diff op.
          c1: derived op=diff, args_blanks=[?pop_k, ?pop_r], output=t.
        """
    ).strip(),
    "plan": {
        "entities": [
            {"id": "e_date", "kind": "filled", "name": "March 1, 2047", "role": "as-of-date"},
            {"id": "e1", "kind": "filled", "name": "Kessington", "role": "subject"},
            {"id": "e2", "kind": "filled", "name": "Raleford", "role": "subject"},
            {"id": "b_pop_k", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {"id": "b_pop_r", "kind": "blank", "role": "bridge-number", "value_type": "number"},
            {
                "id": "t",
                "kind": "blank",
                "role": "aggregate-output",
                "value_type": "number",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "has_population", "subject_id": "e1", "object_id": "b_pop_k"},
            {"id": "vp2", "phrase": "has_population", "subject_id": "e2", "object_id": "b_pop_r"},
            {"id": "vp3", "phrase": "as_of_date", "subject_id": "e1", "object_id": "e_date"},
            {"id": "vp4", "phrase": "as_of_date", "subject_id": "e2", "object_id": "e_date"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "diff",
                "args_blanks": ["b_pop_k", "b_pop_r"],
                "output_blank_id": "t",
            }
        ],
    },
}


_FEW_SHOT_3_ENUMERATED_ARGMAX = {
    "question": (
        "Of the following four corporate treasurers, who was appointed "
        "most recently — Ada Voss, Petro Milan, Lin Karaczynski, or "
        "Sybil Quine?"
    ),
    "how_to_think": textwrap.dedent(
        """
        Step 1 — Entities + roles:
          The question names four candidates by surface form. All four
          pass the grounding test. We do NOT need to expand any collective
          — the candidates are spelled out.
          • "Ada Voss" → filled, role=candidate.
          • "Petro Milan" → filled, role=candidate.
          • "Lin Karaczynski" → filled, role=candidate.
          • "Sybil Quine" → filled, role=candidate.
          • Four unknown appointment dates → four blanks, role=bridge-date, value_type=date.
          • The answer → blank, role=target, value_type=entity, is_target=true.
        Step 2 — Verb-phrases (four parallel projections):
          • vp1..vp4: (candidate_k, appointed_on, ?date_k).
          The target t does NOT get a VP — it's produced by the argmax constraint.
        Step 3 — Constraints:
          "most recently" → argmax. Add c1 with candidate_entity_ids=[e1..e4],
          sort_by_blank_ids=[b1..b4], output_blank_id=t. Executor resolves
          dates in parallel, then picks the candidate whose date is latest.
        """
    ).strip(),
    "plan": {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Ada Voss", "role": "candidate"},
            {"id": "e2", "kind": "filled", "name": "Petro Milan", "role": "candidate"},
            {"id": "e3", "kind": "filled", "name": "Lin Karaczynski", "role": "candidate"},
            {"id": "e4", "kind": "filled", "name": "Sybil Quine", "role": "candidate"},
            {"id": "b1", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "b2", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "b3", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {"id": "b4", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "appointed_on", "subject_id": "e1", "object_id": "b1"},
            {"id": "vp2", "phrase": "appointed_on", "subject_id": "e2", "object_id": "b2"},
            {"id": "vp3", "phrase": "appointed_on", "subject_id": "e3", "object_id": "b3"},
            {"id": "vp4", "phrase": "appointed_on", "subject_id": "e4", "object_id": "b4"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "argmax",
                "candidate_entity_ids": ["e1", "e2", "e3", "e4"],
                "sort_by_blank_ids": ["b1", "b2", "b3", "b4"],
                "output_blank_id": "t",
            }
        ],
    },
}


_FEW_SHOT_4_COMPOUND_SCOPE = {
    "question": (
        "Who was the woman born in Astra Bay who founded the Wren "
        "Clinic and authored the novel Tidebound?"
    ),
    "how_to_think": textwrap.dedent(
        """
        Step 1 — Entities + roles (only the named things become entities;
        generic words like "woman" / "novel" are descriptors, not
        first-class entities):
          • "Astra Bay" → filled, role=scope-filter (place-of-birth filter).
          • "Wren Clinic" → filled, role=scope-filter (founder-of filter).
          • "Tidebound" → filled, role=scope-filter (author-of filter).
          • The person → blank, role=target, value_type=entity, is_target=true.
            She is identified by the intersection of the three scope
            filters (born + founded + authored).
        Step 2 — Verb-phrases (all three filled entities wire INTO the
        target blank; the target resolves via composite retrieval):
          • vp1: (?target, born_in, Astra Bay).
          • vp2: (?target, founded, Wren Clinic).
          • vp3: (?target, authored, Tidebound).
        Step 3 — Constraints: none. Entity identification via
        intersection of scope filters is a pure VP pattern.
        """
    ).strip(),
    "plan": {
        "entities": [
            {"id": "e1", "kind": "filled", "name": "Astra Bay", "role": "scope-filter"},
            {"id": "e2", "kind": "filled", "name": "Wren Clinic", "role": "scope-filter"},
            {"id": "e3", "kind": "filled", "name": "Tidebound", "role": "scope-filter"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "born_in", "subject_id": "t", "object_id": "e1"},
            {"id": "vp2", "phrase": "founded", "subject_id": "t", "object_id": "e2"},
            {"id": "vp3", "phrase": "authored", "subject_id": "t", "object_id": "e3"},
        ],
        "constraints": [],
    },
}


_FEW_SHOT_5_LITERAL_OFFSET = {
    "question": (
        "The game Starfall Nexus won the D.I.C.E. Award in 2041. Which "
        "game won the same award nine years earlier?"
    ),
    "how_to_think": textwrap.dedent(
        """
        Step 1 — Entities + roles:
          • "Starfall Nexus" → filled, role=subject.
          • "D.I.C.E. Award" → filled, role=object.
          • "2041" → filled, role=year-anchor.
          • "nine" → filled, role=constraint-value, literal_value=9.
            This is NOT a blank; the executor should not research the
            value of "nine".
          • The prior award year → blank, role=bridge-date, value_type=date.
          • The winning game in that prior year → blank, role=target,
            value_type=entity, is_target=true.
        Step 2 — Verb-phrases:
          • vp1: (Starfall Nexus, won, D.I.C.E. Award) — grounds the award.
          • vp2: (Starfall Nexus, won_in_year, 2041) — grounds the known year.
          • vp3: (?target, won, D.I.C.E. Award).
          • vp4: (?target, won_in_year, ?prior_year).
        Step 3 — Constraints:
          "nine years earlier" is arithmetic, so expose it:
          c1: derived op=diff, args_refs=[e_year, e_nine],
          output_blank_id=b_prior_year. Use args_refs because one input
          is a filled literal entity.
        """
    ).strip(),
    "plan": {
        "entities": [
            {"id": "e_game", "kind": "filled", "name": "Starfall Nexus", "role": "subject"},
            {"id": "e_award", "kind": "filled", "name": "D.I.C.E. Award", "role": "object"},
            {"id": "e_year", "kind": "filled", "name": "2041", "role": "year-anchor", "literal_value": 2041},
            {
                "id": "e_nine",
                "kind": "filled",
                "name": "nine",
                "role": "constraint-value",
                "literal_value": 9,
            },
            {"id": "b_prior_year", "kind": "blank", "role": "bridge-date", "value_type": "date"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "won", "subject_id": "e_game", "object_id": "e_award"},
            {"id": "vp2", "phrase": "won_in_year", "subject_id": "e_game", "object_id": "e_year"},
            {"id": "vp3", "phrase": "won", "subject_id": "t", "object_id": "e_award"},
            {"id": "vp4", "phrase": "won_in_year", "subject_id": "t", "object_id": "b_prior_year"},
        ],
        "constraints": [
            {
                "id": "c1",
                "kind": "derived",
                "op": "diff",
                "args_refs": ["e_year", "e_nine"],
                "output_blank_id": "b_prior_year",
            }
        ],
    },
}


_FEW_SHOT_6_LIST_INTERSECTION = {
    "question": (
        "Which player appears on both the North League career home run "
        "leaders list and the North League career stolen base leaders list?"
    ),
    "how_to_think": textwrap.dedent(
        """
        Step 1 — Entities + roles:
          • "North League career home run leaders" → filled,
            role=list-header, category=true.
          • "North League career stolen base leaders" → filled,
            role=list-header, category=true.
          • Each list's members → two list blanks.
          • The player in both lists → blank, role=target, value_type=entity.
        Step 2 — Verb-phrases:
          Do NOT attach both list clues directly to TARGET as a
          target-hub. First retrieve the two lists, then identify the
          shared member using the list blanks as retrieval/context.
          • vp1: (home run leaders list, members, ?home_run_leaders).
          • vp2: (stolen base leaders list, members, ?stolen_base_leaders).
          • vp3: (?target, appears_in, ?home_run_leaders).
          • vp4: (?target, appears_in, ?stolen_base_leaders).
        Step 3 — Constraints:
          None. The current executor has no entity-valued intersection
          op, but the plan still exposes the two list blanks instead of
          hiding the structure in direct target clues.
        """
    ).strip(),
    "plan": {
        "entities": [
            {
                "id": "e_hr_list",
                "kind": "filled",
                "name": "North League career home run leaders",
                "role": "list-header",
                "category": True,
            },
            {
                "id": "e_sb_list",
                "kind": "filled",
                "name": "North League career stolen base leaders",
                "role": "list-header",
                "category": True,
            },
            {"id": "b_hr_list", "kind": "blank", "role": "list-member", "value_type": "list"},
            {"id": "b_sb_list", "kind": "blank", "role": "list-member", "value_type": "list"},
            {
                "id": "t",
                "kind": "blank",
                "role": "target",
                "value_type": "entity",
                "is_target": True,
            },
        ],
        "verb_phrases": [
            {"id": "vp1", "phrase": "members", "subject_id": "e_hr_list", "object_id": "b_hr_list"},
            {"id": "vp2", "phrase": "members", "subject_id": "e_sb_list", "object_id": "b_sb_list"},
            {"id": "vp3", "phrase": "appears_in", "subject_id": "t", "object_id": "b_hr_list"},
            {"id": "vp4", "phrase": "appears_in", "subject_id": "t", "object_id": "b_sb_list"},
        ],
        "constraints": [],
    },
}


_FEW_SHOTS = [
    _FEW_SHOT_1_TEMPORAL_BRIDGE,
    _FEW_SHOT_2_AS_OF_DATE_DIFF,
    _FEW_SHOT_3_ENUMERATED_ARGMAX,
    _FEW_SHOT_4_COMPOUND_SCOPE,
    _FEW_SHOT_5_LITERAL_OFFSET,
    _FEW_SHOT_6_LIST_INTERSECTION,
]

_BLANK_QUALIFIER_INSTRUCTION = (
    "## Blank qualifiers\n"
    "When a BLANK bridge entity is described in the question by a category or "
    "type word (e.g. 'the environmental group', 'the film', 'the German city'), "
    "capture that category:\n"
    "  1. Emit a FILLED entity for the category with category:true, "
    "role:'scope-filter', and name = the exact question substring.\n"
    "  2. Wire the blank to it with a verb-phrase {subject_id: <blank>, "
    "phrase: 'instance_of', object_id: <category entity>}.\n"
    "  3. Give the blank a descriptive id (e.g. b_environmental_group); the "
    "blank's own name stays null.\n"
    "This is the same scope-filter pattern used for target qualifiers like "
    "'rock climber'; apply it to intermediate blanks too."
)

# NOTE: keep this example SYNTHETIC/fictional. Do NOT use a real FRAMES eval
# question here — a real question's decomposition in the planner prompt leaks the
# answer shape into the evaluation (the same mistake the v2 few-shots made; see
# the module header).
_BLANK_HINTS_INSTRUCTION = (
    "## Blank hints (retrieval_hint + verification_claim)\n"
    "For EVERY blank entity, also emit two fields:\n"
    "  1. `retrieval_hint` — a natural-language search question that, on its "
    "own, would let a searcher find this blank's value. Write it the way a "
    "human types into a search box, NOT as a snake_case relation.\n"
    "  2. `verification_claim` — one precise sentence stating exactly what the "
    "candidate value must prove.\n"
    "Rules:\n"
    "  - Include temporal scope such as 'as of <DATE>' when the question is "
    "time-bound.\n"
    "  - Include source constraints such as 'according to the <named ranking "
    "or source>' when the question names a ranking, index, or source.\n"
    "  - For values that depend on an upstream blank, reference it with a "
    "placeholder `{<upstream_blank_id>}` (e.g. `{b_winner}`); it is substituted "
    "with the resolved value before retrieval. Never write a raw blank id except "
    "inside such a placeholder.\n"
    "  - SPECIFICITY: prefer the MOST SPECIFIC entity the evidence supports, "
    "not only the broad category — name the specific organization / body / work "
    "the evidence identifies rather than its generic type.\n"
    "  - UNIT CONSISTENCY (Hard rule 7): when this blank is a numeric operand "
    "feeding an arithmetic constraint together with another numeric blank, name "
    "the SAME unit in both retrieval_hints (e.g. both 'in feet', both "
    "'in meters') so the executor never combines mismatched units. Use the "
    "unit the question's answer implies when there is one; otherwise pick any "
    "single unit and repeat it on every blank feeding that constraint.\n"
    "  - QUALIFIER PRESERVATION (Hard rule 8): if the question attaches a "
    "qualifier word ('full', 'exact', 'official', 'original', 'legal', "
    "'formal', 'complete', 'maiden') to the noun this blank resolves, keep "
    "that qualifier verbatim in BOTH retrieval_hint and verification_claim "
    "-- write 'the full first name', never just 'the first name'. A "
    "nickname or shortened form is a WRONG answer when the question said "
    "'full' (or 'exact'/'official'/...).\n"
    "  - The target blank gets a retrieval_hint/verification_claim too.\n"
    "Example (illustrative, fictional entities): "
    "b_winner.retrieval_hint = 'Which club won the 1931 Veldt Cup?'; "
    "t.retrieval_hint = 'Who captained {b_winner}?'; t.verification_claim = "
    "'The candidate must be the captain of {b_winner} (the 1931 Veldt Cup winner).' "
    "Unit example: for 'How many times taller is the Ashgrove Spire than the "
    "Ferrymoor Lighthouse?', b_spire_height.retrieval_hint = 'What is the "
    "height of the Ashgrove Spire, in feet?' and b_lighthouse_height."
    "retrieval_hint = 'What is the height of the Ferrymoor Lighthouse, in "
    "feet?' — both name feet, so the div constraint over them is unit-safe. "
    "Qualifier example: for 'What is the full first name of Ada Voss's "
    "understudy?', b_understudy_first.retrieval_hint = 'What is the full "
    "first name of Ada Voss's understudy?' and verification_claim = 'The "
    "candidate must be the FULL first name (not a nickname) of Ada Voss's "
    "understudy.' — dropping 'full' would let a nickname wrongly pass."
)

# ---------------------------------------------------------------------------
# User prompt rendering
# ---------------------------------------------------------------------------


def _format_few_shot(example: dict) -> str:
    parts = [f"Question: {example['question']}", ""]
    if "how_to_think" in example:
        parts.append("How to think:")
        parts.append(example["how_to_think"])
        parts.append("")
    parts.append("Plan:")
    parts.append(json.dumps(example["plan"], indent=2))
    return "\n".join(parts)


def build_planner_messages(
    question: str,
    *,
    blank_qualifier: bool = False,
    blank_hints: bool = False,
) -> list[dict]:
    """Build the system + user message list for the planner LLM call.

    Each few-shot demonstrates at least one v4 rule: grounding,
    date-anchor handling, literal constants, or avoiding risky target
    hubs. The user prompt reminds the model to self-apply the grounding
    test before emitting JSON.
    """
    few_shots_rendered = "\n\n---\n\n".join(_format_few_shot(ex) for ex in _FEW_SHOTS)
    few_shot_count = "six"
    if _target_hub_mode() == "v4":
        target_structure_summary = (
            "avoid risky target hubs when intermediate blanks are needed"
        )
    else:
        target_structure_summary = (
            "use direct TARGET intersections for simple entity/list "
            "retrieval; use intermediate blanks and constraints when "
            "the target must be computed, compared, counted, or formatted"
        )
    grounding_summary = (
        "grounding (every filled entity.name appears in the question), "
    )
    grounding_instruction = "self-apply the grounding test to each filled entity, "
    user = (
        f"Here are {few_shot_count} worked examples. Each shows the three-step "
        "thinking protocol (How to think) followed by the corresponding "
        "Plan JSON. Pay attention to how each example honours the hard "
        f"rules: {grounding_summary}date-anchor (dates become filled entities), literal "
        "constants (use filled constraint-value entities with "
        "literal_value + args_refs), narrow list coverage, and target structure "
        f"({target_structure_summary}).\n\n"
        "For the new question, walk the same three steps internally, "
        f"{grounding_instruction}choose "
        "the literal final-answer TARGET, preserve temporal anchors, "
        "use full date blanks for age/elapsed-year questions and never "
        "compute age-at-event from birth_year/event_year alone, apply "
        "narrow coverage only for explicit multi-item/count questions, then "
        "return ONLY the Plan JSON — no prose.\n\n"
        f"{few_shots_rendered}\n\n"
        "---\n\n"
        f"Question: {question}\n"
        "Plan:"
    )
    system = _planner_system_for_mode()
    if blank_qualifier:
        system = system + "\n\n" + _BLANK_QUALIFIER_INSTRUCTION
    if blank_hints:
        system = system + "\n\n" + _BLANK_HINTS_INSTRUCTION
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


REPAIR_SYSTEM = textwrap.dedent(
    """
    Your previous response was not valid JSON matching the GSWPlan
    schema, or it violated a plan rule. Re-emit a single JSON object
    that:

    1. Matches the GSWPlan schema exactly.
    2. Has at least one verb-phrase or constraint edge touching every
       entity id (no dangling entities).
    3. Assigns a `role` label to every entity.
    4. Honours the grounding test: every filled entity's `name` appears
       as a substring of the question text (case-insensitive), UNLESS
       it carries a `state="expanded_from_collective:<group>"` breadcrumb.
    5. Promotes any explicit date / year / month token from the
       question to a filled entity with role=year-anchor,
       date-anchor, or as-of-date, and wires it into at least one
       verb-phrase. For age, elapsed-year, "on the day", "by the time",
       or anniversary questions, uses full `date` blanks for retrieved
       events/births instead of year-only number blanks. For "how old
       ... when" or election/anniversary age questions, computes age
       from full birth/event dates, never event_year - birth_year.
    6. Does not create blanks for literal constants. Numeric literals
       used in constraints should be filled `constraint-value` entities
       with `literal_value` and referenced through `args_refs`.
    7. For explicit multi-item/count questions, uses simple list
       coverage: a candidate-list blank, a filtered-list blank when
       needed, and a list or count TARGET. Does not expand unknown
       members as filled entities.
    8. Uses the minimal sufficient TARGET shape: TARGET is the literal
       answer requested by the question; direct TARGET VPs are
       valid for directly retrievable entity/list intersections; create
       intermediate blanks/constraints when the target itself must be
       computed, ranked, counted, rounded, temporally compared, or
       formatted as a compound answer.

    Do not include any prose or Markdown fences.
    """
).strip()

_REPAIR_RULE_8_HYBRID = (
    "8. Uses the minimal sufficient TARGET shape: TARGET is the literal\n"
    "   answer requested by the question; direct TARGET VPs are\n"
    "   valid for directly retrievable entity/list intersections; create\n"
    "   intermediate blanks/constraints when the target itself must be\n"
    "   computed, ranked, counted, rounded, temporally compared, or\n"
    "   formatted as a compound answer."
)

_REPAIR_RULE_8_V4_COMPAT = (
    "8. Avoids risky TARGET hubs: if the question implies computation,\n"
    "   list intersection, temporal filtering, ranking, or a compound\n"
    "   answer, make TARGET the literal requested answer and create\n"
    "   intermediate blanks/constraints instead of wiring every clue\n"
    "   directly to TARGET."
)


def _repair_system_for_mode() -> str:
    if _target_hub_mode() == "hybrid":
        return REPAIR_SYSTEM
    return REPAIR_SYSTEM.replace(_REPAIR_RULE_8_HYBRID, _REPAIR_RULE_8_V4_COMPAT)


def build_repair_messages(
    question: str,
    bad_response: str,
    error: str,
    *,
    attempt: int = 1,
    max_attempts: int = 1,
    blank_hints: bool = False,
) -> list[dict]:
    """Messages for the repair retry on parse failure.

    ``attempt`` is 1-indexed; ``max_attempts`` is the total repair budget.
    Later attempts get a progressively sharper preface so the LLM knows
    it has already failed earlier rounds and must change its output.
    """
    # NOTE: blank_hints IS forwarded (repair re-emits a full plan, so repaired
    # blanks must carry hints). blank_qualifier is intentionally NOT forwarded
    # here — that asymmetry is pre-existing.
    msgs = build_planner_messages(
        question,
        blank_hints=blank_hints,
    )
    msgs.append({"role": "assistant", "content": bad_response})

    if max_attempts <= 1 or attempt <= 1:
        preface = _repair_system_for_mode()
    elif attempt < max_attempts:
        preface = (
            f"This is repair attempt {attempt} of {max_attempts}. "
            "Your previous repaired output ALSO failed validation. "
            "Read the validation error carefully — the named field "
            "is the one to fix. Do NOT echo your previous response.\n\n"
            f"{_repair_system_for_mode()}"
        )
    else:
        preface = (
            f"FINAL repair attempt ({attempt}/{max_attempts}). "
            "All earlier repairs have failed. If this attempt does not "
            "validate, the question will be answered without a plan. "
            "Address the SPECIFIC field named in the validation error — "
            "if it complains about empty `args_refs`/`args_blanks`, "
            "populate them; if it complains about `literal_value`, add "
            "a typed literal_value to the filled constraint-value entity; "
            "if it complains about missing `left_ref`, supply it; if it "
            "complains about a dangling entity, add a verb-phrase that "
            "uses it. Re-emit the entire JSON object from scratch.\n\n"
            f"{_repair_system_for_mode()}"
        )

    msgs.append(
        {
            "role": "user",
            "content": (
                f"{preface}\n\nValidation error was:\n{error}\n\n"
                "Return the corrected JSON now."
            ),
        }
    )
    return msgs
