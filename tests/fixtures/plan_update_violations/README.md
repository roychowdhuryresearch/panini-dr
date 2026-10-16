# Post-update contract violations

Two plans with the structure of plans that a `request_plan_update` produced in
a FRAMES-824 research run, captured AFTER the update revised them. Each carries
a BINDING contract violation that nothing in the pipeline detected at the time,
because `update_plan`'s acceptance test checked liveness (did the op apply and
change the plan) rather than correctness. The plan structure is kept as logged;
entity names and question wording were replaced with fictional ones, so no
benchmark question text ships with the tests.

In that run: 191 post-update plan versions, 36 BINDING violations (34
`ordering`, 2 `acyclic`), 19 questions executing a violating plan, and 15
updates that increased the violation count.

`acyclic.json` is the more damning of the two: emit-time produced zero acyclic
violations across 400 logged plans; the updater manufactures a failure class the
planner never does. `acyclic` here does not mean a verb-phrase graph cycle — it
is `_acyclic_violations` (`_plan_contract.py`) catching a constraint whose
`output_blank_id` appears among its own operands, i.e. a constraint that reads
the very blank it writes and so can never resolve. The diff vocabulary edits one
blank/VP/constraint at a time with no such check re-run globally after the
edit, so this self-reference can be introduced by an otherwise-valid op and ship
undetected.
