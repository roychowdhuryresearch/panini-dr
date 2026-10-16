"""The per-question event-order gate: recorded release order, no deadlock, timeouts."""

from __future__ import annotations

import random
import threading
import time

from panini_dr.replay.gate import EventOrderGate, root_of


def _entries():
    # One question, three calls on two threads (recorded):
    #   A: request t=0.0 ............ response t=3.0
    #   B: request t=1.0 .. response t=2.0
    #   C: request t=2.5 (B's thread, after B's response) ... response t=4.0
    # plus a call of another question and one outside any question.
    return [
        {"i": 0, "lineage": "q:1/_solve_one:a", "t0": 0.0, "t1": 3.0},
        {"i": 1, "lineage": "q:1/_solve_one:b", "t0": 1.0, "t1": 2.0},
        {"i": 2, "lineage": "q:1/_solve_one:b", "t0": 2.5, "t1": 4.0},
        {"i": 3, "lineage": "q:2", "t0": 0.0, "t1": 9.0},
        {"i": 4, "lineage": "main", "t0": 0.0, "t1": 0.5},
    ]


def _replay(gate: EventOrderGate, lock, entries, order_of_arrival, jitter=0.0):
    released: list[int] = []
    errors: list[BaseException] = []

    def worker(e):
        try:
            if jitter:
                time.sleep(random.random() * jitter)
            with lock:                       # what Cassette.take does when a request arrives
                gate.note_arrival(e["i"])
            gate.release(e)
            with lock:
                released.append(e["i"])
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    by_i = {e["i"]: e for e in entries}
    threads = [threading.Thread(target=worker, args=(by_i[i],)) for i in order_of_arrival]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not any(t.is_alive() for t in threads), "gate deadlocked"
    assert not errors, errors
    return released


def test_responses_of_one_question_are_released_in_recorded_completion_order():
    for seed in range(20):
        random.seed(seed)
        entries = _entries()
        lock = threading.RLock()
        gate = EventOrderGate(entries, lock, order="recorded", timeout_s=5.0)
        arrival = [0, 1, 2, 3, 4]
        random.shuffle(arrival)
        released = _replay(gate, lock, entries, arrival, jitter=0.01)
        q1 = [i for i in released if i in (0, 1, 2)]
        assert q1 == [1, 0, 2], (seed, arrival, released)          # by recorded t1: B, A, C
        assert gate.diverged == set() and gate.stats["timeouts"] == 0


def test_a_response_waits_for_a_request_recorded_before_it():
    # A (t1=3.0) must not be released before C's request (t0=2.5) has arrived:
    # that request proves C's thread finished the work it did after B's response.
    entries = _entries()
    lock = threading.RLock()
    gate = EventOrderGate(entries, lock, order="recorded", timeout_s=5.0)
    with lock:
        gate.note_arrival(0)
        gate.note_arrival(1)
    gate.release(entries[1])                                      # B: nothing earlier -> immediate
    assert gate.blocked("q:1", entries[0]) == "request"           # A waits for C's request
    done = threading.Event()
    threading.Thread(target=lambda: (gate.release(entries[0]), done.set()), daemon=True).start()
    assert not done.wait(0.3)
    with lock:
        gate.note_arrival(2)
    assert done.wait(2.0)
    assert gate.stats["waited_request"] == 1


def test_questions_are_independent_and_main_is_never_gated():
    entries = _entries()
    lock = threading.RLock()
    gate = EventOrderGate(entries, lock, order="recorded", timeout_s=5.0)
    t = time.monotonic()
    gate.release(entries[4])        # "main": immediate
    gate.release(entries[3])        # q:2 has nothing earlier of its own; q:1 is irrelevant
    assert time.monotonic() - t < 0.5
    assert root_of("q:7/_solve_one:x#2") == "q:7" and root_of(None) == "main"


def test_a_stuck_question_times_out_and_is_ungated():
    entries = _entries()
    lock = threading.RLock()
    gate = EventOrderGate(entries, lock, order="recorded", timeout_s=0.2)
    t = time.monotonic()
    gate.release(entries[2])        # C waits for A and B, which never come
    assert 0.15 < time.monotonic() - t < 2.0
    assert gate.diverged == {"q:1"} and gate.stats["timeouts"] == 1
    t = time.monotonic()
    gate.release(entries[0])        # the question is ungated from now on
    assert time.monotonic() - t < 0.1


def test_mark_diverged_unblocks_waiters():
    entries = _entries()
    lock = threading.RLock()
    gate = EventOrderGate(entries, lock, order="recorded", timeout_s=30.0)
    done = threading.Event()
    threading.Thread(target=lambda: (gate.release(entries[2]), done.set()), daemon=True).start()
    assert not done.wait(0.2)
    gate.mark_diverged("q:1/_solve_one:a")                        # e.g. a miss in this question
    assert done.wait(2.0)


def test_free_order_releases_immediately():
    entries = _entries()
    gate = EventOrderGate(entries, threading.RLock(), order="free")
    t = time.monotonic()
    for e in reversed(entries):
        gate.release(e)
    assert time.monotonic() - t < 0.1 and gate.released == {0, 1, 2, 3, 4}
