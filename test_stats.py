"""
test_stats.py
=============

Tests for the Wilson-CI helpers. Run:

    PYTHONPATH=. python3 test_stats.py
"""

import math

from stats import fmt_pct_ci, rate_with_ci, wilson_ci

_failures = []


def check(name, fn):
    try:
        fn()
        print(f"PASS  {name}")
    except AssertionError as e:
        _failures.append(name)
        print(f"FAIL  {name}: {e}")


def test_known_values():
    # canonical reference: 50/100 -> Wilson 95% ~ [0.404, 0.596]
    lo, hi = wilson_ci(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3, (lo, hi)
    # 80/100 -> ~ [0.711, 0.867]
    lo, hi = wilson_ci(80, 100)
    assert abs(lo - 0.7112) < 1e-3 and abs(hi - 0.8666) < 1e-3, (lo, hi)


def test_boundaries_stay_in_unit_interval():
    # Wald would give [0,0] and [1,1]; Wilson gives a real interval inside [0,1]
    lo, hi = wilson_ci(0, 20)
    assert lo == 0.0 and 0.0 < hi < 1.0, (lo, hi)
    lo, hi = wilson_ci(20, 20)
    assert hi == 1.0 and 0.0 < lo < 1.0, (lo, hi)


def test_point_estimate_inside_interval():
    for k, n in [(1, 3), (7, 10), (44, 80), (968, 1000), (3, 130)]:
        lo, hi = wilson_ci(k, n)
        assert lo <= k / n <= hi, (k, n, lo, hi)
        assert 0.0 <= lo <= hi <= 1.0


def test_wider_for_smaller_n():
    # same proportion, smaller n => wider interval
    w_small = wilson_ci(4, 8)
    w_big = wilson_ci(400, 800)
    assert (w_small[1] - w_small[0]) > (w_big[1] - w_big[0])


def test_zero_n_and_bad_k():
    assert wilson_ci(0, 0) is None
    try:
        wilson_ci(5, 3)
        assert False, "k>n should raise"
    except ValueError:
        pass


def test_rate_with_ci_and_fmt():
    d = rate_with_ci(44, 80)
    assert d["rate"] == 44 / 80 and d["n"] == 80 and d["k"] == 44
    assert len(d["ci"]) == 2 and d["ci"][0] <= 44 / 80 <= d["ci"][1]
    assert rate_with_ci(0, 0)["rate"] is None
    s = fmt_pct_ci(44, 80)
    assert "55.0%" in s and "(44/80)" in s
    assert fmt_pct_ci(0, 0) == "n/a (0)"


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        check(name, fn)
    print(f"\n{len(tests) - len(_failures)} passed, {len(_failures)} failed, "
          f"{len(tests)} total")
    raise SystemExit(1 if _failures else 0)
