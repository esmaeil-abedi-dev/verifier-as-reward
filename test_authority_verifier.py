"""
test_authority_verifier.py
==========================

Hand-built ground-truth cases for the authority verifier. Each test encodes a
scenario a human has verified by hand, covering the four decision paths:
structural integrity, per-hop activity (expiry/revocation), per-hop
attenuation (scope narrowing / escalation), and final-scope permission.

Run: python -m pytest test_authority_verifier.py   (or python test_authority_verifier.py)
"""

import math
from authority_verifier import (
    Grant, Scope, Delegation, Action, RootAuthority, verify, Decision,
)

# --------------------------------------------------------------------------
# Fixtures: a common root principal and some scopes
# --------------------------------------------------------------------------

def root_owner():
    # Owner may do anything on the acme org, up to a $1000 budget.
    return RootAuthority(
        principal="owner",
        scope=Scope((Grant("*", "acme:*", 1000.0),)),
    )

def scope_email():
    return Scope((Grant("email.*", "acme:*", 100.0),))

def scope_email_narrow():
    return Scope((Grant("email.send", "acme:support/*", 50.0),))


# --------------------------------------------------------------------------
# Group A: the happy path (AUTHORIZED)
# --------------------------------------------------------------------------

def test_direct_root_action_authorized():
    root = root_owner()
    act = Action("owner", "email.send", "acme:support/ticket1", 10.0, t=5)
    v = verify(act, [], root)
    assert v.authorized, v

def test_single_valid_delegation_authorized():
    root = root_owner()
    chain = [Delegation("owner", "assistant", scope_email(), issued_at=0)]
    act = Action("assistant", "email.send", "acme:support/x", 20.0, t=5)
    v = verify(act, chain, root)
    assert v.authorized, v

def test_two_hop_valid_narrowing_authorized():
    root = root_owner()
    chain = [
        Delegation("owner", "assistant", scope_email(), issued_at=0),
        Delegation("assistant", "subagent", scope_email_narrow(), issued_at=1),
    ]
    act = Action("subagent", "email.send", "acme:support/ticket9", 25.0, t=5)
    v = verify(act, chain, root)
    assert v.authorized, v


# --------------------------------------------------------------------------
# Group B: structural failures (UNAUTHORIZED, chain integrity)
# --------------------------------------------------------------------------

def test_wrong_root_rejected():
    root = root_owner()
    chain = [Delegation("intruder", "assistant", scope_email(), issued_at=0)]
    act = Action("assistant", "email.send", "acme:support/x", 10.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized and v.failing_hop == 0, v

def test_broken_chain_rejected():
    root = root_owner()
    chain = [
        Delegation("owner", "assistant", scope_email(), issued_at=0),
        Delegation("someone_else", "subagent", scope_email_narrow(), issued_at=1),
    ]
    act = Action("subagent", "email.send", "acme:support/x", 10.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized and v.failing_hop == 1, v

def test_chain_not_ending_at_agent_rejected():
    root = root_owner()
    chain = [Delegation("owner", "assistant", scope_email(), issued_at=0)]
    act = Action("ghost", "email.send", "acme:support/x", 10.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized, v

def test_no_chain_non_root_rejected():
    root = root_owner()
    act = Action("randobot", "email.send", "acme:support/x", 10.0, t=5)
    v = verify(act, [], root)
    assert not v.authorized and v.failing_hop is None, v


# --------------------------------------------------------------------------
# Group C: activity failures (expiry / revocation)
# --------------------------------------------------------------------------

def test_action_before_issue_rejected():
    root = root_owner()
    chain = [Delegation("owner", "assistant", scope_email(), issued_at=10)]
    act = Action("assistant", "email.send", "acme:support/x", 10.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized and v.failing_hop == 0, v

def test_expired_delegation_rejected():
    root = root_owner()
    chain = [Delegation("owner", "assistant", scope_email(),
                        issued_at=0, expires_at=4)]
    act = Action("assistant", "email.send", "acme:support/x", 10.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized and v.failing_hop == 0, v

def test_revoked_then_use_rejected():
    root = root_owner()
    chain = [Delegation("owner", "assistant", scope_email(),
                        issued_at=0, revoked_at=3)]
    act = Action("assistant", "email.send", "acme:support/x", 10.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized and v.failing_hop == 0, v

def test_use_just_before_revocation_authorized():
    root = root_owner()
    chain = [Delegation("owner", "assistant", scope_email(),
                        issued_at=0, revoked_at=6)]
    act = Action("assistant", "email.send", "acme:support/x", 10.0, t=5)
    v = verify(act, chain, root)
    assert v.authorized, v

def test_second_hop_revoked_rejected():
    root = root_owner()
    chain = [
        Delegation("owner", "assistant", scope_email(), issued_at=0),
        Delegation("assistant", "subagent", scope_email_narrow(),
                   issued_at=1, revoked_at=4),
    ]
    act = Action("subagent", "email.send", "acme:support/x", 10.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized and v.failing_hop == 1, v


# --------------------------------------------------------------------------
# Group D: attenuation failures (scope escalation)
# --------------------------------------------------------------------------

def test_scope_escalation_action_rejected():
    # child tries to grant a broader ACTION than it received
    root = root_owner()
    broad = Scope((Grant("email.send", "acme:support/*", 50.0),))
    escalated = Scope((Grant("payment.charge", "acme:*", 50.0),))
    chain = [
        Delegation("owner", "assistant", broad, issued_at=0),
        Delegation("assistant", "subagent", escalated, issued_at=1),
    ]
    act = Action("subagent", "payment.charge", "acme:billing", 10.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized and v.failing_hop == 1, v

def test_scope_escalation_budget_rejected():
    # child tries to grant a LARGER budget than it received
    root = root_owner()
    parent = Scope((Grant("email.send", "acme:*", 50.0),))
    bigger = Scope((Grant("email.send", "acme:*", 500.0),))
    chain = [
        Delegation("owner", "assistant", parent, issued_at=0),
        Delegation("assistant", "subagent", bigger, issued_at=1),
    ]
    act = Action("subagent", "email.send", "acme:x", 200.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized and v.failing_hop == 1, v

def test_scope_escalation_resource_rejected():
    # child widens the RESOURCE pattern beyond parent
    root = root_owner()
    parent = Scope((Grant("email.send", "acme:support/*", 50.0),))
    wider = Scope((Grant("email.send", "acme:*", 50.0),))
    chain = [
        Delegation("owner", "assistant", parent, issued_at=0),
        Delegation("assistant", "subagent", wider, issued_at=1),
    ]
    act = Action("subagent", "email.send", "acme:finance/secret", 10.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized and v.failing_hop == 1, v

def test_first_hop_escalates_above_root_rejected():
    root = root_owner()  # root capped at acme:* and $1000
    overbroad = Scope((Grant("*", "*", 100.0),))  # resource "*" exceeds "acme:*"
    chain = [Delegation("owner", "assistant", overbroad, issued_at=0)]
    act = Action("assistant", "email.send", "external:evil", 10.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized and v.failing_hop == 0, v


# --------------------------------------------------------------------------
# Group E: final-scope permission failures
# --------------------------------------------------------------------------

def test_action_outside_final_scope_resource_rejected():
    root = root_owner()
    chain = [Delegation("owner", "assistant", scope_email_narrow(), issued_at=0)]
    # narrow scope only covers acme:support/*, this hits finance
    act = Action("assistant", "email.send", "acme:finance/x", 10.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized and v.failing_hop == 0, v

def test_action_over_budget_rejected():
    root = root_owner()
    chain = [Delegation("owner", "assistant", scope_email_narrow(), issued_at=0)]
    # narrow scope cap is $50
    act = Action("assistant", "email.send", "acme:support/x", 75.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized and v.failing_hop == 0, v

def test_wrong_action_type_rejected():
    root = root_owner()
    chain = [Delegation("owner", "assistant", scope_email(), issued_at=0)]
    act = Action("assistant", "payment.charge", "acme:support/x", 10.0, t=5)
    v = verify(act, chain, root)
    assert not v.authorized and v.failing_hop == 0, v


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def _run_all():
    import sys
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    passed = failed = 0
    for name, fn in fns:
        try:
            fn(); passed += 1
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"\n{passed} passed, {failed} failed, {len(fns)} total")
    return failed

if __name__ == "__main__":
    import sys
    sys.exit(1 if _run_all() else 0)
