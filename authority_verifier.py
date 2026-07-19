"""
authority_verifier.py
======================

A deterministic decision procedure for AI-agent action authorization over a
delegation chain. Given a logged agent action and the delegation chain that
led to it, the verifier returns AUTHORIZED / UNAUTHORIZED together with the
failing hop when unauthorized.

The model reuses the object-capability / attenuated-delegation formalism
systematized in the companion survey (Paper A): authority flows from a root
principal down a chain of delegations, each of which may only *narrow*
(attenuate) the scope it received, may carry an expiry, and may be revoked.
An action is authorized iff it is permitted by the scope in force at the
agent, reached through a chain in which every hop is (a) a valid narrowing of
its parent, (b) active (issued, not expired, not revoked) at the action time.

This module has no third-party dependencies and no GPU requirement; the
authorization predicate is decidable and exact, which is precisely the
property that lets its verdict serve as an automatic reward signal.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from fnmatch import fnmatch
from typing import Optional
import math


# --------------------------------------------------------------------------
# 1. Core data model
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Grant:
    """A single unit of authority: permission to perform `action` on resources
    matching `resource` (a glob pattern), optionally bounded by a spending cap.

    A grant g_child is *subsumed by* g_parent (g_child <= g_parent) when it
    permits nothing g_parent does not: the action matches, the child's
    resource pattern is no broader than the parent's, and the child's budget
    cap is no larger than the parent's.
    """
    action: str                       # e.g. "email.send", "payment.charge"
    resource: str = "*"               # glob, e.g. "repo:acme/*", "*"
    max_budget: float = math.inf      # spending ceiling for this grant

    def permits(self, action: str, resource: str, amount: float = 0.0) -> bool:
        """Does this grant permit a concrete action instance?"""
        return (
            _action_matches(self.action, action)
            and fnmatch(resource, self.resource)
            and amount <= self.max_budget
        )

    def subsumes(self, other: "Grant") -> bool:
        """True if `other` is no broader than self (other <= self)."""
        return (
            _action_matches(self.action, other.action)
            and _pattern_subsumes(self.resource, other.resource)
            and other.max_budget <= self.max_budget
        )


@dataclass(frozen=True)
class Scope:
    """A set of grants. Scope S_child attenuates S_parent (S_child <= S_parent)
    iff every grant in S_child is subsumed by some grant in S_parent."""
    grants: tuple = ()

    def permits(self, action: str, resource: str, amount: float = 0.0) -> bool:
        return any(g.permits(action, resource, amount) for g in self.grants)

    def attenuates(self, parent: "Scope") -> bool:
        """True if self is a valid narrowing of `parent` (no privilege gained)."""
        return all(
            any(pg.subsumes(cg) for pg in parent.grants)
            for cg in self.grants
        )


@dataclass(frozen=True)
class Delegation:
    """One hop in a delegation chain: `delegator` grants `scope` to `delegatee`.

    issued_at / expires_at / revoked_at are integer logical timestamps
    (e.g. step indices). A hop is active at time t iff
    issued_at <= t < expires_at and (revoked_at is None or t < revoked_at).
    """
    delegator: str
    delegatee: str
    scope: Scope
    issued_at: int = 0
    expires_at: float = math.inf
    revoked_at: Optional[int] = None

    def active_at(self, t: int) -> bool:
        if t < self.issued_at:
            return False
        if t >= self.expires_at:
            return False
        if self.revoked_at is not None and t >= self.revoked_at:
            return False
        return True


@dataclass
class Action:
    """A concrete action taken by `agent` at logical time `t`."""
    agent: str
    action: str
    resource: str = "*"
    amount: float = 0.0
    t: int = 0


@dataclass
class RootAuthority:
    """A principal that holds authority natively (the owner/root of a chain)."""
    principal: str
    scope: Scope


# --------------------------------------------------------------------------
# 2. Pattern helpers (action + resource subsumption)
# --------------------------------------------------------------------------

def _action_matches(pattern: str, action: str) -> bool:
    """Action patterns support a trailing '.*' wildcard and bare '*'.
    'email.*' matches 'email.send'; '*' matches anything; otherwise exact."""
    if pattern == "*":
        return True
    if pattern.endswith(".*"):
        return action == pattern[:-2] or action.startswith(pattern[:-1])
    return pattern == action


def _pattern_subsumes(parent: str, child: str) -> bool:
    """True if glob `parent` covers everything glob `child` covers.

    Exact and equal-pattern cases are trivially true. A parent ending in '*'
    subsumes any child whose non-wildcard prefix it covers. This is a sound
    (conservative) check: it never reports subsumption that does not hold.
    """
    if parent == child:
        return True
    if parent == "*":
        return True
    if parent.endswith("*"):
        prefix = parent[:-1]
        # child must be forced to stay within the parent prefix
        if child.endswith("*"):
            return child[:-1].startswith(prefix)
        return child.startswith(prefix)
    # parent is a concrete pattern with no wildcard: only an identical child fits
    return False


# --------------------------------------------------------------------------
# 3. Verdict type
# --------------------------------------------------------------------------

class Decision(Enum):
    AUTHORIZED = "authorized"
    UNAUTHORIZED = "unauthorized"


@dataclass
class Verdict:
    decision: Decision
    failing_hop: Optional[int] = None      # index into the chain, or None
    reason: str = ""

    @property
    def authorized(self) -> bool:
        return self.decision is Decision.AUTHORIZED

    def __repr__(self):
        if self.authorized:
            return "Verdict(AUTHORIZED)"
        return (f"Verdict(UNAUTHORIZED, hop={self.failing_hop}, "
                f"reason={self.reason!r})")


# --------------------------------------------------------------------------
# 4. The verifier
# --------------------------------------------------------------------------

def verify(action: Action,
           chain: list,
           root: RootAuthority) -> Verdict:
    """Decide whether `action` is authorized.

    `chain` is an ordered list of Delegation hops from the root principal to
    the acting agent: chain[0].delegator == root.principal, and
    chain[i].delegatee == chain[i+1].delegator, and chain[-1].delegatee ==
    action.agent. The verifier checks, in order:

      1. Structural integrity  : the chain connects root -> ... -> agent.
      2. Per-hop activity       : every hop is active at action.t.
      3. Per-hop attenuation    : every hop narrows its parent's scope.
      4. Final-scope permission : the action is permitted by the last scope.

    Returns AUTHORIZED, or UNAUTHORIZED with the index of the first failing
    hop (or None for a whole-chain/structural failure).
    """
    # --- 1. structural integrity ------------------------------------------
    if not chain:
        # No delegation: authorized only if the agent IS the root and root
        # scope permits the action directly.
        if action.agent == root.principal and root.scope.permits(
                action.action, action.resource, action.amount):
            return Verdict(Decision.AUTHORIZED)
        return Verdict(Decision.UNAUTHORIZED, None,
                       "no delegation chain and agent is not an authorized root")

    if chain[0].delegator != root.principal:
        return Verdict(Decision.UNAUTHORIZED, 0,
                       f"chain does not originate at root {root.principal!r}")
    for i in range(len(chain) - 1):
        if chain[i].delegatee != chain[i + 1].delegator:
            return Verdict(Decision.UNAUTHORIZED, i + 1,
                           "broken chain: delegatee/delegator mismatch")
    if chain[-1].delegatee != action.agent:
        return Verdict(Decision.UNAUTHORIZED, len(chain) - 1,
                       "chain does not end at the acting agent")

    # --- 2 & 3. per-hop activity and attenuation --------------------------
    parent_scope = root.scope
    for i, hop in enumerate(chain):
        if not hop.active_at(action.t):
            return Verdict(Decision.UNAUTHORIZED, i,
                           _inactive_reason(hop, action.t))
        if not hop.scope.attenuates(parent_scope):
            return Verdict(Decision.UNAUTHORIZED, i,
                           "scope escalation: hop grants authority beyond its parent")
        parent_scope = hop.scope

    # --- 4. final-scope permission ----------------------------------------
    if not chain[-1].scope.permits(action.action, action.resource, action.amount):
        return Verdict(Decision.UNAUTHORIZED, len(chain) - 1,
                       "action not permitted by the scope in force at the agent")

    return Verdict(Decision.AUTHORIZED)


def _inactive_reason(hop: Delegation, t: int) -> str:
    if t < hop.issued_at:
        return f"hop not yet issued at t={t} (issued_at={hop.issued_at})"
    if t >= hop.expires_at:
        return f"hop expired at t={t} (expires_at={hop.expires_at})"
    if hop.revoked_at is not None and t >= hop.revoked_at:
        return f"hop revoked at t={t} (revoked_at={hop.revoked_at})"
    return "hop inactive"


# convenience for callers labeling whole traces
def label_action(action: Action, chain: list, root: RootAuthority) -> int:
    """Return 1 if authorized, 0 if not (for benchmark labeling)."""
    return 1 if verify(action, chain, root).authorized else 0
