# `<family_id>` — <short title>

| Field | Value |
|---|---|
| Family ID | `<family_id>` |
| Domain | banking_payments / jobs_recruitment / ecommerce_delivery |
| Version | 0.1 |
| Status | draft |
| Pilot | yes / no |
| Split | unassigned |
| Channel | outbound call / inbound call / chat |

## Family summary

Two or three sentences: the shared setting, and what separates the fraud case from the legitimate one.

## Shared setting

Facts that are identical in both cases.

- Organisation:
- Recipient profile (kept neutral, sampled at generation):
- Pretext / reason for contact:

---

## Case `<family_id>_F` — fraud attempt

**Scenario label:** `fraud_attempt`

**Label rationale:** Why this is fraud, stated in terms of the caller's observable action.

**Caller objective (private, never emitted):**

**Stable facts:**

-

**Required observable evidence (must appear before `branch_after`):**

| ID | Evidence | Why it counts |
|---|---|---|
| F-E1 | | |

**Dialogue-act plan (prefix):**

1.

**`branch_after`:** turn containing F-E1 (or the next recipient turn)

## Case `<family_id>_L` — legitimate interaction

**Scenario label:** `legitimate`

**Label rationale:**

**Caller objective:**

**Stable facts:**

-

**Legitimacy markers (visible, but not required to *prove* legitimacy):**

| ID | Marker |
|---|---|
| L-M1 | |

**Dialogue-act plan (prefix):**

1.

**`branch_after`:** turn at the matching point in the conversation

---

## Recipient continuations

| Condition | Fraud case `_F` | Legitimate case `_L` |
|---|---|---|
| cooperative | | |
| clarifying | | |
| resistant | | |
| suspicious | | |

**Allowed endings:**

**Caller continuation limits:** The fraud caller may press, reassure, or drop the call, but must not newly introduce a *different* fraud mechanism. The legitimate caller must not introduce any fraud-like request after the branch.

## Generation constraints

- **must_preserve:**
- **may_vary:**
- **must_not_emit:**

## Ambiguities and disallowed assumptions

-

## Evidence references

| Claim | Source | Retrieved | Status |
|---|---|---|---|
| | | | VERIFY |

## Sign-off

| Role | Name | Date | Notes |
|---|---|---|---|
| Author | | | |
| Reviewer | | | |
