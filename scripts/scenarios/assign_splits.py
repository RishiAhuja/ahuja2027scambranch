#!/usr/bin/env python3
"""Assign 60 scenario families to train/dev/test at family level.

Near-duplicate clusters are atomic. The script writes
``manifests/splits_v1.json`` and ``manifests/SPLITS.md``, updates the
``split`` field on each ``scenarios/json/<family>.json``, and checks
cluster integrity, the pilot-not-in-test rule, and a 60-family total.

Usage:
  .venv/bin/python scripts/scenarios/assign_splits.py
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON = REPO_ROOT / "scenarios" / "json"
DEFAULT_MANIFEST = REPO_ROOT / "manifests" / "splits_v1.json"
DEFAULT_MARKDOWN = REPO_ROOT / "manifests" / "SPLITS.md"

VERSION = "v1"
SEED = 20260925
CREATED = "2026-09-25"

DOMAINS = ("banking_payments", "jobs_recruitment", "ecommerce_delivery")
SPLITS = ("train", "dev", "test")
TARGET_OVERALL = {"train": 36, "dev": 12, "test": 12}
TARGET_PER_DOMAIN = {"train": 12, "dev": 4, "test": 4}

# Given near-duplicate clusters. Family order is the input order.
GIVEN_CLUSTERS: dict[str, tuple[str, ...]] = {
    "C1": ("bp_12", "ec_09", "ec_14"),
    "C2": ("bp_13", "ec_06", "bp_10", "bp_19"),
    "C3": ("bp_03", "ec_05"),
    "C4": ("bp_04", "ec_11", "ec_13"),
    "C5": ("bp_09", "ec_04"),
    "C6": ("bp_16", "jr_16"),
    "C7": ("jr_13", "jr_18"),
    "C8": ("jr_06", "jr_12"),
    "C9": ("jr_07", "jr_19"),
    "C10": ("jr_01", "jr_10"),
    "C11": ("ec_07", "ec_20"),
    "C12": ("bp_02", "bp_07", "bp_11", "bp_20"),
}

PILOTS = ("bp_01", "bp_02", "jr_01", "jr_02", "ec_01", "ec_02")

# Keyword-confound families forced into test (must be singleton, non-pilot).
FORCED_TEST = ("ec_08", "jr_14", "jr_08")

# After forced families, fill each domain's test slots in this order.
# Ranked for distinct mechanisms not already covered by a train cluster.
TEST_FILL: dict[str, tuple[str, ...]] = {
    "banking_payments": (
        "bp_08",
        "bp_05",
        "bp_15",
        "bp_14",
        "bp_06",
        "bp_17",
        "bp_18",
    ),
    "jobs_recruitment": (
        "jr_03",
        "jr_17",
        "jr_05",
        "jr_15",
        "jr_04",
        "jr_09",
        "jr_11",
        "jr_20",
    ),
    "ecommerce_delivery": (
        "ec_18",
        "ec_19",
        "ec_10",
        "ec_15",
        "ec_03",
        "ec_12",
        "ec_16",
        "ec_17",
    ),
}

MECHANISMS: dict[str, str] = {
    "bp_01": "OTP requested to unblock a card after a disputed transaction",
    "bp_02": "KYC-update link plus net-banking login",
    "bp_03": "UPI PIN entered to 'receive' a refund via collect request",
    "bp_04": "Install a screen-sharing app for 'support'",
    "bp_05": "Electricity bill paid to a personal UPI ID",
    "bp_06": "Advance processing fee to a personal account before disbursal",
    "bp_07": "Reward-points link that asks for full card details",
    "bp_08": "Send a 'mistaken' credit back to the caller's UPI",
    "bp_09": "Move money to a 'safe account' under a crime-link threat",
    "bp_10": "Fee to release an insurance maturity bonus",
    "bp_11": "Tax-refund link that asks for card details",
    "bp_12": "Fake helpline asks for PIN or CVV",
    "bp_13": "Fee paid to claim a cashback prize",
    "bp_14": "OTP requested to continue pension 'verification'",
    "bp_15": "CVV and expiry requested for a 'free card upgrade'",
    "bp_16": "Guaranteed-return investment paid into a personal account",
    "bp_17": "Fee paid to an agent for EMI relief",
    "bp_18": "Dormant-account reactivation fee plus OTP",
    "bp_19": "Premature-FD 'penalty' paid to a personal UPI",
    "bp_20": "Merger re-registration via link and PIN",
    "jr_01": "Registration fee for a guaranteed job",
    "jr_02": "Deposits required to unlock task earnings",
    "jr_03": "Visa processing fee to a personal account",
    "jr_04": "Laptop security deposit after a fake offer",
    "jr_05": "Payment for 'backdoor' government selection",
    "jr_06": "Paid mandatory certification before joining",
    "jr_07": "Candidate-paid background-verification fee",
    "jr_08": "Job is receiving and forwarding other people's funds",
    "jr_09": "Forfeitable security deposit for data-entry work",
    "jr_10": "Campus-drive registration payment",
    "jr_11": "Uniform and training fees after a phone-only 'selection'",
    "jr_12": "Starter kit must be bought before a stipend internship",
    "jr_13": "OTP disclosure to 'verify' onboarding documents",
    "jr_14": "Gig onboarding fee paid to an agent's UPI",
    "jr_15": "Paid 'priority placement' with a guaranteed shortlist",
    "jr_16": "Trading 'job' that requires investing the candidate's own money",
    "jr_17": "Chat interview, then unofficial app install and in-app payment",
    "jr_18": "Net-banking login to 'credit' a joining bonus",
    "jr_19": "Pre-employment medical paid to an individual's UPI",
    "jr_20": "Upfront tutor-registration fee for guaranteed students",
    "ec_01": "Bank OTP collected as if it were a delivery code",
    "ec_02": "QR code presented as a way to receive payment",
    "ec_03": "Redelivery fee via unofficial link plus card details",
    "ec_04": "Contraband or arrest threat; pay to close the 'case'",
    "ec_05": "UPI PIN entered to 'receive' a marketplace refund",
    "ec_06": "Lucky-draw prize tax paid to a UPI ID",
    "ec_07": "Classifieds seller demands full advance to a personal UPI",
    "ec_08": "'Cancel COD' by sharing a payment-app or bank code",
    "ec_09": "Fake support number asks for full card details",
    "ec_10": "'Posted officer' buyer asks the seller to pay security",
    "ec_11": "Install a 'refund helper' app from a link",
    "ec_12": "Small 'verification payment' before compensation",
    "ec_13": "Screen-share to cancel a subscription",
    "ec_14": "Card details on a call to 'lock' an exchange price",
    "ec_15": "Seller KYC link plus OTP",
    "ec_16": "Phone pre-booking paid to a personal UPI",
    "ec_17": "Paid 'warranty activation' via unofficial link",
    "ec_18": "Address confirmation by dialling a forwarding code",
    "ec_19": "Buy gift cards and read out the claim codes",
    "ec_20": "Vehicle token paid before any viewing",
}

CONSIDERED_NOT_CLUSTERED = [
    "OTP-disclosure families bp_01, bp_14, bp_18, and ec_15 share a credential surface but differ in pretext (card block, pension, dormancy, seller KYC); not clustered.",
    "bp_06 and bp_17 both demand an unofficial fee to unlock a credit product (origination vs restructuring); kept separate from each other and from C2 (prize/payout/release).",
    "ec_03, ec_12, and ec_17 are small unofficial e-commerce fees for different actions (redelivery, compensation, warranty); not clustered.",
    "ec_16 is an unofficial advance for a new-phone pre-order, not a classifieds hold, so it was not added to C11.",
    "jr_04 and jr_09 are both deposits (equipment vs accuracy collateral); cards treat them as distinct.",
    "jr_11 (post-selection uniform/training) and jr_20 (tutor leads) were not merged into C8 or C10.",
]

SPLIT_LINE_RE = re.compile(
    r'^  "split": "(unassigned|train|dev|test)",$',
    re.MULTILINE,
)


def load_families(json_dir: Path) -> dict[str, dict]:
    files = sorted(json_dir.glob("*.json"))
    families: dict[str, dict] = {}
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        family_id = record["family_id"]
        if family_id != path.stem:
            raise SystemExit(f"{path.name}: family_id {family_id!r} != stem")
        families[family_id] = record
    if len(families) != 60:
        raise SystemExit(f"expected 60 families, found {len(families)}")
    return families


def cluster_index(clusters: dict[str, tuple[str, ...]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cluster_id, members in clusters.items():
        for family_id in members:
            if family_id in mapping:
                raise SystemExit(f"{family_id} listed in {mapping[family_id]} and {cluster_id}")
            mapping[family_id] = cluster_id
    return mapping


def legitimate_blob(record: dict) -> str:
    legit = record["cases"]["legitimate"]
    parts = [
        legit.get("label_rationale") or "",
        legit.get("caller_objective") or "",
        *(legit.get("stable_facts") or []),
        *(item.get("text") or "" for item in legit.get("markers") or []),
    ]
    for turn in legit.get("dialogue_plan") or []:
        parts.append(turn.get("act") or "")
    return "\n".join(parts)


def domain_of(family_id: str, families: dict[str, dict]) -> str:
    return families[family_id]["domain"]


def counts_by_split_domain(assignment: dict[str, str], families: dict[str, dict]) -> dict[str, dict[str, int]]:
    counts = {split: {domain: 0 for domain in DOMAINS} for split in SPLITS}
    for family_id, split in assignment.items():
        counts[split][domain_of(family_id, families)] += 1
    return counts


def assign(families: dict[str, dict], clusters: dict[str, tuple[str, ...]], seed: int) -> dict[str, str]:
    """Deterministic family-level assignment.

    Policy:
    1. Every multi-family cluster is one unit and goes to train (largest
       clusters first by construction; all of them fit in train).
    2. Pilot families, and any cluster that contains a pilot, never go to test.
    3. Test and development are singleton-only and 4 families per domain.
    4. Forced test families cover the three keyword confounds.
    5. Remaining test slots follow TEST_FILL (distinct-mechanism ranking).
    6. Development takes leftover pilots first, then remaining singletons
       in family-id order, 4 per domain.
    7. Everyone else is train. Seed is used only to break equal ranks.
    """
    rng = random.Random(seed)
    owner = cluster_index(clusters)
    assignment: dict[str, str] = {}

    # Largest clusters (size 4, then 3, then 2) all go to train.
    by_size = sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))
    for _cluster_id, members in by_size:
        for family_id in members:
            assignment[family_id] = "train"

    def is_singleton(family_id: str) -> bool:
        return family_id not in owner

    def eligible_test(family_id: str) -> bool:
        record = families[family_id]
        if record.get("pilot"):
            return False
        if not is_singleton(family_id):
            return False
        if family_id in assignment:
            return False
        return True

    for family_id in FORCED_TEST:
        if family_id not in families:
            raise SystemExit(f"forced test family missing: {family_id}")
        if not eligible_test(family_id) and assignment.get(family_id) != "test":
            raise SystemExit(f"forced test family is not an eligible singleton: {family_id}")
        assignment[family_id] = "test"

    for domain, ranked in TEST_FILL.items():
        have = sum(
            1
            for fid, split in assignment.items()
            if split == "test" and domain_of(fid, families) == domain
        )
        need = TARGET_PER_DOMAIN["test"] - have
        # Group remaining ranked families by position so equal-rank ties
        # (same tuple index is unique here) can still consult the seed.
        candidates = [fid for fid in ranked if eligible_test(fid) and domain_of(fid, families) == domain]
        # Stable unique ranking; shuffle is a no-op unless we add ties later.
        scored = [(rank, fid) for rank, fid in enumerate(candidates)]
        # Seed consult: shuffle a copy of any equal-rank bucket.
        buckets: dict[int, list[str]] = defaultdict(list)
        for rank, fid in scored:
            buckets[rank].append(fid)
        ordered: list[str] = []
        for rank in sorted(buckets):
            bucket = list(buckets[rank])
            rng.shuffle(bucket)
            ordered.extend(bucket)
        for family_id in ordered:
            if need <= 0:
                break
            assignment[family_id] = "test"
            need -= 1
        if need > 0:
            raise SystemExit(f"could not fill test for {domain}: short {need}")

    unassigned = [fid for fid in families if fid not in assignment]
    unassigned.sort()
    for domain in DOMAINS:
        pool = [fid for fid in unassigned if domain_of(fid, families) == domain]
        # Pilots first, then family id. Shuffle only within the same key.
        keyed: dict[tuple[int, str], list[str]] = defaultdict(list)
        for fid in pool:
            keyed[(0 if families[fid].get("pilot") else 1, fid)].append(fid)
        ordered = []
        for key in sorted(keyed):
            bucket = list(keyed[key])
            rng.shuffle(bucket)
            ordered.extend(bucket)
        taken = 0
        for family_id in ordered:
            if taken >= TARGET_PER_DOMAIN["dev"]:
                break
            assignment[family_id] = "dev"
            taken += 1

    for family_id in families:
        assignment.setdefault(family_id, "train")
    return assignment


def check_assignment(
    assignment: dict[str, str],
    families: dict[str, dict],
    clusters: dict[str, tuple[str, ...]],
) -> list[str]:
    errors: list[str] = []
    if len(assignment) != 60 or set(assignment) != set(families):
        errors.append(f"assignment covers {len(assignment)} families, expected 60")
    unknown = {split for split in assignment.values() if split not in SPLITS}
    if unknown:
        errors.append(f"unknown split values: {sorted(unknown)}")

    for cluster_id, members in clusters.items():
        splits = {assignment[fid] for fid in members}
        if len(splits) != 1:
            errors.append(f"{cluster_id} spans splits {sorted(splits)}: {members}")

    for family_id in PILOTS:
        if assignment.get(family_id) == "test":
            errors.append(f"pilot {family_id} assigned to test")

    totals = Counter(assignment.values())
    if sum(totals.values()) != 60:
        errors.append(f"counts total {sum(totals.values())}, expected 60")
    for split, target in TARGET_OVERALL.items():
        if totals[split] != target:
            errors.append(f"{split} has {totals[split]} families, expected {target}")

    for family_id in FORCED_TEST:
        if assignment.get(family_id) != "test":
            errors.append(f"confound family {family_id} is not in test")

    # Confound keywords must appear in the legitimate case of the chosen families.
    otp_blob = legitimate_blob(families["ec_08"])
    if not re.search(r"\bOTP\b", otp_blob, re.IGNORECASE):
        errors.append("ec_08 legitimate case does not mention OTP")
    fee_blob = legitimate_blob(families["jr_14"])
    if not re.search(r"\bfee\b", fee_blob, re.IGNORECASE):
        errors.append("jr_14 legitimate case does not mention a fee")
    acct_blob = legitimate_blob(families["jr_08"])
    if not re.search(r"account number", acct_blob, re.IGNORECASE):
        errors.append("jr_08 legitimate case does not mention an account number")
    return errors


def apply_json_splits(json_dir: Path, assignment: dict[str, str]) -> int:
    updated = 0
    for family_id, split in assignment.items():
        path = json_dir / f"{family_id}.json"
        text = path.read_text(encoding="utf-8")
        new_text, n = SPLIT_LINE_RE.subn(f'  "split": "{split}",', text, count=1)
        if n != 1:
            raise SystemExit(f"{path.name}: could not replace split field")
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            updated += 1
    return updated


def deviation(counts: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        split: {domain: counts[split][domain] - TARGET_PER_DOMAIN[split] for domain in DOMAINS}
        for split in SPLITS
    }


def build_notes(counts: dict[str, dict[str, int]], added_clusters: dict[str, tuple[str, ...]]) -> dict:
    return {
        "policy": (
            "All multi-family clusters go to train. Test and development are "
            "singleton-only, 4 families per domain. Seed 20260925 breaks equal ranks."
        ),
        "added_clusters": {cid: list(members) for cid, members in added_clusters.items()},
        "considered_not_clustered": CONSIDERED_NOT_CLUSTERED,
        "domain_target": "12 train / 4 dev / 4 test per domain",
        "domain_deviation": deviation(counts),
        "pilots": {
            "never_test": list(PILOTS),
            "clusters_pulled_from_test": {"C12": list(GIVEN_CLUSTERS["C12"]), "C10": list(GIVEN_CLUSTERS["C10"])},
        },
        "test_confounds": {
            "otp_in_legitimate_case": {
                "family": "ec_08",
                "note": (
                    "Only ec_01_L actually requests a delivery OTP; it is a pilot "
                    "so it cannot be test (placed in dev). ec_08_L discusses the "
                    "delivery OTP (recipient asks; agent says it is only for "
                    "accepting a parcel). bp_14_L also contains the OTP keyword."
                ),
            },
            "fee_disclosed_legitimately": {
                "family": "jr_14",
                "note": (
                    "jr_14_L discloses an onboarding fee shown in the app and "
                    "deducted from weekly earnings. bp_06_L (fee in the KFS, "
                    "dev) is the banking counterpart."
                ),
            },
            "account_number_requested_legitimately": {
                "family": "jr_08",
                "note": (
                    "jr_18_L is the strongest live payroll request (account "
                    "number, IFSC, name) but sits in C7, which is train. "
                    "jr_02_L also requests account number and IFSC on the call "
                    "but is a pilot (dev). jr_08_L states that account number "
                    "and IFSC are needed for salary after an offer. jr_14_L "
                    "also has bank account number and IFSC entered in the app."
                ),
            },
        },
    }


def write_manifest(
    path: Path,
    clusters: dict[str, tuple[str, ...]],
    assignment: dict[str, str],
    counts: dict[str, dict[str, int]],
    notes: dict,
    seed: int,
) -> None:
    payload = {
        "version": VERSION,
        "seed": seed,
        "created": CREATED,
        "clusters": {cid: list(members) for cid, members in clusters.items()},
        "assignment": dict(sorted(assignment.items())),
        "counts": counts,
        "notes": notes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _md_table(rows: list[tuple[str, str, str, str]]) -> str:
    lines = [
        "| Family | Domain | Cluster | Mechanism |",
        "|---|---|---|---|",
    ]
    for family_id, domain, cluster, mechanism in rows:
        lines.append(f"| `{family_id}` | `{domain}` | {cluster} | {mechanism} |")
    return "\n".join(lines)


def write_markdown(
    path: Path,
    clusters: dict[str, tuple[str, ...]],
    assignment: dict[str, str],
    families: dict[str, dict],
    counts: dict[str, dict[str, int]],
    notes: dict,
    seed: int,
) -> None:
    owner = cluster_index(clusters)
    created = date.fromisoformat(CREATED).isoformat()
    devi = notes["domain_deviation"]
    zero_dev = all(devi[split][domain] == 0 for split in SPLITS for domain in DOMAINS)

    lines = [
        "# Scenario-family splits (v1)",
        "",
        f"Generated `{created}` by `scripts/scenarios/assign_splits.py` with seed `{seed}`.",
        "Assignment is at **family** level: both cases, later histories, branches,",
        "and language versions of a family stay in the same split.",
        "",
        "## Policy",
        "",
        "- Near-duplicate clusters C1–C12 are atomic. No family was deleted or edited.",
        "- Largest clusters go to train. Test and development are singleton-only,",
        "  so the test set covers 12 distinct mechanisms.",
        "- Pilot families `bp_01`, `bp_02`, `jr_01`, `jr_02`, `ec_01`, `ec_02` are",
        "  never test. `bp_02` brings C12; `jr_01` brings C10.",
        "- Target: 36 / 12 / 12 families, and 12 / 4 / 4 per domain.",
        "",
        "## Counts",
        "",
        "| Split | banking_payments | jobs_recruitment | ecommerce_delivery | Total |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        row = counts[split]
        total = sum(row.values())
        lines.append(
            f"| {split} | {row['banking_payments']} | {row['jobs_recruitment']} | "
            f"{row['ecommerce_delivery']} | {total} |"
        )
    lines.append(
        f"| **all** | 20 | 20 | 20 | {sum(sum(counts[s].values()) for s in SPLITS)} |"
    )
    lines.extend(
        [
            "",
            "## Deviation from 12 / 4 / 4 per domain",
            "",
        ]
    )
    if zero_dev:
        lines.append(
            "None. Every domain has exactly 12 train, 4 development, and 4 test families."
        )
        lines.append(
            "Cross-domain clusters (C1–C6, C11) are all in train, so they do not"
        )
        lines.append("unbalance the held-out sets.")
    else:
        lines.append("| Split | Domain | Deviation |")
        lines.append("|---|---|---:|")
        for split in SPLITS:
            for domain in DOMAINS:
                delta = devi[split][domain]
                if delta:
                    lines.append(f"| {split} | `{domain}` | {delta:+d} |")
    lines.extend(
        [
            "",
            "## Test keyword confounds",
            "",
            "| Confound | Test family | Why this family |",
            "|---|---|---|",
            "| OTP in the legitimate case | `ec_08` | "
            "`ec_01_L` is the only card that *requests* a delivery OTP, but it is a "
            "pilot (development). `ec_08_L` discusses the delivery OTP: the recipient "
            "asks whether they must give a code; the agent says the OTP is only for "
            "accepting a parcel. `bp_14_L` also contains the OTP keyword. |",
            "| Fee disclosed legitimately | `jr_14` | "
            "Onboarding fee is shown in the app and deducted from weekly earnings. "
            "`bp_06_L` (processing fee in the KFS) is the banking counterpart, in development. |",
            "| Account number requested legitimately | `jr_08` | "
            "Recruiter states that account number and IFSC are needed for salary after "
            "an offer. `jr_18` (strongest live payroll request) is in C7 / train; "
            "`jr_02_L` requests them on the call but is a pilot (development). "
            "`jr_14_L` also has account number and IFSC entered in the app. |",
            "",
            "## Clusters",
            "",
            "No extra clusters were added. Pairs that share a surface (OTP, unofficial "
            "fee, deposit) but differ in pretext or required action stay singletons; "
            "see `notes.considered_not_clustered` in `splits_v1.json`.",
            "",
        ]
    )
    for cluster_id, members in clusters.items():
        split = assignment[members[0]]
        listed = ", ".join(f"`{fid}`" for fid in members)
        lines.append(f"- **{cluster_id}** → {split}: {listed}")
    lines.append("")

    for split in SPLITS:
        lines.extend([f"## {split.capitalize()}", ""])
        rows = []
        for family_id in sorted(fid for fid, sp in assignment.items() if sp == split):
            cluster = owner.get(family_id, "—")
            rows.append(
                (
                    family_id,
                    families[family_id]["domain"],
                    cluster,
                    MECHANISMS[family_id],
                )
            )
        lines.append(_md_table(rows))
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def extra_clusters_from_review() -> dict[str, tuple[str, ...]]:
    """Hook for added near-duplicates. Review found none that should merge."""
    return {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--no-apply",
        action="store_true",
        help="write manifests only; leave scenario JSON split fields unchanged",
    )
    args = parser.parse_args(argv)

    families = load_families(args.json_dir)
    added = extra_clusters_from_review()
    clusters = dict(GIVEN_CLUSTERS)
    clusters.update(added)

    seen = [fid for members in clusters.values() for fid in members]
    missing = [fid for fid in seen if fid not in families]
    if missing:
        raise SystemExit(f"cluster members not on disk: {missing}")

    assignment = assign(families, clusters, args.seed)
    errors = check_assignment(assignment, families, clusters)
    if errors:
        print("assignment checks failed:", file=sys.stderr)
        for item in errors:
            print(f"  - {item}", file=sys.stderr)
        return 1

    counts = counts_by_split_domain(assignment, families)
    notes = build_notes(counts, added)
    write_manifest(args.manifest, clusters, assignment, counts, notes, args.seed)
    write_markdown(args.markdown, clusters, assignment, families, counts, notes, args.seed)

    updated = 0
    if not args.no_apply:
        updated = apply_json_splits(args.json_dir, assignment)

    print(f"wrote {args.manifest.relative_to(REPO_ROOT)}")
    print(f"wrote {args.markdown.relative_to(REPO_ROOT)}")
    if not args.no_apply:
        print(f"updated split field on {updated} JSON files")
    print("checks: clusters atomic; pilots not in test; 60 families; 36/12/12")
    for split in SPLITS:
        parts = " ".join(f"{domain}={counts[split][domain]}" for domain in DOMAINS)
        print(f"  {split}: {parts} (n={sum(counts[split].values())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
