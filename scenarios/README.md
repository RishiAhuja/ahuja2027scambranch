# Scenario cards

Scenario cards are the human-owned truth specification for generation (plan Section 8). The generator never sees anything that contradicts a card, and the detector never sees the card.

## Layout

```text
scenarios/
  README.md                 this file: conventions and status
  TEMPLATE.md               blank card to copy for each family
  draft/                    authored, not yet reviewed
    banking_payments/
      README.md             domain notes + catalog of 20 families
      bp_01_*.md            one file per family (fraud + legitimate case)
    jobs_recruitment/
    ecommerce_delivery/
  approved/                 cards move here only after author + reviewer sign-off
```

One file = one **scenario family** = one fraud case (`_F`) + one legitimate counterpart (`_L`). Both cases share a domain, a surface setting, and a comparable branch point, so the pair differs in the caller's action, not in topic.

## ID conventions

| Item | Pattern | Example |
|---|---|---|
| Family | `<domain>_<nn>` | `bp_01` |
| Fraud case | `<family>_F` | `bp_01_F` |
| Legitimate case | `<family>_L` | `bp_01_L` |
| Domain prefixes | `bp` banking/payments, `jr` jobs/recruitment, `ec` e-commerce/delivery | |

## Authoring rules

1. **Fictional entities only.** Use the organisation names in `fictional_entities.md`. Links use the reserved `.invalid` TLD (RFC 2606). Phone numbers, account numbers, and UPI IDs are placeholders such as `[PHONE_1]`, `[ACCT_LAST4]`.
2. **No operational uplift.** Cards describe what a caller asks for at the level a public awareness leaflet would. Do not script working persuasion techniques, real app names for remote access, or real payment flows beyond generic descriptions.
3. **Evidence is an action in context, not a keyword.** "OTP", "urgent", or "fee" alone never decides the label. Several legitimate cases deliberately contain those words.
4. **Recipient reactions are never evidence.** A suspicious recipient does not make a call fraudulent; a cooperative one does not make it legitimate.
5. **Evidence before the branch point.** For every fraud case, the fraud-establishing request must be visible before `branch_after`. For every legitimate case, nothing fraud-like may be introduced after the branch.
6. **Domain facts need a source.** Any claim about how real institutions behave (for example "banks do not ask for OTPs on calls") is listed under *Evidence references* with a URL and must be checked against the source, with a retrieval date, before sign-off. Unchecked references are marked `VERIFY`.
7. **Cards are language-neutral.** Write cards in English. Language profiles handle localisation; a card must not depend on a particular language pair.

## Pilot families

Six families are authored first (plan Stage C). Pilot families may later enter train or development, never the final test.

| Family | Domain | Title | Status |
|---|---|---|---|
| `bp_01` | Banking/payments | Card "blocked" OTP request vs. fraud-monitoring verification | draft |
| `bp_02` | Banking/payments | KYC update link vs. re-KYC reminder via official channels | draft |
| `jr_01` | Jobs/recruitment | Registration fee for guaranteed job vs. fee-free interview scheduling | draft |
| `jr_02` | Jobs/recruitment | Paid "task" work-from-home vs. contracted part-time work | draft |
| `ec_01` | E-commerce/delivery | Bank OTP disguised as delivery code vs. genuine delivery OTP | draft |
| `ec_02` | E-commerce/delivery | Marketplace "scan to receive" QR vs. ordinary buyer payment | draft |

All 60 families are drafted (status `draft` in each domain's `README.md`). Split assignment (36/12/12) happens only after deduplication.

## Sources

Real-world claims are verified against the registers in `references/`: `source_register.md` (shared, S/C IDs), `sources_banking_payments.md` (SB/CB), `sources_jobs_recruitment.md` (SJ/CJ), and `sources_ecommerce_delivery.md` (SE/CE). Claims marked `VERIFY` or "reported/secondary" still need a direct check.
