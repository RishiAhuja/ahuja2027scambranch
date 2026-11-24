# Scenario-family splits (v1)

Generated `2026-09-25` by `scripts/scenarios/assign_splits.py` with seed `20260925`.
Assignment is at **family** level: both cases, later histories, branches,
and language versions of a family stay in the same split.

## Policy

- Near-duplicate clusters C1–C12 are atomic. No family was deleted or edited.
- Largest clusters go to train. Test and development are singleton-only,
  so the test set covers 12 distinct mechanisms.
- Pilot families `bp_01`, `bp_02`, `jr_01`, `jr_02`, `ec_01`, `ec_02` are
  never test. `bp_02` brings C12; `jr_01` brings C10.
- Target: 36 / 12 / 12 families, and 12 / 4 / 4 per domain.

## Counts

| Split | banking_payments | jobs_recruitment | ecommerce_delivery | Total |
|---|---:|---:|---:|---:|
| train | 12 | 12 | 12 | 36 |
| dev | 4 | 4 | 4 | 12 |
| test | 4 | 4 | 4 | 12 |
| **all** | 20 | 20 | 20 | 60 |

## Deviation from 12 / 4 / 4 per domain

None. Every domain has exactly 12 train, 4 development, and 4 test families.
Cross-domain clusters (C1–C6, C11) are all in train, so they do not
unbalance the held-out sets.

## Test keyword confounds

| Confound | Test family | Why this family |
|---|---|---|
| OTP in the legitimate case | `ec_08` | `ec_01_L` is the only card that *requests* a delivery OTP, but it is a pilot (development). `ec_08_L` discusses the delivery OTP: the recipient asks whether they must give a code; the agent says the OTP is only for accepting a parcel. `bp_14_L` also contains the OTP keyword. |
| Fee disclosed legitimately | `jr_14` | Onboarding fee is shown in the app and deducted from weekly earnings. `bp_06_L` (processing fee in the KFS) is the banking counterpart, in development. |
| Account number requested legitimately | `jr_08` | Recruiter states that account number and IFSC are needed for salary after an offer. `jr_18` (strongest live payroll request) is in C7 / train; `jr_02_L` requests them on the call but is a pilot (development). `jr_14_L` also has account number and IFSC entered in the app. |

## Clusters

No extra clusters were added. Pairs that share a surface (OTP, unofficial fee, deposit) but differ in pretext or required action stay singletons; see `notes.considered_not_clustered` in `splits_v1.json`.

- **C1** → train: `bp_12`, `ec_09`, `ec_14`
- **C2** → train: `bp_13`, `ec_06`, `bp_10`, `bp_19`
- **C3** → train: `bp_03`, `ec_05`
- **C4** → train: `bp_04`, `ec_11`, `ec_13`
- **C5** → train: `bp_09`, `ec_04`
- **C6** → train: `bp_16`, `jr_16`
- **C7** → train: `jr_13`, `jr_18`
- **C8** → train: `jr_06`, `jr_12`
- **C9** → train: `jr_07`, `jr_19`
- **C10** → train: `jr_01`, `jr_10`
- **C11** → train: `ec_07`, `ec_20`
- **C12** → train: `bp_02`, `bp_07`, `bp_11`, `bp_20`

## Train

| Family | Domain | Cluster | Mechanism |
|---|---|---|---|
| `bp_02` | `banking_payments` | C12 | KYC-update link plus net-banking login |
| `bp_03` | `banking_payments` | C3 | UPI PIN entered to 'receive' a refund via collect request |
| `bp_04` | `banking_payments` | C4 | Install a screen-sharing app for 'support' |
| `bp_07` | `banking_payments` | C12 | Reward-points link that asks for full card details |
| `bp_09` | `banking_payments` | C5 | Move money to a 'safe account' under a crime-link threat |
| `bp_10` | `banking_payments` | C2 | Fee to release an insurance maturity bonus |
| `bp_11` | `banking_payments` | C12 | Tax-refund link that asks for card details |
| `bp_12` | `banking_payments` | C1 | Fake helpline asks for PIN or CVV |
| `bp_13` | `banking_payments` | C2 | Fee paid to claim a cashback prize |
| `bp_16` | `banking_payments` | C6 | Guaranteed-return investment paid into a personal account |
| `bp_19` | `banking_payments` | C2 | Premature-FD 'penalty' paid to a personal UPI |
| `bp_20` | `banking_payments` | C12 | Merger re-registration via link and PIN |
| `ec_04` | `ecommerce_delivery` | C5 | Contraband or arrest threat; pay to close the 'case' |
| `ec_05` | `ecommerce_delivery` | C3 | UPI PIN entered to 'receive' a marketplace refund |
| `ec_06` | `ecommerce_delivery` | C2 | Lucky-draw prize tax paid to a UPI ID |
| `ec_07` | `ecommerce_delivery` | C11 | Classifieds seller demands full advance to a personal UPI |
| `ec_09` | `ecommerce_delivery` | C1 | Fake support number asks for full card details |
| `ec_11` | `ecommerce_delivery` | C4 | Install a 'refund helper' app from a link |
| `ec_13` | `ecommerce_delivery` | C4 | Screen-share to cancel a subscription |
| `ec_14` | `ecommerce_delivery` | C1 | Card details on a call to 'lock' an exchange price |
| `ec_15` | `ecommerce_delivery` | — | Seller KYC link plus OTP |
| `ec_16` | `ecommerce_delivery` | — | Phone pre-booking paid to a personal UPI |
| `ec_17` | `ecommerce_delivery` | — | Paid 'warranty activation' via unofficial link |
| `ec_20` | `ecommerce_delivery` | C11 | Vehicle token paid before any viewing |
| `jr_01` | `jobs_recruitment` | C10 | Registration fee for a guaranteed job |
| `jr_06` | `jobs_recruitment` | C8 | Paid mandatory certification before joining |
| `jr_07` | `jobs_recruitment` | C9 | Candidate-paid background-verification fee |
| `jr_10` | `jobs_recruitment` | C10 | Campus-drive registration payment |
| `jr_11` | `jobs_recruitment` | — | Uniform and training fees after a phone-only 'selection' |
| `jr_12` | `jobs_recruitment` | C8 | Starter kit must be bought before a stipend internship |
| `jr_13` | `jobs_recruitment` | C7 | OTP disclosure to 'verify' onboarding documents |
| `jr_15` | `jobs_recruitment` | — | Paid 'priority placement' with a guaranteed shortlist |
| `jr_16` | `jobs_recruitment` | C6 | Trading 'job' that requires investing the candidate's own money |
| `jr_18` | `jobs_recruitment` | C7 | Net-banking login to 'credit' a joining bonus |
| `jr_19` | `jobs_recruitment` | C9 | Pre-employment medical paid to an individual's UPI |
| `jr_20` | `jobs_recruitment` | — | Upfront tutor-registration fee for guaranteed students |

## Dev

| Family | Domain | Cluster | Mechanism |
|---|---|---|---|
| `bp_01` | `banking_payments` | — | OTP requested to unblock a card after a disputed transaction |
| `bp_06` | `banking_payments` | — | Advance processing fee to a personal account before disbursal |
| `bp_17` | `banking_payments` | — | Fee paid to an agent for EMI relief |
| `bp_18` | `banking_payments` | — | Dormant-account reactivation fee plus OTP |
| `ec_01` | `ecommerce_delivery` | — | Bank OTP collected as if it were a delivery code |
| `ec_02` | `ecommerce_delivery` | — | QR code presented as a way to receive payment |
| `ec_03` | `ecommerce_delivery` | — | Redelivery fee via unofficial link plus card details |
| `ec_12` | `ecommerce_delivery` | — | Small 'verification payment' before compensation |
| `jr_02` | `jobs_recruitment` | — | Deposits required to unlock task earnings |
| `jr_04` | `jobs_recruitment` | — | Laptop security deposit after a fake offer |
| `jr_05` | `jobs_recruitment` | — | Payment for 'backdoor' government selection |
| `jr_09` | `jobs_recruitment` | — | Forfeitable security deposit for data-entry work |

## Test

| Family | Domain | Cluster | Mechanism |
|---|---|---|---|
| `bp_05` | `banking_payments` | — | Electricity bill paid to a personal UPI ID |
| `bp_08` | `banking_payments` | — | Send a 'mistaken' credit back to the caller's UPI |
| `bp_14` | `banking_payments` | — | OTP requested to continue pension 'verification' |
| `bp_15` | `banking_payments` | — | CVV and expiry requested for a 'free card upgrade' |
| `ec_08` | `ecommerce_delivery` | — | 'Cancel COD' by sharing a payment-app or bank code |
| `ec_10` | `ecommerce_delivery` | — | 'Posted officer' buyer asks the seller to pay security |
| `ec_18` | `ecommerce_delivery` | — | Address confirmation by dialling a forwarding code |
| `ec_19` | `ecommerce_delivery` | — | Buy gift cards and read out the claim codes |
| `jr_03` | `jobs_recruitment` | — | Visa processing fee to a personal account |
| `jr_08` | `jobs_recruitment` | — | Job is receiving and forwarding other people's funds |
| `jr_14` | `jobs_recruitment` | — | Gig onboarding fee paid to an agent's UPI |
| `jr_17` | `jobs_recruitment` | — | Chat interview, then unofficial app install and in-app payment |
