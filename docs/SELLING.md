# Selling this — commercial and EU regulatory notes

**Status of this document: research notes, not legal advice.** Nothing here
establishes that the product complies with anything. Every item below is a
question to put to a Polish lawyer and a tax adviser before you take money,
with the primary source checked as it stands on the day you launch. Where a
date or an article number appears, verify it — regulation moves and this file
does not.

---

## 1. The positioning, and why it is the whole product

You are selling a **statistical audit of a past track record**. You are not
selling signals, forecasts, portfolios, or an opinion about what anyone should
do with money.

That is not modesty. It is the line that keeps this a software product instead
of a regulated financial service, and everything in the codebase is built to
sit on the safe side of it:

- no output names an instrument to buy or sell
- no output is tailored to a person's circumstances
- the tool takes a record that already exists and reports what could be chance
- every report carries the "audit, not advice" footer, and it is tested

**Do not cross that line in marketing.** "Find profitable strategies", "our
tool identifies winning systems", "increase your returns" — each of those
turns an audit tool into something that looks like a personal recommendation,
and the regulator reads the landing page, not the source code.

The honest pitch is stronger anyway: *most backtests are luck, and this tells
you which one yours is.* It is unusual, checkable, and it is the only claim
the code can actually support.

---

## 2. Financial-services perimeter (check first)

| question | why it matters |
|---|---|
| Does any output constitute a **personal recommendation** under MiFID II (investment advice)? | Investment advice is a licensed activity in Poland (KNF). The design answer is no — the tool never recommends an instrument or a transaction — but it is the first question counsel should be asked, and the answer must stay no as features are added. |
| Is any output an **investment recommendation** under the Market Abuse Regulation? | MAR and its delegated regulation impose disclosure duties on anyone producing or disseminating recommendations, including non-professionals. Again the design answer is no; again, keep it no. |
| Does marketing make it **look** like either? | Perimeter is assessed on substance and presentation together. |

Hard rules for the product, whatever counsel says:

1. Never add a "which strategy should I run" feature.
2. Never publish audits of *your own* live strategy alongside a sales page.
3. Keep the disclaimer in the report, not only in the terms.

---

## 3. Sales model consequences

The chosen model is **local, self-hosted, perpetual-ish licence with an expiry
date**. That choice removes most of the hard obligations before they start:

| obligation | why it is light here |
|---|---|
| **GDPR as a processor** | Customer data never reaches you. The user's CSV is read on their machine and the report is written next to it. You are a controller only for the order record: name, email, company, licence ref. |
| **Hosting, uptime, backups** | None. There is no service to be down. |
| **Data breach exposure** | Your worst case is a customer list, not a fund's positions. |

What you still owe:

- a **privacy notice** for the order data you do hold (lawful basis: performance
  of the contract, plus a legal obligation for invoices);
- **retention**: Polish accounting rules require invoices to be kept for years;
  the marketing list is not the same data and should not inherit that period.

---

## 4. Consumer law — the part that bites

If you sell to **consumers** in the EU (and a solo trader buying a tool for
personal use often is one), at least three regimes apply. Confirm each:

**Right of withdrawal.** Digital content supplied without a physical medium
carries a 14-day withdrawal right under the Consumer Rights Directive. It can
be lost, but only if the consumer gives **prior express consent** to immediate
supply *and* **acknowledges** losing the right. Both have to happen at
checkout, as a deliberate action, and be recorded. Most payment processors have
a setting for this; most sellers never switch it on, and then owe refunds.

**Conformity of digital content.** Directive (EU) 2019/770 gives buyers
remedies when digital content does not conform to the contract, for a period
after supply. Practical consequences: your description must match what the
software does, and "it rejected my strategy" is not non-conformity — *but a
verdict produced by a genuine bug is*. The test suite is a commercial asset,
not just an engineering one.

**Pre-contract information.** Main characteristics, total price including VAT,
your identity and address, the complaints route, functionality and
interoperability of the digital content (it needs Python 3.11+, it runs
offline, it is not a subscription service). Missing information extends the
withdrawal period.

**Business customers** (sp. z o.o., a fund, a prop firm) do not get the
withdrawal right, and B2B is the better market for this product anyway.

---

## 5. VAT

- **B2C digital services in the EU**: taxed where the customer is. The
  One-Stop-Shop (OSS) exists so you file once instead of registering in every
  member state. There is a small-turnover simplification with a threshold —
  check the current figure and whether you are under it.
- **B2B in the EU** with a valid VAT number: reverse charge, invoice notes it.
- **Outside the EU**: different rules per country; most sellers use a
  **merchant of record** (Paddle, Lemon Squeezy, FastSpring) precisely so this
  becomes someone else's problem. For a one-person operation that is usually
  the right trade: a few percent of revenue against a compliance function you
  do not have.

---

## 6. Product liability

Directive (EU) 2024/2853 brings **software within the definition of a product**
for defective-product liability. Transposition is due in member states, and
it applies to products placed on the market after that point — **verify the
Polish implementation and its date before you launch.**

Practical consequences, none of them exotic:

- keep the disclaimer accurate and prominent (it limits expectations, it does
  not exclude liability for defects);
- keep the test suite green and keep releases traceable — which version a
  customer ran, and what changed;
- never claim the tool prevents losses, validates a strategy as safe, or
  certifies anything.

---

## 7. EU AI Act

The tool contains **no AI system**: permutation tests, bootstraps and a
Bonferroni correction are deterministic statistics, and a ridge logistic
regression elsewhere in this repository is not part of the audit path. On the
current reading it is out of scope.

Say so if asked; do not advertise it as AI. Calling deterministic statistics
"AI" to ride a trend would pull an out-of-scope product into scope for no
commercial gain, and it would be untrue.

---

## 8. Pricing — reasoning, not a recommendation

The value is a *decision*: whether to put capital behind a rule, or to stop
maintaining one that does not work. Price against the cost of being wrong once,
not against the cost of the compute.

| segment | shape | reasoning |
|---|---|---|
| **Individual trader** | one-off, single seat, 12-month updates | Impulse-range purchase, high support burden per euro. Keep the evaluation mode generous so nobody buys and refunds. |
| **Prop firm / trading educator** | per-seat annual | They audit many strategies and many people's strategies. Recurring, and the multiplicity check is exactly their problem. |
| **Fund / family office** | annual site licence, invoiced | The report is an artefact they can hand to an investment committee. This is where the price is defensible, and where B2B removes the consumer-law surface entirely. |

Do not set the numbers from this file. Set them from five conversations with
people in the segment you choose, and raise them once you have the first three
customers.

### What actually sells it

Not features. Two things:

1. **The demo rejects a good-looking strategy.** `python run.py audit --example`
   produces a +135% / Sharpe 1.7 track record on a random walk and the tool
   rejects it. Nobody argues with that; everybody immediately wants to know
   what it says about theirs.
2. **The tool's own project has a negative headline finding.** A vendor whose
   research says "our hypothesis did not survive" is making a credibility claim
   no marketing copy can buy. Lead with it.

---

## 9. Before the first sale — checklist

- [ ] Counsel confirms the product sits outside investment advice and MAR.
- [ ] Landing page reviewed against §1; no performance or profit claims.
- [ ] Terms, privacy notice and pre-contract information published.
- [ ] Withdrawal-right consent and acknowledgement implemented at checkout.
- [ ] VAT route decided: OSS yourself, or a merchant of record.
- [ ] Vendor signing key generated, stored offline, backed up in two places.
- [ ] `PUBLIC_KEY_B64` set in the shipped build; evaluation mode verified with
      and without a key.
- [ ] A licence issued to yourself and verified on a clean machine.
- [ ] Release tagged, test suite green, version recorded.
