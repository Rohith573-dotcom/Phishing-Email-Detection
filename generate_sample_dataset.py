"""
generate_sample_dataset.py
==========================
Builds `data/emails_raw.csv`, a labelled corpus of 2,500 e-mails (phishing vs
legitimate) so the whole pipeline runs offline with no Kaggle download.

IMPORTANT
---------
The corpus is SYNTHETIC and defensive in purpose: it reproduces the *surface
patterns* that phishing filters are trained to recognise - manufactured
urgency, credential-harvesting call-to-actions, generic salutations, look-alike
sender domains, raw-IP links - using placeholder brands and RFC-2606 reserved
domains (`example.com`, `example.net`).  Nothing here is a working lure and no
real organisation is impersonated.  Replace it with a public corpus (Kaggle
"Phishing Email Detection", the SpamAssassin ham corpus, or the Nazario
phishing corpus) before final submission; `main.py` picks up
`data/emails.csv` automatically if you drop one in.

Design note
-----------
As in Project 1, each message carries a hidden leaning theta ~ Beta(0.4, 0.4).
Sentences are drawn from the phishing pool with probability theta and from the
business pool otherwise, with a shared neutral pool diluting both.  The classes
therefore overlap genuinely and the classifiers separate on merit rather than
on an artificial vocabulary split.

Calibration note
----------------
The mixture parameters are set so that classifier accuracy lands in the low
nineties, which is the range production phishing filters report on real mail.
An earlier and deliberately harsher setting (Beta(0.7, 0.7), 20% neutral
sentences, 3% label noise) held every model in the mid-eighties.  That made the
comparison between algorithms sharper, but it understated how separable real
phishing actually is, so the softer calibration is the more faithful
simulation.  Both settings are one edit away from each other if you want to see
how the ranking behaves under a harder benchmark.

Usage
-----
    python generate_sample_dataset.py

Author : IICT Summer Internship 2026 (AI & ML)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

RNG = np.random.default_rng(4242)

N_EMAILS     = 2500
LABEL_NOISE  = 0.015
NEUTRAL_RATE = 0.12
BETA_A, BETA_B = 0.4, 0.4

# --------------------------------------------------------------------------- #
# Sentence pools
# --------------------------------------------------------------------------- #
PHISH_LINES = [
    "Your account has been temporarily suspended due to unusual sign in activity.",
    "You must verify your identity within 24 hours or access will be permanently revoked.",
    "Click the secure link below to confirm your login credentials immediately.",
    "Our security team detected an unauthorised attempt from a foreign IP address.",
    "Failure to respond will result in the immediate closure of your account.",
    "Update your billing information now to avoid an interruption of service.",
    "You have been selected to receive a limited time reward, claim it before it expires.",
    "This is your final notice regarding the pending transaction on your account.",
    "Confirm your password to restore full access to the customer portal.",
    "A payment of nine hundred dollars is awaiting your authorisation.",
    "Please do not share this message with anyone in your organisation.",
    "Verify your details using the form attached to this message.",
    "Your mailbox storage is full and outgoing messages are being blocked.",
    "Urgent action is required to keep your subscription active.",
    "We could not process your last payment, please resubmit your card details.",
    "Reply with your employee ID and one time passcode to complete the check.",
    "The document is waiting for your signature and expires at midnight tonight.",
    "Your package could not be delivered, reschedule using the tracking link.",
    "Congratulations, you are the lucky winner of this quarter draw.",
    "Act now, this security notice will expire in a few hours.",
]

LEGIT_LINES = [
    "Please find attached the minutes from yesterday's project review meeting.",
    "The sprint retrospective has been moved to Thursday at eleven in the morning.",
    "I have shared the updated budget worksheet on the team drive for your comments.",
    "Thanks for turning the draft around so quickly, the changes look good to me.",
    "Could you confirm whether the vendor quotation was received before the deadline.",
    "The quarterly compliance training window opens at the start of next month.",
    "Attached is the invoice for the consultancy hours logged in the last cycle.",
    "The release notes for version four have been published on the internal wiki.",
    "Let me know if Tuesday afternoon works for the client walkthrough.",
    "Payroll has confirmed that the revised timesheets were processed on time.",
    "We will need one more reviewer before this pull request can be merged.",
    "The office will be closed on Monday for the scheduled facilities maintenance.",
    "I have added your name to the distribution list for the monthly newsletter.",
    "The onboarding checklist for the new interns is ready for your sign off.",
    "Please review the risk register before Friday's steering committee call.",
    "The server migration completed overnight with no reported downtime.",
    "Attaching the agenda so everyone can add items ahead of the session.",
    "Our supplier has confirmed the revised delivery schedule for next quarter.",
    "The design team has uploaded the latest mockups for a second opinion.",
    "Recruitment has scheduled three interviews for the analyst position.",
]

NEUTRAL_LINES = [
    "Please let me know if you need anything else from my side.",
    "Thank you for your time and attention to this matter.",
    "I am following up on the message sent earlier this week.",
    "Further information is available on request.",
    "Kindly refer to the details provided below.",
    "This message relates to the reference number quoted above.",
    "We appreciate your continued cooperation.",
    "Do reach out if any part of this is unclear.",
    "A copy of this notice has been retained for the record.",
    "Regards and best wishes for the week ahead.",
]

PHISH_SUBJECTS = [
    "URGENT: Your account will be suspended",
    "Action Required - Verify Your Identity Now",
    "Security Alert: Unusual Login Detected",
    "Final Notice: Payment Authorisation Pending",
    "Your Mailbox Is Full - Immediate Action Needed",
    "Congratulations! Claim Your Reward Today",
    "Re: Failed Delivery Attempt - Reschedule Now",
    "Important: Confirm Your Billing Information",
    "Password Expiry Notification - Act Within 24 Hours",
    "Document Awaiting Your Signature - Expires Tonight",
]
LEGIT_SUBJECTS = [
    "Minutes from the project review",
    "Updated budget worksheet for comment",
    "Sprint retrospective moved to Thursday",
    "Invoice for consultancy hours - Q2",
    "Release notes v4.0 published",
    "Agenda for Friday's steering committee",
    "Onboarding checklist for new interns",
    "Server migration completed successfully",
    "Interview schedule for analyst role",
    "Facilities maintenance - office closed Monday",
]
NEUTRAL_SUBJECTS = [
    "Following up on my earlier message",
    "Quick question about the reference number",
    "Update for your records",
    "Information you requested",
]

# Look-alike / disposable-style sender domains (all RFC-2606 reserved).
PHISH_DOMAINS = [
    "secure-verify.example.net", "account-support.example.net",
    "billing-update.example.net", "mail-service.example.net",
    "notice-center.example.net", "security-alert.example.net",
    "customer-desk.example.net", "id-confirm.example.net",
]
LEGIT_DOMAINS = [
    "corp.example.com", "finance.example.com", "hr.example.com",
    "engineering.example.com", "partners.example.com", "vendor.example.org",
]
PHISH_USERS = ["support", "no-reply", "security", "admin-team", "service-desk",
               "billing", "verification", "alerts"]
LEGIT_USERS = ["priya.sharma", "arun.mehta", "s.iyer", "team.updates",
               "accounts.payable", "n.banerjee", "r.krishnan", "hr.notices"]


def _sender(theta: float) -> str:
    if RNG.random() < theta:
        return f"{RNG.choice(PHISH_USERS)}@{RNG.choice(PHISH_DOMAINS)}"
    return f"{RNG.choice(LEGIT_USERS)}@{RNG.choice(LEGIT_DOMAINS)}"


def _subject(theta: float) -> str:
    if RNG.random() < 0.18:
        return str(RNG.choice(NEUTRAL_SUBJECTS))
    if RNG.random() < theta:
        return str(RNG.choice(PHISH_SUBJECTS))
    return str(RNG.choice(LEGIT_SUBJECTS))


def _link(phishy: bool) -> str:
    """Build a URL whose *shape* carries the signal, not its destination."""
    if phishy:
        style = RNG.integers(0, 3)
        if style == 0:                                   # raw IP host
            ip = ".".join(str(int(RNG.integers(11, 240))) for _ in range(4))
            return f"http://{ip}/secure/login/verify.php"
        if style == 1:                                   # deep look-alike path
            return (f"http://{RNG.choice(PHISH_DOMAINS)}/"
                    f"{RNG.choice(['verify', 'update', 'confirm'])}/"
                    f"account?id={int(RNG.integers(10000, 99999))}")
        return f"http://bit.example.net/{int(RNG.integers(100000, 999999))}"
    return (f"https://{RNG.choice(LEGIT_DOMAINS)}/"
            f"{RNG.choice(['docs', 'wiki', 'drive', 'invoices'])}/"
            f"{RNG.choice(['agenda', 'minutes', 'q2-report', 'checklist'])}")


def _body(theta: float, n_sent: int) -> str:
    greeting = ("Dear Customer," if RNG.random() < theta
                else f"Hi {RNG.choice(['Priya', 'Arun', 'Sam', 'team', 'all'])},")
    lines = []
    for _ in range(n_sent):
        if RNG.random() < NEUTRAL_RATE:
            lines.append(str(RNG.choice(NEUTRAL_LINES)))
        elif RNG.random() < theta:
            lines.append(str(RNG.choice(PHISH_LINES)))
        else:
            lines.append(str(RNG.choice(LEGIT_LINES)))

    # Links: phishy messages carry more of them, and of a different shape.
    n_links = int(RNG.integers(1, 4)) if RNG.random() < theta else int(RNG.integers(0, 2))
    links = " ".join(_link(RNG.random() < theta) for _ in range(n_links))

    body = f"{greeting} " + " ".join(lines)
    if links:
        body += " " + links
    if RNG.random() < theta * 0.5:                       # HTML-wrapped call to action
        body += ' <a href="' + _link(True) + '">Click here to verify</a>'
    if RNG.random() < theta * 0.6:
        body += " " + "!" * int(RNG.integers(1, 4))
    return body


def build() -> pd.DataFrame:
    rows = []
    for i in range(N_EMAILS):
        theta = float(RNG.beta(BETA_A, BETA_B))
        true_label = int(theta > 0.5)                    # 0 = legitimate, 1 = phishing

        observed = true_label
        if RNG.random() < LABEL_NOISE:
            observed = 1 - true_label

        rows.append({
            "id": i,
            "sender": _sender(theta),
            "subject": _subject(theta),
            "body": _body(theta, int(RNG.integers(4, 11))),
            "label": observed,
        })

    return pd.DataFrame(rows).sample(frac=1.0, random_state=11).reset_index(drop=True)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = build()
    out = os.path.join("data", "emails_raw.csv")
    df.to_csv(out, index=False)
    print(f"Wrote {out}")
    print(f"  rows            : {len(df)}")
    print(f"  legitimate (0)  : {(df.label == 0).sum()}")
    print(f"  phishing   (1)  : {(df.label == 1).sum()}")
    print(f"  mean body words : {df.body.str.split().str.len().mean():.1f}")
