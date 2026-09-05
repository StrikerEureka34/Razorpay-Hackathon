from datetime import date
from decimal import Decimal

from src.controller.extract import extract_evidence, name_similarity, normalize_name
from src.controller.models import BankLine


def _line(desc: str) -> BankLine:
    return BankLine(
        txn_id="TXN-000001", value_date=date(2025, 1, 20), posted_date=date(2025, 1, 20),
        amount_signed=Decimal("100.00"), currency="INR", description=desc,
        running_balance=Decimal("0"),
    )


def test_extracts_full_hyphenated_reference():
    ev = extract_evidence(_line("NEFT/INV-000086/USD/KAPOOR AND SONS"))
    assert "INV-000086" in ev.full_refs
    assert ev.currency_hint == "USD"


def test_extracts_reference_without_hyphen():
    # template "NEFT-{ifsc}-{company}-INV{ref_short}" produces INV000123
    ev = extract_evidence(_line("NEFT-HDFC0001234-ACME-INV000123"))
    assert "INV-000123" in ev.full_refs


def test_bare_six_digit_number_is_an_invoice_reference():
    # template "NEFT CR {ref_short}" produces the bare invoice number
    ev = extract_evidence(_line("NEFT CR 000063"))
    assert "INV-000063" in ev.full_refs


def test_ifsc_digits_are_not_mistaken_for_a_reference():
    ev = extract_evidence(_line("NEFT-HDFC0001234-ACME CORP"))
    assert ev.full_refs == ()
    assert ev.prefix_refs == ()


def test_truncated_reference_becomes_a_prefix_constraint():
    ev = extract_evidence(_line("RTGS/CHOUDHURY, BAKSHI AN/INV-00"))
    assert ev.full_refs == ()
    assert "INV-00" in ev.prefix_refs


def test_detects_split_suffix():
    ev = extract_evidence(_line("UPI-ACME-INV-000134-A-PAYMENT"))
    assert "INV-000134" in ev.full_refs
    assert ev.split_suffix == "A"





def test_resolves_every_reference_variant_to_the_same_invoice():
    """build_narration picks one of six forms (generate_dataset.py:184):
    INV-000068, INV68, 000068, 68, REF0068, or nothing. int() normalization
    collapses all of them onto the same invoice id."""
    for desc in (
        "NEFT/INV-000068/ACME",
        "NEFT-HDFC0001234-ACME-INV68",
        "NEFT CR HDFC0001234 ACME 000068",
        "IMPS/REF0068/ACME/INR",
        "NEFT CR 0068",
        "NEFT-HDFC0001234-ACME-INV0068",
        "CR-INV000068#ACME",
    ):
        assert "INV-000068" in extract_evidence(_line(desc)).full_refs, desc


def test_a_deposit_amount_is_not_a_reference():
    """build_messy_narration emits 'DEPOSIT {int(amount)}' — that number is
    money, not an invoice number."""
    ev = extract_evidence(_line("DEPOSIT 41224"))
    assert ev.full_refs == (), f"read the amount as {ev.full_refs}"


def test_ifsc_and_utility_digits_are_still_not_references():
    assert extract_evidence(_line("NEFT-HDFC0001234-ACME CORP")).full_refs == ()
    assert extract_evidence(_line("RTGS-ICIC0002345-UPADHYAY GROUP")).full_refs == ()


def test_a_reference_to_an_invoice_that_was_never_issued_resolves_cleanly():
    """Orphan credits quote numbers in the 900s. They should resolve to a
    well-formed id that simply matches no invoice, not to nothing."""
    ev = extract_evidence(_line("UPI/INV-000900/SIDHU-WADHWA@okaxis"))
    assert ev.full_refs == ("INV-000900",)


def test_extracts_payout_reference():
    assert "000016" in extract_evidence(_line("RAZORPAY PAYOUT RPY-000016")).payout_refs
    assert "000026" in extract_evidence(_line("STRIPE PAYOUT po_000026")).payout_refs


def test_extracts_every_processor_template():
    """All six PROCESSOR_NARRATION_TEMPLATES (generate_dataset.py:125)."""
    for desc in (
        "RAZORPAY PAYOUT RPY-000016",
        "RAZORPAY*ACME RPY000016",
        "RAZORPAY SETTLEMENT-000016",
        "STRIPE TRANSFER ST-000016",
        "STRIPE PAYOUT po_000016",
        "CASHFREE PAY-000016",
    ):
        assert "000016" in extract_evidence(_line(desc)).payout_refs, desc


def test_a_payout_id_is_never_read_as_an_invoice_reference():
    """po_000026 uppercases to PO_000026; the underscore must not let the
    bare-digit rule invent INV-000026 and point at an unrelated invoice."""
    for desc in (
        "STRIPE PAYOUT po_000026",
        "RAZORPAY SETTLEMENT-000026",
        "RAZORPAY PAYOUT RPY-000026",
        "CASHFREE PAY-000026",
    ):
        ev = extract_evidence(_line(desc))
        assert ev.full_refs == (), f"{desc} invented {ev.full_refs}"
        assert ev.prefix_refs == (), f"{desc} invented {ev.prefix_refs}"


def test_identifies_rail():
    assert extract_evidence(_line("IMPS/INV-000012/ACME/INR")).rail == "IMPS"
    assert extract_evidence(_line("RAZORPAY PAYOUT RPY-000016")).rail == "PROCESSOR"


def test_counterparty_drops_a_truncated_reference_fragment():
    """Truncation can leave a bare "INV" or "INV-" tail glued to the company.
    Decoy pairs are separated by name alone, so this fragment must not survive."""
    from src.controller.extract import extract_evidence as ev
    assert ev(_line("NEFT CR HDFC0001234 KAUL PLC INV-")).counterparty == "KAUL"
    # Punctuation still delimits, so only the leading fragment survives here —
    # that is fine, it is a prefix of the real name and scores 0.97.
    cp = ev(_line("RTGS CR IDFB0008901 CHOUDHURY, BAKSHI AN INV")).counterparty
    assert "INV" not in cp.split(), f"reference fragment survived in {cp!r}"
    assert cp.startswith("CHOUDHURY")
    assert name_similarity(cp, "Choudhury, Bakshi and Maharaj") > 0.9


# ── name normalization: these are exactly the transforms name_variant() applies ──

def test_normalize_strips_corporate_suffixes():
    assert normalize_name("Acme Pvt Ltd") == "ACME"
    assert normalize_name("ACME PRIVATE LIMITED") == "ACME"
    assert normalize_name("M/S Acme") == "ACME"
    assert normalize_name("Acme India") == "ACME"


def test_name_similarity_survives_space_removal():
    # name_variant() emits name.replace(" ", "")
    assert name_similarity("KONDA GADE AND SHERE", "KONDAGADEANDSHERE") > 0.95


def test_name_similarity_survives_truncation():
    # narration truncates the company to 20 chars
    assert name_similarity("KONDA GADE AND SHERE", "KONDA GADE AND SHER") > 0.9


def test_name_similarity_survives_a_truncated_middle_fragment():
    """Narration truncates the company on BOTH ends: 'KONDA, GADE AND SHERE'
    can arrive as 'GADE AND SHER', an infix rather than a prefix. Decoys are
    separated by name alone, so an infix has to score as strong evidence."""
    assert name_similarity("GADE AND SHER", "Konda, Gade and Shere") > 0.9
    assert name_similarity("BAKSHI AN", "Choudhury, Bakshi and Maharaj") > 0.9


def test_name_similarity_rejects_different_companies():
    assert name_similarity("ACME TRADING", "ZENITH LOGISTICS") < 0.5
