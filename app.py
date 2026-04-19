import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover
    genai = None


APP_TITLE = "FeeKhoj Karnataka"
APP_SUBTITLE = "Prototype court-fee estimator for selected Karnataka civil suits"


@dataclass
class ParsedQuery:
    case_type: Optional[str]
    claim_amount: Optional[float] = None
    property_market_value: Optional[float] = None
    relief_value: Optional[float] = None
    title_denied: Optional[bool] = None
    explanation: str = ""
    confidence: str = "low"
    raw_mode: str = "heuristic"


@st.cache_data

def load_knowledge_base() -> Dict[str, Any]:
    base_dir = os.path.dirname(__file__)
    with open(os.path.join(base_dir, "knowledge_base.json"), "r", encoding="utf-8") as f:
        return json.load(f)


KB = load_knowledge_base()
SUPPORTED = {item["id"]: item for item in KB["supported_case_types"]}
SLABS = KB["ad_valorem_slabs"]


def format_inr(value: float) -> str:
    rounded = round(float(value), 2)
    negative = rounded < 0
    rounded = abs(rounded)
    integer_part = int(rounded)
    decimal_part = int(round((rounded - integer_part) * 100))
    if decimal_part == 100:
        integer_part += 1
        decimal_part = 0

    s = str(integer_part)
    if len(s) > 3:
        last3 = s[-3:]
        rest = s[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        s = ",".join(parts + [last3])

    sign = "-" if negative else ""
    return f"{sign}₹{s}.{decimal_part:02d}"


def clean_number(text: str) -> Optional[float]:
    if not text:
        return None
    raw = text.lower().strip()

    multiplier = 1
    if re.search(r"\b(crore|cr)\b", raw):
        multiplier = 10000000
    elif re.search(r"\b(lakh|lakhs|lac|lacs)\b", raw):
        multiplier = 100000
    elif re.search(r"\b(thousand|k)\b", raw):
        multiplier = 1000

    raw = raw.replace('₹', '').replace('rs.', '').replace('rs', '').strip()
    raw = re.sub(r"\b(crore|cr|lakh|lakhs|lac|lacs|thousand|k)\b", '', raw).strip()
    raw = raw.replace(',', '')
    raw = re.sub(r"[^0-9.]", '', raw)
    raw = raw.strip('.')
    if not raw:
        return None

    # Indian currency amounts are usually integers with commas; avoid treating multiple dots as decimals.
    if raw.count('.') > 1:
        raw = raw.replace('.', '')

    try:
        return float(raw) * multiplier
    except ValueError:
        return None


AMOUNT_REGEX = re.compile(
    r"(?:(?:₹|rs\.?)\s*)?([0-9]{1,3}(?:,[0-9]{2,3})+(?:\.[0-9]+)?|[0-9]+(?:\.[0-9]+)?)\s*(crore|cr|lakh|lakhs|lac|lacs|thousand|k)?",
    flags=re.IGNORECASE,
)


def extract_amounts(text: str) -> List[float]:
    values: List[float] = []
    for match in AMOUNT_REGEX.finditer(text):
        full = match.group(0)
        prefix_or_suffix = ('₹' in full) or ('rs' in full.lower()) or bool(match.group(2)) or (',' in match.group(1))
        if not prefix_or_suffix:
            continue
        val = clean_number(full)
        if val is not None and val > 0:
            values.append(val)
    seen: List[float] = []
    for v in values:
        if v not in seen:
            seen.append(v)
    return seen


CASE_HINTS = {
    "money_suit": ["recovery", "money", "damages", "compensation", "arrears", "refund", "loan", "dues"],
    "declaration_possession": ["declaration and possession", "declare title and possession", "declaration with possession", "possession"],
    "declaration_injunction_immovable": ["declaration and injunction", "declaration with injunction", "consequential injunction"],
    "other_declaration": ["declaration", "declaratory"],
    "injunction_title_denied": ["injunction", "title denied", "title dispute", "encroachment", "interference"],
    "other_injunction": ["injunction", "restrain", "stay"]
}


def heuristic_parse(user_text: str) -> ParsedQuery:
    text = user_text.lower()
    amounts = extract_amounts(text)
    title_denied = any(phrase in text for phrase in ["title denied", "title dispute", "denies my title", "ownership dispute", "encroachment"])

    if any(h in text for h in ["recovery", "money suit", "damages", "compensation", "arrears", "refund", "loan"]):
        return ParsedQuery(
            case_type="money_suit",
            claim_amount=amounts[0] if amounts else None,
            explanation="Heuristic parser identified a money / recovery style claim.",
            confidence="medium",
            raw_mode="heuristic"
        )

    if "declaration" in text and "possession" in text:
        return ParsedQuery(
            case_type="declaration_possession",
            property_market_value=amounts[0] if amounts else None,
            explanation="Heuristic parser identified a declaration + possession suit.",
            confidence="medium",
            raw_mode="heuristic"
        )

    if "declaration" in text and "injunction" in text and any(w in text for w in ["property", "site", "land", "flat", "house", "immovable"]):
        return ParsedQuery(
            case_type="declaration_injunction_immovable",
            property_market_value=amounts[0] if amounts else None,
            explanation="Heuristic parser identified declaration + consequential injunction concerning immovable property.",
            confidence="medium",
            raw_mode="heuristic"
        )

    if "injunction" in text and any(w in text for w in ["property", "site", "land", "flat", "house", "immovable"]):
        return ParsedQuery(
            case_type="injunction_title_denied" if title_denied else "other_injunction",
            property_market_value=amounts[0] if (title_denied and amounts) else None,
            relief_value=amounts[0] if (not title_denied and amounts) else None,
            title_denied=title_denied,
            explanation="Heuristic parser identified an injunction dispute. It used title-denial cues to choose between sections 26(a) and 26(c).",
            confidence="medium",
            raw_mode="heuristic"
        )

    if "declaration" in text:
        return ParsedQuery(
            case_type="other_declaration",
            relief_value=amounts[0] if amounts else None,
            explanation="Heuristic parser identified a declaratory suit but not one clearly tied to possession or consequential injunction.",
            confidence="low",
            raw_mode="heuristic"
        )

    return ParsedQuery(
        case_type=None,
        explanation="The parser could not confidently classify the query into one of the supported categories.",
        confidence="low",
        raw_mode="heuristic"
    )


LLM_PROMPT = """
You are helping a narrow legal-tech prototype that only supports selected Karnataka court-fee categories.
Read the user query and return STRICT JSON only with these keys:
- case_type: one of [money_suit, declaration_possession, declaration_injunction_immovable, other_declaration, injunction_title_denied, other_injunction, unsupported]
- claim_amount: number or null
- property_market_value: number or null
- relief_value: number or null
- title_denied: true, false, or null
- confidence: one of [high, medium, low]
- explanation: very short explanation

Rules:
- money_suit is for money recovery, damages, compensation, arrears, refund, loan recovery or similar money claims.
- declaration_possession is for declaration plus possession of immovable property.
- declaration_injunction_immovable is for declaration plus consequential injunction concerning immovable property.
- other_declaration is any other declaratory suit supported by section 24(d).
- injunction_title_denied is an injunction concerning immovable property where title is denied or ownership/title is in issue.
- other_injunction is an injunction suit not falling under the title-denied immovable property bucket.
- unsupported means the query is too ambiguous or outside these categories.
- Prefer null instead of guessing numbers.
- Convert rupee amounts into plain numeric values without commas.
"""


def llm_parse(user_text: str, api_key: str) -> Optional[ParsedQuery]:
    if not api_key or genai is None:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([LLM_PROMPT, user_text])
        raw = response.text.strip()
        raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        case_type = data.get("case_type")
        if case_type == "unsupported":
            case_type = None

        regex_amounts = extract_amounts(user_text)
        extracted_amount = regex_amounts[0] if regex_amounts else None

        claim_amount = data.get("claim_amount")
        property_market_value = data.get("property_market_value")
        relief_value = data.get("relief_value")

        def reconcile(model_value: Optional[float], fallback_value: Optional[float]) -> Optional[float]:
            # For rupee amounts, prefer the deterministic extractor whenever it found a value in the text.
            # The LLM is still useful for suit classification and field selection, but numeric parsing should be rule-based.
            if fallback_value is not None:
                return fallback_value
            if model_value is None:
                return None
            try:
                model_num = float(model_value)
            except Exception:
                return None
            return model_num if model_num > 0 else None

        if case_type == "money_suit":
            claim_amount = reconcile(claim_amount, extracted_amount)
        elif case_type in {"declaration_possession", "declaration_injunction_immovable", "injunction_title_denied"}:
            property_market_value = reconcile(property_market_value, extracted_amount)
        elif case_type in {"other_declaration", "other_injunction"}:
            relief_value = reconcile(relief_value, extracted_amount)

        return ParsedQuery(
            case_type=case_type,
            claim_amount=claim_amount,
            property_market_value=property_market_value,
            relief_value=relief_value,
            title_denied=data.get("title_denied"),
            explanation=data.get("explanation", "LLM parser used."),
            confidence=data.get("confidence", "low"),
            raw_mode="llm"
        )
    except Exception:
        return None


def compute_ad_valorem(amount: float) -> Tuple[float, Dict[str, Any]]:
    if amount <= 0:
        raise ValueError("Amount must be positive.")
    for slab in SLABS:
        upper = slab["max_inclusive"]
        if upper is None or amount <= upper:
            fee = slab["base_fee"] + slab["rate"] * (amount - slab["base_amount"])
            return fee, slab
    raise RuntimeError("No slab matched.")


def compute_fee(parsed: ParsedQuery) -> Dict[str, Any]:
    if not parsed.case_type or parsed.case_type not in SUPPORTED:
        raise ValueError("Unsupported or unclassified case type.")

    meta = SUPPORTED[parsed.case_type]
    rule = meta["charging_rule"]
    result: Dict[str, Any] = {
        "case_type": meta["name"],
        "section": meta["section"],
        "basis_amount": None,
        "fee": None,
        "breakdown": "",
    }

    if rule == "ad_valorem_on_claim_amount":
        amount = parsed.claim_amount
        if not amount:
            raise ValueError("This category needs a claim amount.")
        fee, slab = compute_ad_valorem(amount)
        result.update({
            "basis_amount": amount,
            "fee": fee,
            "breakdown": f"Section 21 sends the suit into the ad valorem table. Under Schedule I Article 1, the amount claimed ({format_inr(amount)}) falls in the slab {format_inr(slab['base_amount'])} to {format_inr(slab['max_inclusive']) if slab['max_inclusive'] else 'above ' + format_inr(slab['base_amount'])}. Fee = base {format_inr(slab['base_fee'])} + {slab['rate']*100:.1f}% of the excess over {format_inr(slab['base_amount'])}."
        })
        return result

    if rule == "ad_valorem_on_market_value_min_1000":
        amount = parsed.property_market_value
        if not amount:
            raise ValueError("This category needs the market value of the property.")
        basis = max(amount, 1000)
        fee, slab = compute_ad_valorem(basis)
        result.update({
            "basis_amount": basis,
            "fee": fee,
            "breakdown": f"The section uses the full market value of the property, subject to a minimum basis of {format_inr(1000)}. On the present inputs, the fee-bearing value is {format_inr(basis)} and the Article 1 slab calculation applies from there."
        })
        return result

    if rule == "ad_valorem_on_half_market_value_min_1000":
        amount = parsed.property_market_value
        if not amount:
            raise ValueError("This category needs the market value of the property.")
        half_value = amount / 2
        basis = max(half_value, 1000)
        fee, slab = compute_ad_valorem(basis)
        result.update({
            "basis_amount": basis,
            "fee": fee,
            "breakdown": f"The section directs computation on one-half of the property market value, subject to a minimum basis of {format_inr(1000)}. Half of the supplied market value is {format_inr(half_value)}; the fee-bearing value is therefore {format_inr(basis)}. The Article 1 ad valorem table is then applied to that fee-bearing value."
        })
        return result

    if rule == "ad_valorem_on_relief_value_min_1000":
        amount = parsed.relief_value
        if not amount:
            raise ValueError("This category needs the value at which the relief is stated in the plaint.")
        basis = max(amount, 1000)
        fee, slab = compute_ad_valorem(basis)
        result.update({
            "basis_amount": basis,
            "fee": fee,
            "breakdown": f"The section uses the plaintiff's stated valuation of the relief, subject to a floor of {format_inr(1000)}. On these inputs, the fee-bearing value is {format_inr(basis)} and Article 1 governs the ad valorem amount."
        })
        return result

    raise RuntimeError("Unknown charging rule.")



st.set_page_config(page_title=APP_TITLE, page_icon="⚖️", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .small-muted {color: #9aa0a6; font-size: 0.95rem;}
    .result-card {
        border: 1px solid rgba(250,250,250,0.12);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        background: rgba(255,255,255,0.02);
        margin-bottom: 1rem;
    }
    .step-label {
        font-size: 0.9rem;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: #9aa0a6;
        margin-bottom: 0.15rem;
    }
    .big-fee {
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.15;
        margin: 0.1rem 0 0.7rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title(APP_TITLE)
st.caption("Estimate court fees for selected Karnataka civil suits using plain-language input.")
st.markdown(
    "<div class='small-muted'>This is an educational prototype. It supports only a limited set of Karnataka suit categories and does not provide legal advice.</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Setup")
    env_api_key = os.getenv("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "Gemini API key (optional)",
        type="password",
        help="If added, the app uses Gemini for intake-side classification. Amount extraction and fee computation remain rule-based.",
    )
    api_key = api_key_input or env_api_key
    if env_api_key and not api_key_input:
        st.success("Using API key from environment / Streamlit secrets.")

    st.markdown("### Supported scope")
    supported_labels = [item["name"] for item in KB["supported_case_types"]]
    for label in supported_labels:
        st.markdown(f"- {label}")
    st.info("Best results come from concrete, single-relief queries. Mixed-relief disputes may need manual review.")

left, right = st.columns([1.45, 1], gap="large")

with left:
    st.markdown("<div class='step-label'>Step 1</div>", unsafe_allow_html=True)
    st.subheader("Describe your case")
    st.markdown(
        "<div class='small-muted'>Use ordinary language, but name the relief clearly where possible.</div>",
        unsafe_allow_html=True,
    )

    example_text = "I want to file a money recovery suit in Bengaluru for unpaid invoices of Rs. 5,50,000."
    user_query = st.text_area(
        "Describe the dispute or relief sought",
        value=example_text,
        height=170,
        placeholder="Example: I want to file a money recovery suit for unpaid invoices of Rs. 5,50,000.",
        label_visibility="collapsed",
        help="Examples work best when they clearly state the relief, such as money recovery, declaration, possession, or injunction.",
    )

    with st.expander("Examples of supported queries"):
        st.markdown("- Money recovery suit for unpaid invoices of Rs. 5,50,000")
        st.markdown("- Declaration and possession concerning immovable property worth Rs. 40 lakh")
        st.markdown("- Declaration and injunction concerning an immovable property dispute")
        st.markdown("- Injunction concerning immovable property where title is denied")

    st.markdown("<div class='step-label'>Step 2</div>", unsafe_allow_html=True)
    st.subheader("Review or edit inputs")
    manual_mode = st.toggle("Edit details manually instead of using the parser", value=False)

    manual_case_type = None
    manual_claim = None
    manual_property = None
    manual_relief = None

    if manual_mode:
        options = {item["name"]: item["id"] for item in KB["supported_case_types"]}
        selected_name = st.selectbox("Suit category", list(options.keys()))
        manual_case_type = options[selected_name]
        needs = SUPPORTED[manual_case_type]["requires"]

        manual_col1, manual_col2 = st.columns(2)
        with manual_col1:
            if "claim_amount" in needs:
                manual_claim = st.number_input("Claim amount (₹)", min_value=0.0, step=1000.0)
            if "property_market_value" in needs:
                manual_property = st.number_input("Property market value (₹)", min_value=0.0, step=1000.0)
        with manual_col2:
            if "relief_value" in needs:
                manual_relief = st.number_input("Value placed on relief in plaint (₹)", min_value=0.0, step=1000.0)

        st.caption("Manual mode is useful where the parser misclassifies the relief or misses a material valuation field.")

    run = st.button("Estimate court fee", type="primary")

with right:
    st.markdown("<div class='step-label'>How the estimate is generated</div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("**Legal basis used**")
        st.markdown(f"- **Jurisdiction encoded:** {KB['jurisdiction']}")
        st.markdown(f"- **Supported categories:** {len(KB['supported_case_types'])}")
        st.markdown("- **Classification:** plain-language parser / optional Gemini")
        st.markdown("- **Numeric extraction:** deterministic")
        st.markdown("- **Fee logic:** deterministic statutory computation")

    with st.expander("See encoded statutory categories"):
        st.json({
            "jurisdiction": KB["jurisdiction"],
            "supported_case_types": [
                {"id": x["id"], "section": x["section"], "requires": x["requires"]}
                for x in KB["supported_case_types"]
            ]
        })

    with st.expander("See ad valorem slab table (Schedule I, Article 1)"):
        st.dataframe(SLABS)

parsed = None
result = None
error_message = None

if run:
    if manual_mode:
        parsed = ParsedQuery(
            case_type=manual_case_type,
            claim_amount=manual_claim if manual_claim and manual_claim > 0 else None,
            property_market_value=manual_property if manual_property and manual_property > 0 else None,
            relief_value=manual_relief if manual_relief and manual_relief > 0 else None,
            explanation="Manual override used.",
            confidence="high",
            raw_mode="manual"
        )
    else:
        parsed = llm_parse(user_query, api_key) or heuristic_parse(user_query)

    try:
        result = compute_fee(parsed)
    except Exception as exc:
        error_message = str(exc)

if run:
    st.markdown("---")
    st.markdown("<div class='step-label'>Step 3</div>", unsafe_allow_html=True)
    st.subheader("Review parsed details")

    review_left, review_right = st.columns([1.05, 1], gap="large")
    with review_left:
        with st.container(border=True):
            st.markdown("**Parsed intake details**")
            if parsed:
                display_case = SUPPORTED.get(parsed.case_type, {}).get("name") if parsed.case_type in SUPPORTED else None
                st.write(f"**Detected suit category:** {display_case or 'Unsupported / unclassified'}")
                if parsed.claim_amount is not None:
                    st.write(f"**Claim amount detected:** {format_inr(parsed.claim_amount)}")
                if parsed.property_market_value is not None:
                    st.write(f"**Property value detected:** {format_inr(parsed.property_market_value)}")
                if parsed.relief_value is not None:
                    st.write(f"**Relief valuation detected:** {format_inr(parsed.relief_value)}")
                st.write(f"**Parser route used:** {parsed.raw_mode.upper()}")
                st.write(f"**Confidence:** {parsed.confidence.capitalize()}")
                st.caption(parsed.explanation)

        with st.expander("Show raw extracted fields"):
            st.json(parsed.__dict__)

    with review_right:
        st.markdown("<div class='step-label'>Step 4</div>", unsafe_allow_html=True)
        st.subheader("Estimated court fee")

        if result:
            st.markdown(
                f"""
                <div class='result-card'>
                    <div class='small-muted'>Approximate fee</div>
                    <div class='big-fee'>{format_inr(result["fee"])}</div>
                    <div><strong>Suit category:</strong> {result["case_type"]}</div>
                    <div><strong>Statutory basis:</strong> {result["section"]}</div>
                    <div><strong>Fee-bearing value:</strong> {format_inr(result["basis_amount"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.container(border=True):
                st.markdown("**Calculation path used**")
                st.write(result["breakdown"])
            st.warning(
                "This estimate is limited to selected Karnataka suit categories. It does not resolve jurisdiction, limitation, maintainability, valuation disputes, or mixed-relief complexity."
            )
        else:
            st.error(error_message or "Unsupported or unclassified case type.")
            st.info("Please restate the relief more clearly, or switch to manual input and confirm the category and valuation fields yourself.")

st.markdown("---")
with st.expander("Why this is not just a chatbot"):
    st.write(
        "The app uses a structured knowledge base for selected Karnataka suit categories, an optional LLM only for intake-side classification, deterministic amount extraction, and deterministic statutory logic for the actual fee calculation."
    )

with st.expander("Source notes"):
    for source in KB["sources"]:
        st.write(f"- **{source['label']}**: {source['summary']} ({source['citation']})")
