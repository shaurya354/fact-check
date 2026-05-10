"""
TOP-TIER High-Precision Verification Engine (FINAL VERSION)

This module implements STRICT classification with:
- Mandatory FALSE for exaggerated claims
- Post-processing to enforce rules
- Confidence correction (never 0 if evidence exists)
- Strong source quality enforcement
- Decisive, non-cautious behavior
"""

import logging
import time
import re
from typing import List, Tuple, Dict
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.models import Claim, Evidence, VerificationResult
from src.nvidia_client import get_llm_client

logger = logging.getLogger("fact_checker.verifier")


class VerificationError(Exception):
    """Exception raised when verification fails."""
    pass


# API call tracking
_api_stats = {"llm_calls": 0, "search_calls": 0}


def get_api_stats() -> Dict[str, int]:
    """Get API usage statistics."""
    return _api_stats.copy()


def reset_api_stats():
    """Reset API usage statistics."""
    global _api_stats
    _api_stats = {"llm_calls": 0, "search_calls": 0}


# ELITE-LEVEL VERIFICATION PROMPT WITH REFINED SEVERITY CLASSIFICATION
VERIFICATION_PROMPT = """You are an ELITE fact-checking expert with advanced reasoning capabilities.
Apply REFINED SEVERITY CLASSIFICATION to distinguish between OUTDATED and FALSE claims.

🔴 CRITICAL DISTINCTION: OUTDATED vs FALSE

**OUTDATED → INACCURATE:**
- Previously true but no longer accurate
- Old statistics that have changed
- Data from past years
- Once valid, now incorrect
→ Classify as: INACCURATE

**UNREALISTIC/EXAGGERATED → FALSE:**
- Never true, contradicts reality
- Exaggerated claims
- Unrealistic projections
- Misleading information
→ Classify as: FALSE

🔴 MANDATORY RULE:
Do NOT classify outdated statistics as FALSE.
Outdated = INACCURATE | Exaggerated = FALSE

🔴 EXAMPLES (ENFORCE THESE):

**Example 1: OUTDATED → INACCURATE**
Claim: "Global internet users reached 3 billion in 2024"
Actual: 5.3 billion (2024)
Analysis: The claim is outdated; internet users have grown significantly
Difference: 43%
→ INACCURATE (outdated data, not exaggerated)
Confidence: 70%

**Example 2: OUTDATED → INACCURATE**
Claim: "India's population is 1.2 billion as of 2023"
Actual: 1.4 billion (2023)
Analysis: The claim uses old census data; population has grown
Difference: 14%
→ INACCURATE (outdated, was true in ~2015)
Confidence: 75%

**Example 3: EXAGGERATED → FALSE**
Claim: "The global AI market size is expected to reach $50 billion by 2025"
Actual: Already exceeded $100 billion in 2023
Analysis: The claim is unrealistic; market already far exceeds projection
Difference: 100%+
→ FALSE (unrealistic projection, not just outdated)
Confidence: 90%

**Example 4: EXAGGERATED → FALSE**
Claim: "Python is used by over 95% of developers worldwide"
Actual: ~30% according to Stack Overflow surveys
Analysis: The claim is clearly exaggerated; no evidence supports 95%
Difference: 217%
→ FALSE (exaggerated, never true)
Confidence: 90%

🔴 SEVERITY-AWARE CLASSIFICATION:

**1. SMALL DIFFERENCE (<10%):**
- Minor rounding differences
- Close approximation
→ Classify as: VERIFIED or INACCURATE

Examples:
✔ 4.5 vs 4.54 billion → VERIFIED (0.9% diff, rounding)
✔ 72 vs 70 years → INACCURATE (2.8% diff, minor error)

**2. MODERATE DIFFERENCE (10-30%):**
- Outdated data (was true, now incorrect)
- Noticeable but explainable difference
→ Classify as: INACCURATE

Examples:
✔ 3 billion vs 5.3 billion users → INACCURATE (43% diff, outdated growth)
✔ India 1.2B vs 1.4B → INACCURATE (14% diff, old census)

**3. LARGE DIFFERENCE (>30%) + EXAGGERATED:**
- Unrealistic claims
- Never true
- Contradicts reality
→ Classify as: FALSE

Examples:
❌ Python 95% vs ~30% → FALSE (217% diff, exaggerated)
❌ AI market $50B vs $100B+ → FALSE (100%+ diff, unrealistic)

**IMPORTANT:** Large differences can be INACCURATE if outdated, or FALSE if exaggerated.
Check context to determine which.

🔴 CLASSIFICATION GUIDELINES:

**VERIFIED:**
- Exact match OR minor difference (<5%)
- Acceptable rounding (3B vs 3.2B)
- 2+ reliable sources agree
- Recent data (<3 years)

**INACCURATE:**
- Outdated but once valid
- Moderate difference (10-30%)
- Old statistics that have changed
- Previously true, now incorrect
- Minor factual errors

**FALSE:**
- Exaggerated claims (never true)
- Unrealistic projections
- Contradicts reality
- Misleading information
- No credible evidence
- Extreme claims unsupported by data

🔴 CONFIDENCE CALIBRATION (ALIGNED WITH SEVERITY):

**VERIFIED:**
- Strong evidence (exact match): 85-95
- Approximate match (<5% diff): 70-85

**INACCURATE:**
- Outdated data: 60-75
- Moderate error: 50-70
- Maximum 75%

**FALSE:**
- Exaggerated/unrealistic: 80-95
- Clear contradiction: 85-95

**NEVER return 0 confidence if evidence exists**
**Minimum 40 if any sources found**

🔴 SOURCE PRIORITIZATION (RANKED):

**TIER 1 (Highest Authority):**
- WHO, UN, World Bank, IMF, OECD
- Government (.gov, .edu)
- Official statistics agencies

**TIER 2 (Reliable):**
- Statista, research journals
- Major news (BBC, Reuters, AP, NYT)
- Academic institutions

**TIER 3 (Use with caution):**
- General news sites
- Industry reports
- Wikipedia (for dates/basic facts)

**REJECT:**
- Blogs, forums, social media
- Calculators, converters
- E-commerce sites

🔴 EXPLANATION QUALITY (MUST DISTINGUISH OUTDATED vs FALSE):

Structure your explanation:
1. **Compare:** "Claim states X, but sources indicate Y"
2. **Quantify:** "This is a Z% difference"
3. **Assess Nature:** "This is [outdated/exaggerated/unrealistic]"
4. **Classify:** "This qualifies as [status] because [reasoning]"
5. **Justify:** "Confidence is [score] due to [reasons]"

**GOOD EXAMPLE (OUTDATED → INACCURATE):**
"The claim states 3 billion internet users in 2024, but authoritative sources (ITU, Statista) report 5.3 billion in 2024. This is a 43% underestimate. The claim appears to be based on outdated data from several years ago when internet adoption was lower. This qualifies as INACCURATE rather than FALSE because the claim was likely true in the past but is now outdated due to rapid internet growth. Confidence is 75% based on strong source agreement."

**GOOD EXAMPLE (EXAGGERATED → FALSE):**
"The claim states Python is used by 95% of developers worldwide, but authoritative sources (Stack Overflow Developer Survey, JetBrains) report approximately 30% usage. This is a 217% overestimate. The claim is clearly exaggerated and has no basis in reality; no credible survey has ever shown 95% Python usage. This qualifies as FALSE because it is unrealistic and misleading, not merely outdated. Confidence is 90% based on consistent data from multiple developer surveys."

🔴 REASONING STRUCTURE:

Provide step-by-step reasoning with NATURE ASSESSMENT:
1. **Sources:** "WHO reports X, UN reports Y, Statista reports Z"
2. **Analysis:** "The claim differs by N% from the consensus"
3. **Nature:** "This appears to be [outdated/exaggerated/unrealistic] because..."
4. **Classification:** "This is [status] because [outdated vs false logic]"
5. **Confidence:** "Score is [X] due to [source quality + agreement + recency]"

🔴 OUTPUT FORMAT (STRICT JSON):

Return ONLY a JSON array (NO extra text):
[
  {
    "claim_index": 0,
    "status": "Verified|Inaccurate|False",
    "confidence": 40-95,
    "explanation": "Clear explanation distinguishing outdated from false",
    "correct_fact": "Precise correction with source attribution",
    "source_agreement": 0.0-1.0,
    "reasoning": "1) Sources report X. 2) Claim differs by Y%. 3) This is [outdated/exaggerated]. 4) Status is [X] because [logic]. 5) Confidence is Z.",
    "has_conflicts": true|false,
    "is_exaggerated": true|false,
    "source_quality": "high|medium|low"
  }
]

🔴 FINAL CHECKLIST:

Before returning, verify:
✅ Distinguished between OUTDATED (→ INACCURATE) and EXAGGERATED (→ FALSE)
✅ Outdated statistics classified as INACCURATE, not FALSE
✅ Exaggerated/unrealistic claims classified as FALSE
✅ Explanation clearly states whether claim is outdated or exaggerated
✅ Confidence aligned: INACCURATE (50-75), FALSE (80-95)
✅ Reasoning includes nature assessment (outdated vs exaggerated)

REMEMBER: Outdated ≠ False. Distinguish between old data (INACCURATE) and exaggerated claims (FALSE)."""


def _detect_severity_and_override(
    result: dict,
    claim: Claim,
    evidence: List[Evidence]
) -> dict:
    """
    Detect severity of error and override classification if needed.
    
    REFINED SEVERITY CLASSIFICATION:
    1. OUTDATED: Previously true but no longer accurate → INACCURATE
    2. UNREALISTIC/EXAGGERATED: Never true, contradicts reality → FALSE
    
    CRITICAL DISTINCTION:
    - Outdated statistics (e.g., "3 billion users" when it's now 5B) → INACCURATE
    - Exaggerated claims (e.g., "95% of developers") → FALSE
    
    This prevents outdated data from being incorrectly labeled as FALSE.
    
    Args:
        result: LLM result
        claim: Original claim
        evidence: Evidence list
        
    Returns:
        Updated result with refined severity-aware classification
    """
    status = result.get("status", "Inaccurate")
    explanation = result.get("explanation", "")
    reasoning = result.get("reasoning", "")
    
    # Normalize status
    status_map = {
        "verified": "Verified",
        "inaccurate": "Inaccurate",
        "false": "False",
        "VERIFIED": "Verified",
        "INACCURATE": "Inaccurate",
        "FALSE": "False"
    }
    status = status_map.get(status, status)
    
    claim_text = claim.text.lower()
    combined_text = f"{explanation} {reasoning}".lower()
    
    # Extract percentage differences mentioned in reasoning
    pct_diff_patterns = [
        r'(\d+(?:\.\d+)?)\s*%\s*(?:difference|diff|error|deviation|higher|lower|more|less)',
        r'differs?\s+by\s+(\d+(?:\.\d+)?)\s*%',
        r'(\d+(?:\.\d+)?)\s*%\s+(?:under|over)estimate',
        r'(\d+(?:\.\d+)?)\s*%\s+(?:increase|decrease)'
    ]
    
    detected_percentage = None
    for pattern in pct_diff_patterns:
        match = re.search(pattern, combined_text)
        if match:
            detected_percentage = float(match.group(1))
            logger.info(f"SEVERITY CHECK: Detected {detected_percentage}% difference in reasoning")
            break
    
    # Check for OUTDATED indicators (should be INACCURATE, not FALSE)
    outdated_indicators = [
        'outdated', 'old data', 'previous', 'was true', 'once valid', 'formerly',
        'no longer accurate', 'has since', 'now shows', 'updated to', 'currently',
        'as of 20', 'in 20', 'recent data', 'latest data', 'newer data'
    ]
    
    # Check for FALSE indicators (unrealistic/exaggerated)
    false_indicators = [
        'exaggerated', 'unrealistic', 'impossible', 'clearly wrong', 'never true',
        'contradicts reality', 'not even close', 'wildly inaccurate', 'absurd',
        'misleading', 'false claim', 'fabricated', 'no evidence', 'unsupported'
    ]
    
    has_outdated_indicator = any(ind in combined_text for ind in outdated_indicators)
    has_false_indicator = any(ind in combined_text for ind in false_indicators)
    
    # Check for extreme claims in the original claim (likely FALSE, not outdated)
    extreme_claim_patterns = [
        r'9[0-9]%',  # 90-99%
        r'100%',
        r'all\s+(?:developers|people|users)',
        r'every\s+(?:developer|person|user)',
        r'\d+\s*trillion'
    ]
    
    has_extreme_claim = any(re.search(pattern, claim_text) for pattern in extreme_claim_patterns)
    
    # REFINED SEVERITY DECISION LOGIC
    should_be_false = False
    should_be_inaccurate = False
    severity_reason = ""
    
    # Rule 1: If claim has OUTDATED indicators → should be INACCURATE (not FALSE)
    if has_outdated_indicator and not has_false_indicator:
        if status == "False":
            should_be_inaccurate = True
            severity_reason = "outdated data (previously true, now incorrect)"
            logger.info(f"SEVERITY REFINEMENT: Outdated data detected → should be INACCURATE, not FALSE")
    
    # Rule 2: If claim has extreme values + FALSE indicators → should be FALSE
    if has_extreme_claim and (has_false_indicator or detected_percentage and detected_percentage > 50):
        if status == "Inaccurate":
            should_be_false = True
            severity_reason = "extreme/exaggerated claim contradicting reality"
            logger.warning(f"SEVERITY OVERRIDE: Extreme claim detected → forcing FALSE")
    
    # Rule 3: Very large percentage differences (>50%) without outdated context → FALSE
    if detected_percentage is not None and detected_percentage > 50:
        # Check if it's just outdated or actually exaggerated
        if has_outdated_indicator and not has_extreme_claim:
            # Large difference but outdated → keep as INACCURATE
            if status == "False":
                should_be_inaccurate = True
                severity_reason = f"large difference ({detected_percentage}%) but outdated, not exaggerated"
                logger.info(f"SEVERITY REFINEMENT: {detected_percentage}% diff but outdated → INACCURATE")
        elif not has_outdated_indicator or has_extreme_claim:
            # Large difference and exaggerated → FALSE
            if status == "Inaccurate":
                should_be_false = True
                severity_reason = f"large difference ({detected_percentage}%) indicating exaggeration"
                logger.warning(f"SEVERITY OVERRIDE: {detected_percentage}% diff + exaggeration → FALSE")
    
    # Rule 4: Moderate differences (20-50%) with outdated context → INACCURATE
    if detected_percentage is not None and 20 < detected_percentage <= 50:
        if has_outdated_indicator:
            if status == "False":
                should_be_inaccurate = True
                severity_reason = f"moderate difference ({detected_percentage}%) with outdated context"
                logger.info(f"SEVERITY REFINEMENT: {detected_percentage}% diff + outdated → INACCURATE")
    
    # Rule 5: Check for specific patterns in claim text
    # "AI market $50B" type claims (unrealistic projections) → FALSE
    if 'market' in claim_text or 'revenue' in claim_text or 'worth' in claim_text:
        if detected_percentage and detected_percentage > 50 and not has_outdated_indicator:
            if status == "Inaccurate":
                should_be_false = True
                severity_reason = "unrealistic market/financial claim"
                logger.warning(f"SEVERITY OVERRIDE: Unrealistic financial claim → FALSE")
    
    # Apply overrides
    if should_be_false and status != "False":
        logger.warning(f"SEVERITY OVERRIDE: Changing '{status}' to 'False' due to {severity_reason}")
        result["status"] = "False"
        result["confidence"] = max(result.get("confidence", 70), 80)
        
        if "This is clearly exaggerated or unrealistic" not in explanation:
            result["explanation"] = f"This is clearly exaggerated or unrealistic. {explanation}"
    
    elif should_be_inaccurate and status == "False":
        logger.info(f"SEVERITY REFINEMENT: Changing 'False' to 'Inaccurate' due to {severity_reason}")
        result["status"] = "Inaccurate"
        result["confidence"] = min(result.get("confidence", 70), 75)
        
        if "This claim is outdated" not in explanation:
            result["explanation"] = f"This claim is outdated but was previously valid. {explanation}"
    
    return result


def _detect_exaggeration(claim_text: str) -> bool:
    """
    Detect if claim contains obvious exaggeration.
    
    Patterns:
    - Extreme percentages (>90%)
    - Trillion-scale numbers (unrealistic)
    - Superlatives with extreme claims
    
    Args:
        claim_text: Claim text to check
        
    Returns:
        True if exaggerated, False otherwise
    """
    text = claim_text.lower()
    
    # Check for extreme percentages (>90%)
    pct_match = re.search(r'(\d+)%', text)
    if pct_match:
        pct = int(pct_match.group(1))
        if pct >= 90:
            # Check if it's about something unlikely to be 90%+
            unlikely_contexts = ['developers', 'people', 'users', 'population', 'companies']
            if any(ctx in text for ctx in unlikely_contexts):
                logger.info(f"Detected exaggeration: {pct}% in context")
                return True
    
    # Check for trillion-scale numbers (usually unrealistic)
    if 'trillion' in text:
        match = re.search(r'(\d+)\s*trillion', text)
        if match:
            num = int(match.group(1))
            # Most trillion claims are exaggerated (except GDP, debt, etc.)
            if num >= 5 and 'gdp' not in text and 'debt' not in text:
                logger.info(f"Detected exaggeration: {num} trillion")
                return True
    
    # Check for extreme superlatives
    extreme_patterns = [
        r'all\s+(?:developers|people|users)',
        r'every\s+(?:developer|person|user)',
        r'100%\s+of',
        r'universally',
        r'always\s+(?:use|uses)'
    ]
    
    for pattern in extreme_patterns:
        if re.search(pattern, text):
            logger.info(f"Detected exaggeration: extreme superlative")
            return True
    
    return False


def _enforce_minimum_confidence(
    confidence: int,
    source_count: int,
    status: str
) -> int:
    """
    Enforce minimum and maximum confidence rules aligned with SEVERITY.
    
    Rules (SEVERITY-ALIGNED):
    - If sources exist, minimum 40
    - VERIFIED: minimum 70, maximum 95
    - FALSE with sources: minimum 80, maximum 95 (increased for major errors)
    - INACCURATE: minimum 50, maximum 75 (reduced to reflect uncertainty)
    
    Args:
        confidence: Original confidence score
        source_count: Number of sources
        status: Verification status
        
    Returns:
        Adjusted confidence score
    """
    # If no sources, confidence can be 0
    if source_count == 0:
        return 0
    
    # Enforce minimums and maximums based on status (SEVERITY-ALIGNED)
    if status == "Verified":
        confidence = max(confidence, 70)
        confidence = min(confidence, 95)
    elif status == "False":
        # FALSE should have HIGH confidence (80-95) for major errors
        confidence = max(confidence, 80)  # Increased from 70 to 80
        confidence = min(confidence, 95)
    elif status == "Inaccurate":
        # INACCURATE should have lower confidence to reflect uncertainty
        confidence = max(confidence, 50)
        confidence = min(confidence, 75)  # Cap at 75% for inaccurate claims
    
    # Global minimum if sources exist
    confidence = max(confidence, 40)
    
    return confidence


def _post_process_result(
    result: dict,
    claim: Claim,
    evidence: List[Evidence]
) -> dict:
    """
    Post-process LLM result to enforce strict rules with SEVERITY-AWARE classification.
    
    SAFETY POST-PROCESSING:
    1. Detect severity and override if major error labeled as "Inaccurate"
    2. Detect exaggeration → override to FALSE
    3. Enforce minimum confidence
    4. Validate source quality
    5. Ensure decisive classification
    
    Args:
        result: Raw LLM result
        claim: Original claim
        evidence: Evidence list
        
    Returns:
        Post-processed result
    """
    # STEP 1: Severity-aware classification (NEW - CRITICAL)
    result = _detect_severity_and_override(result, claim, evidence)
    
    status = result.get("status", "Inaccurate")
    
    # Normalize status to proper case
    status_map = {
        "verified": "Verified",
        "inaccurate": "Inaccurate", 
        "false": "False",
        "VERIFIED": "Verified",
        "INACCURATE": "Inaccurate",
        "FALSE": "False"
    }
    status = status_map.get(status, status)  # Normalize if needed
    
    confidence = result.get("confidence", 50)
    explanation = result.get("explanation", "")
    
    # STEP 2: Detect exaggeration and override to FALSE
    is_exaggerated = _detect_exaggeration(claim.text)
    
    if is_exaggerated and status != "False":
        logger.warning(f"POST-PROCESSING: Overriding '{status}' to 'False' for exaggerated claim")
        status = "False"
        confidence = max(confidence, 80)  # High confidence for exaggerations
        explanation = f"This claim contains clear exaggeration. {explanation}"
        result["status"] = status
        result["explanation"] = explanation
    
    # STEP 3: Enforce minimum confidence
    source_count = len(evidence)
    original_confidence = confidence
    confidence = _enforce_minimum_confidence(confidence, source_count, status)
    
    if confidence != original_confidence:
        logger.info(f"POST-PROCESSING: Adjusted confidence from {original_confidence} to {confidence}")
        result["confidence"] = confidence
    
    # STEP 4: Validate source quality
    source_quality = result.get("source_quality", "medium")
    
    if status == "Verified" and source_quality == "low":
        logger.warning("POST-PROCESSING: Cannot verify with low-quality sources, changing to Inaccurate")
        result["status"] = "Inaccurate"
        result["confidence"] = min(confidence, 60)
        result["explanation"] = f"Sources are not authoritative enough for verification. {explanation}"
    
    # STEP 5: Ensure FALSE has high confidence if strongly contradicted
    if status == "False" and source_count >= 2:
        source_agreement = float(result.get("source_agreement", 0.5))
        if source_agreement >= 0.7:
            result["confidence"] = max(confidence, 80)  # Increased from 75 to 80
            logger.info("POST-PROCESSING: Increased FALSE confidence due to strong contradiction")
    
    return result


def calculate_confidence_final(
    source_count: int,
    source_agreement: float,
    recency_score: float,
    source_quality_score: float,
    has_conflicts: bool,
    status: str
) -> int:
    """
    Calculate confidence score with ELITE-LEVEL calibration.
    
    CONFIDENCE RANGES:
    - VERIFIED (strong): 85-95
    - VERIFIED (approx): 70-85
    - INACCURATE: 55-75
    - FALSE: 70-95
    
    Formula:
    Start at 50
    +25 if multiple sources strongly agree (>0.8)
    +15 if authoritative sources (>0.8 quality)
    +10 if recent data (>0.8 recency)
    -15 if conflicts
    -10 if outdated
    
    Then enforce minimums based on status.
    
    Args:
        source_count: Number of sources
        source_agreement: Agreement level (0.0-1.0)
        recency_score: Recency score (0.0-1.0)
        source_quality_score: Quality score (0.0-1.0)
        has_conflicts: Whether sources conflict
        status: Verification status
        
    Returns:
        Confidence score (40-95)
    """
    # Start at 50
    confidence = 50
    
    # +25 if multiple sources strongly agree (increased from +20)
    if source_count >= 2 and source_agreement >= 0.8:
        confidence += 25
    elif source_count >= 2 and source_agreement >= 0.6:
        confidence += 12
    
    # +15 if authoritative sources
    if source_quality_score >= 0.8:
        confidence += 15
    elif source_quality_score >= 0.6:
        confidence += 8
    
    # +10 if recent data
    if recency_score >= 0.8:
        confidence += 10
    elif recency_score >= 0.5:
        confidence += 5
    
    # -15 if conflicts
    if has_conflicts:
        confidence -= 15
    
    # -10 if outdated
    if recency_score < 0.3:
        confidence -= 10
    
    # Enforce minimums based on status (ELITE CALIBRATION)
    if status == "Verified":
        # VERIFIED with strong evidence: 85-95
        if source_agreement >= 0.8 and source_quality_score >= 0.8:
            confidence = max(confidence, 85)
        # VERIFIED with approximate match: 70-85
        else:
            confidence = max(confidence, 70)
    elif status == "False":
        # FALSE with clear contradiction: 70-95
        confidence = max(confidence, 70)
    elif status == "Inaccurate":
        # INACCURATE: 55-75
        confidence = max(confidence, 55)
    
    # Global minimum if sources exist
    if source_count > 0:
        confidence = max(confidence, 40)
    
    # Clamp to [40, 95]
    return max(40, min(95, confidence))


def calculate_recency_score(evidence: List[Evidence]) -> float:
    """Calculate recency score based on publication dates."""
    if not evidence:
        return 0.5
    
    current_year = datetime.now().year
    recent_count = 0
    total_with_dates = 0
    
    for ev in evidence:
        if ev.published_date:
            try:
                pub_date = datetime.fromisoformat(ev.published_date.replace('Z', '+00:00'))
                total_with_dates += 1
                
                age = current_year - pub_date.year
                if age <= 3:
                    recent_count += 1
            except (ValueError, AttributeError):
                pass
    
    if total_with_dates == 0:
        return 0.5
    
    return recent_count / total_with_dates


def calculate_source_quality_score(evidence: List[Evidence]) -> float:
    """Calculate average source quality score with STRICT requirements."""
    if not evidence:
        return 0.0
    
    # High-quality domains (STRICT)
    high_quality = [
        '.gov', '.edu',
        'who.int', 'un.org', 'worldbank.org', 'imf.org', 'oecd.org',
        'nih.gov', 'cdc.gov', 'fda.gov',
        'statista.com', 'nature.com', 'science.org',
        'reuters.com', 'apnews.com', 'bbc.com', 'bbc.co.uk'
    ]
    
    # Low-quality domains (REJECT)
    low_quality = [
        'blog', 'forum', 'wiki', 'quora', 'reddit',
        'amazon', 'ebay', 'calculator', 'converter',
        'pinterest', 'instagram', 'facebook'
    ]
    
    scores = []
    for ev in evidence:
        domain = ev.domain.lower()
        
        if any(hq in domain for hq in high_quality):
            scores.append(1.0)
        elif any(lq in domain for lq in low_quality):
            scores.append(0.1)  # Very low for rejected sources
        else:
            scores.append(0.5)
    
    return sum(scores) / len(scores)


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def _call_llm_batch_verification(
    claims_with_evidence: List[Tuple[Claim, List[Evidence]]]
) -> List[dict]:
    """
    Verify multiple claims with ULTRA-STRICT prompt.
    
    Args:
        claims_with_evidence: List of (Claim, Evidence list) tuples
        
    Returns:
        List of verification results
        
    Raises:
        VerificationError: If API call fails
    """
    global _api_stats
    _api_stats["llm_calls"] += 1
    
    logger.info(f"API CALL: Batch verification ({len(claims_with_evidence)} claims)")
    
    # Build batch prompt with quality indicators
    batch_input = []
    for i, (claim, evidence) in enumerate(claims_with_evidence):
        evidence_text = []
        for j, ev in enumerate(evidence[:3]):
            # Calculate quality
            domain = ev.domain.lower()
            if any(hq in domain for hq in ['.gov', '.edu', 'who.int', 'un.org', 'worldbank.org']):
                quality = "HIGH"
            elif any(lq in domain for lq in ['blog', 'forum', 'wiki', 'calculator']):
                quality = "LOW"
            else:
                quality = "MEDIUM"
            
            evidence_text.append(
                f"  Source {j+1} [{quality} QUALITY] ({ev.domain}):\n"
                f"  {ev.snippet[:200]}..."
            )
        
        batch_input.append(f"""
Claim {i}:
Text: {claim.text}
Context: {claim.context}
Evidence ({len(evidence)} sources):
{chr(10).join(evidence_text) if evidence_text else "  No evidence found"}
""")
    
    full_prompt = f"{VERIFICATION_PROMPT}\n\n{''.join(batch_input)}\n\nVerify each claim. Return ONLY JSON array."
    
    try:
        time.sleep(1)  # Rate limiting
        
        client = get_llm_client()
        
        result = client.generate_json(
            prompt=full_prompt,
            temperature=0.15,  # Very low for maximum consistency
            max_tokens=4096
        )
        
        if not isinstance(result, list):
            logger.warning("Expected list, got dict. Wrapping.")
            result = [result]
        
        logger.info(f"Batch verification completed for {len(result)} claims")
        
        return result
        
    except TimeoutError as e:
        logger.warning("LLM timeout, will retry...")
        raise
    except ConnectionError as e:
        logger.warning(f"Connection error: {str(e)}, will retry...")
        raise
    except Exception as e:
        logger.error(f"LLM verification failed: {str(e)}", exc_info=True)
        raise VerificationError(f"Verification failed: {str(e)}")


def verify_claims_batch(
    claims: List[Claim],
    evidence_list: List[List[Evidence]],
    batch_size: int = 5
) -> List[VerificationResult]:
    """
    Verify claims with TOP-TIER precision and post-processing.
    
    FEATURES:
    - Ultra-strict classification
    - Exaggeration detection
    - Confidence correction
    - Source quality enforcement
    - Post-processing safety checks
    
    Args:
        claims: List of Claim objects
        evidence_list: List of Evidence lists
        batch_size: Batch size (default: 5)
        
    Returns:
        List of VerificationResult objects
    """
    if len(claims) != len(evidence_list):
        raise ValueError("Claims and evidence lists must have same length")
    
    results = []
    
    for batch_start in range(0, len(claims), batch_size):
        batch_end = min(batch_start + batch_size, len(claims))
        batch_claims = claims[batch_start:batch_end]
        batch_evidence = evidence_list[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}: claims {batch_start}-{batch_end-1}")
        
        claims_with_evidence = list(zip(batch_claims, batch_evidence))
        
        try:
            # Call LLM
            api_results = _call_llm_batch_verification(claims_with_evidence)
            
            # Process each result with POST-PROCESSING
            for i, api_result in enumerate(api_results):
                if i >= len(batch_claims):
                    break
                
                claim = batch_claims[i]
                evidence = batch_evidence[i]
                
                # POST-PROCESS to enforce rules
                api_result = _post_process_result(api_result, claim, evidence)
                
                # Extract fields
                status = api_result.get("status", "Inaccurate")
                explanation = api_result.get("explanation", "Verification completed")
                correct_fact = api_result.get("correct_fact", "")
                source_agreement = float(api_result.get("source_agreement", 0.5))
                reasoning = api_result.get("reasoning", "No reasoning provided")
                has_conflicts = api_result.get("has_conflicts", False)
                
                # Clamp source_agreement
                source_agreement = max(0.0, min(1.0, source_agreement))
                
                # Calculate scores
                recency_score = calculate_recency_score(evidence)
                quality_score = calculate_source_quality_score(evidence)
                source_count = len(evidence)
                
                # Calculate confidence with strict formula
                confidence_score = calculate_confidence_final(
                    source_count=source_count,
                    source_agreement=source_agreement,
                    recency_score=recency_score,
                    source_quality_score=quality_score,
                    has_conflicts=has_conflicts,
                    status=status
                )
                
                # Use post-processed confidence if higher (but never 0 if we have sources)
                if "confidence" in api_result:
                    llm_confidence = api_result["confidence"]
                    # Only use LLM confidence if it's reasonable (not 0 when we have sources)
                    if llm_confidence > 0 or source_count == 0:
                        confidence_score = max(confidence_score, llm_confidence)
                    else:
                        logger.warning(f"LLM returned 0 confidence with {source_count} sources, using calculated: {confidence_score}")
                
                # Final safety check: never 0 if we have sources
                if source_count > 0 and confidence_score == 0:
                    confidence_score = 40  # Minimum confidence with sources
                    logger.warning(f"Forced confidence to 40 (had {source_count} sources but confidence was 0)")
                
                # Log for debugging
                logger.info(
                    f"Claim {batch_start + i}: {status} (confidence={confidence_score}%, "
                    f"sources={source_count}, quality={quality_score:.2f})"
                )
                
                # Create result
                result = VerificationResult(
                    claim=claim.text,
                    status=status,
                    confidence_score=confidence_score,
                    explanation=explanation,
                    correct_fact=correct_fact if correct_fact else None,
                    sources=[ev.url for ev in evidence],
                    reasoning=reasoning
                )
                
                results.append(result)
                
        except VerificationError as e:
            logger.error(f"Batch verification failed: {str(e)}")
            
            # Add error results with minimum confidence
            for claim, evidence in claims_with_evidence:
                source_count = len(evidence)
                results.append(VerificationResult(
                    claim=claim.text,
                    status="Inaccurate",
                    confidence_score=40 if source_count > 0 else 0,
                    explanation=f"Verification failed: {str(e)}",
                    correct_fact=None,
                    sources=[ev.url for ev in evidence] if evidence else [],
                    reasoning="Verification process error"
                ))
    
    return results


def detect_conflicts(
    claims: List[Claim],
    results: List[VerificationResult]
) -> List[Tuple[int, int]]:
    """Detect conflicting claims within the same document."""
    conflicts = []
    
    for i in range(len(claims)):
        for j in range(i + 1, len(claims)):
            claim1 = claims[i].text.lower()
            claim2 = claims[j].text.lower()
            
            words1 = set(claim1.split())
            words2 = set(claim2.split())
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0.5:
                if results[i].status != results[j].status:
                    conflicts.append((i, j))
                    logger.warning(f"Conflict detected between claims {i} and {j}")
    
    return conflicts
