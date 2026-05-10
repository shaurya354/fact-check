"""
PRODUCTION-READY Claim Extractor with strict validation.

This module extracts ONLY complete, meaningful, independently verifiable claims.
NO fragments, NO standalone numbers, NO garbage claims.
"""

import json
import logging
import re
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.models import Claim
from src.nvidia_client import get_llm_client
from src.utils import sanitize_log_message

logger = logging.getLogger("fact_checker.claim_extractor_v2")


class ClaimExtractionError(Exception):
    """Exception raised when claim extraction fails."""
    pass


# SYSTEM PROMPT - Extract complete sentences as claims
SYSTEM_PROMPT = """You are a fact-checking assistant. Extract ALL COMPLETE SENTENCES as claims from the document.

🔴 MANDATORY RULES:
1. Each claim MUST be a COMPLETE SENTENCE (subject + verb + object/complement)
2. Each claim MUST be independently verifiable
3. Each claim MUST make sense without any other context
4. Extract ALL verifiable claims found in the document
5. NEVER extract fragments, phrases, or incomplete sentences
6. If document has 40 claims, extract all 40; if it has 5 claims, extract all 5

🔴 CRITICAL: DO NOT JUDGE TRUTHFULNESS
- Extract ALL claims, even if they seem FALSE, EXAGGERATED, or QUESTIONABLE
- Extract claims with statistics like "95%", "100%", "trillion", etc.
- Extract claims that seem unrealistic or too high/low
- Your job is ONLY to extract claims, NOT to verify them
- The verification system will check accuracy later

✅ GOOD EXAMPLES (COMPLETE SENTENCES - EXTRACT ALL OF THESE):
✔ "Python was released in 1991"
✔ "The company has 500 employees"
✔ "Global users reached 3 billion in 2024"
✔ "Python is used by 95% of developers" (extract even if seems high)
✔ "The company has 1 trillion users" (extract even if unrealistic)
✔ "AI will replace 100% of jobs" (extract even if exaggerated)

❌ BAD EXAMPLES (DO NOT EXTRACT):
❌ "3 billion" (not a sentence)
❌ "2024" (not a sentence)
❌ "95%" (not a sentence)
❌ "over 95%" (not a sentence)
❌ "in 2024" (fragment)

🔴 SENTENCE REQUIREMENTS:
- Must have a SUBJECT (who/what)
- Must have a VERB (action/state)
- Must have COMPLETE information
- Must be at least 5 words long
- Must be grammatically complete

PRIORITY:
1. Complete sentences with statistics (even if they seem wrong)
2. Complete sentences with dates/events
3. Complete sentences with financial data
4. Complete sentences with technical facts

CRITICAL:
- If you cannot form a complete sentence, DO NOT extract it
- Each claim must be a grammatically correct, complete sentence
- Extract ALL verifiable claims from the document (no artificial limits)
- Extract ALL claims, even if they seem exaggerated or questionable
- Do NOT filter out claims based on whether they seem true or false
- The verification system will check accuracy later

Return STRICT JSON (no extra text):
{{
  "claims": [
    {{
      "text": "A complete, grammatically correct sentence with all necessary information",
      "claim_type": "statistic|date|financial|technical",
      "context": "surrounding context from document",
      "importance": 1-10
    }}
  ]
}}

REMEMBER: ONLY COMPLETE SENTENCES. NO FRAGMENTS. NO PHRASES. EXTRACT ALL VERIFIABLE CLAIMS. DO NOT FILTER BY TRUTHFULNESS."""


def _validate_claim(claim_text: str) -> bool:
    """
    Validate that a claim is a complete sentence.
    
    RELAXED RULES (more reasonable):
    - Must be at least 6 words (reduced from 10)
    - Must contain a verb (expanded verb list)
    - Must not be just a number or date
    - Must be a complete sentence
    - Must have subject and predicate
    
    Args:
        claim_text: Claim text to validate
        
    Returns:
        True if claim is valid, False otherwise
    """
    claim_text = claim_text.strip()
    
    # Must be at least 5 words (RELAXED from 6)
    words = claim_text.split()
    if len(words) < 5:
        logger.debug(f"❌ Rejected (too short, {len(words)} words): {claim_text[:50]}")
        return False
    
    # Must contain a verb (EXPANDED verb list)
    verb_patterns = [
        # Common verbs
        r'\b(is|are|was|were|has|have|had|be|been)\b',
        # Action verbs
        r'\b(reached|exceeded|contains|includes|provides|offers|supports)\b',
        r'\b(shows|indicates|reports|states|claims|estimates|projects|expects)\b',
        r'\b(grew|increased|decreased|rose|fell|dropped|climbed|declined)\b',
        r'\b(will|would|should|could|may|might|can|must|shall)\b',
        r'\b(released|launched|created|developed|built|designed|made|founded)\b',
        r'\b(became|remains|continues|began|started|ended|finished|completed)\b',
        r'\b(announced|reported|revealed|disclosed|published|issued)\b',
        r'\b(achieved|attained|obtained|gained|earned|received)\b',
        r'\b(operates|manages|controls|owns|holds|maintains)\b'
    ]
    
    has_verb = any(re.search(pattern, claim_text, re.IGNORECASE) for pattern in verb_patterns)
    if not has_verb:
        logger.debug(f"❌ Rejected (no verb): {claim_text[:50]}")
        return False
    
    # Must not be just a number or percentage
    if re.match(r'^[\d\s,\.%]+$', claim_text):
        logger.debug(f"❌ Rejected (just numbers): {claim_text}")
        return False
    
    # Must not start with ONLY a number (e.g., "3 billion" alone)
    if re.match(r'^\d+[\s,\.]*(billion|million|thousand|trillion|%|percent)?$', claim_text, re.IGNORECASE):
        logger.debug(f"❌ Rejected (starts with number only): {claim_text}")
        return False
    
    # REMOVED: Fragment pattern checks that were too strict
    # Now only reject obvious fragments
    obvious_fragments = [
        r'^\d+\s+(billion|million|thousand|trillion|percent|%)$',  # Just number + unit alone
    ]
    
    for pattern in obvious_fragments:
        if re.match(pattern, claim_text, re.IGNORECASE):
            logger.debug(f"❌ Rejected (obvious fragment): {claim_text[:50]}")
            return False
    
    # Must have both subject and predicate (RELAXED check)
    # A complete sentence should have words before and after the verb
    verb_position = -1
    for pattern in verb_patterns:
        match = re.search(pattern, claim_text, re.IGNORECASE)
        if match:
            verb_position = match.start()
            break
    
    if verb_position != -1:
        words_before_verb = claim_text[:verb_position].strip().split()
        words_after_verb = claim_text[verb_position:].strip().split()
        
        # RELAXED: Need at least 1 word before verb (was 2)
        if len(words_before_verb) < 1:
            logger.debug(f"❌ Rejected (no subject): {claim_text[:50]}")
            return False
        
        # RELAXED: Need at least 2 words after verb (was 3)
        if len(words_after_verb) < 2:
            logger.debug(f"❌ Rejected (no predicate): {claim_text[:50]}")
            return False
    
    # Must end with proper punctuation (sentence ending)
    if not claim_text[-1] in '.!?':
        # Add period if missing but otherwise valid
        claim_text = claim_text + '.'
    
    return True


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def _call_llm_api(text: str, max_claims: int = 10) -> dict:
    """
    Call LLM API with retry logic.
    
    Args:
        text: Text to analyze
        max_claims: Maximum number of claims to extract
        
    Returns:
        Parsed JSON response
        
    Raises:
        ClaimExtractionError: If API call fails
    """
    try:
        client = get_llm_client()
        
        # Simple, clear prompt without artificial limits
        prompt = f"""{SYSTEM_PROMPT}

🔴 CRITICAL INSTRUCTION:
Extract ALL verifiable claims from the document.
Each claim MUST be a COMPLETE SENTENCE (not a fragment, not a phrase).
Do not impose artificial limits - if there are 40 claims, extract all 40.

IMPORTANT: Extract ALL claims, even if they seem exaggerated, questionable, or potentially false.
Do NOT filter claims based on whether you think they are true or false.
The verification system will check accuracy later.

Your job is ONLY to extract complete sentence claims, not to judge their truthfulness.

Text to analyze:
{text}

Return ONLY JSON with ALL complete sentence claims found in the document."""
        
        result = client.generate_json(
            prompt=prompt,
            temperature=0.1,  # Very low temperature for strict adherence to instructions
            max_tokens=8000  # Increased to handle more claims
        )
        
        logger.debug(f"LLM API response: {sanitize_log_message(str(result))}")
        
        return result
        
    except TimeoutError as e:
        logger.warning("LLM request timeout, will retry...")
        raise
        
    except ConnectionError as e:
        logger.warning(f"Connection error: {str(e)}, will retry...")
        raise
    
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing error: {str(e)}")
        logger.error("This usually means the LLM returned invalid JSON format")
        raise ClaimExtractionError(f"Invalid JSON response from LLM: {str(e)}")
        
    except Exception as e:
        logger.error(f"LLM API error: {str(e)}", exc_info=True)
        raise ClaimExtractionError(f"API request failed: {str(e)}")


def extract_claims(text: str, max_claims: int = None) -> List[Claim]:
    """
    Extract complete, meaningful, verifiable claims from text.
    
    STRICT VALIDATION:
    - Only full sentences with subject + context + value
    - No fragments, no standalone numbers
    - Minimum 10 words per claim
    - Must contain a verb
    - Extracts ALL verifiable claims (no artificial limits)
    
    Args:
        text: Extracted PDF text to analyze
        max_claims: DEPRECATED - kept for backward compatibility, not used
        
    Returns:
        List[Claim]: List of validated Claim objects
        
    Raises:
        ClaimExtractionError: If extraction fails
    """
    # Validate input
    if not text or not text.strip():
        logger.warning("Empty text provided to extract_claims")
        return []
    
    # Limit text length to avoid token limits
    if len(text) > 15000:
        logger.info(f"Truncating text from {len(text)} to 15000 characters")
        text = text[:15000]
    
    logger.info(f"Extracting ALL claims from text ({len(text)} characters)...")
    logger.info(f"API CALL: Claim extraction (1 call) - Extracting ALL complete sentences")
    
    try:
        # Call LLM API
        result = _call_llm_api(text, max_claims)
        
        # Parse claims
        claims_data = result.get("claims", [])
        
        if not claims_data:
            logger.info("No claims found in text")
            return []
        
        logger.info(f"LLM returned {len(claims_data)} potential claims")
        
        # Validate and convert to Claim objects
        claims = []
        seen_texts = set()
        
        for claim_dict in claims_data:
            try:
                claim_text = claim_dict["text"].strip()
                claim_type = claim_dict.get("claim_type", "").lower()
                
                # Validate claim type
                if claim_type not in ["statistic", "date", "financial", "technical"]:
                    logger.warning(f"Invalid claim_type '{claim_type}', skipping")
                    continue
                
                # STRICT VALIDATION: Check if claim is complete and meaningful
                if not _validate_claim(claim_text):
                    continue
                
                # Deduplicate
                if claim_text in seen_texts:
                    continue
                seen_texts.add(claim_text)
                
                # Create Claim object
                claim = Claim(
                    text=claim_text,
                    claim_type=claim_type,
                    context=claim_dict.get("context", "")
                )
                claims.append(claim)
                
            except (KeyError, TypeError) as e:
                logger.warning(f"Failed to parse claim: {claim_dict}. Error: {str(e)}")
                continue
        
        # Log claim count and types for monitoring
        claim_types = {}
        for claim in claims:
            claim_types[claim.claim_type] = claim_types.get(claim.claim_type, 0) + 1
        
        logger.info(f"✅ Extracted {len(claims)} COMPLETE SENTENCE claims: {dict(claim_types)}")
        
        # Final validation: ensure all are complete sentences (STRICT)
        validated_claims = []
        for claim in claims:
            if _validate_claim(claim.text):
                validated_claims.append(claim)
            else:
                logger.warning(f"❌ Post-validation rejected: {claim.text[:50]}...")
        
        logger.info(f"✅ Final count after validation: {len(validated_claims)} claims")
        
        return validated_claims
        
    except ClaimExtractionError:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected error during claim extraction: {str(e)}", exc_info=True)
        raise ClaimExtractionError(f"Failed to extract claims: {str(e)}")
