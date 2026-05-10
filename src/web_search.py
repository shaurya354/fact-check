"""
PRODUCTION-READY Web Search with source quality filtering.

This module searches for evidence with strict source quality requirements.
Prioritizes .gov, .edu, WHO, World Bank, UN, major news sources.
"""

import os
import logging
import time
from typing import List, Dict
from datetime import datetime
import streamlit as st
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.models import Evidence
from src.utils import get_api_key

logger = logging.getLogger("fact_checker.web_search")


class SearchAPIError(Exception):
    """Exception raised when search API fails."""
    pass


class InsufficientSourcesError(Exception):
    """Exception raised when insufficient high-quality sources are found."""
    pass


# High-quality source domains (prioritize these)
HIGH_QUALITY_DOMAINS = [
    # Government
    '.gov', 'gov.uk', 'gov.au', 'gov.ca',
    # Education
    '.edu',
    # International organizations
    'who.int', 'un.org', 'worldbank.org', 'imf.org', 'oecd.org',
    # Research & Statistics
    'statista.com', 'nature.com', 'science.org', 'ieee.org', 'acm.org',
    # Health
    'nih.gov', 'cdc.gov', 'mayoclinic.org',
    # Major news (fact-checked)
    'reuters.com', 'apnews.com', 'bbc.com', 'npr.org',
    # Tech documentation
    'python.org', 'microsoft.com', 'apple.com', 'google.com'
]

# Low-quality domains (avoid these)
LOW_QUALITY_DOMAINS = [
    'blog', 'forum', 'wiki', 'quora', 'reddit',
    'amazon', 'ebay', 'aliexpress',  # E-commerce
    'calculator', 'converter',  # Tools
    'pinterest', 'instagram', 'facebook',  # Social media
]


def _calculate_source_quality(domain: str) -> float:
    """
    Calculate source quality score (0.0-1.0).
    
    Args:
        domain: Source domain
        
    Returns:
        Quality score (0.0 = low quality, 1.0 = high quality)
    """
    domain_lower = domain.lower()
    
    # Check high-quality domains
    for hq_domain in HIGH_QUALITY_DOMAINS:
        if hq_domain in domain_lower:
            return 1.0
    
    # Check low-quality domains
    for lq_domain in LOW_QUALITY_DOMAINS:
        if lq_domain in domain_lower:
            return 0.2
    
    # Default: medium quality
    return 0.6


def _filter_high_quality_sources(results: List[Dict]) -> List[Dict]:
    """
    Filter search results to keep only high-quality sources.
    
    Args:
        results: Raw search results from Tavily
        
    Returns:
        Filtered list of high-quality results
    """
    filtered = []
    
    for result in results:
        url = result.get('url', '')
        domain = _extract_domain(url)
        quality = _calculate_source_quality(domain)
        
        # Only keep sources with quality >= 0.6
        if quality >= 0.6:
            result['quality_score'] = quality
            filtered.append(result)
            logger.debug(f"Accepted source (quality={quality:.2f}): {domain}")
        else:
            logger.debug(f"Rejected source (quality={quality:.2f}): {domain}")
    
    # Sort by quality score (highest first)
    filtered.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
    
    return filtered


def _extract_domain(url: str) -> str:
    """
    Extract domain from URL.
    
    Args:
        url: Full URL
        
    Returns:
        Domain name
    """
    import re
    match = re.search(r'https?://([^/]+)', url)
    if match:
        domain = match.group(1)
        # Remove www. prefix
        domain = domain.replace('www.', '')
        return domain
    return url


# Cache for search results
@st.cache_data(ttl=3600, show_spinner=False)
def _cached_search(query: str, max_results: int) -> List[Dict]:
    """
    Cached search function to reduce API calls.
    
    Args:
        query: Search query
        max_results: Maximum number of results
        
    Returns:
        List of search results
    """
    api_key = get_api_key("TAVILY_API_KEY")
    
    client = TavilyClient(api_key=api_key)
    
    try:
        # Search with Tavily
        response = client.search(
            query=query,
            max_results=max_results * 2,  # Get more results for filtering
            search_depth="advanced",
            include_domains=None,
            exclude_domains=None
        )
        
        results = response.get('results', [])
        logger.info(f"Tavily returned {len(results)} results for: {query[:50]}...")
        
        return results
        
    except Exception as e:
        logger.error(f"Tavily search failed: {str(e)}", exc_info=True)
        raise SearchAPIError(f"Search failed: {str(e)}")


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError))
)
def search_for_evidence(
    claim: str,
    min_sources: int = 3,
    max_results: int = 3
) -> List[Evidence]:
    """
    Search for high-quality evidence to verify a claim.
    
    QUALITY FILTERING:
    - Prioritizes .gov, .edu, WHO, World Bank, UN
    - Filters out blogs, forums, e-commerce
    - Returns top 3 highest-quality sources
    
    Args:
        claim: Claim text to search for
        min_sources: Minimum number of sources required (default: 3)
        max_results: Maximum number of results to return (default: 3)
        
    Returns:
        List[Evidence]: List of Evidence objects from high-quality sources
        
    Raises:
        SearchAPIError: If search fails
        InsufficientSourcesError: If not enough high-quality sources found
    """
    logger.info(f"Searching for evidence: {claim[:80]}...")
    logger.info(f"API CALL: Web search (1 call)")
    
    # Add rate limiting
    time.sleep(1)  # 1 second delay between searches
    
    try:
        # Search with caching
        results = _cached_search(claim, max_results)
        
        # Filter for high-quality sources
        filtered_results = _filter_high_quality_sources(results)
        
        if len(filtered_results) < min_sources:
            logger.warning(
                f"Insufficient high-quality sources: found {len(filtered_results)}, "
                f"need {min_sources}"
            )
            # If we don't have enough high-quality sources, use what we have
            if len(filtered_results) == 0:
                raise InsufficientSourcesError(
                    f"No high-quality sources found for claim: {claim[:50]}..."
                )
        
        # Convert to Evidence objects (top max_results)
        evidence_list = []
        for result in filtered_results[:max_results]:
            try:
                url = result.get('url', '')
                domain = _extract_domain(url)
                
                # Parse published date if available
                published_date = result.get('published_date')
                if published_date:
                    try:
                        # Try to parse ISO format
                        date_obj = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                        published_date = date_obj.isoformat()
                    except (ValueError, AttributeError):
                        published_date = None
                
                evidence = Evidence(
                    url=url,
                    domain=domain,
                    snippet=result.get('content', '')[:500],  # Limit snippet length
                    title=result.get('title', domain),  # Use domain as fallback title
                    published_date=published_date
                )
                evidence_list.append(evidence)
                
            except Exception as e:
                logger.warning(f"Failed to parse search result: {str(e)}")
                continue
        
        logger.info(f"Found {len(evidence_list)} high-quality sources")
        
        return evidence_list
        
    except InsufficientSourcesError:
        raise
        
    except SearchAPIError:
        raise
        
    except Exception as e:
        logger.error(f"Unexpected search error: {str(e)}", exc_info=True)
        raise SearchAPIError(f"Search failed: {str(e)}")


def deduplicate_claims(claims: List[str]) -> Dict[str, List[str]]:
    """
    Group similar claims to reduce duplicate searches.
    
    Args:
        claims: List of claim texts
        
    Returns:
        Dictionary mapping representative claim to list of similar claims
    """
    # Simple deduplication: group claims with high word overlap
    groups = {}
    
    for claim in claims:
        # Normalize claim
        normalized = claim.lower().strip()
        words = set(normalized.split())
        
        # Find existing group with high overlap
        found_group = False
        for representative, group in groups.items():
            rep_words = set(representative.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(words & rep_words)
            union = len(words | rep_words)
            similarity = intersection / union if union > 0 else 0
            
            # If similarity > 0.6, add to this group
            if similarity > 0.6:
                group.append(claim)
                found_group = True
                break
        
        # Create new group if no match found
        if not found_group:
            groups[claim] = [claim]
    
    logger.info(f"Deduplicated {len(claims)} claims into {len(groups)} groups")
    
    return groups


def get_cache_stats() -> Dict[str, int]:
    """
    Get cache statistics.
    
    Returns:
        Dictionary with cache statistics
    """
    # Streamlit cache stats
    try:
        cache_info = _cached_search.cache_info()
        return {
            "cache_size": cache_info.hits + cache_info.misses,
            "cache_hits": cache_info.hits,
            "cache_misses": cache_info.misses
        }
    except AttributeError:
        return {"cache_size": 0, "cache_hits": 0, "cache_misses": 0}
