"""
Data models for the Fact-Checking Web App.

This module defines the core data structures used throughout the application:
- Claim: Represents a verifiable claim extracted from text
- Evidence: Represents evidence from web search results
- VerificationResult: Represents the verification result for a claim
- ProcessingState: Tracks processing state for UI updates
"""

from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime


@dataclass
class Claim:
    """
    Represents a verifiable claim extracted from text.
    
    Attributes:
        text: The claim statement
        claim_type: One of: "statistic", "date", "financial", "technical"
        context: Surrounding text for context (1-2 sentences)
    """
    text: str
    claim_type: str
    context: str
    
    def __hash__(self):
        """
        Make Claim hashable for caching purposes.
        
        Returns:
            Hash of the claim text
        """
        return hash(self.text)


@dataclass
class Evidence:
    """
    Represents evidence from web search.
    
    Attributes:
        url: Source URL
        snippet: Relevant text excerpt from the source
        title: Title of the source document
        published_date: Publication date (ISO format string, optional)
        domain: Domain name of the source
    """
    url: str
    snippet: str
    title: str
    published_date: Optional[str]
    domain: str
    
    def is_recent(self, years: int = 2) -> bool:
        """
        Check if evidence is from the last N years.
        
        Args:
            years: Number of years to consider as "recent" (default: 2)
            
        Returns:
            True if evidence is from the last N years, False otherwise
            
        Note:
            Returns False if published_date is None or cannot be parsed
        """
        if not self.published_date:
            return False
        try:
            pub_date = datetime.fromisoformat(self.published_date)
            return (datetime.now() - pub_date).days < (years * 365)
        except (ValueError, TypeError):
            return False


@dataclass
class VerificationResult:
    """
    Represents the verification result for a claim.
    
    Attributes:
        claim: The claim text that was verified
        status: One of "Verified", "Inaccurate", or "False"
        confidence_score: Confidence score between 0 and 100
        explanation: Explanation for the verification decision
        correct_fact: The correct fact if claim is Inaccurate or False (optional)
        sources: List of source URLs used for verification
        reasoning: Step-by-step justification for the decision (optional)
    """
    claim: str
    status: str
    confidence_score: int
    explanation: str
    correct_fact: Optional[str]
    sources: List[str]
    reasoning: Optional[str] = None
    
    def to_dict(self) -> dict:
        """
        Convert verification result to dictionary for export.
        
        Returns:
            Dictionary containing all verification result fields
        """
        return {
            "claim": self.claim,
            "status": self.status,
            "confidence_score": self.confidence_score,
            "explanation": self.explanation,
            "correct_fact": self.correct_fact,
            "sources": self.sources,
            "reasoning": self.reasoning if self.reasoning else "No reasoning provided"
        }
    
    def get_color_code(self) -> str:
        """
        Get color code for UI display based on verification status.
        
        Returns:
            Hex color code string:
            - "#28a745" (green) for Verified
            - "#ffc107" (yellow) for Inaccurate
            - "#dc3545" (red) for False
            
        Raises:
            KeyError: If status is not one of the valid values
        """
        return {
            "Verified": "#28a745",    # Green
            "Inaccurate": "#ffc107",  # Yellow
            "False": "#dc3545"        # Red
        }[self.status]


@dataclass
class ProcessingState:
    """
    Tracks processing state for UI updates.
    
    Attributes:
        total_claims: Total number of claims to process
        processed_claims: Number of claims processed so far
        current_claim: The claim currently being processed (optional)
        errors: List of error messages encountered during processing
    """
    total_claims: int
    processed_claims: int
    current_claim: Optional[str]
    errors: List[str]
    
    def progress_percentage(self) -> float:
        """
        Calculate progress as a percentage.
        
        Returns:
            Progress percentage between 0.0 and 100.0
            Returns 0.0 if total_claims is 0
        """
        if self.total_claims == 0:
            return 0.0
        return (self.processed_claims / self.total_claims) * 100
