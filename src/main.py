"""
PRODUCTION-READY Streamlit UI for Fact-Checking Web App.

This is the high-accuracy, production-ready version with:
- Strict claim validation (no garbage claims)
- Source quality filtering
- Multi-source verification
- Batch processing
- Conflict detection
- Comprehensive error handling
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# Load .env from project root (parent directory of src/)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=env_path)

import streamlit as st
import pandas as pd
import json
import csv
import io
from typing import List
import logging

from utils import setup_logging, validate_api_keys
from pdf_processor import (
    extract_text_from_pdf,
    PDFExtractionError,
    EmptyPDFError,
    CorruptedPDFError,
    PasswordProtectedPDFError
)
from claim_extractor import extract_claims, ClaimExtractionError
from web_search import (
    search_for_evidence,
    SearchAPIError,
    InsufficientSourcesError,
    deduplicate_claims,
    get_cache_stats
)
from verifier import (
    verify_claims_batch,
    VerificationError,
    detect_conflicts,
    get_api_stats,
    reset_api_stats
)
from models import VerificationResult

# Set up logging
logger = setup_logging()


def validate_configuration():
    """Validate required API keys on startup."""
    valid, missing_keys = validate_api_keys()
    
    if not valid:
        st.error(f"❌ Missing required API keys: {', '.join(missing_keys)}")
        st.info("Please set the following environment variables:")
        for key in missing_keys:
            st.code(f"export {key}=your_api_key_here")
        st.markdown("""
        **Setup Instructions:**
        1. Create a `.env` file in the project root
        2. Add your API keys:
           ```
           NVIDIA_API_KEY=your_nvidia_key_here
           TAVILY_API_KEY=your_tavily_key_here
           ```
        3. Restart the application
        
        **Get API Keys:**
        - NVIDIA API: https://build.nvidia.com/
        - Tavily API: https://tavily.com/
        """)
        st.stop()


def display_api_usage():
    """Display API usage statistics in sidebar."""
    stats = get_api_stats()
    cache_stats = get_cache_stats()
    
    st.sidebar.markdown("### 📊 API Usage Stats")
    st.sidebar.metric("LLM Calls", stats["llm_calls"])
    st.sidebar.metric("Search Calls", stats.get("search_calls", 0))
    st.sidebar.metric("Cache Hits", cache_stats.get("cache_hits", 0))
    
    # Calculate efficiency
    total_claims = st.session_state.get('total_claims', 0)
    if total_claims > 0 and stats["llm_calls"] > 0:
        efficiency = ((total_claims - stats["llm_calls"]) / total_claims * 100)
        st.sidebar.metric("API Efficiency", f"{efficiency:.0f}%")


def display_results(results: List[VerificationResult]):
    """Display verification results in formatted table."""
    if not results:
        st.warning("No results to display.")
        return
    
    def normalize_status(status: str) -> str:
        """Normalize status to title case."""
        status_map = {
            "verified": "Verified",
            "inaccurate": "Inaccurate",
            "false": "False",
            "VERIFIED": "Verified",
            "INACCURATE": "Inaccurate",
            "FALSE": "False"
        }
        return status_map.get(status, status.title())
    
    def get_status_indicator(status: str) -> str:
        status = normalize_status(status)
        indicators = {
            "Verified": "✔ 🟢 Verified",
            "Inaccurate": "⚠ 🟡 Inaccurate",
            "False": "✖ 🔴 False"
        }
        return indicators.get(status, status)
    
    def format_sources(sources: List[str]) -> str:
        if not sources:
            return "N/A"
        links = [f"[{i+1}]({url})" for i, url in enumerate(sources[:5])]
        return " | ".join(links)
    
    def format_confidence(score: int) -> str:
        if score < 50:
            return f"{score}% ⚠️"
        return f"{score}%"
    
    # Create DataFrame
    df_data = []
    for result in results:
        df_data.append({
            "Status": get_status_indicator(result.status),
            "Claim": result.claim,
            "Confidence": format_confidence(result.confidence_score),
            "Explanation": result.explanation,
            "Reasoning": result.reasoning if result.reasoning else "N/A",
            "Correct Fact": result.correct_fact if result.correct_fact else "N/A",
            "Sources": format_sources(result.sources)
        })
    
    df = pd.DataFrame(df_data)
    
    st.subheader("📊 Verification Results")
    
    # Summary metrics - normalize status for counting
    col1, col2, col3, col4 = st.columns(4)
    
    verified_count = sum(1 for r in results if normalize_status(r.status) == "Verified")
    inaccurate_count = sum(1 for r in results if normalize_status(r.status) == "Inaccurate")
    false_count = sum(1 for r in results if normalize_status(r.status) == "False")
    avg_confidence = sum(r.confidence_score for r in results) / len(results) if results else 0
    
    col1.metric("✔ Verified", verified_count)
    col2.metric("⚠ Inaccurate", inaccurate_count)
    col3.metric("✖ False", false_count)
    col4.metric("📈 Avg Confidence", f"{avg_confidence:.0f}%")
    
    # Display table
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn("Status", width="medium"),
            "Claim": st.column_config.TextColumn("Claim", width="large"),
            "Confidence": st.column_config.TextColumn("Confidence", width="small"),
            "Explanation": st.column_config.TextColumn("Explanation", width="large"),
            "Reasoning": st.column_config.TextColumn("Reasoning", width="large"),
            "Correct Fact": st.column_config.TextColumn("Correct Fact", width="medium"),
            "Sources": st.column_config.TextColumn("Sources", width="medium")
        }
    )


def export_to_csv(results: List[VerificationResult]) -> str:
    """Export results to CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        "Claim", "Status", "Confidence Score", "Explanation",
        "Reasoning", "Correct Fact", "Sources"
    ])
    
    for result in results:
        writer.writerow([
            result.claim,
            result.status,
            result.confidence_score,
            result.explanation,
            result.reasoning if result.reasoning else "N/A",
            result.correct_fact if result.correct_fact else "N/A",
            ", ".join(result.sources) if result.sources else "N/A"
        ])
    
    return output.getvalue()


def export_to_json(results: List[VerificationResult]) -> str:
    """Export results to JSON format."""
    from datetime import datetime
    
    def normalize_status(status: str) -> str:
        """Normalize status to title case."""
        status_map = {
            "verified": "Verified",
            "inaccurate": "Inaccurate",
            "false": "False",
            "VERIFIED": "Verified",
            "INACCURATE": "Inaccurate",
            "FALSE": "False"
        }
        return status_map.get(status, status.title())
    
    verified_count = sum(1 for r in results if normalize_status(r.status) == "Verified")
    inaccurate_count = sum(1 for r in results if normalize_status(r.status) == "Inaccurate")
    false_count = sum(1 for r in results if normalize_status(r.status) == "False")
    
    api_stats = get_api_stats()
    
    report = {
        "report_metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_claims": len(results),
            "verified_count": verified_count,
            "inaccurate_count": inaccurate_count,
            "false_count": false_count,
            "api_usage": api_stats,
            "version": "v2-production"
        },
        "results": [result.to_dict() for result in results]
    }
    
    return json.dumps(report, indent=2)


def main():
    """Main Streamlit application entry point."""
    validate_configuration()
    
    st.set_page_config(
        page_title="Fact-Checking Web App",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Fact-Checking Web App")
    st.caption("🚀 AI-Powered Document Verification with Multi-Source Evidence")
    
    st.markdown("""
    **Features:**
    - ✅ Strict claim validation (no incomplete claims)
    - ✅ High-quality source filtering (.gov, WHO, UN, etc.)
    - ✅ Multi-source verification
    - ✅ Efficient batch processing
    - ✅ Conflict detection
    - ✅ Confidence scoring (0–100)
    
    **How it works:**
    1. 📄 Upload a PDF document
    2. 🔍 Extract meaningful, verifiable claims
    3. 🌐 Retrieve supporting evidence from trusted sources
    4. ✅ Verify claims with reasoning and confidence score
    5. 📊 View structured results
    """)
    
    # Display API usage in sidebar
    display_api_usage()
    
    st.divider()
    
    uploaded_file = st.file_uploader(
        "Upload PDF Document",
        type=["pdf"],
        help="Upload a PDF file containing factual claims to verify"
    )
    
    if not uploaded_file:
        st.info("👆 Please upload a PDF file to get started.")
        return
    
    # Reset API stats for new document
    reset_api_stats()
    
    st.subheader("📝 Step 1: Extract Text from PDF")
    
    try:
        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)
        
        st.success(f"✅ Successfully extracted text from PDF")
        
        with st.expander("📄 View Extracted Text", expanded=False):
            char_count = len(extracted_text)
            page_count = extracted_text.count("--- Page Break ---") + 1
            
            col1, col2 = st.columns(2)
            col1.metric("Characters", f"{char_count:,}")
            col2.metric("Pages", page_count)
            
            st.text_area(
                "Extracted Text",
                value=extracted_text[:2000] + "..." if len(extracted_text) > 2000 else extracted_text,
                height=300,
                disabled=True,
                label_visibility="collapsed"
            )
        
    except (EmptyPDFError, CorruptedPDFError, PasswordProtectedPDFError, PDFExtractionError) as e:
        st.error(f"❌ PDF processing error: {str(e)}")
        logger.error(f"PDF error: {uploaded_file.name}", exc_info=True)
        return
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
        logger.error(f"Unexpected error: {uploaded_file.name}", exc_info=True)
        return
    
    st.divider()
    
    st.subheader("🔍 Step 2: Extract Claims")
    st.caption("Extracting meaningful, verifiable claims from the document")
    
    if st.button("Extract Claims", type="primary", width='stretch'):
        try:
            with st.spinner("Analyzing text and extracting verifiable claims..."):
                claims = extract_claims(extracted_text)
            
            if not claims:
                st.warning("⚠️ No verifiable claims found in the document.")
                logger.info(f"No claims found in: {uploaded_file.name}")
                return
            
            st.success(f"✅ Extracted {len(claims)} verified claims")
            st.info(f"📊 API Calls: 1 LLM call for claim extraction")
            
            st.session_state['claims'] = claims
            st.session_state['total_claims'] = len(claims)
            
            st.markdown("### Extracted Claims")
            
            for i, claim in enumerate(claims, 1):
                type_colors = {
                    "statistic": "🔢",
                    "date": "📅",
                    "financial": "💰",
                    "technical": "⚙️"
                }
                badge = type_colors.get(claim.claim_type, "📌")
                
                st.markdown(f"**{i}. {badge} [{claim.claim_type.upper()}]** {claim.text}")
                with st.expander("Context"):
                    st.caption(claim.context)
            
        except ClaimExtractionError as e:
            st.error(f"❌ Failed to extract claims: {str(e)}")
            logger.error(f"Claim extraction failed for: {uploaded_file.name}", exc_info=True)
            return
        except Exception as e:
            st.error(f"❌ Unexpected error during claim extraction: {str(e)}")
            logger.error(f"Unexpected claim extraction error: {uploaded_file.name}", exc_info=True)
            return
    
    if 'claims' in st.session_state and st.session_state['claims']:
        st.divider()
        st.subheader("✅ Step 3: Verify Claims")
        st.caption("Retrieving evidence from trusted sources and verifying with confidence scoring")
        
        if st.button("Verify All Claims", type="primary", width='stretch'):
            claims = st.session_state['claims']
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Search for evidence
            status_text.text("🌐 Retrieving supporting evidence from trusted sources...")
            evidence_map = {}
            
            for i, claim in enumerate(claims):
                try:
                    evidence = search_for_evidence(claim.text, min_sources=2, max_results=3)
                    evidence_map[claim.text] = evidence
                    logger.info(f"Found {len(evidence)} sources for claim {i+1}")
                except (InsufficientSourcesError, SearchAPIError) as e:
                    logger.warning(f"Search failed for claim {i+1}: {str(e)}")
                    evidence_map[claim.text] = []
                
                progress_bar.progress((i + 1) / len(claims) * 0.5)
            
            # Prepare evidence list
            evidence_list = [evidence_map.get(c.text, []) for c in claims]
            
            # Batch verify
            status_text.text("✅ Verifying claims with reasoning and confidence scoring...")
            
            try:
                results = verify_claims_batch(claims, evidence_list, batch_size=5)
                
                progress_bar.progress(0.9)
                
                # Detect conflicts
                status_text.text("🔍 Detecting conflicts...")
                conflicts = detect_conflicts(claims, results)
                
                if conflicts:
                    st.warning(f"⚠️ Detected {len(conflicts)} potential conflict(s) between claims")
                
                progress_bar.progress(1.0)
                status_text.empty()
                progress_bar.empty()
                
                # Display API stats
                api_stats = get_api_stats()
                st.success(f"✅ Verification complete!")
                st.info(f"""
                📊 **API Usage:**
                - LLM Calls: {api_stats['llm_calls']} (efficient batch processing)
                - Efficiency: ~{((len(claims) - api_stats['llm_calls']) / len(claims) * 100):.0f}% fewer calls
                """)
                
                st.session_state['results'] = results
                
            except VerificationError as e:
                st.error(f"❌ Verification failed: {str(e)}")
                logger.error("Verification error", exc_info=True)
                return
    
    if 'results' in st.session_state and st.session_state['results']:
        st.divider()
        
        display_results(st.session_state['results'])
        
        st.divider()
        
        st.subheader("📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = export_to_csv(st.session_state['results'])
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="fact_check_report_v2.csv",
                mime="text/csv",
                width='stretch'
            )
        
        with col2:
            json_data = export_to_json(st.session_state['results'])
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name="fact_check_report_v2.json",
                mime="application/json",
                width='stretch'
            )


if __name__ == "__main__":
    main()
