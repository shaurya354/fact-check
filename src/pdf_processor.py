"""
PDF Processor Module

This module handles PDF text extraction with robust error handling.
Uses PyMuPDF (fitz) as the primary library with pdfplumber as fallback.

Requirements: 1.2, 1.3, 1.4, 1.5, 7.2, 9.1
"""

from typing import BinaryIO
import logging

logger = logging.getLogger("fact_checker.pdf_processor")


# Custom Exception Classes

class PDFExtractionError(Exception):
    """
    Base exception for PDF text extraction failures.
    
    Raised when text extraction from a PDF fails for any reason.
    This is the parent class for more specific PDF-related exceptions.
    
    Validates: Requirements 1.4, 9.1
    """
    pass


class EmptyPDFError(PDFExtractionError):
    """
    Exception raised when a PDF contains no extractable text.
    
    This occurs when:
    - The PDF has no text content (e.g., blank pages)
    - The PDF contains only images without OCR-readable text
    - All pages are empty after text extraction
    
    Validates: Requirements 1.4, 9.1, 9.4
    """
    pass


class CorruptedPDFError(PDFExtractionError):
    """
    Exception raised when a PDF file is corrupted or invalid.
    
    This occurs when:
    - The PDF file structure is damaged
    - The file is not a valid PDF format
    - The PDF cannot be opened by the extraction library
    
    Validates: Requirements 1.4, 9.1
    """
    pass


class PasswordProtectedPDFError(PDFExtractionError):
    """
    Exception raised when a PDF is password-protected.
    
    This occurs when:
    - The PDF requires a password to open
    - The PDF has encryption that prevents text extraction
    - Access to the PDF content is restricted
    
    Validates: Requirements 1.4, 9.1
    """
    pass


# PDF Text Extraction Function

def extract_text_from_pdf(file: BinaryIO) -> str:
    """
    Extract text from PDF file using PyMuPDF with pdfplumber fallback.
    
    This function attempts to extract text using PyMuPDF (fitz) first. If that fails,
    it falls back to pdfplumber for complex layouts. It handles various error conditions
    including corrupted PDFs, password-protected PDFs, and empty PDFs.
    
    Args:
        file: Binary file object from Streamlit file uploader or similar source
        
    Returns:
        Extracted text as string with pages separated by newlines
        
    Raises:
        EmptyPDFError: If PDF contains no extractable text
        CorruptedPDFError: If PDF file is corrupted or invalid
        PasswordProtectedPDFError: If PDF is password-protected
        PDFExtractionError: If text extraction fails for any other reason
        
    Validates: Requirements 1.2, 1.3, 1.5, 7.2
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF (fitz) is not installed")
        raise PDFExtractionError("PyMuPDF library is not available")
    
    # Read the binary content from the file object once
    pdf_bytes = file.read()
    
    # Try PyMuPDF extraction first
    try:
        # Open PDF from binary stream
        logger.info("Opening PDF document with PyMuPDF")
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Get page count
        page_count = len(pdf_document)
        logger.info(f"PDF has {page_count} page(s)")
        
        # Extract text from all pages
        extracted_text_parts = []
        
        for page_num in range(page_count):
            page = pdf_document[page_num]
            page_text = page.get_text()
            
            # Add page separator if there's content
            if page_text.strip():
                extracted_text_parts.append(page_text)
                logger.debug(f"Extracted {len(page_text)} characters from page {page_num + 1}")
        
        # Close the PDF document
        pdf_document.close()
        
        # Concatenate all page text with page separators
        extracted_text = "\n\n--- Page Break ---\n\n".join(extracted_text_parts)
        
        # Check if extracted text is empty
        if not extracted_text.strip():
            logger.warning("PDF contains no extractable text with PyMuPDF")
            raise EmptyPDFError("The PDF contains no extractable text content")
        
        # Log extraction metadata
        char_count = len(extracted_text)
        logger.info(f"Successfully extracted {char_count} characters from {page_count} page(s) using PyMuPDF")
        
        return extracted_text
        
    except fitz.FileDataError as e:
        # Handle corrupted PDF files
        logger.error(f"Corrupted PDF file: {str(e)}")
        raise CorruptedPDFError(f"The PDF file is corrupted or invalid: {str(e)}")
        
    except RuntimeError as e:
        # Handle password-protected PDFs
        error_msg = str(e).lower()
        if "password" in error_msg or "encrypted" in error_msg:
            logger.warning(f"Password-protected PDF: {str(e)}")
            raise PasswordProtectedPDFError("The PDF is password-protected and cannot be processed")
        else:
            # Other runtime errors - try fallback
            logger.warning(f"PyMuPDF runtime error: {str(e)}, attempting pdfplumber fallback")
            return _extract_with_pdfplumber(pdf_bytes)
            
    except EmptyPDFError:
        # Re-raise EmptyPDFError without wrapping
        raise
        
    except CorruptedPDFError:
        # Re-raise CorruptedPDFError without wrapping
        raise
        
    except PasswordProtectedPDFError:
        # Re-raise PasswordProtectedPDFError without wrapping
        raise
        
    except Exception as e:
        # Catch any other unexpected errors and try fallback
        logger.warning(f"PyMuPDF extraction failed: {str(e)}, attempting pdfplumber fallback")
        return _extract_with_pdfplumber(pdf_bytes)


def _extract_with_pdfplumber(pdf_bytes: bytes) -> str:
    """
    Extract text from PDF using pdfplumber as fallback.
    
    This function is called when PyMuPDF extraction fails. It uses pdfplumber
    which can handle more complex PDF layouts.
    
    Args:
        pdf_bytes: PDF file content as bytes
        
    Returns:
        Extracted text as string with pages separated by newlines
        
    Raises:
        EmptyPDFError: If PDF contains no extractable text
        PDFExtractionError: If text extraction fails
        
    Validates: Requirements 1.5
    """
    try:
        import pdfplumber
        import io
    except ImportError:
        logger.error("pdfplumber is not installed")
        raise PDFExtractionError("pdfplumber library is not available for fallback extraction")
    
    try:
        logger.info("Attempting PDF extraction with pdfplumber fallback")
        
        # Open PDF from bytes using pdfplumber
        pdf_stream = io.BytesIO(pdf_bytes)
        
        with pdfplumber.open(pdf_stream) as pdf:
            page_count = len(pdf.pages)
            logger.info(f"pdfplumber: PDF has {page_count} page(s)")
            
            # Extract text from all pages
            extracted_text_parts = []
            
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                
                # Add page separator if there's content
                if page_text and page_text.strip():
                    extracted_text_parts.append(page_text)
                    logger.debug(f"pdfplumber: Extracted {len(page_text)} characters from page {page_num + 1}")
            
            # Concatenate all page text with page separators
            extracted_text = "\n\n--- Page Break ---\n\n".join(extracted_text_parts)
            
            # Check if extracted text is empty
            if not extracted_text.strip():
                logger.warning("pdfplumber: PDF contains no extractable text")
                raise EmptyPDFError("The PDF contains no extractable text content")
            
            # Log extraction metadata
            char_count = len(extracted_text)
            logger.info(f"Successfully extracted {char_count} characters from {page_count} page(s) using pdfplumber fallback")
            
            return extracted_text
            
    except EmptyPDFError:
        # Re-raise EmptyPDFError without wrapping
        raise
        
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"pdfplumber fallback extraction failed: {str(e)}", exc_info=True)
        raise PDFExtractionError(f"Both PyMuPDF and pdfplumber extraction failed: {str(e)}")
