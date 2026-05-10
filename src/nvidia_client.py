"""
NVIDIA API client for the Fact-Checking Web App.

This module provides a unified interface for NVIDIA API calls with fallback
support for Gemini and OpenAI.
"""

import os
import json
import logging
import re
from typing import Dict, Any, Optional
from openai import OpenAI

from src.utils import get_api_key

logger = logging.getLogger("fact_checker.nvidia_client")


class LLMClient:
    """
    Unified LLM client with provider switching support.
    
    Supports:
    - NVIDIA API (primary)
    - Google Gemini (fallback, commented)
    - OpenAI (fallback, commented)
    """
    
    def __init__(self, provider: str = "nvidia"):
        """
        Initialize LLM client with specified provider.
        
        Args:
            provider: LLM provider ("nvidia", "gemini", "openai")
        """
        self.provider = provider.lower()
        self.client = None
        
        if self.provider == "nvidia":
            self._init_nvidia()
        # elif self.provider == "gemini":
        #     self._init_gemini()
        # elif self.provider == "openai":
        #     self._init_openai()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def _init_nvidia(self):
        """Initialize NVIDIA API client."""
        api_key = get_api_key("NVIDIA_API_KEY")
        
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        self.model = "meta/llama-3.3-70b-instruct"
        logger.info(f"Initialized NVIDIA client with model: {self.model}")
    
    # COMMENTED: Gemini fallback
    # def _init_gemini(self):
    #     """Initialize Google Gemini client."""
    #     from google import genai
    #     api_key = os.getenv("GEMINI_API_KEY")
    #     if not api_key:
    #         raise ValueError("GEMINI_API_KEY not found in environment")
    #     
    #     self.client = genai.Client(api_key=api_key)
    #     self.model = "gemini-2.5-flash"
    #     logger.info(f"Initialized Gemini client with model: {self.model}")
    
    # COMMENTED: OpenAI fallback
    # def _init_openai(self):
    #     """Initialize OpenAI client."""
    #     api_key = os.getenv("OPENAI_API_KEY")
    #     if not api_key:
    #         raise ValueError("OPENAI_API_KEY not found in environment")
    #     
    #     self.client = OpenAI(api_key=api_key)
    #     self.model = "gpt-4o-mini"
    #     logger.info(f"Initialized OpenAI client with model: {self.model}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text using the configured LLM provider.
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            Generated text response
            
        Raises:
            Exception: If API call fails
        """
        if self.provider == "nvidia":
            return self._generate_nvidia(prompt, temperature, max_tokens, system_prompt)
        # elif self.provider == "gemini":
        #     return self._generate_gemini(prompt, temperature, max_tokens, system_prompt)
        # elif self.provider == "openai":
        #     return self._generate_openai(prompt, temperature, max_tokens, system_prompt)
    
    def _generate_nvidia(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        system_prompt: Optional[str]
    ) -> str:
        """Generate text using NVIDIA API."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    # COMMENTED: Gemini implementation
    # def _generate_gemini(
    #     self,
    #     prompt: str,
    #     temperature: float,
    #     max_tokens: int,
    #     system_prompt: Optional[str]
    # ) -> str:
    #     """Generate text using Gemini API."""
    #     full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    #     
    #     response = self.client.models.generate_content(
    #         model=self.model,
    #         contents=full_prompt,
    #         config={
    #             "temperature": temperature,
    #             "max_output_tokens": max_tokens
    #         }
    #     )
    #     
    #     return response.text
    
    # COMMENTED: OpenAI implementation
    # def _generate_openai(
    #     self,
    #     prompt: str,
    #     temperature: float,
    #     max_tokens: int,
    #     system_prompt: Optional[str]
    # ) -> str:
    #     """Generate text using OpenAI API."""
    #     messages = []
    #     
    #     if system_prompt:
    #         messages.append({"role": "system", "content": system_prompt})
    #     
    #     messages.append({"role": "user", "content": prompt})
    #     
    #     response = self.client.chat.completions.create(
    #         model=self.model,
    #         messages=messages,
    #         temperature=temperature,
    #         max_tokens=max_tokens
    #     )
    #     
    #     return response.choices[0].message.content
    
    def generate_json(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate JSON response using the configured LLM provider.
        
        Automatically extracts JSON from markdown code blocks if present.
        Includes automatic JSON repair for common issues.
        
        Args:
            prompt: User prompt
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            Parsed JSON dictionary
            
        Raises:
            json.JSONDecodeError: If response cannot be parsed as JSON
        """
        # Add explicit JSON instruction to prompt
        json_instruction = "\n\nIMPORTANT: Return ONLY valid JSON. No extra text before or after the JSON."
        prompt_with_instruction = prompt + json_instruction
        
        response = self.generate(prompt_with_instruction, temperature, max_tokens, system_prompt)
        
        # Extract JSON from markdown code blocks if present
        content = response.strip()
        
        logger.debug(f"Raw LLM response (first 500 chars): {content[:500]}")
        
        # Try to extract from ```json ... ``` (non-greedy)
        if "```json" in content:
            json_match = re.search(r'```json\s*(\{.*\}|\[.*\])\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
                logger.debug("Extracted JSON from ```json block")
        # Try to extract from ``` ... ``` (non-greedy)
        elif "```" in content:
            json_match = re.search(r'```\s*(\{.*\}|\[.*\])\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
                logger.debug("Extracted JSON from ``` block")
        
        # If no code blocks, try to find JSON object/array in the text
        if not content.startswith('{') and not content.startswith('['):
            # Look for first { or [ and extract to matching closing bracket
            json_match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
            if json_match:
                content = json_match.group(1).strip()
                logger.debug("Extracted JSON from raw text")
        
        # Clean up common issues
        content = content.strip()
        
        # Fix double curly braces (LLM sometimes copies template format)
        if content.startswith('{{'):
            content = content[1:]  # Remove first {
        if content.endswith('}}'):
            content = content[:-1]  # Remove last }
        
        content = content.strip()
        
        # Remove any trailing text after the JSON
        if content.startswith('{'):
            # Find the matching closing brace
            brace_count = 0
            for i, char in enumerate(content):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        content = content[:i+1]
                        break
        elif content.startswith('['):
            # Find the matching closing bracket
            bracket_count = 0
            for i, char in enumerate(content):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        content = content[:i+1]
                        break
        
        # Parse JSON
        try:
            parsed = json.loads(content)
            logger.debug(f"Successfully parsed JSON with {len(str(parsed))} characters")
            return parsed
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON. Error: {e}")
            logger.error(f"Content (first 500 chars): {content[:500]}")
            logger.error(f"Content (last 200 chars): {content[-200:]}")
            
            # Try to fix common JSON issues
            logger.info("Attempting to fix common JSON issues...")
            
            # Fix 1: Remove trailing commas
            fixed_content = re.sub(r',(\s*[}\]])', r'\1', content)
            
            # Fix 2: Replace single quotes with double quotes (if any)
            fixed_content = fixed_content.replace("'", '"')
            
            # Fix 3: Remove comments (// or /* */)
            fixed_content = re.sub(r'//.*?\n', '\n', fixed_content)
            fixed_content = re.sub(r'/\*.*?\*/', '', fixed_content, flags=re.DOTALL)
            
            # Fix 4: Ensure property names are quoted
            # This regex finds unquoted property names like {text: "value"} and converts to {"text": "value"}
            fixed_content = re.sub(r'(\{|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_content)
            
            try:
                parsed = json.loads(fixed_content)
                logger.info("✅ Successfully parsed JSON after fixes")
                return parsed
            except json.JSONDecodeError as e2:
                logger.error(f"Still failed after fixes: {e2}")
                logger.error(f"Fixed content (first 500 chars): {fixed_content[:500]}")
                
                # Last resort: Try to extract just the claims array
                claims_match = re.search(r'"claims"\s*:\s*\[(.*?)\]', content, re.DOTALL)
                if claims_match:
                    logger.info("Attempting to extract claims array only...")
                    try:
                        claims_json = '{"claims": [' + claims_match.group(1) + ']}'
                        parsed = json.loads(claims_json)
                        logger.info("✅ Successfully extracted claims array")
                        return parsed
                    except:
                        pass
                
                raise json.JSONDecodeError(
                    f"Invalid JSON response from LLM: {str(e)}",
                    content,
                    e.pos
                )


# Global client instance (lazy initialization)
_client_instance: Optional[LLMClient] = None


def get_llm_client(provider: str = None) -> LLMClient:
    """
    Get or create the global LLM client instance.
    
    Args:
        provider: LLM provider ("nvidia", "gemini", "openai").
                 If None, uses LLM_PROVIDER env var or defaults to "nvidia"
    
    Returns:
        LLMClient instance
    """
    global _client_instance
    
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "nvidia")
    
    if _client_instance is None or _client_instance.provider != provider:
        _client_instance = LLMClient(provider=provider)
    
    return _client_instance
