"""
Fact-Checking PoC App
Validates Gemini URL Context for hallucination detection and claim correction.

Compares two flows:
1. Baseline: Perplexity → Answer Formatter → Final Answer
2. Fact-Checked: Perplexity → Gemini Fact-Check → Answer Formatter → Final Answer + Analysis
"""

import os
import re
import time
import hashlib
import streamlit as st
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Set
from urllib.parse import urlparse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_anthropic import ChatAnthropic


# ============================================================================
# Simple LLM Factory (replaces safe_llm_core dependency)
# ============================================================================

def get_llm(model_name: str, temperature: float = 0.0):
    """
    Get a LangChain LLM by model name.
    
    Supported models:
    - "GPT-4o": Azure OpenAI GPT-4o
    - "GPT-4o Mini": Azure OpenAI GPT-4o-mini
    - "GPT-5 Mini": Azure OpenAI GPT-5-mini (Sweden endpoint)
    - "Claude 4 Sonnet": Anthropic Claude 4 Sonnet
    - "Claude 4.5 Sonnet": Anthropic Claude 4.5 Sonnet
    """
    if model_name == "GPT-4o":
        return AzureChatOpenAI(
            azure_deployment="gpt-4o",
            model_name="gpt-4o",
            temperature=temperature,
            streaming=False,
            seed=42,
        )
    
    elif model_name == "GPT-4o Mini":
        return AzureChatOpenAI(
            azure_deployment="gpt-4o-mini",
            model_name="gpt-4o-mini",
            temperature=temperature,
            streaming=False,
            seed=42,
        )
    
    elif model_name == "GPT-5 Mini":
        sweden_api_key = os.getenv("AZURE_OPENAI_SWEDEN_API_KEY")
        sweden_endpoint = os.getenv("AZURE_OPENAI_SWEDEN_ENDPOINT")
        return AzureChatOpenAI(
            azure_deployment="gpt-5-mini",
            model_name="gpt-5-mini",
            temperature=temperature,
            streaming=False,
            seed=42,
            api_key=sweden_api_key,
            azure_endpoint=sweden_endpoint,
        )
    
    elif model_name == "Claude 4 Sonnet":
        return ChatAnthropic(
            model="claude-sonnet-4-20250514",
            temperature=temperature,
            max_tokens=12288,
        )
    
    elif model_name == "Claude 4.5 Sonnet":
        return ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=temperature,
            max_tokens=12288,
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported: GPT-4o, GPT-4o Mini, GPT-5 Mini, Claude 4 Sonnet, Claude 4.5 Sonnet")

# ============================================================================
# Page Config
# ============================================================================
st.set_page_config(
    page_title="Fact-Checking PoC",
    page_icon="🔍",
    layout="wide"
)

# ============================================================================
# Rule-based Source Processing Utilities
# ============================================================================

def generate_source_id(url: str) -> str:
    """
    Generate a unique but consistent source ID from URL.
    Returns a short ID like 'WEB123' for web sources.
    
    Args:
        url: The full URL of the source
    Returns:
        A unique source ID (e.g., 'WEB1a3')
    """
    hash_value = hashlib.md5(url.encode()).hexdigest()
    return f"WEB{hash_value[:3]}"


def get_domain_label(url: str) -> str:
    """
    Extract a readable domain label from a URL.
    
    Args:
        url: The full URL
    Returns:
        A readable domain name (e.g., 'cbinsights.com')
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        # Remove www. if present
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return url


def build_source_registry(sources: List[str]) -> Dict[int, dict]:
    """
    Build an indexed registry from source URLs with rule-based metadata.
    
    Args:
        sources: List of source URLs from Perplexity
    Returns:
        Dictionary mapping 1-based index to source metadata:
        {
            1: {"id": "WEB123", "url": "https://...", "label": "domain.com"},
            2: {"id": "WEB456", "url": "https://...", "label": "gartner.com"},
        }
    Note: source_type is determined by LLM during fact-checking.
    """
    registry = {}
    for i, url in enumerate(sources, 1):  # 1-based indexing to match [1], [2] citations
        source_id = generate_source_id(url)
        label = get_domain_label(url)
        
        registry[i] = {
            "id": source_id,
            "url": url,
            "label": label,
        }
    
    return registry


def extract_used_citations(text: str) -> Set[int]:
    """
    Extract all citation indices used in the text.
    
    Args:
        text: Text containing citations like [1], [2], etc.
    Returns:
        Set of citation indices (as integers)
    """
    citations = re.findall(r"\[(\d+)\]", text)
    return {int(c) for c in citations}


def replace_citation_ids(text: str, registry: Dict[int, dict]) -> str:
    """
    Replace numeric citations [1], [2] with source IDs [WEB123], [WEB456].
    
    Args:
        text: Text with numeric citations
        registry: Source registry with id mappings
    Returns:
        Text with replaced citation IDs
    """
    result = text
    for idx, source_meta in registry.items():
        result = result.replace(f"[{idx}]", f"[{source_meta['id']}]")
    return result


def generate_citation_id(snippet: str, source_index: int) -> str:
    """
    Generate a unique citation ID from snippet and source index.
    Uses hash for uniqueness.
    
    Args:
        snippet: The snippet text
        source_index: The source index
    Returns:
        Citation ID like "CIT_a1b2c3"
    """
    content = f"{source_index}:{snippet}"
    hash_value = hashlib.md5(content.encode()).hexdigest()
    return f"CIT_{hash_value[:6]}"


def build_citation_registry(
    claims: List,  # List[ClaimVerification]
    url_registry: Dict[int, dict]
) -> Dict[str, dict]:
    """
    Build citation registry from claims. Each (claim, source_evidence) pair = one citation.
    
    Args:
        claims: List of ClaimVerification objects from Gemini
        url_registry: URL registry with base source metadata
    
    Returns:
        Dictionary mapping citation_id to citation metadata:
        {
            "CIT_a1b2c3": {
                "id": "CIT_a1b2c3",
                "claim_text": "Market grew 15%",
                "source_index": 1,
                "source_url": "https://...",
                "source_label": "cbinsights.com",
                "snippet": "The market expanded by 15%...",
                "authority": "High",
                "source_type": "Database",
                "year": "2024",
                "grounding_status": "GROUNDED",
            },
            ...
        }
    """
    citations = {}
    
    for claim in claims:
        for evidence in claim.source_evidences:
            citation_id = generate_citation_id(evidence.snippet, evidence.source_index)
            
            # Get URL metadata from registry
            url_meta = url_registry.get(evidence.source_index, {})
            
            citations[citation_id] = {
                "id": citation_id,
                "claim_text": claim.claim,
                "grounding_status": claim.grounding_status,
                "source_index": evidence.source_index,
                "source_url": url_meta.get("url", ""),
                "source_label": url_meta.get("label", f"Source {evidence.source_index}"),
                "snippet": evidence.snippet,
                "authority": evidence.authority,
                "source_type": evidence.source_type,
                "year": evidence.year,
            }
    
    return citations


def get_claim_citation_ids(claim, url_registry: Dict[int, dict]) -> List[str]:
    """Get all citation IDs for a claim (one per source_evidence). For two-step mode."""
    return [
        generate_citation_id(ev.snippet, ev.source_index)
        for ev in claim.source_evidences
    ]


def build_citation_registry_from_single_call(
    analysis,  # FactCheckAnalysisWithCitations
    url_registry: Dict[int, dict]
) -> Dict[str, dict]:
    """
    Build citation registry from single-call result using Gemini-assigned citation_ids.
    
    Args:
        analysis: FactCheckAnalysisWithCitations from single-call Gemini response
        url_registry: URL registry with base source metadata
    
    Returns:
        Dictionary mapping citation_id to citation metadata
    """
    citations = {}
    
    for claim in analysis.claims:
        for evidence in claim.source_evidences:
            # Use Gemini-assigned citation_id directly
            citation_id = evidence.citation_id
            
            # Get URL metadata from registry
            url_meta = url_registry.get(evidence.source_index, {})
            
            citations[citation_id] = {
                "id": citation_id,
                "claim_text": claim.claim,
                "grounding_status": claim.grounding_status,
                "source_index": evidence.source_index,
                "source_url": url_meta.get("url", ""),
                "source_label": url_meta.get("label", f"Source {evidence.source_index}"),
                "snippet": evidence.snippet,
                "authority": evidence.authority,
                "source_type": evidence.source_type,
                "year": evidence.year,
            }
    
    return citations


# ============================================================================
# Pydantic Models for Structured Output
# ============================================================================

# --- Models for Gemini Fact-Check ---

class SourceEvidence(BaseModel):
    """Evidence from a single source for a claim."""
    source_index: int = Field(description="The source index (1-based) this evidence comes from")
    snippet: str = Field(description="Relevant excerpt/quote from THIS specific source")
    authority: Literal["High", "Medium", "Low"] = Field(
        description="Authority/quality assessment of this specific source"
    )
    source_type: str = Field(
        description="Type of source: 'Market Report', 'Database', 'News', 'Article', 'Research Paper', 'Government', 'Blog', or 'Other'"
    )
    year: Optional[str] = Field(
        default=None,
        description="Publication year if extractable from this source (e.g., '2024', 'Jan 2025')"
    )


class ClaimVerification(BaseModel):
    """A claim verified against one or more sources."""
    claim: str = Field(description="The specific claim being verified")
    grounding_status: Literal["GROUNDED", "PARTIALLY_GROUNDED", "NOT_GROUNDED", "UNVERIFIABLE"] = Field(
        description="GROUNDED=fully supported, PARTIALLY_GROUNDED=partial support, NOT_GROUNDED=contradicted/unsupported, UNVERIFIABLE=no source found"
    )
    source_evidences: List[SourceEvidence] = Field(
        description="Evidence from each supporting source. Each source has its own snippet, authority, type, and year."
    )


class FactCheckAnalysis(BaseModel):
    """Complete fact-check analysis result from Gemini (two-step mode)."""
    reasoning: str = Field(description="Overall synthesis explaining how the analysis was derived from sources")
    confidence_score: int = Field(description="Confidence score from 1-5", ge=1, le=5)
    confidence_label: Literal["Very Low Confidence", "Low Confidence", "Medium Confidence", "High Confidence", "Very High Confidence"] = Field(
        description="Human-readable confidence label"
    )
    claims: List[ClaimVerification] = Field(description="List of claim verifications")


class AnalysisOnlyResult(BaseModel):
    """Result from analysis-only call (two-step approach - Gemini only analyzes)."""
    analysis: FactCheckAnalysis = Field(description="The fact-check analysis")


# --- Models for Single-Call (with Gemini-assigned citation IDs) ---

class SourceEvidenceWithCitation(BaseModel):
    """Evidence from a single source with Gemini-assigned citation ID (single-call mode)."""
    citation_id: str = Field(description="Unique citation ID assigned by you, e.g., 'CIT_001', 'CIT_002'. Use sequential numbering.")
    source_index: int = Field(description="The source index (1-based) this evidence comes from")
    snippet: str = Field(description="Relevant excerpt/quote from THIS specific source")
    authority: Literal["High", "Medium", "Low"] = Field(
        description="Authority/quality assessment of this specific source"
    )
    source_type: str = Field(
        description="Type of source: 'Market Report', 'Database', 'News', 'Article', 'Research Paper', 'Government', 'Blog', or 'Other'"
    )
    year: Optional[str] = Field(
        default=None,
        description="Publication year if extractable from this source (e.g., '2024', 'Jan 2025')"
    )


class ClaimVerificationWithCitations(BaseModel):
    """A claim verified against one or more sources with citation IDs (single-call mode)."""
    claim: str = Field(description="The specific claim being verified")
    grounding_status: Literal["GROUNDED", "PARTIALLY_GROUNDED", "NOT_GROUNDED", "UNVERIFIABLE"] = Field(
        description="GROUNDED=fully supported, PARTIALLY_GROUNDED=partial support, NOT_GROUNDED=contradicted/unsupported, UNVERIFIABLE=no source found"
    )
    source_evidences: List[SourceEvidenceWithCitation] = Field(
        description="Evidence from each supporting source. Each has a unique citation_id."
    )


class FactCheckAnalysisWithCitations(BaseModel):
    """Complete fact-check analysis with citation IDs (single-call mode)."""
    reasoning: str = Field(description="Overall synthesis explaining how the analysis was derived from sources")
    confidence_score: int = Field(description="Confidence score from 1-5", ge=1, le=5)
    confidence_label: Literal["Very Low Confidence", "Low Confidence", "Medium Confidence", "High Confidence", "Very High Confidence"] = Field(
        description="Human-readable confidence label"
    )
    claims: List[ClaimVerificationWithCitations] = Field(description="List of claim verifications with citation IDs")


class SingleCallResult(BaseModel):
    """Result from single-call approach (Gemini assigns citation IDs and uses them in answer)."""
    analysis: FactCheckAnalysisWithCitations = Field(description="The fact-check analysis with citation IDs")
    corrected_answer: str = Field(
        description="The corrected answer with inline citation IDs like [CIT_001], [CIT_002]. Use the exact citation_ids from source_evidences."
    )
    citations_used: List[str] = Field(
        description="List of citation IDs actually used in the corrected_answer, e.g., ['CIT_001', 'CIT_002']"
    )


# --- Models for Formatting LLM (two-step approach) ---

class FormattedOutput(BaseModel):
    """Structured output from formatting LLM with inline citations."""
    formatted_answer: str = Field(
        description="The formatted answer with inline citation IDs like [CIT_abc123]"
    )
    citations_used: List[str] = Field(
        description="List of citation IDs that were used in the formatted_answer"
    )


# --- Legacy model for backward compatibility with UI display ---

class Evidence(BaseModel):
    """Enriched evidence for UI display (combines rule-based + LLM data)."""
    claim: str = Field(description="The specific claim being verified")
    source_id: str = Field(description="Unique source ID (e.g., 'WEB123')")
    source_url: str = Field(description="The URL of the source")
    source_label: str = Field(description="Readable domain label (e.g., 'cbinsights.com')")
    source_type: Optional[str] = Field(default=None, description="Type of source: 'Market Report', 'Database', etc.")
    year: Optional[str] = Field(default=None, description="Publication year")
    authority: Literal["High", "Medium", "Low"] = Field(description="Source quality/reliability")
    grounding_status: Literal["GROUNDED", "PARTIALLY_GROUNDED", "NOT_GROUNDED", "UNVERIFIABLE"] = Field(
        description="Whether the claim is supported by this source"
    )
    snippet: str = Field(description="Relevant excerpt from the source")


# ============================================================================
# Section Presets (Trends, Growth Drivers)
# ============================================================================

SECTION_PRESETS = {
    "Trends Analyzer": {
        "search_prompt": """You are a market research analyst gathering comprehensive trend data.
For the query: {query}

Focus on identifying and documenting:
- Current market trends and momentum
- Emerging trends (next 2-3 years)
- Technology and innovation trends
- Consumer behavior shifts
- Sustainability and ESG trends
- Economic and financial patterns
- Geographic/regional variations
- Future outlook and predictions

Provide specific data points, statistics, and cite your sources clearly.
Structure your response to cover these trend dimensions comprehensively.""",
        
        "format_prompt": """Based on the following research data, create a well-structured trend analysis.

Research Data:
{search_results}

Sources: {sources}

Format the analysis with:
1. Executive Summary (2-3 sentences)
2. Key Trends (organized by category)
3. Supporting data points and statistics
4. Future outlook

Keep the tone professional and analytical. Include specific numbers and timeframes where available."""
    },
    
    "Growth Drivers": {
        "search_prompt": """You are a market research analyst analyzing growth drivers and catalysts.
For the query: {query}

Focus on identifying:
- Primary growth catalysts (top 3-5 high-impact drivers)d
- Demand-side drivers (customer demand, market expansion)
- Supply-side drivers (operational capabilities, infrastructure)
- Economic growth drivers (macro conditions, financial factors)
- Technological growth drivers (innovation, digital transformation)
- Competitive dynamics driving growth
- Future growth potential and projections

Provide quantified impact where possible, cite sources, and explain the mechanisms behind each driver.""",
        
        "format_prompt": """Based on the following research data, create a comprehensive growth drivers analysis.

Research Data:
{search_results}

Sources: {sources}

Format the analysis with:
1. Executive Summary
2. Primary Growth Catalysts (ranked by impact)
3. Detailed breakdown by driver category
4. Growth projections and timeline
5. Key risks to growth trajectory

Be specific with numbers, percentages, and timeframes. Maintain analytical rigor."""
    }
}


# ============================================================================
# API Clients & Functions
# ============================================================================

def call_perplexity(
    query: str,
    system_prompt: str,
    model_name: str = "sonar-pro"
) -> dict:
    """Direct Perplexity API call."""
    api_key = os.getenv("PERPLEXITY_API_KEY")
    if not api_key:
        raise ValueError("PERPLEXITY_API_KEY environment variable not set")
    
    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        "stream": False,
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        return {
            "answer": data["choices"][0]["message"]["content"],
            "sources": data.get("citations", []),
        }
    else:
        raise Exception(f"Perplexity API error: {response.status_code} - {response.text}")


def call_gemini_fact_check(
    answer: str,
    sources: List[str],
    source_registry: Dict[int, dict],
    mode: Literal["single_call", "analysis_only"],
    model_id: str = "gemini-2.5-flash"
) -> dict:
    """
    Call Gemini with URL context for fact-checking.
    
    Args:
        answer: The answer text to verify
        sources: List of source URLs
        source_registry: Pre-built registry mapping indices to source metadata
        mode: "single_call" (analysis + correction) or "analysis_only" (just analysis)
        model_id: Gemini model to use
    
    Returns:
        Dict with "result" (parsed model), "url_metadata", and "source_registry"
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key)
    
    # Build tools
    tools = [{"url_context": {}}]
    
    # Build indexed sources text for prompt (so Gemini knows the indices)
    sources_text = "\n".join(
        f"[{i}] {url} (domain: {source_registry[i]['label']})"
        for i, url in enumerate(sources[:20], 1)
    )
    
    if mode == "single_call":
        prompt = f"""Analyze the following answer for factual accuracy by checking against the provided source URLs.

ANSWER TO VERIFY:
{answer}

INDEXED SOURCE URLs:
{sources_text}

Your task:
1. Identify each distinct claim in the answer
2. For each claim, verify against the sources:
   - Find which source(s) support it
   - For EACH supporting source, provide a separate source_evidence entry with:
     * citation_id: assign a UNIQUE ID like "CIT_001", "CIT_002", etc. (sequential numbering)
     * source_index: the source number (1, 2, 3...)
     * snippet: the relevant quote from THAT specific source
     * authority: quality of THAT source (High/Medium/Low)
     * source_type: type of THAT source (Market Report, News, Database, etc.)
     * year: publication year if extractable from THAT source
   - Determine overall grounding status for the claim:
     * GROUNDED: Claim is fully supported
     * PARTIALLY_GROUNDED: Claim is partially supported
     * NOT_GROUNDED: Claim is contradicted or unsupported
     * UNVERIFIABLE: No source found
3. Provide overall reasoning synthesizing your findings
4. Give a confidence score (1-5)
5. Generate a CORRECTED VERSION of the answer:
   - PRESERVE the original structure, flow, and approximate length
   - Use the EXACT citation_ids you assigned: [CIT_001], [CIT_002], etc.
   - Keep GROUNDED claims with their citation IDs
   - Keep PARTIALLY_GROUNDED claims but add hedging language (e.g., "approximately", "some sources suggest")
   - ONLY remove claims that are clearly NOT_GROUNDED (contradicted by sources) or fabricated
   - Keep transitional text, context, and non-factual statements (they don't need citations)
   - Place citations immediately after the specific facts they support
6. List which citation_ids you used in citations_used

IMPORTANT:
- Each source_evidence MUST have a unique citation_id (CIT_001, CIT_002, CIT_003, etc.)
- Use these SAME citation_ids in the corrected_answer - this ensures each inline citation maps to a specific snippet
- The corrected answer should be SIMILAR IN LENGTH to the original - don't over-trim
- Only remove content that is demonstrably false or unsupported"""
        schema = SingleCallResult
    else:
        prompt = f"""Analyze the following answer for factual accuracy by checking against the provided source URLs.

ANSWER TO VERIFY:
{answer}

INDEXED SOURCE URLs:
{sources_text}

Your task:
1. Identify each distinct claim in the answer
2. For each claim, verify against the sources:
   - Find which source(s) support it
   - For EACH supporting source, provide a separate source_evidence entry with:
     * source_index: the source number (1, 2, 3...)
     * snippet: the relevant quote from THAT specific source
     * authority: quality of THAT source (High/Medium/Low)
     * source_type: type of THAT source (Market Report, News, Database, Article, Research Paper, Government, Blog, Other)
     * year: publication year if extractable from THAT source
   - Determine overall grounding status for the claim:
     * GROUNDED: Claim is fully supported
     * PARTIALLY_GROUNDED: Claim is partially supported
     * NOT_GROUNDED: Claim is contradicted or unsupported
     * UNVERIFIABLE: No source found
3. Provide overall reasoning synthesizing your findings
4. Give a confidence score (1-5)

IMPORTANT: Each source supporting a claim must have its OWN snippet and metadata in source_evidences array.
Be thorough and cite specific evidence from the sources."""
        schema = AnalysisOnlyResult
    
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=tools,
            response_mime_type="application/json",
            response_schema=schema,
        )
    )
    
    # Parse response
    result = schema.model_validate_json(response.text)
    
    # Get URL metadata
    url_metadata = None
    if hasattr(response.candidates[0], 'url_context_metadata'):
        url_metadata = response.candidates[0].url_context_metadata
    
    return {
        "result": result,
        "url_metadata": url_metadata,
        "source_registry": source_registry,
    }


def call_formatter_llm(
    search_results: str,
    sources: List[str],
    format_prompt: str,
    analysis: Optional[FactCheckAnalysis] = None,
    llm_name: str = "GPT-4o"
) -> str:
    """Format the answer using LLM via safe_llm_core factory (baseline - no fact-checking)."""
    
    # Build the prompt
    sources_str = ", ".join(sources[:10]) if sources else "N/A"
    prompt = format_prompt.format(
        search_results=search_results,
        sources=sources_str
    )
    
    # Get LLM
    llm = get_llm(llm_name)
    
    messages = [
        SystemMessage(content="You are a professional market research analyst. Format the provided data into a clear, well-structured analysis."),
        HumanMessage(content=prompt)
    ]
    
    response = llm.invoke(messages)
    return response.content


def call_correction_llm(
    original_answer: str,
    analysis: FactCheckAnalysis,
    url_registry: Dict[int, dict],
    citation_registry: Dict[str, dict],
    llm_name: str = "GPT-4o"
) -> FormattedOutput:
    """
    Generate corrected answer with inline citations based on fact-check analysis.
    Uses citation IDs (each with its own snippet) for precise sourcing.
    
    Args:
        original_answer: The original Perplexity answer
        analysis: Fact-check analysis from Gemini
        url_registry: Mapping of source indices to URL metadata
        citation_registry: Mapping of citation IDs to full citation metadata (including snippets)
        llm_name: LLM to use for formatting
    
    Returns:
        FormattedOutput with formatted_answer and citations_used
    """
    
    # Build citations list with IDs for the LLM
    citations_for_prompt = []
    for claim in analysis.claims:
        if claim.grounding_status in ["GROUNDED", "PARTIALLY_GROUNDED"]:
            status_emoji = "✅" if claim.grounding_status == "GROUNDED" else "⚠️"
            
            # Each source_evidence is a separate citable unit
            for ev in claim.source_evidences:
                cit_id = generate_citation_id(ev.snippet, ev.source_index)
                url_meta = url_registry.get(ev.source_index, {})
                citations_for_prompt.append(f"""
{status_emoji} Citation ID: [{cit_id}]
   Claim: {claim.claim}
   Source: [{ev.source_index}] {url_meta.get('label', 'Unknown')}
   Type: {ev.source_type} | Authority: {ev.authority} | Year: {ev.year or 'Unknown'}
   Snippet: "{ev.snippet[:200]}..."
""")
    
    citations_text = "\n".join(citations_for_prompt)
    
    # Also list claims that should be excluded
    excluded_claims = []
    for claim in analysis.claims:
        if claim.grounding_status in ["NOT_GROUNDED", "UNVERIFIABLE"]:
            status_emoji = "❌" if claim.grounding_status == "NOT_GROUNDED" else "❓"
            excluded_claims.append(f"{status_emoji} {claim.claim} ({claim.grounding_status})")
    
    excluded_text = "\n".join(excluded_claims) if excluded_claims else "None"
    
    prompt = f"""Rewrite the following answer using ONLY verified information with inline citations.

ORIGINAL ANSWER:
{original_answer}

AVAILABLE CITATIONS (use these exact citation IDs):
{citations_text}

CLAIMS TO EXCLUDE OR CORRECT:
{excluded_text}

FACT-CHECK SUMMARY:
Confidence: {analysis.confidence_label} ({analysis.confidence_score}/5)
Reasoning: {analysis.reasoning}

INSTRUCTIONS:
1. PRESERVE the original structure, flow, and approximate LENGTH of the answer
2. Add inline citations using the EXACT citation IDs in brackets, e.g., [CIT_abc123]
3. Place citation immediately after the specific fact it supports
4. Each citation ID corresponds to a specific source+snippet pair
5. If a fact is supported by multiple sources, use multiple citation IDs: [CIT_abc][CIT_def]
6. For PARTIALLY_GROUNDED claims: KEEP them but add hedging language (e.g., "approximately", "some estimates suggest")
7. ONLY remove claims listed as NOT_GROUNDED that are clearly contradicted or fabricated
8. Keep transitional text, context, and non-factual statements (they don't need citations)
9. Maintain professional structure and flow
10. List ALL citation IDs you used in citations_used

IMPORTANT: The corrected answer should be SIMILAR IN LENGTH to the original. Don't over-trim - only remove demonstrably false content.

Example: "The market grew 15% [CIT_a1b2c3], driven by AI adoption [CIT_d4e5f6][CIT_g7h8i9]."

Generate the corrected answer:"""

    # Get LLM with structured output
    llm = get_llm(llm_name)
    structured_llm = llm.with_structured_output(FormattedOutput)
    
    messages = [
        SystemMessage(content="You are a fact-checker and editor. Rewrite content using ONLY the provided citation IDs. Each citation ID maps to a specific source and snippet."),
        HumanMessage(content=prompt)
    ]
    
    response = structured_llm.invoke(messages)
    return response


# ============================================================================
# Conversion Helpers  
# ============================================================================

# Note: claims_to_evidences removed - now using display_claim_card directly with source_evidences


# ============================================================================
# UI Display Functions
# ============================================================================

def display_claim_card(claim, url_registry: Dict[int, dict], index: int):
    """Display a single claim verification card with all supporting sources."""
    # Status styling
    status_config = {
        "GROUNDED": {"color": "green", "icon": "✅", "bg": "#e8f5e9"},
        "PARTIALLY_GROUNDED": {"color": "orange", "icon": "⚠️", "bg": "#fff3e0"},
        "NOT_GROUNDED": {"color": "red", "icon": "❌", "bg": "#ffebee"},
        "UNVERIFIABLE": {"color": "gray", "icon": "❓", "bg": "#f5f5f5"},
    }
    config = status_config.get(claim.grounding_status, status_config["UNVERIFIABLE"])
    
    # Build sources section - each source_evidence gets its own entry with snippet
    sources_html_parts = []
    for ev in claim.source_evidences:
        url_meta = url_registry.get(ev.source_index, {})
        label = url_meta.get("label", f"Source {ev.source_index}")
        url = url_meta.get("url", "")
        
        authority_color = '#2e7d32' if ev.authority == 'High' else '#f57c00' if ev.authority == 'Medium' else '#c62828'
        
        # Check if citation_id is available (single-call mode)
        citation_badge = ""
        if hasattr(ev, 'citation_id') and ev.citation_id:
            citation_badge = f'<span style="background-color: #00897b; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold;">{ev.citation_id}</span>'
        
        # Build HTML without indentation to avoid markdown code block interpretation
        source_html = (
            f'<div style="background-color: #fafafa; padding: 8px; border-radius: 4px; margin-top: 8px; border: 1px solid #e0e0e0;">'
            f'<div style="display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px;">'
            f'{citation_badge}'
            f'<span style="background-color: #1976d2; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">[{ev.source_index}] {label}</span>'
            f'<span style="background-color: #6a1b9a; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{ev.source_type}</span>'
            f'<span style="background-color: #455a64; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{ev.year or "Unknown"}</span>'
            f'<span style="background-color: {authority_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{ev.authority}</span>'
            f'</div>'
            f'<div style="font-size: 12px; color: #555; font-style: italic; margin-bottom: 4px;">'
            f'📝 "{ev.snippet[:250]}{"..." if len(ev.snippet) > 250 else ""}"'
            f'</div>'
            f'<a href="{url}" target="_blank" style="font-size: 11px; color: #1976d2;">🔗 View Source</a>'
            f'</div>'
        )
        sources_html_parts.append(source_html)
    
    sources_html = "".join(sources_html_parts) if sources_html_parts else '<div style="color: #757575; font-size: 12px;">No sources found</div>'
    
    # Build main card HTML without indentation
    card_html = (
        f'<div style="background-color: {config["bg"]}; padding: 12px; border-radius: 8px; margin-bottom: 12px; border-left: 4px solid {config["color"]};">'
        f'<div style="font-weight: 600; margin-bottom: 8px; color: #1a1a1a;">'
        f'{config["icon"]} Claim: {claim.claim}'
        f'</div>'
        f'<div style="font-size: 13px; color: #1a1a1a; font-weight: 500; margin-bottom: 4px;">'
        f'Supporting Sources ({len(claim.source_evidences)}):'
        f'</div>'
        f'{sources_html}'
        f'</div>'
    )
    
    with st.container():
        st.markdown(card_html, unsafe_allow_html=True)


def display_analysis(analysis, source_registry: Dict[int, dict]):
    """Display the full fact-check analysis."""
    # Confidence score display
    stars = "⭐" * analysis.confidence_score + "☆" * (5 - analysis.confidence_score)
    
    confidence_colors = {
        "Very High Confidence": "#2e7d32",
        "High Confidence": "#558b2f",
        "Medium Confidence": "#f57c00",
        "Low Confidence": "#e65100",
        "Very Low Confidence": "#c62828",
    }
    conf_color = confidence_colors.get(analysis.confidence_label, "#757575")
    
    st.markdown(f"""
<div style="background-color: #f5f5f5; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
    <h4 style="margin-top: 0; color: #1a1a1a;">🎯 Reasoning:</h4>
    <p style="color: #333;">{analysis.reasoning}</p>
</div>
""", unsafe_allow_html=True)
    
    st.markdown(f"""
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
    <span style="font-weight: 600; color: #1a1a1a;">Data Confidence Score: {analysis.confidence_score}/5</span>
    <span style="font-size: 20px;">{stars}</span>
    <span style="background-color: {conf_color}; color: white; padding: 4px 12px; border-radius: 16px; font-size: 13px;">
        ✓ {analysis.confidence_label}
    </span>
</div>
""", unsafe_allow_html=True)
    
    # Count by status
    grounded = sum(1 for c in analysis.claims if c.grounding_status == "GROUNDED")
    partial = sum(1 for c in analysis.claims if c.grounding_status == "PARTIALLY_GROUNDED")
    not_grounded = sum(1 for c in analysis.claims if c.grounding_status == "NOT_GROUNDED")
    unverifiable = sum(1 for c in analysis.claims if c.grounding_status == "UNVERIFIABLE")
    
    st.markdown(f"""
<div style="display: flex; gap: 12px; margin-bottom: 16px;">
    <span style="color: #2e7d32;">✅ Grounded: {grounded}</span>
    <span style="color: #f57c00;">⚠️ Partial: {partial}</span>
    <span style="color: #c62828;">❌ Not Grounded: {not_grounded}</span>
    <span style="color: #757575;">❓ Unverifiable: {unverifiable}</span>
</div>
""", unsafe_allow_html=True)
    
    # Claim cards
    st.markdown("### Claims Analysis:")
    for i, claim in enumerate(analysis.claims):
        display_claim_card(claim, source_registry, i)


def display_url_metadata(url_metadata):
    """Display URL retrieval status."""
    if url_metadata and hasattr(url_metadata, 'url_metadata'):
        with st.expander("🔗 Source Retrieval Status"):
            for url_info in url_metadata.url_metadata:
                status = url_info.url_retrieval_status
                url = url_info.retrieved_url
                
                if "SUCCESS" in status:
                    st.success(f"✓ {url}")
                elif "UNSAFE" in status:
                    st.error(f"✗ {url} (unsafe content)")
                else:
                    st.warning(f"? {url} ({status})")


# ============================================================================
# Main App
# ============================================================================

st.title("🔍 Fact-Checking PoC")
st.caption("Compare baseline vs fact-checked flows for hallucination detection")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Section preset
    section_type = st.selectbox(
        "Section Type",
        list(SECTION_PRESETS.keys()),
        index=0
    )
    
    # Fact-check mode
    st.subheader("Fact-Check Mode")
    fact_check_mode = st.radio(
        "Approach",
        ["single_call", "two_step"],
        format_func=lambda x: "Single-call (Gemini all-in-one)" if x == "single_call" else "Two-step (Analysis → Correction)",
        index=1,
        help="Single-call: Gemini does analysis + correction together.\nTwo-step: Gemini analyzes, then LLM corrects."
    )
    
    # Formatter LLM
    formatter_llm = st.selectbox(
        "Formatter LLM",
        ["GPT-4o", "GPT-4o Mini", "GPT-5 Mini", "Claude 4 Sonnet", "Claude 4.5 Sonnet"],
        index=0
    )
    
    # Gemini model for fact-checking
    gemini_model = st.selectbox(
        "Gemini Model (Fact-Check)",
        ["gemini-2.5-pro", "gemini-3-flash-preview"],
        index=0
    )
    
    st.divider()
    
    # Environment check
    st.subheader("🔑 API Keys Status")
    perplexity_ok = bool(os.getenv("PERPLEXITY_API_KEY"))
    gemini_ok = bool(os.getenv("GEMINI_API_KEY"))
    openai_ok = bool(os.getenv("OPENAI_API_KEY"))
    anthropic_ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    st.markdown(f"- Perplexity: {'✅' if perplexity_ok else '❌'}")
    st.markdown(f"- Gemini: {'✅' if gemini_ok else '❌'}")
    st.markdown(f"- OpenAI: {'✅' if openai_ok else '❌'}")
    st.markdown(f"- Anthropic: {'✅' if anthropic_ok else '❌'}")
    
    if not all([perplexity_ok, gemini_ok]):
        st.warning("Perplexity & Gemini keys required. Other keys needed based on Formatter LLM selection.")

# Main content
query = st.text_input(
    "🔎 Enter your query",
    placeholder="e.g., AI healthcare market trends in 2024",
    help="Enter a market research query to analyze"
)

# Show selected preset info
with st.expander("📋 View Selected Prompts"):
    preset = SECTION_PRESETS[section_type]
    st.markdown("**Search Prompt:**")
    st.code(preset["search_prompt"], language="text")
    st.markdown("**Format Prompt:**")
    st.code(preset["format_prompt"], language="text")

# Run button
run_button = st.button("🚀 Run Comparison", type="primary", use_container_width=True)

if run_button:
    if not query.strip():
        st.error("Please enter a query")
    elif not os.getenv("PERPLEXITY_API_KEY"):
        st.error("PERPLEXITY_API_KEY not set")
    elif not os.getenv("GEMINI_API_KEY"):
        st.error("GEMINI_API_KEY not set")
    else:
        preset = SECTION_PRESETS[section_type]
        
        # Initialize timing metrics
        timing_metrics = {}
        
        # ========================================
        # Step 1: Perplexity Search
        # ========================================
        with st.spinner("🔍 Searching with Perplexity..."):
            try:
                search_prompt = preset["search_prompt"].format(query=query)
                
                start_time = time.time()
                perplexity_result = call_perplexity(
                    query=query,
                    system_prompt=search_prompt
                )
                timing_metrics["perplexity_search"] = time.time() - start_time
                
                perplexity_answer = perplexity_result["answer"]
                sources = perplexity_result["sources"]
                
                st.success(f"✅ Perplexity returned {len(sources)} sources ({timing_metrics['perplexity_search']:.2f}s)")
            except Exception as e:
                st.error(f"Perplexity error: {e}")
                st.stop()
        
        # ========================================
        # Step 2: Build Source Registry (Rule-based)
        # ========================================
        source_registry = build_source_registry(sources)
        
        # Show Perplexity raw output with registry info
        with st.expander("📄 Raw Perplexity Output"):
            st.markdown(perplexity_answer)
            st.markdown("**Sources Registry:**")
            for idx, meta in source_registry.items():
                st.markdown(f"[{idx}] **{meta['id']}** - {meta['label']}")
                st.caption(f"   {meta['url']}")
        
        # Create placeholder for timing metrics (will be filled after all steps complete)
        timing_placeholder = st.empty()
        
        # Create two columns for comparison
        col1, col2 = st.columns(2)
        
        # ========================================
        # Column 1: Baseline (No Fact-Check)
        # ========================================
        with col1:
            st.subheader("📊 Baseline")
            st.caption("Perplexity → Formatter → Answer")
            
            with st.spinner("Formatting baseline answer..."):
                try:
                    start_time = time.time()
                    baseline_answer = call_formatter_llm(
                        search_results=perplexity_answer,
                        sources=sources,
                        format_prompt=preset["format_prompt"],
                        llm_name=formatter_llm
                    )
                    timing_metrics["baseline_formatting"] = time.time() - start_time
                    
                    st.markdown("---")
                    st.markdown(baseline_answer)
                    
                except Exception as e:
                    st.error(f"Formatter error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # ========================================
        # Column 2: Fact-Checked
        # ========================================
        with col2:
            st.subheader("✅ Fact-Checked")
            st.caption(f"Perplexity → Gemini → {'Correction' if fact_check_mode == 'single_call' else 'Correction LLM'} → Answer")
            
            # Step 3: Fact-check with Gemini
            with st.spinner("🔍 Fact-checking with Gemini..."):
                try:
                    if fact_check_mode == "single_call":
                        # Single call: Gemini assigns citation IDs and uses them directly
                        start_time = time.time()
                        fc_result = call_gemini_fact_check(
                            answer=perplexity_answer,
                            sources=sources,
                            source_registry=source_registry,
                            mode="single_call",
                            model_id=gemini_model
                        )
                        timing_metrics["single_call_total"] = time.time() - start_time
                        
                        analysis = fc_result["result"].analysis
                        url_metadata = fc_result["url_metadata"]
                        
                        # Build citation registry using Gemini-assigned citation_ids
                        citation_registry = build_citation_registry_from_single_call(analysis, source_registry)
                        
                        # Gemini already uses citation IDs in the answer - use directly
                        corrected_answer = fc_result["result"].corrected_answer
                        citations_used = set(fc_result["result"].citations_used)
                        
                    else:
                        # Two-step: analysis first, then correction LLM
                        start_time = time.time()
                        fc_result = call_gemini_fact_check(
                            answer=perplexity_answer,
                            sources=sources,
                            source_registry=source_registry,
                            mode="analysis_only",
                            model_id=gemini_model
                        )
                        timing_metrics["two_step_analysis"] = time.time() - start_time
                        
                        analysis = fc_result["result"].analysis
                        url_metadata = fc_result["url_metadata"]
                        
                        # Build citation registry from claims
                        citation_registry = build_citation_registry(analysis.claims, source_registry)
                        
                        # Now call correction LLM with citation IDs
                        with st.spinner("📝 Generating corrected answer with citations..."):
                            start_time = time.time()
                            formatted_result = call_correction_llm(
                                original_answer=perplexity_answer,
                                analysis=analysis,
                                url_registry=source_registry,
                                citation_registry=citation_registry,
                                llm_name=formatter_llm
                            )
                            timing_metrics["two_step_correction"] = time.time() - start_time
                            
                            corrected_answer = formatted_result.formatted_answer
                            citations_used = set(formatted_result.citations_used)
                    
                    # Display corrected answer with inline citations
                    st.markdown("---")
                    st.markdown("### 📝 Corrected Answer")
                    st.markdown(corrected_answer)
                    
                    # Show citations used with their snippets
                    with st.expander("📚 Citations Used"):
                        for cit_id in sorted(citations_used):
                            if cit_id in citation_registry:
                                cit = citation_registry[cit_id]
                                st.markdown(f"""
**[{cit_id}]** - {cit.get('source_label', 'Unknown')}
- Type: {cit.get('source_type', 'Unknown')} | Authority: {cit.get('authority', 'Unknown')} | Year: {cit.get('year', 'Unknown')}
- Snippet: _{cit.get('snippet', '')[:200]}..._
- [View Source]({cit.get('source_url', '')})
---
""")
                    
                    st.markdown("---")
                    
                    # Display analysis with source registry for enrichment
                    st.markdown("### 📊 Claim Analysis")
                    display_analysis(analysis, source_registry)
                    
                    # Display URL metadata
                    display_url_metadata(url_metadata)
                    
                except Exception as e:
                    st.error(f"Fact-check error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # ========================================
        # Timing Metrics Summary (rendered in placeholder at top)
        # ========================================
        with timing_placeholder.container():
            st.subheader("⏱️ Latency Metrics")
            
            # Build metrics display
            metrics_cols = st.columns(4)
            
            with metrics_cols[0]:
                perplexity_time = timing_metrics.get("perplexity_search", 0)
                st.metric("🔍 Perplexity Search", f"{perplexity_time:.2f}s")
            
            with metrics_cols[1]:
                baseline_time = timing_metrics.get("baseline_formatting", 0)
                st.metric("📊 Baseline Formatting", f"{baseline_time:.2f}s")
            
            with metrics_cols[2]:
                if fact_check_mode == "single_call":
                    single_call_time = timing_metrics.get("single_call_total", 0)
                    st.metric("⚡ Single-Call (Total)", f"{single_call_time:.2f}s")
                else:
                    analysis_time = timing_metrics.get("two_step_analysis", 0)
                    st.metric("🔬 Claim Analysis", f"{analysis_time:.2f}s")
            
            with metrics_cols[3]:
                if fact_check_mode == "single_call":
                    # Show overhead comparison
                    baseline_time = timing_metrics.get("baseline_formatting", 0)
                    single_call_time = timing_metrics.get("single_call_total", 0)
                    overhead = single_call_time - baseline_time if baseline_time > 0 else single_call_time
                    st.metric("📈 Fact-Check Overhead", f"+{overhead:.2f}s")
                else:
                    correction_time = timing_metrics.get("two_step_correction", 0)
                    st.metric("✏️ Answer Correction", f"{correction_time:.2f}s")
            
            # Detailed breakdown in expander
            with st.expander("📊 Detailed Timing Breakdown"):
                perplexity_time = timing_metrics.get("perplexity_search", 0)
                baseline_time = timing_metrics.get("baseline_formatting", 0)
                
                st.markdown("#### Common Steps")
                st.markdown(f"- **Perplexity Search**: {perplexity_time:.2f}s")
                
                st.markdown("#### Baseline Flow")
                baseline_total = perplexity_time + baseline_time
                st.markdown(f"- **Formatting**: {baseline_time:.2f}s")
                st.markdown(f"- **Total**: **{baseline_total:.2f}s**")
                
                st.markdown("#### Fact-Checked Flow")
                if fact_check_mode == "single_call":
                    single_call_time = timing_metrics.get("single_call_total", 0)
                    fc_total = perplexity_time + single_call_time
                    st.markdown(f"- **Gemini Single-Call** (analysis + correction): {single_call_time:.2f}s")
                    st.markdown(f"- **Total**: **{fc_total:.2f}s**")
                    
                    overhead_pct = ((fc_total - baseline_total) / baseline_total * 100) if baseline_total > 0 else 0
                    st.markdown(f"- **Overhead vs Baseline**: +{fc_total - baseline_total:.2f}s ({overhead_pct:.1f}%)")
                else:
                    analysis_time = timing_metrics.get("two_step_analysis", 0)
                    correction_time = timing_metrics.get("two_step_correction", 0)
                    two_step_total = analysis_time + correction_time
                    fc_total = perplexity_time + two_step_total
                    
                    st.markdown(f"- **Gemini Analysis**: {analysis_time:.2f}s")
                    st.markdown(f"- **Correction LLM**: {correction_time:.2f}s")
                    st.markdown(f"- **Fact-Check Subtotal**: {two_step_total:.2f}s")
                    st.markdown(f"- **Total**: **{fc_total:.2f}s**")
                    
                    overhead_pct = ((fc_total - baseline_total) / baseline_total * 100) if baseline_total > 0 else 0
                    st.markdown(f"- **Overhead vs Baseline**: +{fc_total - baseline_total:.2f}s ({overhead_pct:.1f}%)")
            
            st.markdown("---")

