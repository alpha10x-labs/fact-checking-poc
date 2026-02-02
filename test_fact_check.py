"""
Test script for Gemini URL Context fact-checking.
Run this independently to verify the fact-checking works before using in the app.
"""

import os
from pathlib import Path

# Load .env from project root
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Literal

# ============================================================================
# Pydantic Models (same as poc_app.py)
# ============================================================================

class Evidence(BaseModel):
    """Individual evidence supporting or contradicting a claim."""
    claim: str = Field(description="The specific claim being verified")
    source_name: str = Field(
        description="Name of the source (e.g., 'CB Insights', 'Crunchbase')"
    )
    source_url: str = Field(description="The URL of the source")
    year: str = Field(
        description="Publication year or recency (e.g., '2023', 'Jan 2026')"
    )
    source_type: str = Field(
        description="Type of source: Market Report, Database, Article, News"
    )
    authority: Literal["High", "Medium", "Low"] = Field(
        description="Source quality/reliability"
    )
    grounding_status: Literal[
        "GROUNDED", "PARTIALLY_GROUNDED", "NOT_GROUNDED"
    ] = Field(description="Whether the claim is supported by this source")
    snippet: str = Field(
        description="Relevant excerpt from the source"
    )


class FactCheckAnalysis(BaseModel):
    """Complete fact-check analysis result."""
    reasoning: str = Field(
        description="Overall synthesis explaining how the analysis was derived"
    )
    confidence_score: int = Field(
        description="Confidence score from 1-5", ge=1, le=5
    )
    confidence_label: Literal[
        "Very Low Confidence",
        "Low Confidence",
        "Medium Confidence",
        "High Confidence",
        "Very High Confidence"
    ] = Field(description="Human-readable confidence label")
    evidences: List[Evidence] = Field(
        description="List of evidence items for each claim"
    )


class AnalysisOnlyResult(BaseModel):
    """Result from analysis-only call."""
    analysis: FactCheckAnalysis = Field(description="The fact-check analysis")


class SingleCallResult(BaseModel):
    """Result from single-call approach (analysis + correction)."""
    analysis: FactCheckAnalysis = Field(description="The fact-check analysis")
    corrected_answer: str = Field(
        description="The corrected/verified answer"
    )


# ============================================================================
# Test Function
# ============================================================================

def test_fact_check(
    answer: str,
    sources: List[str],
    mode: str = "analysis_only",
    model_id: str = "gemini-2.5-flash"
):
    """Test the Gemini fact-check call."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    print(f"Testing fact-check with model: {model_id}")
    print(f"Mode: {mode}")
    print(f"Sources: {sources}")
    print("-" * 50)

    client = genai.Client(api_key=api_key)
    tools = [{"url_context": {}}]
    sources_text = "\n".join(f"- {url}" for url in sources[:20])

    if mode == "single_call":
        prompt = f"""Analyze the following answer for factual accuracy.

ANSWER TO VERIFY:
{answer}

SOURCE URLs:
{sources_text}

Your task:
1. Identify each distinct claim in the answer
2. For each claim, verify against the sources
3. Provide an overall confidence score and reasoning
4. Generate a corrected version of the answer

Be thorough and cite specific evidence from the sources."""
        schema = SingleCallResult
    else:
        prompt = f"""Analyze the following answer for factual accuracy.

ANSWER TO VERIFY:
{answer}

SOURCE URLs:
{sources_text}

Your task:
1. Identify each distinct claim in the answer
2. For each claim, verify against the sources
3. Provide overall reasoning synthesizing your findings
4. Give a confidence score (1-5) based on how well grounded

Be thorough and cite specific evidence from the sources."""
        schema = AnalysisOnlyResult

    print("Calling Gemini API...")
    try:
        response = client.models.generate_content(
            model=model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=tools,
                response_mime_type="application/json",
                response_schema=schema,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            )
        )

        print("✅ API call successful!")
        print("-" * 50)

        # Parse response
        result = schema.model_validate_json(response.text)
        print("✅ Response parsed successfully!")
        print("-" * 50)

        # Display results
        print(f"Reasoning: {result.analysis.reasoning}")
        print(f"Confidence: {result.analysis.confidence_score}/5")
        print(f"Label: {result.analysis.confidence_label}")
        print(f"Number of evidences: {len(result.analysis.evidences)}")

        for i, ev in enumerate(result.analysis.evidences, 1):
            print(f"\n--- Evidence {i} ---")
            print(f"  Claim: {ev.claim[:100]}...")
            print(f"  Status: {ev.grounding_status}")
            print(f"  Source: {ev.source_name} ({ev.authority})")
            print(f"  Snippet: {ev.snippet[:100]}...")

        if mode == "single_call":
            print(f"\n--- Corrected Answer ---")
            print(result.corrected_answer[:500])

        # Check URL metadata
        if hasattr(response.candidates[0], 'url_context_metadata'):
            url_meta = response.candidates[0].url_context_metadata
            if url_meta and hasattr(url_meta, 'url_metadata'):
                print("\n--- URL Retrieval Status ---")
                for url_info in url_meta.url_metadata:
                    status = url_info.url_retrieval_status
                    url = url_info.retrieved_url
                    status_icon = "✅" if "SUCCESS" in status else "❌"
                    print(f"  {status_icon} {url}")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Test with real URLs and a claim to verify
    test_answer = """
    The global AI healthcare market is experiencing rapid growth.
    According to recent reports, the market was valued at approximately
    $15 billion in 2023 and is projected to reach $188 billion by 2030,
    growing at a CAGR of over 37%. Key drivers include increased adoption
    of AI in diagnostics, drug discovery, and personalized medicine.
    Major players include Google Health, IBM Watson Health, and NVIDIA.
    """

    test_sources = [
        "https://www.grandviewresearch.com/industry-analysis/artificial-intelligence-ai-healthcare-market",
        "https://www.marketsandmarkets.com/Market-Reports/artificial-intelligence-healthcare-market-54679303.html",
    ]

    print("=" * 60)
    print("FACT-CHECK TEST")
    print("=" * 60)
    print(f"\nAnswer to verify:\n{test_answer}")
    print(f"\nSources: {test_sources}")
    print("=" * 60)

    result = test_fact_check(
        answer=test_answer,
        sources=test_sources,
        mode="analysis_only",  # Try "single_call" for full test
        model_id="gemini-3-flash-preview"
    )

    if result:
        print("\n" + "=" * 60)
        print("TEST PASSED ✅")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("TEST FAILED ❌")
        print("=" * 60)

