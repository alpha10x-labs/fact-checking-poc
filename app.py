import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Literal

st.set_page_config(page_title="Fact Checker", page_icon="🔍", layout="wide")

st.title("Fact Checker")
st.caption("Verify claims against sources using Gemini")


# Pydantic models for structured output
class ClaimAnalysis(BaseModel):
    claim: str = Field(description="The original claim being analyzed")
    status: Literal["GROUNDED", "PARTIALLY_GROUNDED", "NOT_GROUNDED"] = Field(
        description="Whether the claim is supported by the sources"
    )
    evidence: str = Field(description="Specific evidence from sources that supports or contradicts the claim")
    confidence: Literal["High", "Medium", "Low"] = Field(description="Confidence level in the assessment")


class AnalysisResult(BaseModel):
    claims: List[ClaimAnalysis] = Field(description="Analysis of each claim")
    summary: str = Field(description="Brief overall summary of the fact-check")


class CorrectionResult(BaseModel):
    corrected_text: str = Field(description="The corrected version of the text with accurate information")
    changes_made: str = Field(description="Brief description of what was corrected")

# Sidebar for settings
with st.sidebar:
    st.header("Settings")
    
    api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.get("api_key", ""))
    if api_key:
        st.session_state.api_key = api_key
    
    model_id = st.selectbox(
        "Model",
        ["gemini-2.5-flash", "gemini-3-flash-preview", "gemini-2.5-flash-lite"],
        index=0
    )
    
    full_fact_check = st.toggle(
        "Full Fact-Checking",
        value=False,
        help="Enable Google Search to verify beyond provided sources"
    )
    
    output_type = st.radio(
        "Output Type",
        ["Analysis", "Corrected Claims"],
        index=0
    )

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    
    claims = st.text_area(
        "Claims to verify",
        height=150,
        placeholder="Enter the claims or answer you want to fact-check..."
    )
    
    sources = st.text_area(
        "Source URLs (one per line)",
        height=100,
        placeholder="https://example.com/article1\nhttps://example.com/article2"
    )
    
    verify_button = st.button("Verify", type="primary", use_container_width=True)

with col2:
    st.subheader("Result")
    result_container = st.container()

# Build the prompt based on output type
def build_prompt(claims: str, sources: list[str], output_type: str) -> str:
    sources_text = "\n".join(f"- {url}" for url in sources)
    
    if output_type == "Analysis":
        return f"""Analyze the following claims for factual accuracy based on the provided sources.

Claims:
{claims}

Sources:
{sources_text}

Identify each distinct claim and analyze whether it is grounded in the sources."""

    else:  # Corrected Claims
        return f"""Verify the following text against the provided sources. If any information is inaccurate, provide a corrected version.

Text:
{claims}

Sources:
{sources_text}

Return the corrected text (with accurate information from the sources) and briefly note what was changed."""


def verify_claims(claims: str, sources: list[str], model_id: str, full_fact_check: bool, output_type: str, api_key: str):
    """Call Gemini with URL context to verify claims using structured output."""
    
    client = genai.Client(api_key=api_key)
    
    # Build tools list
    tools = [{"url_context": {}}]
    if full_fact_check:
        tools.append({"google_search": {}})
    
    prompt = build_prompt(claims, sources, output_type)
    
    # Choose schema based on output type
    schema = AnalysisResult if output_type == "Analysis" else CorrectionResult
    
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
    
    # Parse structured response
    result = schema.model_validate_json(response.text)
    
    # Extract URL metadata
    url_metadata = None
    if hasattr(response.candidates[0], 'url_context_metadata'):
        url_metadata = response.candidates[0].url_context_metadata
    
    return result, url_metadata


def display_analysis_result(result: AnalysisResult):
    """Display analysis result in a structured way."""
    st.markdown(f"**Summary:** {result.summary}")
    st.divider()
    
    for i, claim in enumerate(result.claims, 1):
        with st.container(border=True):
            st.markdown(f"**Claim {i}:** {claim.claim}")
            
            # Status badge
            status_colors = {
                "GROUNDED": "green",
                "PARTIALLY_GROUNDED": "orange", 
                "NOT_GROUNDED": "red"
            }
            status_labels = {
                "GROUNDED": "✓ Grounded",
                "PARTIALLY_GROUNDED": "~ Partially Grounded",
                "NOT_GROUNDED": "✗ Not Grounded"
            }
            
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"**Status:** :{status_colors[claim.status]}[{status_labels[claim.status]}]")
            with col2:
                st.markdown(f"**Confidence:** {claim.confidence}")
            
            st.markdown(f"**Evidence:** {claim.evidence}")


def display_correction_result(result: CorrectionResult):
    """Display correction result in a structured way."""
    st.markdown("**Corrected Text:**")
    st.markdown(result.corrected_text)
    st.divider()
    st.caption(f"**Changes:** {result.changes_made}")


# Handle verification
if verify_button:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not claims.strip():
        st.error("Please enter claims to verify.")
    elif not sources.strip():
        st.error("Please enter at least one source URL.")
    else:
        # Parse sources
        source_urls = [url.strip() for url in sources.strip().split("\n") if url.strip()]
        
        if len(source_urls) > 20:
            st.error("Maximum 20 URLs allowed per request.")
        else:
            with result_container:
                with st.spinner("Verifying claims..."):
                    try:
                        result, url_metadata = verify_claims(
                            claims=claims,
                            sources=source_urls,
                            model_id=model_id,
                            full_fact_check=full_fact_check,
                            output_type=output_type,
                            api_key=api_key
                        )
                        
                        # Display structured result
                        if output_type == "Analysis":
                            display_analysis_result(result)
                        else:
                            display_correction_result(result)
                        
                        # Show URL retrieval status
                        if url_metadata and hasattr(url_metadata, 'url_metadata'):
                            with st.expander("Source Retrieval Status"):
                                for url_info in url_metadata.url_metadata:
                                    status = url_info.url_retrieval_status
                                    url = url_info.retrieved_url
                                    
                                    if "SUCCESS" in status:
                                        st.success(f"✓ {url}")
                                    elif "UNSAFE" in status:
                                        st.error(f"✗ {url} (unsafe content)")
                                    else:
                                        st.warning(f"? {url} ({status})")
                        
                        if full_fact_check:
                            st.info("Full fact-checking was enabled (Google Search)")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

