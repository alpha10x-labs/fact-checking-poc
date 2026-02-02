# Fact-Checking / Hallucination Detection — Technical Discovery

> **Status**: Discovery  
> **Goal**: Explore approaches to verify that claims from external sources (Perplexity, document-based answers) are grounded in their sources.

---

## The Problem

We use external tools like Perplexity to get answers. Perplexity is essentially web search + LLM answer generation. The LLM can hallucinate — it might generate claims that aren't actually supported by the sources it cites.

**Fact-checking here means**: Verifying that the claims in an answer are grounded in the provided sources.

We can also go further with **full fact-checking**: verifying claims against the broader web, not just the provided sources.

---

## Input and Output

**Input**:
- Answer (containing one or multiple claims)
- Sources (URLs) — these should always be provided

**Output options**:
- **Hallucination analysis**: Is each claim grounded? With evidence and confidence.
- **Corrected claims**: Rewritten claims with accurate information from sources. This is likely the most common output we'll need.

---

## The Tooling Landscape

Before diving into pipelines, here's what's available:

### Search APIs
| Tool | What it does |
|------|--------------|
| Exa | Neural/semantic search, finds pages by meaning |
| Tavily | AI-native search, designed for agents |
| Perplexity | Web search + answer generation |

### URL to Content
| Tool | What it does |
|------|--------------|
| Exa Content | Gets page content from URLs |
| Tavily Extract | Extracts clean content from URLs |
| Jina Reader | Converts any URL to clean markdown |

### Answer APIs
| Tool | What it does |
|------|--------------|
| Exa Answer | Ask a question, get answer with sources |
| Perplexity | Ask a question, get answer with sources |

### Verification / LLM
| Tool | What it does |
|------|--------------|
| Gemini URL Context | Gemini reads URLs directly, can verify claims |
| Gemini + Google Search | Full fact-checking with web access |
| Cerebras / Groq | Ultra-fast LLM inference |

---

## Pipeline Options

### Option A: Gemini URL Context

The simplest approach. Give Gemini the answer, claims, and source URLs. Gemini reads the URLs directly and verifies.

```
Answer + Source URLs → Gemini (URL Context) → Analysis / Corrected Claims
```

**How it works**:
- Gemini fetches and reads the source URLs itself
- You prompt it to analyze claims and/or provide corrections
- Can optionally add Google Search for full fact-checking beyond provided sources

**Considerations**:

| Aspect | Notes |
|--------|-------|
| Page access | Google likely has better access than our own scrapers (cached pages, potentially some restricted content) |
| Simplicity | Single API call |
| URL limit | Maximum 20 URLs per request |
| Transparency | Black box — we don't see what Gemini extracted from the pages |
| Output flexibility | Can do analysis, corrections, or both via prompting |

---

### Option B: Custom Pipeline (Scrape + Fast LLM)

We scrape the source content ourselves, then use a fast LLM (Cerebras or Groq) for verification.

```
Source URLs → Scrape (Exa/Tavily/Jina) → Page Content → Fast LLM → Output
```

**How it works**:
- Use Exa Content, Tavily Extract, or Jina Reader to get page content
- Feed the content + claims to Cerebras or Groq for verification
- Full control over both steps

**Two approaches for claims handling**:

**B1. Batch verification (all claims together)**
```
All Claims + All Page Content → LLM → Analysis / Corrections for all
```
- Simpler, fewer API calls
- Less granular (harder to pinpoint which specific claim failed)

**B2. Individual claim verification**
```
Answer → Claims Extraction (LLM) → Individual Claims
                                         ↓
                    For each claim: Claim + Relevant Content → LLM → Result
                                         ↓
                              Aggregate results
```
- More precise ("claim 3 is not grounded")
- More API calls, more complex
- Requires a claims extraction step first

**Considerations**:

| Aspect | Notes |
|--------|-------|
| Page access | Limited to what our scrapers can access. Paywalled or restricted pages may fail. |
| Control | Full visibility — we see exactly what was scraped |
| Speed | Scraping adds latency, but Cerebras/Groq are very fast |
| URL limit | No limit |
| Quality | Scraping quality varies — some pages may not extract well |

---

## Gemini vs Custom Pipeline: Key Comparison

This is likely the central decision for the team.

| Dimension | Gemini URL Context | Custom Pipeline |
|-----------|-------------------|-----------------|
| Simplicity | Single API call | Multi-step |
| Page access | Likely better (Google's infrastructure) | Limited by scraper capabilities |
| Transparency | Black box | Full visibility |
| URL limits | 20 max | No limit |
| Speed | Gemini latency | Scraping + fast LLM |
| Control | Via prompting | Full control over logic |
| Failure mode | Gemini misreads or can't access page | Scraper fails or extracts poorly |

**Open question**: How much better is Gemini's page access really? This is worth testing empirically.

---

## The Meta-Problem: LLMs Verifying LLMs

We're using an LLM to verify another LLM's output. The verifier itself could hallucinate its verification — confidently saying a claim is grounded when it isn't.

Worth noting: Gemini provides grounding metadata in its responses, which gives some transparency into what it found.

Possible mitigations: require specific quotes from sources, use multiple verification methods, apply confidence thresholds, or human review for high-stakes cases.

---

## Full Fact-Checking (Beyond Provided Sources)

Sometimes we may want to verify against the broader web, not just the sources Perplexity provided.

**Options**:
- Gemini URL Context + Google Search tool (Gemini searches the web itself)
- Exa Answer (ask Exa to verify, it searches and provides its own sources)
- Exa Search + verification pipeline
- OpenAI or Grok with web search and domain filtering

This adds another layer but provides independent verification.

---

## Open Questions for Discussion

1. **Gemini vs Custom Pipeline**: Which path should we prototype first? Is the page access advantage of Gemini significant enough?

2. **Corrected claims format**: How should corrections be presented? Inline replacements? Side-by-side comparison?

3. **Latency budget**: How fast does verification need to be? This affects whether we can afford multi-step pipelines.

4. **Failure handling**: What do we do when scraping fails or Gemini can't access a URL? Fallback? Flag as unverifiable?

5. **Integration point**: When in the Codex flow does fact-checking happen? Before showing to user? As a separate step?
