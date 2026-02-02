# Fact-Checking PoC

A Streamlit application that validates AI-generated responses using Gemini URL Context for hallucination detection and claim correction.

## Features

- **Two-Flow Comparison:**
  1. **Baseline:** Perplexity → Answer Formatter → Final Answer
  2. **Fact-Checked:** Perplexity → Gemini Fact-Check → Answer Formatter → Final Answer + Analysis

- **Customizable Prompts:** Pre-configured templates for market research queries
- **Multiple LLM Support:** Choose from GPT-4o, GPT-4o Mini, Claude models, and more
- **Source Verification:** Automatic fact-checking with source citations

## Setup

### Local Development

1. **Clone the repository:**
```bash
git clone https://github.com/alpha10x-labs/fact-checking-poc.git
cd fact-checking-poc
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure secrets:**
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
   - Add your API keys to `.streamlit/secrets.toml`

5. **Run the app:**
```bash
streamlit run poc_app.py
```

### Streamlit Cloud Deployment

1. **Fork/push this repository to GitHub**

2. **Deploy on Streamlit Cloud:**
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Click "New app"
   - Select your repository and branch
   - Set main file path: `poc_app.py`

3. **Configure secrets:**
   - In your app dashboard, go to Settings → Secrets
   - Add your API keys in TOML format (see `.streamlit/secrets.toml.example`)
   - Required keys:
     - `PERPLEXITY_API_KEY`
     - `GEMINI_API_KEY`
   - Optional keys (based on LLM selection):
     - `OPENAI_API_KEY`
     - `ANTHROPIC_API_KEY`
     - `AZURE_OPENAI_API_KEY`
     - `AZURE_OPENAI_ENDPOINT`

## Required API Keys

- **Perplexity API:** Get your key from [perplexity.ai](https://www.perplexity.ai/)
- **Gemini API:** Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- **OpenAI/Azure OpenAI:** (Optional) For GPT models
- **Anthropic:** (Optional) For Claude models

## Usage

1. Enter your research query
2. Select the section type (Market Size, Growth Drivers, etc.)
3. Choose your formatter LLM
4. Click "Run Comparison" to see both baseline and fact-checked results
5. Review the analysis, corrections, and source citations

## Project Structure

```
fact-checking-poc/
├── poc_app.py              # Main Streamlit application
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── secrets.toml.example  # Example secrets configuration
├── .gitignore
└── README.md
```

## License

MIT

