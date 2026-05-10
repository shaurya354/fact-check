# 📄 Fact-Checking Web App

A production-ready web application that automatically verifies factual claims within PDF documents using real-time web search and LLM-based analysis.

## Features

- **PDF Upload & Processing**: Extract text from single or multi-page PDF documents
- **Automated Claim Extraction**: Identify verifiable claims (statistics, dates, financial figures, technical facts) using GPT-4/GPT-4o-mini
- **Real-Time Web Search**: Gather evidence from multiple independent sources using Tavily API
- **Intelligent Verification**: Compare claims against evidence with confidence scoring (0-100)
- **Intuitive UI**: Color-coded results with visual indicators (✔ Verified, ⚠ Inaccurate, ✖ False)
- **Export Functionality**: Download verification reports in CSV or JSON format
- **Robust Error Handling**: Graceful degradation with clear user feedback

## Technology Stack

- **Frontend**: Streamlit
- **PDF Processing**: PyMuPDF (with pdfplumber fallback)
- **LLM**: NVIDIA API (meta/llama-3.3-70b-instruct)
- **Web Search**: Tavily API
- **Testing**: pytest, Hypothesis (property-based testing)
- **Python**: 3.10+

## API Configuration

### Current Implementation: NVIDIA API

This application currently uses the **NVIDIA API** with the `meta/llama-3.3-70b-instruct` model for claim extraction and verification.

**Why NVIDIA API?**
- **Free tier limitations**: Google Gemini API has a strict limit of ~20 requests, which is insufficient for processing multiple PDFs
- **Cost-effective development**: NVIDIA API provides a generous free tier suitable for development and testing
- **Good performance**: Llama 3.3 70B delivers reliable results for fact-checking tasks

### Production Deployment Recommendation

⚠️ **For production deployment, we recommend using the paid OpenAI API** for the following reasons:

1. **Better accuracy**: GPT-4 and GPT-4o models provide superior reasoning and fact-checking capabilities
2. **Structured outputs**: OpenAI's Structured Outputs feature ensures 100% reliable JSON parsing
3. **Higher rate limits**: Paid tiers offer significantly higher request limits for production workloads
4. **Better reliability**: Enterprise-grade SLA and uptime guarantees
5. **Advanced features**: Function calling, vision capabilities, and fine-tuning options

**Switching to OpenAI API:**

The codebase is designed to support multiple LLM providers. To switch to OpenAI:

1. Update `src/nvidia_client.py` to use OpenAI client (commented code available in the file)
2. Set `OPENAI_API_KEY` environment variable instead of `NVIDIA_API_KEY`
3. Update model references to `gpt-4o-mini` or `gpt-4o`

**Estimated OpenAI API costs** (production):
- **GPT-4o-mini**: ~$0.01-0.05 per document (10-15 claims)
- **GPT-4o**: ~$0.05-0.15 per document (10-15 claims)

The application architecture supports easy API provider switching without major code changes.

## Installation

### Prerequisites

- Python 3.10 or higher
- NVIDIA API key ([Get one here](https://build.nvidia.com/))
- Tavily API key ([Get one here](https://tavily.com/))

> **Note**: For production deployment, consider using OpenAI API instead. See [API Configuration](#api-configuration) section for details.

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fact-checking-web-app
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```
   NVIDIA_API_KEY=your_nvidia_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```
   
   > **Production Note**: For production deployment, use `OPENAI_API_KEY` instead of `NVIDIA_API_KEY` and update the client configuration. See [API Configuration](#api-configuration) for details.

5. **Run the application**:
   ```bash
   streamlit run src/main.py
   ```

6. **Open your browser** to `http://localhost:8501`

## Usage

1. **Upload a PDF**: Click the file uploader and select a PDF document
2. **Extract Text**: The system automatically extracts text from all pages
3. **Extract Claims**: Click "Extract Claims" to identify verifiable statements
4. **Verify Claims**: Click "Verify Claims" to search for evidence and verify each claim
5. **Review Results**: View color-coded verification results with confidence scores
6. **Export Report**: Download results in CSV or JSON format

## Project Structure

```
fact-checking-web-app/
├── src/
│   ├── main.py              # Streamlit UI application
│   ├── pdf_processor.py     # PDF text extraction
│   ├── claim_extractor.py   # LLM-based claim identification
│   ├── web_search.py        # Web search API integration
│   ├── verifier.py          # Claim verification logic
│   ├── models.py            # Data models
│   └── utils.py             # Utilities and helpers
├── tests/
│   ├── property_tests/      # Property-based tests
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── data/                    # Sample PDFs
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Deployment

This application can be deployed to Streamlit Cloud or Render with minimal configuration.

### Prerequisites for Deployment

- GitHub repository with your code
- NVIDIA API key ([Get one here](https://build.nvidia.com/)) or OpenAI API key for production ([Get one here](https://platform.openai.com/api-keys))
- Tavily API key ([Get one here](https://tavily.com/))

> **Production Recommendation**: Use OpenAI API (GPT-4o or GPT-4o-mini) for better accuracy and reliability. See [API Configuration](#api-configuration) section.

### Option 1: Streamlit Cloud (Recommended)

Streamlit Cloud offers free hosting for Streamlit apps with easy deployment.

1. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Go to [Streamlit Cloud](https://streamlit.io/cloud)** and sign in with GitHub

3. **Create a new app**:
   - Click "New app"
   - Repository: Select your GitHub repository
   - Branch: `main`
   - Main file path: `src/main.py`

4. **Configure secrets** (Settings → Secrets):
   ```toml
   NVIDIA_API_KEY = "your_nvidia_api_key_here"
   TAVILY_API_KEY = "your_tavily_api_key_here"
   ```
   
   > **Production**: Use `OPENAI_API_KEY` instead of `NVIDIA_API_KEY` for better results. Update the client configuration accordingly.

5. **Deploy**: Click "Deploy" and wait for the app to start (usually 2-3 minutes)

6. **Access your app**: You'll get a URL like `https://your-app-name.streamlit.app`

**Streamlit Cloud Features:**
- ✅ Free tier available
- ✅ Automatic HTTPS
- ✅ Auto-redeploy on git push
- ✅ Built-in secrets management
- ✅ No server configuration needed

### Option 2: Render

Render provides free web service hosting with automatic deployments.

1. **Ensure `render.yaml` exists** (already included in this repository)

2. **Push your code to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

3. **Go to [Render Dashboard](https://dashboard.render.com/)** and sign up/sign in

4. **Create a new Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml`
   - Or manually configure:
     - Name: `fact-checking-web-app`
     - Environment: `Python 3`
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `streamlit run src/main.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`

5. **Add environment variables** (Environment tab):
   - `NVIDIA_API_KEY`: Your NVIDIA API key (or `OPENAI_API_KEY` for production)
   - `TAVILY_API_KEY`: Your Tavily API key
   - `PYTHON_VERSION`: `3.10.0`
   
   > **Production**: Replace `NVIDIA_API_KEY` with `OPENAI_API_KEY` and update the client configuration for better accuracy.

6. **Deploy**: Click "Create Web Service" and wait for deployment (5-10 minutes)

7. **Access your app**: You'll get a URL like `https://your-app-name.onrender.com`

**Render Features:**
- ✅ Free tier available (750 hours/month)
- ✅ Automatic HTTPS
- ✅ Auto-deploy on git push
- ✅ Environment variable management
- ⚠️ Free tier may spin down after inactivity (30-60 seconds to wake up)

### Deployment Configuration Files

This repository includes pre-configured deployment files:

- **`.streamlit/config.toml`**: Streamlit theme and server settings
- **`render.yaml`**: Render service configuration

### Environment Variables

Both platforms require these environment variables:

| Variable | Description | Required |
|----------|-------------|----------|
| `NVIDIA_API_KEY` | NVIDIA API key for Llama 3.3 70B (development) | ✅ Yes (dev) |
| `OPENAI_API_KEY` | OpenAI API key for GPT-4o/GPT-4o-mini (production) | ✅ Yes (prod) |
| `TAVILY_API_KEY` | Tavily API key for web search | ✅ Yes |

> **Note**: Use either `NVIDIA_API_KEY` (development) or `OPENAI_API_KEY` (production), not both. Update the client configuration in `src/nvidia_client.py` when switching providers.

### Post-Deployment Checklist

After deploying, verify:

- [ ] App loads without errors
- [ ] API keys are correctly configured (check for "Missing API keys" error)
- [ ] PDF upload works
- [ ] Claim extraction completes successfully
- [ ] Verification produces results
- [ ] Export buttons download files correctly

### Troubleshooting Deployment

**"Missing required API keys" error:**
- Verify environment variables are set correctly in your deployment platform
- Check for typos in variable names (must be exact: `NVIDIA_API_KEY` or `OPENAI_API_KEY`, `TAVILY_API_KEY`)
- Restart the app after adding environment variables

**App won't start:**
- Check deployment logs for errors
- Verify `requirements.txt` includes all dependencies
- Ensure Python version is 3.10 or higher

**Slow performance:**
- Free tiers have limited resources
- Consider upgrading to paid tier for production use
- Enable caching (already configured in the app)

**API rate limits:**
- Monitor your API usage in NVIDIA/OpenAI and Tavily dashboards
- Consider implementing request throttling for high-traffic apps
- Upgrade API plans if needed
- **For production**: Switch to paid OpenAI API for higher rate limits and better reliability

## Testing

### Run All Tests
```bash
pytest
```

### Run Property-Based Tests
```bash
pytest -m property_test
```

### Run Unit Tests
```bash
pytest tests/unit/
```

### Run Integration Tests
```bash
pytest tests/integration/
```

### Generate Coverage Report
```bash
pytest --cov=src --cov-report=html
```

## Documentation

### Code Documentation

All modules include comprehensive docstrings following Google style:

- **`src/main.py`**: Streamlit UI and application orchestration
- **`src/pdf_processor.py`**: PDF text extraction with PyMuPDF and pdfplumber fallback
- **`src/claim_extractor.py`**: LLM-based claim identification using NVIDIA API (Llama 3.3 70B)
- **`src/web_search.py`**: Tavily API integration with caching and retry logic
- **`src/verifier.py`**: Claim verification with confidence scoring and tolerance logic
- **`src/nvidia_client.py`**: Unified LLM client with JSON auto-repair (supports NVIDIA/OpenAI)
- **`src/models.py`**: Data models (Claim, Evidence, VerificationResult, ProcessingState)
- **`src/utils.py`**: Logging, configuration, and helper functions

### Sample Data

The `data/` directory contains sample PDF files for testing:

- **`sample_single_page.pdf`**: Simple 1-page document with statistics
- **`sample_multi_page.pdf`**: Multi-page document with various claim types
- **`sample_empty.pdf`**: Empty PDF for error testing

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit UI (main.py)                  │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  PDF Upload → Text Extraction → Claim Extraction →          │
│  Evidence Search → Verification → Results Display           │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│ PDF Processor│    │ Claim Extractor  │    │ Web Search   │
│ (PyMuPDF)    │    │ (NVIDIA/OpenAI)  │    │ (Tavily API) │
└──────────────┘    └──────────────────┘    └──────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │    Verifier      │
                    │ (NVIDIA/OpenAI)  │
                    └──────────────────┘
```

### Key Features Explained

**1. Multi-Provider LLM Support**
- Supports both NVIDIA API (Llama 3.3 70B) and OpenAI API (GPT-4o/GPT-4o-mini)
- Easy switching between providers via environment variables
- Unified client interface in `src/nvidia_client.py`
- JSON auto-repair system for handling malformed LLM responses

**2. Intelligent Caching**
- Web search results cached for 24 hours using Streamlit's `@st.cache_data`
- Reduces API costs by avoiding duplicate searches
- Automatic cache invalidation after TTL expires

**3. Elite-Level Verification with Tolerance Logic**
- **Numeric tolerance**: <5% difference → VERIFIED, 5-15% → INACCURATE, >15% → FALSE
- **Severity classification**: Minor/moderate/major errors
- **Confidence calibration**: VERIFIED (85-95%), INACCURATE (55-75%), FALSE (70-95%)
- **Source prioritization**: WHO/UN/Gov > Research/Statista > News
- **Human-like explanations**: 4-step format (compare, quantify, classify, justify)

**4. Confidence Scoring Algorithm**
```python
confidence = min(100, max(0,
    min(source_count * 20, 40) +    # Base score (max 40)
    source_agreement * 40 +          # Agreement score (max 40)
    recency_score * 20               # Recency score (max 20)
))

# Status-based caps:
# - Inaccurate: max 75
# - False: max 60
```

**5. Error Handling Strategy**
- **Retry Logic**: Exponential backoff for transient failures (2 attempts, 2s-10s wait)
- **Fallback**: PyMuPDF → pdfplumber for PDF extraction
- **Graceful Degradation**: Individual claim failures don't stop entire pipeline
- **User-Friendly Messages**: Clear error messages with actionable guidance

**6. Logging**
- Rotating file handler (10MB max, 5 backups)
- Separate log levels for file (INFO) and console (WARNING)
- Sensitive data sanitization (API keys redacted from logs)
- Structured log format: `[timestamp] [level] [module] message`

## Troubleshooting

### Local Development Issues

**"Missing required API keys" error:**
- Ensure `.env` file exists in the project root
- Verify API keys are correctly set in `.env` (use `NVIDIA_API_KEY` or `OPENAI_API_KEY`)
- Check for typos in variable names
- Restart the application after adding keys

**"Failed to extract text from PDF" error:**
- Ensure PDF is not password-protected
- Try a different PDF file
- Check if PDF contains actual text (not just images)
- Verify PyMuPDF and pdfplumber are installed

**"API rate limit exceeded" error:**
- Wait a few moments and try again
- Check your API usage in NVIDIA/OpenAI/Tavily dashboards
- Consider upgrading your API plan
- Reduce the number of claims being processed
- **For production**: Switch to paid OpenAI API for higher rate limits

**"No verifiable claims found" error:**
- Ensure PDF contains factual statements (statistics, dates, figures)
- Try a different document with more concrete claims
- Check if the text extraction was successful

**Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Streamlit won't start:**
```bash
# Check if port 8501 is already in use
lsof -i :8501  # On macOS/Linux
netstat -ano | findstr :8501  # On Windows

# Use a different port
streamlit run src/main.py --server.port=8502
```

### Testing Issues

**Tests failing:**
```bash
# Ensure test dependencies are installed
pip install pytest hypothesis pytest-cov

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_verifier.py -v
```

**Coverage report not generating:**
```bash
# Install coverage tools
pip install pytest-cov

# Generate HTML coverage report
pytest --cov=src --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
start htmlcov/index.html  # Windows
```

## API Costs

### Current Setup (NVIDIA API - Free Tier)
- **NVIDIA API**: Free tier with generous limits for development
- **Tavily API**: ~$0.02-0.04 per document (2-5 searches per claim)
- **Total**: ~$0.02-0.04 per document

### Production Setup (OpenAI API - Recommended)
Estimated costs per document (10-15 claims):
- **OpenAI GPT-4o-mini**: ~$0.01-0.05 per document
- **OpenAI GPT-4o**: ~$0.05-0.15 per document
- **Tavily API**: ~$0.02-0.04 per document
- **Total (GPT-4o-mini)**: ~$0.03-0.09 per document
- **Total (GPT-4o)**: ~$0.07-0.19 per document

Costs can be reduced by:
- Using caching (enabled by default)
- Limiting claim extraction to fewer claims
- Using GPT-4o-mini instead of GPT-4o for cost optimization

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the web framework
- [NVIDIA](https://build.nvidia.com/) for the NVIDIA API and Llama models
- [OpenAI](https://openai.com/) for GPT-4/GPT-4o API (production recommendation)
- [Tavily](https://tavily.com/) for real-time search API
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing
- [Hypothesis](https://hypothesis.readthedocs.io/) for property-based testing

## Support

For issues, questions, or suggestions, please open an issue on GitHub.
