# AI Service Intelligence Hub

**AI-powered operational intelligence for enterprise incident analysis, executive-ready insights, and stakeholder-aware communication workflows.**

AI Service Intelligence Hub is a production-oriented SaaS MVP that transforms ServiceNow-like incident exports into business-relevant intelligence. It combines deterministic incident analytics with generative AI orchestration to help technology leaders identify recurring issues, prioritize backlog opportunities, and communicate operational risk with clarity.

The project is designed as an enterprise AI portfolio product: practical enough to demo, structured enough to extend, and polished enough to represent senior engineering judgment across product, UX, data, and AI architecture.

## What It Does

The application ingests CSV or Excel incident exports and turns operational noise into decision-ready insight:

- Detects recurring incident patterns, impacted services, probable root causes, and backlog candidates.
- Builds an operational overview with incident volume, open workload, high-priority exposure, recurrence, and P1 candidates.
- Generates executive summaries, leadership actions, backlog recommendations, and stakeholder messages.
- Provides a Communication Assistant for email and call suggestions tailored to the incident context.
- Supports multiple AI providers with graceful fallback to backend-generated intelligence when external providers are unavailable.

## Problem Space

Enterprise service teams often manage incident data that contains strong signals but weak storytelling:

- Recurring technical issues are buried inside ticket noise.
- Executives need concise business impact, not raw operational logs.
- Stakeholders need coordinated communication during service degradation.
- Backlog prioritization often lacks clear evidence from incident history.
- AI analysis must remain resilient when provider quotas, latency, or availability become constraints.

This MVP demonstrates how an AI-native operations hub can bridge incident management, leadership communication, and continuous improvement.

## Core Capabilities

- **AI Incident Analysis**: Converts enriched incident data into structured executive-ready summaries.
- **Operational Overview**: Highlights workload, recurrence, priority, reopened issues, and business-critical exposure.
- **Root Cause Insights**: Infers probable causes using deterministic rules before AI summarization.
- **Backlog Prioritization**: Scores incidents by value, impact, recurrence, and estimated effort.
- **Communication Assistant**: Produces stakeholder-aware email and call suggestions based on business context.
- **Multi-organization Demo Datasets**: Includes Retail, Healthcare, and B2B SaaS scenarios to showcase domain-aware outputs.
- **Resilient AI Orchestration**: Uses Gemini first, Hugging Face as an alternative provider, and backend fallback as the final reliability layer.
- **Responsive SaaS UX**: Includes a polished Streamlit interface, dark mode, guided scroll behavior, and enterprise-style visual hierarchy.

## AI Architecture

The AI layer is intentionally designed for reliability rather than a single-provider happy path.

```text
Uploaded CSV/XLSX
      |
      v
Data validation and enrichment
      |
      v
Deterministic metrics, recurrence, root cause, backlog scoring
      |
      v
Prompt context compaction
      |
      v
Gemini primary analysis
      |
      +--> Hugging Face fallback
              |
              +--> Backend-generated decision brief
```

### Provider Strategy

- **Primary provider**: Gemini via `google-genai`
- **Alternative provider**: Hugging Face Inference / Router
- **Final fallback**: Rule-based backend summary generated locally

The app classifies provider failures into quota, availability, authentication, model support, request, and parsing categories. User-facing messaging stays non-technical while terminal logs retain enough diagnostic detail for development and production troubleshooting.

### Prompt Strategy

The system sends compact operational context rather than full raw exports. It prioritizes:

- aggregate distributions,
- top impacted services,
- probable root causes,
- recurrence patterns,
- selected high-value incident examples,
- structured JSON output for predictable rendering.

This keeps AI usage practical for free-tier development while preserving useful business analysis.

## Demo Datasets

The repository includes three curated sample datasets under `data/`:

| Dataset | Format | Scenario | Analysis Signal |
| --- | --- | --- | --- |
| `sample_incidents_org1.csv` | CSV | Retail / E-commerce | Checkout degradation, payment latency, login recurrence, revenue impact |
| `sample_incidents_org2.xlsx` | Excel | Healthcare / Hospital | EHR instability, lab result delays, clinical operations and compliance risk |
| `sample_incidents_org3.csv` | CSV | B2B SaaS Platform | Kubernetes pressure, API degradation, OAuth issues, tenant isolation risk |

Each dataset keeps the same schema so the parser, dashboards, AI analysis, and Communication Assistant work consistently across industries.

## Technical Stack

- **Python** for backend logic and AI orchestration
- **Streamlit** for the interactive SaaS MVP interface
- **Pandas** for ingestion, validation, enrichment, and tabular analysis
- **Plotly** for operational visualizations
- **Google GenAI SDK** for Gemini integration
- **Hugging Face Inference** for alternative AI generation
- **HTML/CSS templates** for a more product-oriented Streamlit experience
- **python-dotenv** for local environment configuration

## Project Structure

```text
.
├── app.py                      # Streamlit application entry point
├── assets/
│   └── styles.css              # SaaS UI styling and theme system
├── data/                       # Demo incident datasets
├── prompts/
│   └── executive_summary_prompt.txt
├── src/
│   ├── ai_engine.py            # Gemini, Hugging Face, fallback parsing, provider diagnostics
│   ├── communication_assistant.py
│   ├── data_loader.py
│   ├── executive_summary.py
│   ├── incident_analyzer.py
│   ├── prioritization.py
│   └── ui.py
├── templates/                  # Reusable HTML snippets for Streamlit rendering
├── requirements.txt
├── Procfile
└── README.md
```

## Running Locally

### 1. Create a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a local `.env` file from `.env.example`:

```env
INTELLIGENCE_HUB_GEMINI_API_KEY=
INTELLIGENCE_HUB_GEMINI_MODEL=gemini-2.5-flash-lite
INTELLIGENCE_HUB_HF_TOKEN=
INTELLIGENCE_HUB_HF_MODEL=meta-llama/Llama-3.1-8B-Instruct:novita
INTELLIGENCE_HUB_HF_FALLBACK_MODELS=meta-llama/Llama-3.1-8B-Instruct:novita,deepseek-ai/DeepSeek-V4-Flash:novita,google/gemma-4-31B-it:novita
```

For backwards compatibility, the application still accepts `GEMINI_API_KEY_2` and the previous Hugging Face variable names, but the project-specific names above are recommended for a professional deployment.

### 4. Launch the app

```powershell
streamlit run app.py
```

Then upload one of the demo files from `data/`.

## Deployment Notes

The included `Procfile` supports Railway-style deployment:

```text
web: streamlit run app.py --server.address=0.0.0.0 --server.port=$PORT --server.headless=true
```

In production, configure provider tokens as platform environment variables. Do not commit `.env`, Streamlit secrets, provider tokens, or personal credentials.

## Screenshots

Screenshots can be added under a future `docs/screenshots/` folder:

- Landing / Hero experience
- Upload and Operational Overview
- Incident Workspace
- AI Analysis and Communication Assistant
- Dark mode

## Roadmap

- ServiceNow and Jira connectors for live incident ingestion
- Role-aware executive, technical, and customer-facing communication modes
- Observability integrations for incident correlation
- Multi-agent analysis for root cause, backlog, and stakeholder workflows
- Persistent workspace history and saved decision briefs
- Enterprise authentication and multi-tenant access controls
- Exportable leadership briefings for operational reviews

## Portfolio Positioning

This MVP is built to demonstrate more than coding ability. It shows how AI product thinking, enterprise UX, prompt design, fallback architecture, and operational data modeling can come together into a credible SaaS prototype for AI-enabled service leadership.
