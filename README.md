# AI Service Intelligence Hub

AI Service Intelligence Hub is an AI-powered dashboard designed to support IT service teams in analyzing incident data, identifying recurring patterns, and transforming operational pain points into actionable improvement opportunities.

The project simulates how ServiceNow-like incident data can be analyzed using AI to help team leads, product owners and stakeholders make better decisions.

## Key Capabilities

- Group similar incidents and recurring issues
- Identify probable root causes
- Suggest backlog priorities
- Estimate business impact and implementation effort
- Recommend actions to reduce recurrence
- Generate executive summaries for stakeholders
- Support team coordination, workload visibility and decision-making

## Why this project matters

In many enterprise environments, incident data contains valuable signals about process inefficiencies, technical debt, recurring failures and automation opportunities.

This project explores how AI can help transform service management data into business and engineering insights.

## Initial Input

The first version works with CSV/XLSX files exported from tools such as ServiceNow.

Future versions may include API-based integration.

## Technologies

- Python
- Streamlit
- Pandas
- Plotly
- Gemini API
- Hugging Face fallback provider
- Prompt Engineering
- AI-assisted analysis
- IT Service Management
- Backlog prioritization

## AI provider configuration

Create a `.env` file based on `.env.example` and configure the provider credentials as Windows environment variables or local `.env` values:

```env
GEMINI_API_KEY_2=
INTELLIGENCE_HUB_HUGGING_FACE_TOKEN=
INTELLIGENCE_HUB_HUGGING_FACE_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct
```

The app first tries Gemini, then Hugging Face, and finally uses backend-generated results if external AI providers are unavailable.

## Portfolio Context

This project is part of my AI Business & Engineering portfolio, focused on practical AI solutions that connect business needs, technical execution and measurable impact.
