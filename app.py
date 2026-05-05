import logging

import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from src.ai_engine import (
    AIQuotaExceededError,
    AlternativeAIError,
    generate_ai_decision_layer,
    generate_huggingface_decision_layer,
)
from src.communication_assistant import generate_all_emails, generate_call_suggestions
from src.data_loader import calculate_basic_metrics, load_incidents
from src.executive_summary import build_fallback_ai_decision, build_rule_based_summary
from src.incident_analyzer import enrich_incidents, get_pattern_summary
from src.ui import escape_html, load_css, render_ai_card_grid, render_template


logger = logging.getLogger(__name__)

AI_SOURCE_GEMINI = "gemini"
AI_SOURCE_HUGGING_FACE = "huggingFace"
AI_SOURCE_BACKEND_FALLBACK = "backendFallback"

AI_MODE_LABELS = {
    AI_SOURCE_GEMINI: "Gemini AI",
    AI_SOURCE_HUGGING_FACE: "Hugging Face AI",
    AI_SOURCE_BACKEND_FALLBACK: "Backend fallback",
}

GEMINI_QUOTA_MESSAGE = (
    "Gemini quota limit reached. We are generating the analysis using an alternative AI provider."
)
GEMINI_ERROR_MESSAGE = (
    "Gemini could not complete the analysis. We are generating the analysis using an alternative AI provider."
)
AI_FALLBACK_MESSAGE = (
    "The available AI providers could not complete the analysis. "
    "Showing basic backend-generated results instead."
)

TEAM_ICON = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M16 21v-2a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v2" />
  <circle cx="9.5" cy="7" r="4" />
  <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
  <path d="M16 3.13a4 4 0 0 1 0 7.75" />
</svg>
"""

BACKLOG_ICON = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M9 11l3 3L22 4" />
  <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
</svg>
"""

COMMS_ICON = """
<svg viewBox="0 0 24 24" aria-hidden="true">
  <path d="M21 15a4 4 0 0 1-4 4H7l-4 4V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z" />
  <path d="M8 9h8M8 13h5" />
</svg>
"""


CHART_COLORS = ["#215BFF", "#00A6A6", "#8FD14F", "#FFB020", "#FF5A5F", "#6E6EF6", "#7A8BA0"]
PRIORITY_COLORS = {
    "Critical": "#FF5A5F",
    "High": "#FFB020",
    "Medium": "#215BFF",
    "Low": "#8FD14F",
}
BACKLOG_COLORS = ["#215BFF", "#00A6A6", "#8FD14F", "#FFB020", "#6E6EF6", "#FF5A5F"]


def polish_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.92)",
        font=dict(color="#334155", family="Inter, Segoe UI, sans-serif", size=12),
        title_font=dict(color="#0F2F4F", size=16),
        margin=dict(l=24, r=24, t=58, b=28),
        legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(size=11)),
        xaxis=dict(gridcolor="#E5EEF7", zerolinecolor="#E5EEF7"),
        yaxis=dict(gridcolor="#E5EEF7", zerolinecolor="#E5EEF7"),
    )
    return fig


def palette(count: int, colors: list[str]) -> list[str]:
    return [colors[index % len(colors)] for index in range(count)]


def render_preview_table(enriched_df):
    preview_columns = [
        "incident_id",
        "priority",
        "state",
        "service",
        "short_description",
        "opened_by_email",
        "stakeholder_emails",
        "team_emails",
        "recurrence_type",
        "probable_root_cause",
        "estimated_impact",
        "suggested_backlog_priority",
    ]
    available_columns = [column for column in preview_columns if column in enriched_df.columns]
    preview_df = enriched_df[available_columns].reset_index(drop=True)
    st.dataframe(preview_df, use_container_width=True, height=390, hide_index=True)


def render_context_nav(links: list[tuple[str, str]]) -> None:
    link_html = "".join([f'<a class="pill" href="{href}">{label}</a>' for label, href in links])
    render_template("context_nav.html", links=link_html)


def impact_badge(impact: str) -> str:
    impact_lower = str(impact).lower()
    if "high" in impact_lower:
        return "🔥 High impact"
    if "medium" in impact_lower:
        return "⚡ Medium impact"
    return "✅ Low impact"


def priority_badge(priority: str) -> str:
    priority_lower = str(priority).lower()
    if "critical" in priority_lower:
        return "Critical"
    if "high" in priority_lower:
        return "High"
    if "medium" in priority_lower:
        return "Medium"
    return "Low"


def pattern_signal(count: int) -> str:
    if count >= 4:
        return "🔥 Concentrated"
    if count >= 2:
        return "⚡ Recurring"
    return "✅ Monitor"


st.set_page_config(
    page_title="AI Service Intelligence Hub",
    page_icon="🧠",
    layout="wide",
)

load_css()

components.html(
    """
    <script>
      (() => {
        const doc = window.parent.document;
        const storageKey = "aisi-theme";
        const existing = doc.getElementById("aisi-theme-toggle");
        const preferred = window.parent.localStorage.getItem(storageKey) || "light";

        doc.documentElement.dataset.theme = preferred;

        if (existing) {
          existing.dataset.theme = preferred;
          existing.setAttribute("aria-label", preferred === "dark" ? "Switch to light mode" : "Switch to dark mode");
          return;
        }

        const button = doc.createElement("button");
        button.id = "aisi-theme-toggle";
        button.className = "theme-toggle";
        button.type = "button";
        button.dataset.theme = preferred;
        button.setAttribute("aria-label", preferred === "dark" ? "Switch to light mode" : "Switch to dark mode");
        button.innerHTML = "<span class='theme-sun'>☀</span><span class='theme-moon'>☾</span>";
        button.addEventListener("click", () => {
          const next = doc.documentElement.dataset.theme === "dark" ? "light" : "dark";
          doc.documentElement.dataset.theme = next;
          window.parent.localStorage.setItem(storageKey, next);
          button.dataset.theme = next;
          button.setAttribute("aria-label", next === "dark" ? "Switch to light mode" : "Switch to dark mode");
        });
        doc.body.appendChild(button);
      })();
    </script>
    """,
    height=0,
)

render_template("hero.html")


st.markdown('<section id="input-data" class="landing-panel upload-panel">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title landing-title">1. Input Data</div>',
    unsafe_allow_html=True,
)
upload_col, guidance_col = st.columns([0.58, 0.42], gap="large")
with upload_col:
    st.markdown('<div class="upload-card-heading">Incident export</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="upload-helper-copy">Upload a CSV or Excel export from ServiceNow or another ITSM tool</p>',
        unsafe_allow_html=True,
    )
    uploaded_file = st.file_uploader(
        "Incident export file",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )

with guidance_col:
    render_template("input_guidance_card.html")

components.html(
    """
    <script>
      (() => {
        const doc = window.parent.document;
        if (window.parent.__aisiSectionSnapInstalled) return;
        window.parent.__aisiSectionSnapInstalled = true;

        const getSections = () => [
          "ai-service-intelligence-hub",
          "input-data",
          "overview",
          "workspace",
          "ai-engine",
          "ai-results-anchor"
        ].map((id) => doc.getElementById(id)).filter(Boolean);

        const isScrollableChild = (target) => {
          let node = target;
          while (node && node !== doc.body) {
            const style = window.parent.getComputedStyle(node);
            const scrollable = /(auto|scroll)/.test(style.overflowY) && node.scrollHeight > node.clientHeight + 12;
            if (scrollable) return true;
            node = node.parentElement;
          }
          return false;
        };

        let locked = false;
        doc.addEventListener("wheel", (event) => {
          if (Math.abs(event.deltaY) < 24 || locked || isScrollableChild(event.target)) return;
          const sections = getSections();
          if (sections.length < 2) return;

          const currentY = window.parent.scrollY;
          const viewport = window.parent.innerHeight || doc.documentElement.clientHeight;
          const positions = sections.map((section) => ({
            section,
            top: section.getBoundingClientRect().top + currentY
          }));

          let currentIndex = 0;
          positions.forEach((item, index) => {
            if (item.top <= currentY + viewport * 0.34) currentIndex = index;
          });

          const nextIndex = Math.max(0, Math.min(positions.length - 1, currentIndex + (event.deltaY > 0 ? 1 : -1)));
          if (nextIndex === currentIndex) return;

          event.preventDefault();
          locked = true;
          positions[nextIndex].section.scrollIntoView({ behavior: "smooth", block: "start" });
          window.parent.setTimeout(() => { locked = false; }, 850);
        }, { passive: false });
      })();
    </script>
    """,
    height=0,
)

if uploaded_file is None:
    st.markdown("</section>", unsafe_allow_html=True)
    render_template("footer.html")
    st.stop()

st.markdown("</section>", unsafe_allow_html=True)

st.markdown('<div id="uploaded-data-anchor"></div>', unsafe_allow_html=True)

try:
    raw_df = load_incidents(uploaded_file)
except Exception as error:
    render_template(
        "ai_status_banner.html",
        status="error",
        message=f"The uploaded file could not be processed. Please check the required columns and file format. Details: {error}",
    )
    st.stop()


data_signature = f"{uploaded_file.name}:{uploaded_file.size}"
if st.session_state.get("data_signature") != data_signature:
    st.session_state["data_signature"] = data_signature
    st.session_state.pop("ai_result", None)
    st.session_state.pop("ai_mode", None)
    st.session_state.pop("ai_source", None)
    st.session_state.pop("ai_banner_status", None)
    st.session_state.pop("ai_banner_message", None)


enriched_df = enrich_incidents(raw_df)
metrics = calculate_basic_metrics(raw_df)
summary = build_rule_based_summary(enriched_df)
patterns_df = get_pattern_summary(enriched_df)

overview_scroll_script = """
    <script>
        setTimeout(() => {{
            const target = window.parent.document.getElementById("overview-section");
            const signature = "__DATA_SIGNATURE__";
            if (target && window.parent.__aisiUploadScrolled !== signature) {{
                window.parent.__aisiUploadScrolled = signature;
                const top = target.getBoundingClientRect().top + window.parent.scrollY - 20;
                window.parent.scrollTo({{ top, behavior: "smooth" }});
            }}
        }}, 850);
    </script>
    """.replace("__DATA_SIGNATURE__", data_signature)

components.html(
    overview_scroll_script,
    height=0,
)

st.markdown('<section id="overview-section" class="workflow-section overview-section">', unsafe_allow_html=True)
st.markdown('<div id="overview" class="section-title">2. Operational Overview</div>', unsafe_allow_html=True)

metric_columns = st.columns(6)
metric_cards = [
    ("Total incidents", metrics["total"]),
    ("Open / in progress", metrics["open_incidents"]),
    ("Critical / high", metrics["critical_high"]),
    ("Reopened count", metrics["reopened"]),
    ("Recurring issues", summary["recurring_count"]),
    ("P1 candidates", summary["p1_count"]),
]

for column, (label, value) in zip(metric_columns, metric_cards):
    with column:
        render_template("metric_card.html", label=label, value=value)


preview_col, rule_col = st.columns([0.68, 0.32])
with preview_col:
    st.markdown('<div id="file-preview" class="section-title compact-title">File Preview</div>', unsafe_allow_html=True)
    render_preview_table(enriched_df)

with rule_col:
    st.markdown('<div class="section-title compact-title">Initial rule-based signal</div>', unsafe_allow_html=True)
    render_template(
        "overview_signal_card.html",
        main_recommendation=summary["main_recommendation"],
        top_service=summary["top_service"],
        top_root_cause=summary["top_root_cause"],
    )

st.markdown("</section>", unsafe_allow_html=True)

st.markdown('<div id="workspace" class="section-title">3. Incident Workspace</div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Patterns",
        "Backlog priorities",
        "Visual insights",
        "Leadership support",
    ]
)


with tab1:
    render_template(
        "insight_box.html",
        content="""
        Patterns are generated deterministically from service, probable root cause and recurrence signals.
        This keeps the analysis explainable and auditable.
        """,
    )
    styled_patterns_df = patterns_df.copy()
    if "incident_count" in styled_patterns_df.columns:
        styled_patterns_df["pattern_signal"] = styled_patterns_df["incident_count"].apply(pattern_signal)
        styled_patterns_df = styled_patterns_df[
            ["pattern_signal"]
            + [column for column in styled_patterns_df.columns if column != "pattern_signal"]
        ]
    st.dataframe(
        styled_patterns_df,
        use_container_width=True,
        height=360,
        hide_index=True,
        column_config={
            "pattern_signal": st.column_config.TextColumn(
                "Signal",
                help="Fast visual indicator based on incident concentration.",
            ),
            "incident_count": st.column_config.NumberColumn("Incidents"),
        },
    )


with tab2:
    backlog_df = enriched_df.sort_values(
        by=["value_effort_ratio", "value_score"],
        ascending=[False, False],
    ).copy()
    backlog_df["priority_signal"] = backlog_df["priority"].apply(priority_badge)
    backlog_df["impact_signal"] = backlog_df["estimated_impact"].apply(impact_badge)

    st.dataframe(
        backlog_df[
            [
                "incident_id",
                "service",
                "short_description",
                "priority_signal",
                "impact_signal",
                "estimated_effort",
                "value_score",
                "effort_score",
                "value_effort_ratio",
                "suggested_backlog_priority",
                "recommended_action",
            ]
        ].reset_index(drop=True),
        use_container_width=True,
        height=440,
        hide_index=True,
        column_config={
            "impact_signal": st.column_config.TextColumn(
                "Estimated impact",
                help="Impact badge generated from priority, recurrence and business impact signals.",
            ),
            "priority_signal": st.column_config.TextColumn(
                "Priority",
                help="Visual priority signal for quick scanning.",
            ),
            "value_effort_ratio": st.column_config.NumberColumn(
                "Value / effort",
                format="%.2f",
            ),
        },
    )


with tab3:
    st.markdown('<div class="viz-stage">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="viz-cockpit">
          <div>
            <span class="viz-eyebrow">Operational analytics</span>
            <h3>Incident intelligence cockpit</h3>
            <p>Executive-ready views for priority, services, root-cause concentration and backlog focus.</p>
          </div>
          <div class="viz-stat-row">
            <div><b>{metrics["critical_high"]}</b><span>Critical / high</span></div>
            <div><b>{summary["recurring_count"]}</b><span>Recurring signals</span></div>
            <div><b>{summary["p1_count"]}</b><span>P1 candidates</span></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        priority_counts = enriched_df["priority"].value_counts().reset_index()
        priority_counts.columns = ["priority", "count"]

        fig = px.bar(
            priority_counts,
            x="priority",
            y="count",
            text="count",
            title="Incidents by priority",
            color="priority",
            color_discrete_map=PRIORITY_COLORS,
            category_orders={"priority": ["Critical", "High", "Medium", "Low"]},
        )
        fig.update_traces(textposition="inside", marker_line_width=0, hovertemplate="%{x}: %{y}<extra></extra>")
        fig.update_layout(showlegend=False)
        fig.update_layout(transition=dict(duration=650, easing="cubic-in-out"))
        polish_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    with chart_col2:
        service_counts = enriched_df["service"].value_counts().reset_index()
        service_counts.columns = ["service", "count"]

        fig = px.pie(
            service_counts,
            names="service",
            values="count",
            title="Incident volume by service",
            hole=0.58,
            color_discrete_sequence=CHART_COLORS,
        )
        fig.update_traces(
            textinfo="percent",
            marker=dict(line=dict(color="#FFFFFF", width=3)),
            hovertemplate="%{label}: %{value} incidents<extra></extra>",
            pull=[0.035 if index == 0 else 0 for index in range(len(service_counts))],
        )
        fig.update_layout(transition=dict(duration=650, easing="cubic-in-out"))
        polish_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        root_counts = enriched_df["probable_root_cause"].value_counts().reset_index()
        root_counts.columns = ["root_cause", "count"]

        fig = px.bar(
            root_counts,
            x="count",
            y="root_cause",
            orientation="h",
            text="count",
            title="Probable root cause distribution",
        )
        fig.update_traces(
            marker_color=palette(len(root_counts), CHART_COLORS),
            marker_line_width=0,
            textposition="inside",
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
        fig.update_layout(transition=dict(duration=650, easing="cubic-in-out"))
        polish_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

    with chart_col4:
        backlog_counts = enriched_df["suggested_backlog_priority"].value_counts().reset_index()
        backlog_counts.columns = ["backlog_priority", "count"]

        fig = px.bar(
            backlog_counts,
            x="count",
            y="backlog_priority",
            orientation="h",
            text="count",
            title="Suggested backlog priority",
        )
        fig.update_traces(
            marker_color=palette(len(backlog_counts), BACKLOG_COLORS),
            marker_line_width=0,
            textposition="inside",
            hovertemplate="%{y}: %{x}<extra></extra>",
        )
        fig.update_layout(transition=dict(duration=650, easing="cubic-in-out"))
        polish_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


with tab4:
    support_col1, support_col2, support_col3 = st.columns(3)

    with support_col1:
        render_template(
            "leadership_card.html",
            title="Team coordination",
            icon=TEAM_ICON,
            content="""
            Identify workload concentration, recurring services and ownership signals
            to guide follow-up conversations with the right teams.
            """,
        )

    with support_col2:
        render_template(
            "leadership_card.html",
            title="Backlog refinement",
            icon=BACKLOG_ICON,
            content="""
            Translate incidents into backlog candidates using impact, effort,
            recurrence and value/effort ratio.
            """,
        )

    with support_col3:
        render_template(
            "leadership_card.html",
            title="Stakeholder communication",
            icon=COMMS_ICON,
            content="""
            Convert technical incidents into business-readable messages,
            priorities and recommended next actions.
            """,
        )


st.markdown('<div id="ai-engine" class="section-title">4. AI Analysis</div>', unsafe_allow_html=True)
ai_action_col, _ = st.columns([0.56, 0.44])
with ai_action_col:
    st.markdown('<section class="ai-analysis-panel">', unsafe_allow_html=True)
    render_template("cta_ai_box.html")
    st.markdown("</section>", unsafe_allow_html=True)
    st.markdown('<div class="ai-button-wrap">', unsafe_allow_html=True)
    generate_ai = st.button("Generate Insights", type="primary", use_container_width=False)
    st.markdown("</div>", unsafe_allow_html=True)
    ai_loading_placeholder = st.empty()


if generate_ai:
    st.session_state["ai_request_id"] = st.session_state.get("ai_request_id", 0) + 1
    ai_request_key = f"{data_signature}:{st.session_state['ai_request_id']}"
    ai_source = AI_SOURCE_GEMINI
    ai_mode = AI_MODE_LABELS[ai_source]
    banner_status = None
    banner_message = None
    loading_placeholder = ai_loading_placeholder

    with loading_placeholder.container():
        st.markdown('<div id="ai-loading-anchor"></div>', unsafe_allow_html=True)
        render_template("ai_loading.html")
        components.html(
            f"""
            <script>
                setTimeout(() => {{
                    const target = window.parent.document.getElementById("ai-loading-anchor");
                    const key = {ai_request_key!r};
                    if (target && window.parent.__aisiAiLoadingScroll !== key) {{
                        window.parent.__aisiAiLoadingScroll = key;
                        target.scrollIntoView({{ behavior: "smooth", block: "center" }});
                    }}
                }}, 250);
            </script>
            """,
            height=0,
        )

    cached_ai = st.session_state.get("ai_cache", {}).get(data_signature)
    if cached_ai:
        ai_result = cached_ai["result"]
        ai_source = cached_ai["source"]
        ai_mode = AI_MODE_LABELS[ai_source]
        banner_status = cached_ai.get("banner_status")
        banner_message = cached_ai.get("banner_message")
        loading_placeholder.empty()
    else:
        try:
            ai_result = generate_ai_decision_layer(enriched_df)
        except AIQuotaExceededError as error:
            logger.warning("Gemini quota exceeded. Trying Hugging Face fallback.", exc_info=error)
            try:
                ai_result = generate_huggingface_decision_layer(enriched_df)
                ai_source = AI_SOURCE_HUGGING_FACE
                ai_mode = AI_MODE_LABELS[ai_source]
                banner_status = "quota"
                banner_message = GEMINI_QUOTA_MESSAGE
            except AlternativeAIError as hf_error:
                logger.exception("Hugging Face fallback failed after Gemini quota error.", exc_info=hf_error)
                ai_result = build_fallback_ai_decision(enriched_df)
                ai_source = AI_SOURCE_BACKEND_FALLBACK
                ai_mode = AI_MODE_LABELS[ai_source]
                banner_status = "fallback"
                banner_message = AI_FALLBACK_MESSAGE
        except Exception as error:
            logger.exception("Gemini analysis failed. Trying Hugging Face fallback.", exc_info=error)
            try:
                ai_result = generate_huggingface_decision_layer(enriched_df)
                ai_source = AI_SOURCE_HUGGING_FACE
                ai_mode = AI_MODE_LABELS[ai_source]
                banner_status = "error"
                banner_message = GEMINI_ERROR_MESSAGE
            except AlternativeAIError as hf_error:
                logger.exception("Hugging Face fallback failed after Gemini error.", exc_info=hf_error)
                ai_result = build_fallback_ai_decision(enriched_df)
                ai_source = AI_SOURCE_BACKEND_FALLBACK
                ai_mode = AI_MODE_LABELS[ai_source]
                banner_status = "fallback"
                banner_message = AI_FALLBACK_MESSAGE
        finally:
            if banner_status == "quota":
                banner_message = GEMINI_QUOTA_MESSAGE
            elif banner_status == "error":
                banner_message = GEMINI_ERROR_MESSAGE
            elif banner_status == "fallback":
                banner_message = AI_FALLBACK_MESSAGE
            loading_placeholder.empty()

        st.session_state.setdefault("ai_cache", {})[data_signature] = {
            "result": ai_result,
            "source": ai_source,
            "banner_status": banner_status,
            "banner_message": banner_message,
        }

    st.session_state["ai_result"] = ai_result
    st.session_state["ai_mode"] = ai_mode
    st.session_state["ai_source"] = ai_source
    st.session_state["ai_banner_status"] = banner_status
    st.session_state["ai_banner_message"] = banner_message


ai_result = st.session_state.get("ai_result")
ai_mode = st.session_state.get("ai_mode")
ai_source = st.session_state.get("ai_source")

if ai_result:
    st.markdown('<div id="ai-results-anchor"></div>', unsafe_allow_html=True)

    result_scroll_key = f"{data_signature}:{st.session_state.get('ai_request_id', 'initial')}"
    components.html(
        f"""
        <script>
            setTimeout(() => {{
                const target = window.parent.document.getElementById("ai-results-anchor");
                const key = {result_scroll_key!r};
                if (target && window.parent.__aisiAiResultScroll !== key) {{
                    window.parent.__aisiAiResultScroll = key;
                    target.scrollIntoView({{ behavior: "smooth", block: "start" }});
                }}
            }}, 700);
        </script>
        """,
        height=0,
    )

    st.markdown('<div class="ai-results-wrapper">', unsafe_allow_html=True)
    if ai_source == AI_SOURCE_BACKEND_FALLBACK:
        decision_brief_label = "Back Decision Brief"
    elif ai_source == AI_SOURCE_HUGGING_FACE:
        decision_brief_label = "AI Decision Brief · Hugging Face"
    else:
        decision_brief_label = "AI Decision Brief · Gemini"
    render_context_nav(
        [
            (decision_brief_label, "#ai-decision-brief"),
        ]
    )

    banner_status = st.session_state.get("ai_banner_status")
    banner_message = st.session_state.get("ai_banner_message")
    if banner_status and banner_message:
        render_template("ai_status_banner.html", status=banner_status, message=banner_message)

    st.markdown('<div class="section-title">5. Communication Assistant Suggestions</div>', unsafe_allow_html=True)

    emails = generate_all_emails(enriched_df, ai_result, ai_mode)
    email_cols = st.columns(2)
    for index, email in enumerate(emails):
        with email_cols[index % 2]:
            render_template(
                "email_card.html",
                audience=email["audience"],
                subject=email["subject"],
                stakeholder_summary=email["stakeholder_summary"],
                impact_summary=email["impact_summary"],
                gmail_link=email["gmail_link"].replace("&", "&amp;"),
                outlook_link=email["outlook_link"].replace("&", "&amp;"),
            )

    calls = generate_call_suggestions(enriched_df, ai_result, ai_mode)
    call_cols = st.columns(2)
    for index, call in enumerate(calls):
        with call_cols[index % 2]:
            render_template(
                "call_card.html",
                title=call["title"],
                stakeholder_summary=call["stakeholder_summary"],
                purpose=call["purpose"],
                calendar_link=call["calendar_link"].replace("&", "&amp;"),
            )

    st.markdown(
        f'<div id="ai-decision-brief" class="section-title">6. {decision_brief_label}</div>',
        unsafe_allow_html=True,
    )
    render_template(
        "insight_box.html",
        content=f"""
        <b>Executive summary</b><br>
        {escape_html(ai_result.get("executive_summary", "No executive summary returned."))}
        """,
    )

    render_ai_card_grid(
        [
            ("Key patterns", ai_result.get("key_patterns", [])),
            ("Backlog priorities", ai_result.get("backlog_priorities", [])),
            ("Probable root causes", ai_result.get("probable_root_causes", [])),
            ("Leadership actions", ai_result.get("leadership_actions", [])),
        ]
    )

    render_template(
        "insight_box.html",
        content=f"""
        <b>Stakeholder message</b><br>
        {escape_html(ai_result.get("stakeholder_message", "No stakeholder message returned."))}
        """,
    )

    st.markdown("</div>", unsafe_allow_html=True)


render_template("footer.html")
