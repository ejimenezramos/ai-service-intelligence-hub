from pathlib import Path
from html import escape

import streamlit as st


TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"


def load_css() -> None:
    css_path = ASSETS_DIR / "styles.css"

    if not css_path.exists():
        st.warning(f"CSS file not found: {css_path}")
        return

    st.markdown(
        f"<style>{css_path.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
        )


def load_template(template_name: str) -> str:
    template_path = TEMPLATES_DIR / template_name

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    return template_path.read_text(encoding="utf-8")


def render_template(template_name: str, **kwargs) -> None:
    html = load_template(template_name)

    for key, value in kwargs.items():
        html = html.replace(f"{{{{{key}}}}}", str(value))
        html = html.replace(f"{{{{ {key} }}}}", str(value))

    st.markdown(html, unsafe_allow_html=True)


def escape_html(value: object) -> str:
    return escape(str(value), quote=True)


def render_list(items: list) -> str:
    if not items:
        return "<p class='muted'>No AI insight returned.</p>"

    return "<ul class='ai-list'>" + "".join([f"<li>{escape_html(item)}</li>" for item in items]) + "</ul>"


def render_ai_card_grid(cards: list[tuple[str, list | str]]) -> None:
    card_html = []
    for title, items in cards:
        content = render_list(items) if isinstance(items, list) else f"<p>{escape_html(items)}</p>"
        card_html.append(
            f'<div class="ai-card"><h4>{escape_html(title)}</h4>{content}</div>'
        )

    st.markdown(
        f'<div class="ai-card-grid">{"".join(card_html)}</div>',
        unsafe_allow_html=True,
    )
