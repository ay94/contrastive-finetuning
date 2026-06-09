#!/usr/bin/env python3
"""
Takes screenshots of each dashboard page using Playwright.
Run after: streamlit run dashboard/Load.py

Usage:
    python3.10 scripts/take_screenshots.py
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8502"
OUT_DIR = Path(__file__).parent.parent / "assets" / "screenshots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAGES = [
    ("Choose Model",          "01_choose_model.png",          None),
    ("Classification Report", "02_classification_report.png", None),
    ("Average Similarity",    "03_average_similarity.png",    None),
    ("Similarity Matrix",     "04_similarity_matrix.png",     "Generate Similarity Matrix"),
    ("Semantic Map",          "05_semantic_map.png",          "Generate Semantic Map"),
    ("Class Metrics",         "06_class_metrics.png",         "Generate Class Metrics"),
    ("Unmasked Data",         "07_unmasked_data.png",         None),
]


def wait_for_streamlit_idle(page):
    try:
        page.wait_for_selector("[data-testid='stSpinner']", timeout=3000)
        page.wait_for_selector("[data-testid='stSpinner']", state="hidden", timeout=25000)
    except Exception:
        pass
    time.sleep(1.5)


def select_page(page, label):
    page.locator("[data-testid='stSidebar'] [data-testid='stSelectbox']").first.click()
    time.sleep(0.4)
    page.locator(
        f"[data-testid='stSelectboxVirtualDropdown'] li:has-text('{label}')"
    ).first.click()
    wait_for_streamlit_idle(page)


def click_generate_and_capture(page, button_text, out_path):
    """Click the generate button, wait for the chart, then screenshot it in-viewport."""
    btn = page.locator(f"button:has-text('{button_text}')").first
    btn.scroll_into_view_if_needed()
    time.sleep(0.5)
    btn.click()

    wait_for_streamlit_idle(page)

    # Wait for the Plotly container to appear
    try:
        page.wait_for_selector(".js-plotly-plot", timeout=15000)
    except Exception:
        print(f"    (chart selector timed out)")

    # Dispatch a resize event — forces Plotly to recalculate dimensions
    page.evaluate("window.dispatchEvent(new Event('resize'))")
    time.sleep(1)

    # Scroll the chart into view and screenshot it there
    chart = page.locator(".js-plotly-plot").first
    chart.scroll_into_view_if_needed()
    time.sleep(2)

    # Take the full page now that Plotly has been resized and is in view
    page.screenshot(path=str(out_path), full_page=True)
    print(f"    chart captured → {out_path}")

    # Scroll back for next iteration
    page.evaluate("window.scrollTo(0, 0)")
    time.sleep(0.5)


def run():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()

        print(f"Loading {BASE_URL} ...")
        page.goto(BASE_URL, wait_until="networkidle")
        wait_for_streamlit_idle(page)
        time.sleep(2)

        for label, filename, button_text in PAGES:
            print(f"  [{label}]")
            select_page(page, label)
            out_path = OUT_DIR / filename

            if button_text:
                click_generate_and_capture(page, button_text, out_path)
            else:
                page.screenshot(path=str(out_path), full_page=True)
                print(f"    saved → {out_path}")

        browser.close()
        print(f"\nDone. All screenshots in {OUT_DIR}")


if __name__ == "__main__":
    run()
