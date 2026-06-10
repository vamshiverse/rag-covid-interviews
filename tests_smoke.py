"""Headless UI smoke test using Streamlit's official AppTest harness.

Drives the real app script through Streamlit's own state pipeline (reliable,
unlike external DOM events) and asserts no exceptions plus key rendered output.
"""
from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=120)
at.run()
assert not at.exception, f"App raised on load: {at.exception}"
print("[load] OK — no exceptions on initial render")
print("[load] tabs present:", len(at.tabs) if hasattr(at, "tabs") else "n/a")

# ---- Ask tab: set the question text_input and click Answer ----
# text inputs across the script; pick the one in the Ask tab (the QA box)
ti = at.text_input(key=None) if False else at.text_input
print("[ask] text_inputs:", len(ti))
ti[0].set_value("What is MIS-C and how was it treated in children?")
at.run()
assert not at.exception, f"raise after setting question: {at.exception}"

# find + click the Answer button
clicked = False
for b in at.button:
    if "Answer" in b.label:
        b.click(); at.run(); clicked = True; break
assert clicked, "Answer button not found"
assert not at.exception, f"raise after Answer click: {at.exception}"

# the rendered answer should mention MIS-C / immunoglobulin somewhere in markdown
md_blob = " ".join(m.value for m in at.markdown)
assert ("MIS-C" in md_blob) or ("immunoglobulin" in md_blob.lower()), \
    "answer text not found in rendered markdown"
assert ("Citations" in md_blob) or ("relevance" in md_blob.lower()), \
    "citations block not rendered"
assert ("line " in md_blob), "citation line locator not rendered"
print("[ask] OK — answer + citations + line locators rendered for MIS-C query")

# ---- Line-level source viewer: click the first '📄 Source' button -> dialog ----
src_clicked = False
for b in at.button:
    if "Source" in b.label:
        b.click(); at.run(); src_clicked = True; break
assert src_clicked, "no 'Source' button found (line-level viewer missing)"
assert not at.exception, f"raise after opening source dialog: {at.exception}"
print("[viewer] OK — source-transcript dialog opens with no exception")

# ---- Evaluate tab: click 'Load last results' ----
loaded = False
for b in at.button:
    if "Load last results" in b.label:
        b.click(); at.run(); loaded = True; break
assert loaded, "'Load last results' button not found"
assert not at.exception, f"raise after loading eval: {at.exception}"
md_blob2 = " ".join(m.value for m in at.markdown)
assert "RAG Triad" in md_blob2, "RAG Triad section not rendered"
# plotly gauges render as plotly_chart elements
n_charts = len(at.get("plotly_chart")) if "plotly_chart" in [e.type for e in at.get("plotly_chart")] or True else 0
n_df = len(at.dataframe)
assert n_df >= 1, "no dataframe rendered on eval tab"
print(f"[eval] OK - eval results rendered; {n_df} dataframe(s), no Arrow error")

print("\nALL SMOKE TESTS PASSED")
