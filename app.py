"""Window Quote dashboard. Run:  streamlit run app.py

Upload a photo of each side of a building -> every window boxed -> grouped into floors ->
priced per floor ($3, $6, $12 ... by default) -> per-side and building total.
"""
import hashlib
import io
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps

import counter as c

c.load_env()
# Streamlit Cloud secrets, if deployed there
try:
    for k in ("ROBOFLOW_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"):
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = st.secrets[k]
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SIDES = ["Front", "Back", "Left", "Right"]
USD = "\\$"  # Streamlit markdown reads $...$ as maths, so dollar signs in markdown text are escaped
SAMPLES = {
    "1-storey house": "01_1storey_maracaibo.jpg",
    "2-storey house (irregular)": "03_2storey_buffalo.jpg",
    "3-storey townhouse with dormers": "06_apt_karlsruhe.jpg",
    "Victorian house (irregular)": "10_irregular_knox.jpg",
    "Modern house (irregular)": "11_irregular_wien.jpg",
    "Apartment block, angled + balconies (hard)": "07_apt_hirzenbach.jpg",
}

st.set_page_config(page_title="Window Quote", page_icon="🪟", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; max-width: 1400px;}
  div[data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #E3E8F2; border-radius: 12px;
      padding: 14px 18px;}
  div[data-testid="stMetricValue"] {color: #08142D;}
  .check {color:#B26B00; font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ---------- processing (cached: re-pricing never re-calls the APIs) ----------

def _prep(data: bytes) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    img.thumbnail((1600, 1600))  # phone photos are huge; this is plenty for window boxes and much faster
    return img


@st.cache_data(show_spinner=False, max_entries=64)
def analyse(data: bytes, thresh: float, floors: str) -> dict:
    img = _prep(data)
    t = time.time()
    res = c.run_side(img, floors=floors, thresh=thresh)
    return {"boxes": [(b.x0, b.y0, b.x1, b.y1, b.conf, b.floor) for b in res.boxes],
            "n_floors": len(res.floors), "note": res.note, "seconds": round(time.time() - t, 1)}


def to_result(name: str, a: dict, base: float, mult: float) -> c.SideResult:
    boxes = [c.Box(x0, y0, x1, y1, conf, fl) for x0, y0, x1, y1, conf, fl in a["boxes"]]
    res = c.SideResult(name, boxes, note=a["note"])
    res.floors = c.price(boxes, a["n_floors"], base, mult)
    return res


# ---------- sidebar ----------

with st.sidebar:
    st.header("Pricing")
    base = st.number_input("Floor 1 price per window ($)", min_value=0.0, value=3.0, step=0.5)
    mult = st.number_input("Each floor up costs ×", min_value=1.0, value=2.0, step=0.25,
                           help="2 means every floor costs double the one below: $3, $6, $12, $24 …")
    st.caption(" · ".join(f"F{n}: {USD}{base * mult ** (n - 1):g}" for n in range(1, 7)))
    st.divider()
    st.header("Detection")
    floors_mode = st.selectbox("Floor grouping", ["ai", "geometric"],
                               format_func=lambda m: {"ai": "AI + geometry check (best)",
                                                      "geometric": "Geometry only (no AI)"}[m])
    thresh = st.slider("Detector confidence", 0.2, 0.8, 0.4, 0.05,
                       help="Lower finds more windows but risks false ones. 0.4 is the tested default.")
    st.caption(f"Amber boxes are below {c.CHECK_BELOW:.2f} confidence; their floor is flagged to check.")
    missing = [k for k in ("ROBOFLOW_API_KEY", "OPENAI_API_KEY") if not os.environ.get(k)]
    if missing:
        st.error("Missing keys: " + ", ".join(missing))


# ---------- input ----------

st.title("Window Quote from Photos")
st.caption("Photo of each side → every window boxed → grouped into floors → priced per floor → building total. "
           "Only building windows are counted; car windows and neighbouring buildings are dropped.")

src = st.radio("Photos", ["Upload my building", "Try a sample building"], horizontal=True,
               label_visibility="collapsed")
inputs: dict[str, bytes] = {}
if src == "Upload my building":
    cols = st.columns(4)
    for col, side in zip(cols, SIDES):
        f = col.file_uploader(side, type=["jpg", "jpeg", "png", "webp"], key=f"up_{side}")
        if f:
            inputs[side] = f.getvalue()
            col.image(f, width="stretch")
    st.caption("Straight-on photos work best: stand across the street, whole side in frame.")
else:
    pick = st.selectbox("Sample building (never seen by the system before)", list(SAMPLES))
    inputs["Front"] = open(os.path.join(HERE, "photos", SAMPLES[pick]), "rb").read()
    st.image(inputs["Front"], width=360)

go = st.button("Count windows and price it", type="primary", disabled=not inputs)
key = hashlib.md5(b"".join(inputs.values()) + f"{thresh}{floors_mode}".encode()).hexdigest() if inputs else ""

if go:
    t0 = time.time()
    with st.spinner(f"Counting windows on {len(inputs)} side(s)…"):
        with ThreadPoolExecutor(max_workers=4) as ex:  # sides run in parallel
            futs = {name: ex.submit(analyse, data, thresh, floors_mode) for name, data in inputs.items()}
            done = {}
            for name, fut in futs.items():
                try:
                    done[name] = fut.result()
                except Exception as e:
                    st.error(f"{name}: failed ({e})")
    st.session_state.update(results=done, key=key, wall=round(time.time() - t0, 1),
                            images={n: inputs[n] for n in done}, edits={})

if not st.session_state.get("results") or st.session_state.get("key") != key:
    st.stop()


# ---------- results ----------

results: dict = st.session_state["results"]
edits: dict = st.session_state.setdefault("edits", {})
sides = {name: to_result(name, a, base, mult) for name, a in results.items()}

# apply any hand-corrected counts, then re-price
for name, res in sides.items():
    for fl in res.floors:
        if (name, fl.floor) in edits:
            fl.windows = edits[(name, fl.floor)]
            fl.subtotal = fl.windows * fl.price_each

total = sum(r.total for r in sides.values())
windows = sum(r.windows for r in sides.values())
to_check = sum(fl.check for r in sides.values() for fl in r.floors)

m = st.columns(5)
m[0].metric("Building quote", f"${total:,.2f}")
m[1].metric("Windows", windows)
m[2].metric("Sides", len(sides))
m[3].metric("Floors to check", to_check)
m[4].metric("Time", f"{st.session_state['wall']}s")

tabs = st.tabs([f"{n}  ·  ${r.total:,.0f}" for n, r in sides.items()] + ["Building summary"])
for tab, (name, res) in zip(tabs, sides.items()):
    with tab:
        left, right = st.columns([3, 2], gap="large")
        img = _prep(st.session_state["images"][name])
        left.image(c.draw(img, res), width="stretch")
        with right:
            st.subheader(f"{name}: {res.windows} windows, {USD}{res.total:,.2f}")
            df = pd.DataFrame([{"Floor": fl.floor, "Windows": fl.windows, "$ each": fl.price_each,
                                "Subtotal $": fl.subtotal, "Check": "⚠ check" if fl.check else ""}
                               for fl in reversed(res.floors)])
            edited = st.data_editor(df, hide_index=True, width="stretch", key=f"ed_{name}",
                                    disabled=["Floor", "$ each", "Subtotal $", "Check"],
                                    column_config={"Windows": st.column_config.NumberColumn(
                                        help="Wrong count? Type the right number; the price updates.",
                                        min_value=0, step=1)})
            changed = False
            for _, row in edited.iterrows():
                orig = next(fl for fl in res.floors if fl.floor == row["Floor"])
                if int(row["Windows"]) != orig.windows:
                    edits[(name, int(row["Floor"]))] = int(row["Windows"])
                    changed = True
            if changed:
                st.rerun()
            if any(fl.check for fl in res.floors):
                st.markdown('<span class="check">⚠ Amber boxes are windows the detector was unsure about. '
                            'Check those floors before sending the quote.</span>', unsafe_allow_html=True)
            if res.note:
                st.info(res.note)
            st.caption(f"Counted in {results[name]['seconds']}s. Edit the Windows column to correct a count.")

with tabs[-1]:
    n_max = max((len(r.floors) for r in sides.values()), default=0)
    rows = []
    for n in range(n_max, 0, -1):
        row = {"Floor": n, "$ each": base * mult ** (n - 1)}
        for name, r in sides.items():
            fl = next((f for f in r.floors if f.floor == n), None)
            row[name] = fl.windows if fl else 0
        row["Windows"] = sum(row[name] for name in sides)
        row["Subtotal $"] = row["Windows"] * row["$ each"]
        rows.append(row)
    summary = pd.DataFrame(rows)
    st.dataframe(summary, hide_index=True, width="stretch")
    st.markdown(f"### Total: **{USD}{total:,.2f}** for **{windows}** windows")
    st.download_button("Download quote (CSV)", summary.to_csv(index=False).encode(), "window-quote.csv",
                       "text/csv")
