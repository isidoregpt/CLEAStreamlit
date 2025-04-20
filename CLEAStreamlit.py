import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from astropy.io import fits
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

import sunpy.coordinates
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from astropy.time import Time

# Configure page layout
st.set_page_config(layout="wide")
MAX_DISPLAY_WIDTH = 800

# ------------------------------------------------------------------
#                    HELPER FUNCTIONS
# ------------------------------------------------------------------

def extract_zoom_region(img, cx, cy, zoom_size=60, zoom_factor=4):
    half = zoom_size // 2
    left = max(0, int(cx - half))
    top = max(0, int(cy - half))
    right = min(img.width, int(cx + half))
    bottom = min(img.height, int(cy + half))
    region = img.crop((left, top, right, bottom))
    zoomed = region.resize((int(region.width * zoom_factor),
                             int(region.height * zoom_factor)),
                            Image.LANCZOS)
    return zoomed

def add_crosshair(img):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0,0,0,0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.size
    cx, cy = w//2, h//2
    color = (255,0,255,200)
    draw.line([(0,cy),(w,cy)], fill=color, width=2)
    draw.line([(cx,0),(cx,h)], fill=color, width=2)
    for i in range(2):
        draw.rectangle([i,i,w-i-1,h-i-1], outline=color, width=1)
    return Image.alpha_composite(img, overlay).convert("RGB")

def resize_for_display(img, max_w):
    w,h = img.size
    if w>max_w:
        r = max_w/float(w)
        return img.resize((max_w,int(h*r)), Image.LANCZOS), r
    return img, 1.0

def read_fits(file_buf):
    hd = fits.open(file_buf)
    data = hd[0].data
    hdr = hd[0].header
    obs = None
    for k in ['DATE-OBS','DATE_OBS','DATE','OBSDATE']:
        if k in hdr:
            try:
                obs = Time(hdr[k])
                break
            except:
                pass
    # normalize to 0–255
    arr = np.nan_to_num(data)
    arr -= arr.min()
    if arr.max()>0:
        arr = arr/arr.max()
    img = Image.fromarray((arr*255).astype(np.uint8)).convert("RGB")
    hd.close()
    return img, obs

# ------------------------------------------------------------------
#                         APP STATE
# ------------------------------------------------------------------

if 'images' not in st.session_state:
    st.session_state.images = []           # list of (name, PIL.Image, obs_time)
if 'measurements' not in st.session_state:
    st.session_state.measurements = []

# Sidebar: upload
st.sidebar.header("Upload Images")
uploads = st.sidebar.file_uploader("FITS, PNG or JPG", 
                                   type=['fits','fit','png','jpg'],
                                   accept_multiple_files=True)
for f in uploads or []:
    if f.name not in [n for n,_,_ in st.session_state.images]:
        if f.name.lower().endswith(('fits','fit')):
            img, obs = read_fits(f)
        else:
            img = Image.open(f)
            obs = Time(datetime.now())
        st.session_state.images.append((f.name, img, obs))

if not st.session_state.images:
    st.info("Upload at least one image to begin.")
    st.stop()

# Select image
names = [n for n,_,_ in st.session_state.images]
idx = st.sidebar.selectbox("Select image", range(len(names)), format_func=lambda i: names[i])
name, orig_img, obs_time = st.session_state.images[idx]

# Prepare display image
disp_img, ratio = resize_for_display(orig_img, MAX_DISPLAY_WIDTH)
bg_array = np.array(disp_img)  # <-- convert PIL to NumPy

# Layout: columns
left, right = st.columns([3,2])

with left:
    st.subheader(f"Image: {name}")
    st.image(disp_img, use_column_width=False, width=disp_img.width)

    # Draw canvas with NumPy background
    canvas_result = st_canvas(
        background_image=bg_array,
        update_streamlit=True,
        height=disp_img.height,
        width=disp_img.width,
        stroke_color="#FF00FF",
        fill_color="rgba(255,0,255,0.3)",
        stroke_width=2,
        drawing_mode="rect",
        key="canvas",
    )

with right:
    st.subheader("Measurements")
    if canvas_result.json_data and canvas_result.json_data.get("objects"):
        obj = canvas_result.json_data["objects"][-1]
        # rectangle center
        x_disp = obj["left"] + obj["width"]/2
        y_disp = obj["top"]  + obj["height"]/2
        x_orig = x_disp/ratio
        y_orig = y_disp/ratio

        st.write(f"**Coordinates (orig):** x={x_orig:.1f}, y={y_orig:.1f}")

        # zoom
        zoomed = extract_zoom_region(orig_img, x_orig, y_orig)
        zoomed = add_crosshair(zoomed)
        st.image(zoomed, caption="Zoomed Region")

        if st.button("Record measurement"):
            st.session_state.measurements.append({
                "Image": name,
                "X": x_orig,
                "Y": y_orig,
                "Time": obs_time.iso if obs_time else "",
            })

    st.write("---")
    st.subheader("Recorded")
    if st.session_state.measurements:
        df = pd.DataFrame(st.session_state.measurements)
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "measurements.csv")

