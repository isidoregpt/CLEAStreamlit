import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from astropy.io import fits
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# SunPy / Astropy for coordinates
import sunpy.coordinates
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from astropy.time import Time

# Configure page
st.set_page_config(layout="wide")
MAX_DISPLAY_WIDTH = 800

# ---------------------------
# Helper functions
# ---------------------------

def extract_zoom_region(img, cx, cy, zoom_size=60, zoom_factor=4):
    half = zoom_size // 2
    left = max(0, int(cx - half))
    top = max(0, int(cy - half))
    right = min(img.width, int(cx + half))
    bottom = min(img.height, int(cy + half))
    region = img.crop((left, top, right, bottom))
    return region.resize(
        (int(region.width * zoom_factor), int(region.height * zoom_factor)),
        Image.LANCZOS
    )

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
    w, h = img.size
    if w > max_w:
        r = max_w/float(w)
        return img.resize((max_w, int(h*r)), Image.LANCZOS), r
    return img, 1.0

def read_fits_file(fbuf):
    hdul = fits.open(fbuf)
    data = hdul[0].data
    hdr = hdul[0].header
    obs = None
    for k in ['DATE-OBS','DATE_OBS','DATE','OBSDATE']:
        if k in hdr:
            try:
                obs = Time(hdr[k])
                break
            except:
                pass
    arr = np.nan_to_num(data)
    arr -= arr.min()
    if arr.max() > 0:
        arr = arr/arr.max()
    img = Image.fromarray((arr*255).astype(np.uint8)).convert("RGB")
    hdul.close()
    return img, obs

def pixel_to_heliographic(x, y, cx, cy, r):
    x_n, y_n = (x-cx)/r, (y-cy)/r
    d = np.hypot(x_n, y_n)
    if d < 1e-6:
        return 0.0, 0.0
    theta = np.arctan2(y_n, x_n)
    if d > 0.98:
        lon = np.degrees(np.arcsin(np.clip(np.cos(theta), -1, 1)))
        if x_n < 0: lon = -lon
        if abs(y_n) > 0.94: lon = 0.0
        lat = 90*np.sin(theta)
    else:
        lon = np.degrees(np.arcsin(np.clip(x_n, -0.98, 0.98)))
        c = np.cos(np.radians(lon))
        lat = np.sign(y_n)*90 if abs(c)<0.1 else np.degrees(
            np.arcsin(np.clip(y_n/c, -1, 1))
        )
    return lon, lat

def accurate_pixel_to_heliographic(x, y, cx, cy, r, obs_time):
    try:
        if isinstance(obs_time, (str, datetime)):
            obs_time = Time(obs_time)
        scale = 959.63 / r
        x_arc = (x-cx)*scale * u.arcsec
        y_arc = (cy-y)*scale * u.arcsec
        hpc = SkyCoord(x_arc, y_arc,
                       frame=frames.Helioprojective(obstime=obs_time, observer="earth"))
        hgs = hpc.transform_to(frames.HeliographicStonyhurst(obstime=obs_time))
        if np.isnan(hgs.lon.deg) or np.isnan(hgs.lat.deg):
            return pixel_to_heliographic(x, y, cx, cy, r)
        return hgs.lon.deg, hgs.lat.deg
    except:
        return pixel_to_heliographic(x, y, cx, cy, r)

def calculate_selection_center(rect):
    return rect.get("left",0) + rect.get("width",0)/2, rect.get("top",0) + rect.get("height",0)/2

# ---------------------------
# Session state
# ---------------------------

if 'images' not in st.session_state:
    st.session_state.images = {}   # name → (PIL.Image, obs_time)
if 'sun_params' not in st.session_state:
    st.session_state.sun_params = {}
if 'measurements' not in st.session_state:
    st.session_state.measurements = []
if 'animation_running' not in st.session_state:
    st.session_state.animation_running = False
if 'current_image_index' not in st.session_state:
    st.session_state.current_image_index = 0

# ---------------------------
# Sidebar: configuration
# ---------------------------

st.sidebar.header("Configuration")
force_fits_data = st.sidebar.checkbox("Always use FITS header data", value=True)
radius_correction = st.sidebar.slider("Sun boundary scale", 1.0, 2.5, 2.0, 0.05)
center_x_offset = st.sidebar.slider("X offset", -50, 50, -21, 1)
center_y_offset = st.sidebar.slider("Y offset", -50, 50, 0, 1)
st.session_state.selection_mode = st.sidebar.radio(
    "Selection mode", ['rect','point'],
    index=0, format_func=lambda x: "Rectangle" if x=='rect' else "Point"
)
if not force_fits_data:
    st.session_state.detection_method = st.sidebar.radio(
        "Detection", ['center','detect'], index=0,
        format_func=lambda x: "Centered" if x=='center' else "Contour"
    )
else:
    st.session_state.detection_method = 'center'
st.session_state.contour_threshold = st.sidebar.slider("Contour threshold", 1, 50, 15, 1)
show_sun_boundary = st.sidebar.checkbox("Show boundary", value=True)
disable_boundary_check = st.sidebar.checkbox("Disable boundary check", value=False)

# Animation controls
st.sidebar.header("Animation")
st.session_state.animation_speed = st.sidebar.slider("Speed (ms)", 100, 2000, 500, 100)
if st.session_state.animation_running:
    if st.sidebar.button("Pause"):
        st.session_state.animation_running = False
else:
    if st.sidebar.button("Play"):
        st.session_state.animation_running = True
col1, col2 = st.sidebar.columns(2)
if col1.button("Prev"):
    st.session_state.current_image_index = max(0, st.session_state.current_image_index-1)
    st.session_state.animation_running = False
if col2.button("Next"):
    n = len(st.session_state.images)
    st.session_state.current_image_index = min(n-1, st.session_state.current_image_index+1)
    st.session_state.animation_running = False

# Uploader
st.sidebar.header("Load images")
files = st.sidebar.file_uploader("Upload FITS/PNG/JPG", type=['fits','fit','png','jpg'], accept_multiple_files=True)
if files:
    for f in files:
        if f.name not in st.session_state.images:
            if f.name.lower().endswith(('fits','fit')):
                img, obs = read_fits_file(f)
                # optionally read header params here...
            else:
                img = Image.open(f)
                obs = Time(datetime.now())
            st.session_state.images[f.name] = (img, obs)
if not st.session_state.images:
    st.info("No images loaded.")
    st.stop()

# ---------------------------
# Main layout
# ---------------------------

# Select and retrieve
names = list(st.session_state.images.keys())
idx = st.sidebar.slider("Image index", 0, len(names)-1, st.session_state.current_image_index)
st.session_state.current_image_index = idx
name = names[idx]
orig_img, obs_time = st.session_state.images[name]

# Determine sun center/radius
if name in st.session_state.sun_params:
    cx, cy, r = st.session_state.sun_params[name]
else:
    w,h = orig_img.size
    if st.session_state.detection_method=='center':
        cx, cy, r = w/2, h/2, min(w,h)*0.45
    else:
        cx, cy, r = (lambda i,th: i[:3])(None, st.session_state.contour_threshold)  # can use your detect fn
    st.session_state.sun_params[name] = (cx, cy, r)

# Resize for display
disp_img, ratio = resize_for_display(orig_img, MAX_DISPLAY_WIDTH)
scaled_cx, scaled_cy, scaled_r = cx*ratio, cy*ratio, r*ratio
if show_sun_boundary:
    disp_img = disp_img.copy()
    ImageDraw.Draw(disp_img).ellipse(
        (scaled_cx-radius_correction*scaled_r, scaled_cy-radius_correction*scaled_r,
         scaled_cx+radius_correction*scaled_r, scaled_cy+radius_correction*scaled_r),
        outline=(255,0,0), width=2
    )

# Auto‐refresh if animating
if st.session_state.animation_running:
    st_autorefresh(interval=st.session_state.animation_speed, limit=100000, key="anim")
    st.session_state.current_image_index = (st.session_state.current_image_index+1) % len(names)

left, right = st.columns([3,2])

with left:
    st.subheader(f"Current Image: {name}")
    if st.session_state.animation_running:
        st.image(disp_img, width=disp_img.width)
    else:
        drawing_mode = st.session_state.selection_mode
        bg_arr = np.array(disp_img)   # <-- THE FIX
        canvas_result = st_canvas(
            fill_color="rgba(255,0,255,0.3)",
            stroke_width=2,
            stroke_color="#FF00FF",
            background_image=bg_arr,
            update_streamlit=True,
            height=disp_img.height,
            width=disp_img.width,
            drawing_mode=drawing_mode,
            key="canvas"
        )

with right:
    tab1, tab2 = st.tabs(["Measurements","Image Info"])
    with tab1:
        if not st.session_state.animation_running and 'canvas_result' in locals():
            data = canvas_result.json_data or {}
            objs = data.get("objects", [])
            if objs:
                obj = objs[-1]
                if drawing_mode=='rect':
                    x_d, y_d = calculate_selection_center(obj)
                else:
                    x_d, y_d = obj.get("left",0), obj.get("top",0)
                x_o, y_o = x_d/ratio, y_d/ratio
                lon, lat = accurate_pixel_to_heliographic(x_o, y_o, cx, cy, r, obs_time)
                dist = np.hypot(x_o-cx, y_o-cy)
                st.write(f"**Orig coords:** x={x_o:.1f}, y={y_o:.1f}")
                st.write(f"**Helio:** lon={lon:.1f}°, lat={lat:.1f}°")
                st.write(f"**Dist:** {dist:.1f}px ({dist/r*100:.1f}% radius)")

                zoom = extract_zoom_region(orig_img, x_o, y_o)
                zoom = add_crosshair(zoom)
                st.image(zoom, caption="Zoomed")

                label = st.text_input("Label", key="lbl")
                if st.button("Record"):
                    st.session_state.measurements.append({
                        "Image": name,
                        "Time": obs_time.iso if obs_time else "",
                        "X": x_o, "Y": y_o,
                        "Lon": lon, "Lat": lat,
                        "Label": label
                    })
            else:
                st.info("Draw to measure.")
        else:
            st.info("Pause to measure.")

        st.write("---")
        if st.session_state.measurements:
            df = pd.DataFrame(st.session_state.measurements)
            st.dataframe(df)
            st.download_button("Download CSV", df.to_csv(index=False), "data.csv")
    with tab2:
        w,h = orig_img.size
        st.write(f"**Orig size:** {w}×{h}")
        st.write(f"**Display:** {disp_img.width}×{disp_img.height}")
        if obs_time: st.write(f"**Time:** {obs_time.iso}")
        st.write(f"**Center:** ({cx:.1f},{cy:.1f})  **Radius:** {r:.1f}")
