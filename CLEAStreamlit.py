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

st.set_page_config(layout="wide")
MAX_DISPLAY_WIDTH = 800

# ──────────────── ARRAY WRAPPER TO FIX TRUTHINESS ERROR ────────────────

class BGArray(np.ndarray):
    """A tiny subclass of np.ndarray that always truth‐tests True."""
    def __new__(cls, arr):
        obj = np.asarray(arr).view(cls)
        return obj
    def __bool__(self):
        return True

# ──────────────── HELPER FUNCTIONS ────────────────

def read_fits_file(buf):
    hd = fits.open(buf)
    data = hd[0].data
    hdr  = hd[0].header
    obs  = None
    for k in ['DATE-OBS','DATE_OBS','DATE','OBSDATE']:
        if k in hdr:
            try:
                obs = Time(hdr[k])
                break
            except:
                pass
    arr = np.nan_to_num(data)
    arr -= arr.min()
    if arr.max()>0:
        arr = arr/arr.max()
    img = Image.fromarray((arr*255).astype(np.uint8)).convert("RGB")
    hd.close()
    return img, obs

def resize_for_display(img, max_w):
    w,h = img.size
    if w>max_w:
        r = max_w/float(w)
        return img.resize((max_w,int(h*r)), Image.LANCZOS), r
    return img, 1.0

def extract_zoom_region(img, cx, cy, zoom_size=60, zoom_factor=4):
    half = zoom_size//2
    left  = max(0, int(cx-half))
    top   = max(0, int(cy-half))
    right = min(img.width,  int(cx+half))
    bot   = min(img.height, int(cy+half))
    region = img.crop((left,top,right,bot))
    return region.resize(
      (int(region.width*zoom_factor), int(region.height*zoom_factor)),
      Image.LANCZOS
    )

def add_crosshair(img):
    if img.mode!='RGBA':
        img = img.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (0,0,0,0))
    draw    = ImageDraw.Draw(overlay)
    w,h     = img.size
    cx,cy   = w//2, h//2
    color   = (255,0,255,200)
    draw.line([(0,cy),(w,cy)], fill=color, width=2)
    draw.line([(cx,0),(cx,h)], fill=color, width=2)
    for i in range(2):
        draw.rectangle([i,i,w-i-1,h-i-1], outline=color, width=1)
    return Image.alpha_composite(img, overlay).convert('RGB')

def calculate_selection_center(obj):
    return obj.get('left',0)+obj.get('width',0)/2, obj.get('top',0)+obj.get('height',0)/2

def pixel_to_heliographic(x,y,cx,cy,r):
    x_n,y_n = (x-cx)/r,(y-cy)/r
    d = np.hypot(x_n,y_n)
    if d<1e-6: return 0.0,0.0
    θ = np.arctan2(y_n,x_n)
    if d>0.98:
        lon = np.degrees(np.arcsin(np.clip(np.cos(θ),-1,1)))
        if x_n<0: lon=-lon
        if abs(y_n)>0.94: lon=0.0
        lat=90*np.sin(θ)
    else:
        lon = np.degrees(np.arcsin(np.clip(x_n,-0.98,0.98)))
        c = np.cos(np.radians(lon))
        lat = np.sign(y_n)*90 if abs(c)<0.1 else np.degrees(np.arcsin(np.clip(y_n/c,-1,1)))
    return lon,lat

def accurate_pixel_to_heliographic(x,y,cx,cy,r,obs):
    try:
        if isinstance(obs,(str,datetime)): obs=Time(obs)
        scale=959.63/r
        x_arc=(x-cx)*scale*u.arcsec
        y_arc=(cy-y)*scale*u.arcsec
        hpc=SkyCoord(x_arc,y_arc,
                     frame=frames.Helioprojective(obstime=obs,observer="earth"))
        hgs=hpc.transform_to(frames.HeliographicStonyhurst(obstime=obs))
        if np.isnan(hgs.lon.deg) or np.isnan(hgs.lat.deg):
            return pixel_to_heliographic(x,y,cx,cy,r)
        return hgs.lon.deg, hgs.lat.deg
    except:
        return pixel_to_heliographic(x,y,cx,cy,r)

# ──────────────── SESSION STATE ────────────────

if 'images' not in st.session_state:
    st.session_state.images = {}      # filename → (PIL.Image, obs_time)
if 'sun_params' not in st.session_state:
    st.session_state.sun_params = {}
if 'measurements' not in st.session_state:
    st.session_state.measurements = []
if 'anim_running' not in st.session_state:
    st.session_state.anim_running = False
if 'img_idx' not in st.session_state:
    st.session_state.img_idx = 0

# ──────────────── SIDEBAR ────────────────

st.sidebar.header("Load Images")
uploads = st.sidebar.file_uploader(
    "FITS, PNG, JPG", type=['fits','fit','png','jpg'],
    accept_multiple_files=True
)
for f in uploads or []:
    if f.name not in st.session_state.images:
        if f.name.lower().endswith(('fits','fit')):
            img, obs = read_fits_file(f)
        else:
            img = Image.open(f)
            obs = Time(datetime.now())
        st.session_state.images[f.name] = (img, obs)

if not st.session_state.images:
    st.info("Upload at least one image.")
    st.stop()

# Config
force_header = st.sidebar.checkbox("Force FITS header", True)
show_boundary = st.sidebar.checkbox("Show Sun boundary", True)
st.sidebar.header("Animation")
speed = st.sidebar.slider("ms per frame", 100, 2000, 500, 100)
if st.session_state.anim_running:
    if st.sidebar.button("Pause"):
        st.session_state.anim_running=False
else:
    if st.sidebar.button("Play"):
        st.session_state.anim_running=True

# ──────────────── MAIN ────────────────

names = list(st.session_state.images)
n = len(names)
idx = st.sidebar.slider("Image index", 0, n-1, st.session_state.img_idx)
st.session_state.img_idx = idx
fname = names[idx]
orig_img, obs_time = st.session_state.images[fname]

# Determine center & radius once
if fname not in st.session_state.sun_params:
    w,h = orig_img.size
    st.session_state.sun_params[fname] = (w/2, h/2, min(w,h)*0.45)
cx, cy, r = st.session_state.sun_params[fname]

# Resize and overlay boundary
disp_img, ratio = resize_for_display(orig_img, MAX_DISPLAY_WIDTH)
if show_boundary:
    draw = ImageDraw.Draw(disp_img)
    scaled_r = r*ratio
    draw.ellipse(
      (cx*ratio-scaled_r, cy*ratio-scaled_r,
       cx*ratio+scaled_r, cy*ratio+scaled_r),
      outline=(255,0,0), width=2
    )

# Animate?
if st.session_state.anim_running:
    st_autorefresh(interval=speed, limit=100000, key="anim")
    st.session_state.img_idx = (idx+1)%n

col1, col2 = st.columns([3,2])

with col1:
    st.subheader(f"Image: {fname}")
    st.image(disp_img, width=disp_img.width)

    # draw!
    bg = np.array(disp_img)
    bg = BGArray(bg)  # <-- wrap to avoid ValueError
    canvas = st_canvas(
        background_image=bg,
        update_streamlit=True,
        height=disp_img.height,
        width=disp_img.width,
        stroke_color="#FF00FF",
        fill_color="rgba(255,0,255,0.3)",
        stroke_width=2,
        drawing_mode="rect",
        key="canv"
    )

with col2:
    st.subheader("Measurements")
    if canvas.json_data and canvas.json_data.get("objects"):
        o = canvas.json_data["objects"][-1]
        x_d, y_d = calculate_selection_center(o)
        x_o, y_o = x_d/ratio, y_d/ratio
        lon, lat = accurate_pixel_to_heliographic(x_o,y_o,cx,cy,r,obs_time)
        st.write(f"**Orig coords:** {x_o:.1f}, {y_o:.1f}")
        st.write(f"**Helio:** lon={lon:.1f}°, lat={lat:.1f}°")
        zoom = extract_zoom_region(orig_img, x_o, y_o)
        zoom = add_crosshair(zoom)
        st.image(zoom, caption="Zoomed")
        if st.button("Record"):
            st.session_state.measurements.append({
                "Image": fname,
                "Time": obs_time.iso if obs_time else "",
                "X": x_o, "Y": y_o,
                "Lon": lon, "Lat": lat
            })
    else:
        st.info("Draw a rectangle to measure.")

    st.write("---")
    if st.session_state.measurements:
        df = pd.DataFrame(st.session_state.measurements)
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "measurements.csv")
