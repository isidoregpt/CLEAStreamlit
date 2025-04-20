import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import traceback
import logging
import io
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page layout
st.set_page_config(layout="wide")
MAX_DISPLAY_WIDTH = 800

# Add a flag to enable/disable advanced features
USE_FITS = False
USE_SUNPY = False

# Conditional imports
if USE_FITS:
    try:
        from astropy.io import fits
        from astropy.time import Time
        logger.info("Successfully imported astropy modules")
    except ImportError as e:
        logger.error(f"Failed to import astropy: {e}")
        st.error("Astropy module not available. FITS support is disabled.")
        USE_FITS = False

if USE_SUNPY:
    try:
        import sunpy.coordinates
        import sunpy.map
        import astropy.units as u
        from astropy.coordinates import SkyCoord
        from sunpy.coordinates import frames
        logger.info("Successfully imported sunpy modules")
    except ImportError as e:
        logger.error(f"Failed to import sunpy: {e}")
        st.error("SunPy module not available. Solar coordinate support is disabled.")
        USE_SUNPY = False

# Try to import autorefresh
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=10000)  # refresh every 10 seconds
    logger.info("Successfully imported streamlit_autorefresh")
except ImportError as e:
    logger.warning(f"streamlit_autorefresh not available: {e}")

# ------------------------------------------------------------------
#                    HELPER FUNCTIONS
# ------------------------------------------------------------------

def extract_zoom_region(img, cx, cy, zoom_size=60, zoom_factor=4):
    try:
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
    except Exception as e:
        logger.error(f"Error in extract_zoom_region: {e}")
        st.error(f"Failed to extract zoom region: {e}")
        return img  # Return original as fallback

def add_crosshair(img):
    try:
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
    except Exception as e:
        logger.error(f"Error in add_crosshair: {e}")
        st.error(f"Failed to add crosshair: {e}")
        return img  # Return original as fallback

def resize_for_display(img, max_w):
    try:
        w,h = img.size
        if w>max_w:
            r = max_w/float(w)
            return img.resize((max_w,int(h*r)), Image.LANCZOS), r
        return img, 1.0
    except Exception as e:
        logger.error(f"Error in resize_for_display: {e}")
        st.error(f"Failed to resize image: {e}")
        return img, 1.0  # Return original as fallback

def read_fits(file_buf):
    if not USE_FITS:
        st.error("FITS support is disabled.")
        return Image.new("RGB", (100, 100), "black"), None
        
    try:
        hd = fits.open(file_buf)
        data = hd[0].data
        hdr = hd[0].header
        obs = None
        for k in ['DATE-OBS','DATE_OBS','DATE','OBSDATE']:
            if k in hdr:
                try:
                    obs = Time(hdr[k])
                    break
                except Exception as e:
                    logger.warning(f"Failed to parse date {k}: {e}")
                    
        # normalize to 0–255
        arr = np.nan_to_num(data)
        arr -= arr.min()
        if arr.max()>0:
            arr = arr/arr.max()
        img = Image.fromarray((arr*255).astype(np.uint8)).convert("RGB")
        hd.close()
        return img, obs
    except Exception as e:
        logger.error(f"Error in read_fits: {e}")
        st.error(f"Failed to read FITS file: {e}")
        return Image.new("RGB", (100, 100), "black"), None

# ------------------------------------------------------------------
#                         APP STATE
# ------------------------------------------------------------------

if 'images' not in st.session_state:
    st.session_state.images = []           # list of (name, PIL.Image, obs_time)
if 'measurements' not in st.session_state:
    st.session_state.measurements = []

# Sidebar: upload
st.sidebar.header("Upload Images")

try:
    uploads = st.sidebar.file_uploader("PNG or JPG (FITS support optional)", 
                                    type=['fits','fit','png','jpg'],
                                    accept_multiple_files=True)
                                    
    for f in uploads or []:
        if f.name not in [n for n,_,_ in st.session_state.images]:
            try:
                logger.info(f"Processing upload: {f.name}")
                if f.name.lower().endswith(('fits','fit')):
                    if USE_FITS:
                        img, obs = read_fits(f)
                    else:
                        st.warning(f"FITS support is disabled. Skipping {f.name}")
                        continue
                else:
                    try:
                        img = Image.open(f)
                        obs = pd.Timestamp.now()
                    except Exception as e:
                        logger.error(f"Failed to open image {f.name}: {e}")
                        st.error(f"Failed to open {f.name}: {e}")
                        continue
                        
                st.session_state.images.append((f.name, img, obs))
                logger.info(f"Added image: {f.name}, size: {img.width}x{img.height}")
            except Exception as e:
                logger.error(f"Error processing upload {f.name}: {e}")
                st.error(f"Error processing {f.name}: {e}")
except Exception as e:
    logger.error(f"Error in file uploader: {e}")
    st.error(f"File uploader error: {e}")

if not st.session_state.images:
    st.info("Upload at least one image to begin.")
    st.stop()

# Select image
try:
    names = [n for n,_,_ in st.session_state.images]
    idx = st.sidebar.selectbox("Select image", range(len(names)), format_func=lambda i: names[i])
    name, orig_img, obs_time = st.session_state.images[idx]
    logger.info(f"Selected image: {name}")
except Exception as e:
    logger.error(f"Error selecting image: {e}")
    st.error(f"Error selecting image: {e}")
    st.stop()

# Prepare display image
try:
    disp_img, ratio = resize_for_display(orig_img, MAX_DISPLAY_WIDTH)
    bg_array = np.array(disp_img)  # <-- convert PIL to NumPy
    logger.info(f"Display image size: {disp_img.width}x{disp_img.height}, ratio: {ratio}")
except Exception as e:
    logger.error(f"Error preparing display image: {e}")
    st.error(f"Error preparing display image: {e}")
    st.stop()

# Layout: columns
left, right = st.columns([3,2])

with left:
    st.subheader(f"Image: {name}")
    try:
        st.image(disp_img, use_column_width=False, width=disp_img.width)
    except Exception as e:
        logger.error(f"Error displaying image: {e}")
        st.error(f"Error displaying image: {e}")

    # Draw canvas with NumPy background
    try:
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
        logger.info("Canvas initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing canvas: {e}")
        st.error(f"Error initializing canvas: {e}")

with right:
    st.subheader("Measurements")
    try:
        if canvas_result and canvas_result.json_data and canvas_result.json_data.get("objects"):
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
                    "Time": obs_time.iso if hasattr(obs_time, 'iso') else str(obs_time),
                })
                logger.info(f"Recorded measurement: {name}, x={x_orig:.1f}, y={y_orig:.1f}")
    except Exception as e:
        logger.error(f"Error processing canvas result: {e}")
        st.error(f"Error processing canvas result: {e}")

    st.write("---")
    st.subheader("Recorded")
    try:
        if st.session_state.measurements:
            df = pd.DataFrame(st.session_state.measurements)
            st.dataframe(df)
            csv = df.to_csv(index=False)
            st.download_button("Download CSV", csv, "measurements.csv")
            logger.info(f"Displaying {len(df)} measurements")
    except Exception as e:
        logger.error(f"Error displaying measurements: {e}")
        st.error(f"Error displaying measurements: {e}")
