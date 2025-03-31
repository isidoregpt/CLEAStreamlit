import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
from io import BytesIO
import base64
from astropy.io import fits
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# SunPy and Astropy imports for coordinate transformations
import sunpy.coordinates
import sunpy.map
import astropy.units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from astropy.time import Time

# Configure page layout
st.set_page_config(layout="wide")
MAX_DISPLAY_WIDTH = 800

# Add CSS for overlay positioning
st.markdown("""
<style>
.image-canvas-container {
    position: relative;
    width: fit-content;
    margin: 0;
    padding: 0;
}
.image-layer {
    position: absolute;
    top: 0;
    left: 0;
    z-index: 1;
}
.canvas-layer {
    position: absolute;
    top: 0;
    left: 0;
    z-index: 2;
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
#                    NEW HELPER FUNCTIONS FOR ZOOM VIEWER
# ------------------------------------------------------------------

def extract_zoom_region(image, center_x, center_y, zoom_size=60, zoom_factor=4):
    """
    Extract a zoomed region around a point from an image.
    Returns the zoomed image and the (left, top) of the extracted region.
    """
    half_size = zoom_size // 2
    left = max(0, int(center_x - half_size))
    top = max(0, int(center_y - half_size))
    right = min(image.width, int(center_x + half_size))
    bottom = min(image.height, int(center_y + half_size))
    region = image.crop((left, top, right, bottom))
    zoomed = region.resize((int(region.width * zoom_factor), int(region.height * zoom_factor)), Image.LANCZOS)
    return zoomed, (left, top)

def add_crosshair(zoomed_image):
    """
    Add a crosshair overlay to the zoomed image.
    """
    if zoomed_image.mode != 'RGBA':
        zoomed_image = zoomed_image.convert('RGBA')
    overlay = Image.new('RGBA', zoomed_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = zoomed_image.size
    cx, cy = w // 2, h // 2
    line_color = (255, 0, 255, 200)  # Magenta, semi-transparent
    line_width = 2
    draw.line([(0, cy), (w, cy)], fill=line_color, width=line_width)
    draw.line([(cx, 0), (cx, h)], fill=line_color, width=line_width)
    # Optional border
    border_width = 2
    for i in range(border_width):
        draw.rectangle([i, i, w-i-1, h-i-1], outline=line_color, width=1)
    return Image.alpha_composite(zoomed_image, overlay).convert('RGB')

# ------------------------------------------------------------------
#                         HELPER FUNCTIONS
# ------------------------------------------------------------------

def pixel_to_heliographic(x, y, center_x, center_y, radius):
    """
    Convert pixel coordinates to approximate heliographic coordinates.
    """
    x_norm = (x - center_x) / radius
    y_norm = (y - center_y) / radius
    dist = np.sqrt(x_norm**2 + y_norm**2)
    if dist < 1e-6:
        return 0.0, 0.0
    theta = np.arctan2(y_norm, x_norm)
    if dist > 0.98:
        longitude = np.degrees(np.arcsin(np.clip(np.cos(theta), -1.0, 1.0)))
        if x_norm < 0:
            longitude = -longitude
        if abs(y_norm) > 0.94:
            longitude = 0.0
        latitude = 90.0 * np.sin(theta)
    else:
        longitude = np.degrees(np.arcsin(np.clip(x_norm, -0.98, 0.98)))
        cos_lon_rad = np.cos(np.radians(longitude))
        if abs(cos_lon_rad) < 0.1:
            latitude = np.sign(y_norm) * 90.0
        else:
            latitude = np.degrees(np.arcsin(np.clip(y_norm / cos_lon_rad, -1.0, 1.0)))
    return longitude, latitude

def accurate_pixel_to_heliographic(x, y, center_x, center_y, radius, obs_time):
    """
    Use SunPy to convert pixel coordinates to heliographic Stonyhurst coordinates.
    Falls back to a simpler calculation if necessary.
    """
    try:
        if isinstance(obs_time, str) or isinstance(obs_time, datetime):
            obs_time = Time(obs_time)
        x_norm = (x - center_x) / radius
        y_norm = (y - center_y) / radius
        distance_from_center = np.sqrt(x_norm**2 + y_norm**2)
        if distance_from_center > 0.90:
            print(f"Point is near the limb ({distance_from_center:.2f} of radius)")
        scale = 959.63 / radius  # arcsec per pixel at 1 solar radius
        x_arcsec = (x - center_x) * scale * u.arcsec
        y_arcsec = (center_y - y) * scale * u.arcsec  # invert y-axis
        hpc = SkyCoord(x_arcsec, y_arcsec,
                       frame=frames.Helioprojective(obstime=obs_time, observer="earth"))
        hgs = hpc.transform_to(frames.HeliographicStonyhurst(obstime=obs_time))
        if np.isnan(hgs.lon.deg) or np.isnan(hgs.lat.deg):
            return pixel_to_heliographic(x, y, center_x, center_y, radius)
        return hgs.lon.deg, hgs.lat.deg
    except Exception as e:
        print(f"SunPy coordinate transform failed: {e}")
        return pixel_to_heliographic(x, y, center_x, center_y, radius)

def is_point_on_sun(x, y, center_x, center_y, radius, radius_correction=1.0, x_offset=0, y_offset=0):
    """
    Check if (x, y) lies within the adjusted solar disk boundary.
    """
    adjusted_center_x = center_x + x_offset
    adjusted_center_y = center_y + y_offset
    adjusted_radius = radius * radius_correction
    distance = np.hypot(x - adjusted_center_x, y - adjusted_center_y)
    return distance <= (adjusted_radius * 1.05)

def resize_image_for_display(image: Image.Image, max_width: int):
    """
    Resize the image to a maximum width while preserving aspect ratio.
    Returns the resized image and the scale ratio.
    """
    w, h = image.width, image.height
    if w > max_width:
        ratio = max_width / float(w)
        new_w = max_width
        new_h = int(h * ratio)
        resized = image.resize((new_w, new_h), Image.LANCZOS)
        return resized, ratio
    else:
        return image, 1.0

def determine_sun_center_and_radius(image, contour_threshold):
    """
    Determine the Sun's center and radius using a contour-based method.
    """
    img_array = np.array(image.convert('L'))
    height, width = img_array.shape
    try:
        from scipy import ndimage
        from skimage import measure
        threshold = np.percentile(img_array, contour_threshold)
        binary = img_array > threshold
        filled = ndimage.binary_fill_holes(binary)
        contours = measure.find_contours(filled, 0.5)
        if contours:
            largest = max(contours, key=len)
            y_coords, x_coords = largest[:, 0], largest[:, 1]
            cx = np.mean(x_coords)
            cy = np.mean(y_coords)
            distances = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            radius = np.mean(distances)
            return cx, cy, radius
    except Exception:
        pass
    try:
        cx = width / 2
        cy = height / 2
        threshold = np.percentile(img_array, 30)
        from scipy import ndimage
        binary = img_array > threshold
        edges = ndimage.binary_erosion(binary) ^ binary
        y_indices, x_indices = np.where(edges)
        if len(x_indices) > 100:
            distances = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
            radius = np.percentile(distances, 50)
            return cx, cy, radius
    except Exception:
        pass
    cx = width / 2
    cy = height / 2
    radius = min(width, height) * 0.475
    return cx, cy, radius

def calculate_selection_center(rect):
    """
    Given a rectangle (with keys 'left', 'top', 'width', 'height'), return its center.
    """
    left = rect.get('left', 0)
    top = rect.get('top', 0)
    width = rect.get('width', 0)
    height = rect.get('height', 0)
    return left + width/2, top + height/2

def read_fits_file(file_buffer):
    """
    Read a FITS file and return (img, obs_time, sun_params),
    where sun_params = (cx, cy, radius).
    """
    hdul = fits.open(file_buffer)
    data = hdul[0].data
    header = hdul[0].header
    obs_time = None
    for key in ['DATE-OBS','DATE_OBS','DATE','OBSDATE']:
        if key in header:
            try:
                obs_time = Time(header[key])
                break
            except:
                pass
    sun_params = None
    req_keys = ["FNDLMBXC","FNDLMBYC","FNDLMBMI","FNDLMBMA"]
    if all(k in header for k in req_keys):
        try:
            cx = float(header["FNDLMBXC"])
            cy = float(header["FNDLMBYC"])
            minor_axis = float(header["FNDLMBMI"])
            major_axis = float(header["FNDLMBMA"])
            avg_diameter = (minor_axis + major_axis) / 2
            radius_val = avg_diameter / 2
            sun_params = (cx, cy, radius_val)
            print(f"FITS header: center=({cx:.2f},{cy:.2f}), radius={radius_val:.2f}")
        except Exception as e:
            print(f"Error reading solar params from FITS header: {e}")
    hdul.close()
    data = np.nan_to_num(data)
    data = data - data.min()
    if data.max() > 0:
        data = data/data.max()
    data = (data*255).astype(np.uint8)
    img = Image.fromarray(data)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img, obs_time, sun_params

def create_sun_boundary(image, cx, cy, radius, width=2):
    """
    Draw a thin red circle around the solar disk boundary.
    """
    boundary_img = image.copy()
    draw = ImageDraw.Draw(boundary_img)
    left = cx - radius
    top = cy - radius
    right = cx + radius
    bottom = cy + radius
    draw.ellipse((left, top, right, bottom), outline=(255, 0, 0), width=width)
    return boundary_img

# ------------------------------------------------------------------
#                         STREAMLIT APP
# ------------------------------------------------------------------

# Session state initialization
if 'images' not in st.session_state:
    st.session_state['images'] = {}  # {filename: (PIL.Image, obs_time)}
if 'sun_params' not in st.session_state:
    st.session_state['sun_params'] = {}  # {filename: (cx, cy, r)}
if 'measurements' not in st.session_state:
    st.session_state['measurements'] = []
if 'animation_running' not in st.session_state:
    st.session_state['animation_running'] = False
if 'current_image_index' not in st.session_state:
    st.session_state['current_image_index'] = 0
if 'animation_speed' not in st.session_state:
    st.session_state['animation_speed'] = 500
if 'selection_mode' not in st.session_state:
    st.session_state['selection_mode'] = 'rect'
if 'detection_method' not in st.session_state:
    st.session_state['detection_method'] = 'center'
if 'contour_threshold' not in st.session_state:
    st.session_state['contour_threshold'] = 15  # percentile for contour detection
if 'zoom_size' not in st.session_state:
    st.session_state['zoom_size'] = 60
if 'zoom_factor' not in st.session_state:
    st.session_state['zoom_factor'] = 4

# Title & instructions
st.title("Advanced Solar Rotation Analysis")
st.write("""
**How to Use**  
1. Upload your solar images (FITS recommended) in the sidebar.  
2. If your FITS has solar disk keywords (`FNDLMBXC`, etc.), enable "Force FITS header data."  
3. Otherwise, or if header data are missing, set "Detect Sun Edge" and adjust "Contour Threshold" to refine the boundary.  
4. Pause animation to measure sunspots with either "Rectangle Selection" or "Point Selection."  
5. Heliographic coordinates are computed with SunPy (fallback if that fails).  
6. When a region is selected, a zoom viewer will appear next to the main image so you can inspect the area and record the measurement.
""")

# --- Sidebar: Configuration
st.sidebar.header("Configuration")
force_fits_data = st.sidebar.checkbox("Always use FITS header data", value=True)
radius_correction = st.sidebar.slider(
    "Sun Boundary Size Adjustment",
    min_value=1.0, max_value=2.5, value=2.0, step=0.05,
    help="Adjusts the size of the Sun boundary circle (values above 1.0 make it larger)"
)
st.sidebar.markdown("#### Fine-tune Circle Position")
center_x_offset = st.sidebar.slider(
    "Horizontal Center Adjustment",
    min_value=-50, max_value=50, value=-21, step=1,
    help="Adjusts the horizontal position of the circle (negative = left, positive = right)"
)
center_y_offset = st.sidebar.slider(
    "Vertical Center Adjustment",
    min_value=-50, max_value=50, value=0, step=1,
    help="Adjusts the vertical position of the circle (negative = up, positive = down)"
)
st.session_state['selection_mode'] = st.sidebar.radio(
    "Selection Mode",
    ['rect','point'],
    index=(0 if st.session_state['selection_mode']=='rect' else 1),
    format_func=lambda x: "Rectangle" if x=='rect' else "Point"
)
if not force_fits_data:
    st.session_state['detection_method'] = st.sidebar.radio(
        "Sun Detection Method",
        ['center','detect'],
        index=(0 if st.session_state['detection_method']=='center' else 1),
        format_func=lambda x: "Assume Centered" if x=='center' else "Contour Detection"
    )
else:
    st.session_state['detection_method'] = 'center'
st.session_state['contour_threshold'] = st.sidebar.slider(
    "Contour Threshold (Percentile for Method 1)",
    min_value=1, max_value=50, value=st.session_state['contour_threshold'], step=1
)
show_sun_boundary = st.sidebar.checkbox("Show Sun Boundary Circle", value=True)
disable_boundary_check = st.sidebar.checkbox("Disable Boundary Check", value=False)

# --- Sidebar: Animation Controls
st.sidebar.header("Animation Controls")
st.session_state['animation_speed'] = st.sidebar.slider(
    "Animation Speed (ms per frame)",
    min_value=100, max_value=2000, value=st.session_state['animation_speed'], step=100
)
if st.session_state['animation_running']:
    if st.sidebar.button("Pause"):
        st.session_state['animation_running'] = False
else:
    if st.sidebar.button("Play"):
        st.session_state['animation_running'] = True
colA, colB = st.sidebar.columns(2)
if colA.button("Previous"):
    st.session_state['current_image_index'] = max(0, st.session_state['current_image_index']-1)
    st.session_state['animation_running'] = False
if colB.button("Next"):
    st.session_state['current_image_index'] = min(
        len(st.session_state['images'])-1 if st.session_state['images'] else 0,
        st.session_state['current_image_index']+1
    )
    st.session_state['animation_running'] = False

# --- Sidebar: File Uploader
st.sidebar.header("Load Images")
files = st.sidebar.file_uploader(
    "Upload (JPG, PNG, FITS, or FIT)",
    type=["jpg","png","fits","fit"],
    accept_multiple_files=True
)
if files:
    new_loaded = False
    for file in files:
        if file.name not in st.session_state['images']:
            try:
                is_fits = file.name.lower().endswith((".fits",".fit"))
                if is_fits:
                    img, obs_time, header_params = read_fits_file(file)
                else:
                    img = Image.open(file)
                    obs_time = Time(datetime.now())
                    header_params = None
                if obs_time is None:
                    obs_time = Time(datetime.now())
                st.session_state['images'][file.name] = (img, obs_time)
                if is_fits and header_params and force_fits_data:
                    st.session_state['sun_params'][file.name] = header_params
                    st.success(f"Using FITS header for {file.name}")
                else:
                    if st.session_state['detection_method'] == 'center':
                        w, h = img.width, img.height
                        cx, cy = w/2, h/2
                        radius = min(w, h)*0.45
                        st.session_state['sun_params'][file.name] = (cx, cy, radius)
                    else:
                        cx, cy, radius = determine_sun_center_and_radius(img, st.session_state['contour_threshold'])
                        st.session_state['sun_params'][file.name] = (cx, cy, radius)
                    if is_fits and not header_params:
                        st.warning(f"No disk params in header for {file.name}. Used detection.")
                new_loaded = True
            except Exception as e:
                st.error(f"Error with {file.name}: {e}")
    if new_loaded:
        st.session_state['current_image_index'] = 0

if not st.session_state['images']:
    st.info("No images loaded yet.")
    st.stop()

# Sort filenames and ensure current index is valid
filenames_sorted = sorted(list(st.session_state['images'].keys()))
if st.session_state['current_image_index'] >= len(filenames_sorted):
    st.session_state['current_image_index'] = 0
img_index = st.sidebar.slider(
    "Image Index",
    0, len(filenames_sorted)-1,
    st.session_state['current_image_index']
)
st.session_state['current_image_index'] = img_index

if st.session_state['animation_running']:
    st_autorefresh(interval=st.session_state['animation_speed'], limit=100000, key="anim_refresh")
    st.session_state['current_image_index'] = (st.session_state['current_image_index'] + 1) % len(filenames_sorted)

# ------------------------------------------------------------------
#                          MAIN LAYOUT
# ------------------------------------------------------------------
left_col, right_col = st.columns([3, 2])

# Retrieve the selected image & parameters
current_filename = filenames_sorted[st.session_state['current_image_index']]
orig_img, obs_time = st.session_state['images'][current_filename]
cx, cy, radius = st.session_state['sun_params'].get(current_filename, (None, None, None))
if cx is None:
    w, h = orig_img.width, orig_img.height
    cx, cy, radius = w/2, h/2, min(w, h)*0.45
orig_w, orig_h = orig_img.width, orig_img.height

# Make a copy for display and resize it
display_img = orig_img.copy()
resized_img, ratio = resize_image_for_display(display_img, MAX_DISPLAY_WIDTH)
scaled_cx = cx * ratio
scaled_cy = cy * ratio
scaled_radius = radius * ratio
if show_sun_boundary and cx is not None:
    adjusted_cx = scaled_cx + center_x_offset
    adjusted_cy = scaled_cy + center_y_offset
    corrected_radius = scaled_radius * radius_correction
    resized_img = create_sun_boundary(resized_img, adjusted_cx, adjusted_cy, corrected_radius, width=2)

# Left column: Main image/canvas
with left_col:
    st.subheader(f"Current Image: {current_filename}")
    if st.session_state['animation_running']:
        st.image(resized_img, use_column_width=False, width=resized_img.width)
    else:
        drawing_mode = st.session_state['selection_mode']
        
        try:
            # Try a direct approach first (this might work in some Streamlit Cloud environments)
            canvas_result = st_canvas(
                fill_color="rgba(255,0,255,0.3)",
                stroke_width=2,
                stroke_color="#FF00FF",
                background_image=resized_img,
                update_streamlit=True,
                height=resized_img.height,
                width=resized_img.width,
                drawing_mode=drawing_mode,
                key="canvas_measurement"
            )
        except Exception as e:
            st.error(f"Canvas error: {type(e).__name__}. Using fallback approach.")
            
            # Fallback approach using HTML and a separate canvas
            st.image(resized_img, use_column_width=False, width=resized_img.width)
            
            # Create an alternative interface for point selection
            st.write("**Click Selection Coordinates:**")
            col1, col2 = st.columns(2)
            with col1:
                x_input = st.number_input("X coordinate:", 0, resized_img.width, resized_img.width//2)
            with col2:
                y_input = st.number_input("Y coordinate:", 0, resized_img.height, resized_img.height//2)
                
            submit_point = st.button("Use these coordinates")
            
            # Create a synthetic canvas_result to use with the rest of the code
            class DummyObject:
                def __init__(self, x, y):
                    self.left = x
                    self.top = y
                    self.width = 10
                    self.height = 10
                    
            class DummyCanvas:
                def __init__(self, x, y, active=False):
                    self.json_data = {"objects": [DummyObject(x, y)]} if active else {"objects": []}
                    
            if submit_point:
                canvas_result = DummyCanvas(x_input, y_input, active=True)
            else:
                canvas_result = DummyCanvas(0, 0, active=False)

# Right column: Data, measurements, and (if applicable) the zoom viewer
with right_col:
    measure_tab, image_tab = st.tabs(["Measurements & Data", "Image Info"])
    
    with measure_tab:
        # When not animating and a selection has been made
        if not st.session_state['animation_running'] and 'canvas_result' in locals():
            if canvas_result.json_data:
                objs = canvas_result.json_data.get("objects", [])
                if objs:
                    if drawing_mode == 'rect':
                        x_disp, y_disp = calculate_selection_center(objs[-1])
                    else:
                        x_disp = objs[-1].get("left", 0)
                        y_disp = objs[-1].get("top", 0)
                    
                    # Check boundary using adjusted values
                    adjusted_cx = scaled_cx + center_x_offset
                    adjusted_cy = scaled_cy + center_y_offset
                    if disable_boundary_check or is_point_on_sun(
                        x_disp, y_disp,
                        scaled_cx, scaled_cy,
                        scaled_radius,
                        radius_correction=radius_correction,
                        x_offset=center_x_offset,
                        y_offset=center_y_offset
                    ):
                        st.write(f"**Display coords:** x = {x_disp:.2f}, y = {y_disp:.2f}")
                        x_orig = x_disp / ratio
                        y_orig = y_disp / ratio
                        st.write(f"**Original coords:** x = {x_orig:.2f}, y = {y_orig:.2f}")
                        lon, lat = accurate_pixel_to_heliographic(x_orig, y_orig, cx, cy, radius, obs_time)
                        st.write(f"**Heliographic:** Lon = {lon:.2f}°, Lat = {lat:.2f}°")
                        dist = np.hypot(x_orig - cx, y_orig - cy)
                        st.write(f"**Distance from center:** {dist:.1f} px ({dist/radius*100:.1f}% of radius)")
                        
                        st.write("---")
                        # NEW: Display the zoom viewer only when a selection is made
                        zoom_img, (zleft, ztop) = extract_zoom_region(orig_img, x_orig, y_orig,
                                                                       zoom_size=st.session_state['zoom_size'],
                                                                       zoom_factor=st.session_state['zoom_factor'])
                        final_zoom = add_crosshair(zoom_img)
                        zoom_col, data_col = st.columns([1, 1])
                        with zoom_col:
                            st.image(final_zoom, caption="Zoomed Region")
                        with data_col:
                            st.write("### Measurement Data")
                            st.write(f"**Observation Time:** {obs_time.iso if obs_time else 'Unknown'}")
                            st.write(f"**Heliographic Coordinates:** Lon = {lon:.2f}°, Lat = {lat:.2f}°")
                            st.write(f"**Pixel Coords (Original):** ({x_orig:.2f}, {y_orig:.2f})")
                            label = st.text_input("Feature label:", value="", key="label_input")
                            if st.button("Record Measurement", key="record_btn"):
                                measurement = {
                                    "Image": current_filename,
                                    "Observation Time": obs_time.iso if obs_time else "Unknown",
                                    "Pixel X (orig)": x_orig,
                                    "Pixel Y (orig)": y_orig,
                                    "Helio Longitude": lon,
                                    "Helio Latitude": lat,
                                    "Distance (% radius)": dist/radius*100,
                                    "Label": label
                                }
                                st.session_state['measurements'].append(measurement)
                                st.success("Measurement recorded!")
                    else:
                        st.warning("Selected point is outside the solar disk. Uncheck 'Disable Boundary Check' or pick a point inside.")
                        st.write(f"**Debug:** x = {x_disp:.1f}, y = {y_disp:.1f}, center = ({adjusted_cx:.1f}, {adjusted_cy:.1f}), radius = {scaled_radius * radius_correction:.1f}")
                else:
                    st.info("Draw a selection on the image to measure solar features.")
            else:
                st.info("Draw a selection on the image to measure solar features.")
        else:
            if st.session_state['animation_running']:
                st.info("Pause the animation to make measurements.")
            else:
                st.info("Draw a selection on the image to measure solar features.")
        
        st.write("---")
        st.subheader("Recorded Measurements")
        if st.session_state['measurements']:
            df = pd.DataFrame(st.session_state['measurements'])
            if 'Label' in df.columns and df['Label'].nunique() > 1:
                all_labels = ["All"] + list(df['Label'].unique())
                sel_label = st.selectbox("Filter by label:", options=all_labels)
                if sel_label != "All":
                    df = df[df['Label'] == sel_label]
            st.dataframe(df, height=300)
            csv_data = df.to_csv(index=False)
            st.download_button("Download CSV", data=csv_data, file_name="solar_measurements.csv", mime="text/csv")
            
            if (df['Label'].nunique() == 1 and len(df) >= 2 and "Observation Time" in df.columns 
                and df['Label'].iloc[0] != ''):
                st.subheader("Simple Rotation Analysis")
                try:
                    df['Time'] = pd.to_datetime(df['Observation Time'])
                    df = df.sort_values('Time')
                    df['Hours'] = (df['Time'] - df['Time'].iloc[0]).dt.total_seconds() / 3600
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(6, 3))
                    ax.scatter(df['Hours'], df['Helio Longitude'], c='magenta')
                    ax.set_xlabel("Hours since first measurement")
                    ax.set_ylabel("Helio Longitude (°)")
                    ax.set_title(f"Longitude changes: {df['Label'].iloc[0]}")
                    ax.grid(True)
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Analysis error: {e}")
        else:
            st.write("No measurements recorded yet.")
        
        with st.expander("About Solar Differential Rotation", expanded=False):
            st.write("""
            The Sun doesn't rotate like a solid body—rotation speed depends on latitude:
            - ~25 days at the equator
            - ~36 days near the poles

            Tracking sunspots over time reveals this differential rotation.
            """)
    
    with image_tab:
        st.write(f"**Original size:** {orig_w} x {orig_h}")
        st.write(f"**Display size:** {resized_img.width} x {resized_img.height}")
        if obs_time:
            st.write(f"**Observation time:** {obs_time.iso}")
        param_source = "FITS Header" if (force_fits_data and current_filename.lower().endswith((".fits", ".fit"))) else "Detection"
        st.write(f"**Sun center:** ({cx:.1f}, {cy:.1f}) px, **Radius:** {radius:.1f} px")
        st.write(f"*Parameters source: {param_source}*")
        if show_sun_boundary:
            st.write(f"**Scale:** {ratio:.3f}x, scaled center = ({scaled_cx:.1f}, {scaled_cy:.1f}), radius = {scaled_radius:.1f}")
            st.write(f"**Circle adjustments:** Size: {radius_correction:.2f}x, X-offset: {center_x_offset}px, Y-offset: {center_y_offset}px")
