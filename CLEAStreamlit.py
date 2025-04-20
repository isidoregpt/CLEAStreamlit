import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd

# Configure page layout
st.set_page_config(layout="wide")
MAX_DISPLAY_WIDTH = 800

# ------------------------------------------------------------------
#                         APP STATE
# ------------------------------------------------------------------

if 'images' not in st.session_state:
    st.session_state.images = []           # list of (name, PIL.Image, timestamp)
if 'measurements' not in st.session_state:
    st.session_state.measurements = []     # list of measurements

# Sidebar: upload
st.sidebar.header("Upload Images")
uploads = st.sidebar.file_uploader("PNG or JPG", 
                                   type=['png','jpg'],
                                   accept_multiple_files=True)

for f in uploads or []:
    if f.name not in [n for n,_,_ in st.session_state.images]:
        try:
            img = Image.open(f)
            timestamp = pd.Timestamp.now()
            st.session_state.images.append((f.name, img, timestamp))
            st.sidebar.success(f"Loaded: {f.name}")
        except Exception as e:
            st.sidebar.error(f"Error loading {f.name}: {str(e)}")

if not st.session_state.images:
    st.info("Upload at least one image to begin.")
    st.stop()

# Select image
names = [n for n,_,_ in st.session_state.images]
idx = st.sidebar.selectbox("Select image", range(len(names)), format_func=lambda i: names[i])
name, orig_img, timestamp = st.session_state.images[idx]

# Prepare display image
w, h = orig_img.size
if w > MAX_DISPLAY_WIDTH:
    ratio = MAX_DISPLAY_WIDTH / float(w)
    disp_img = orig_img.resize((MAX_DISPLAY_WIDTH, int(h * ratio)))
else:
    disp_img = orig_img
    ratio = 1.0

# Layout: columns
left, right = st.columns([3,2])

with left:
    st.subheader(f"Image: {name}")
    st.image(disp_img, use_column_width=False)

with right:
    st.subheader("Image Information")
    st.write(f"Original size: {orig_img.width} x {orig_img.height}")
    st.write(f"Display size: {disp_img.width} x {disp_img.height}")
    
    # Simple click coordinates
    st.write("Click on the image to record a position")
    if st.button("Record Position (center)"):
        x_orig = orig_img.width / 2
        y_orig = orig_img.height / 2
        st.session_state.measurements.append({
            "Image": name,
            "X": x_orig,
            "Y": y_orig,
            "Time": str(timestamp),
        })
        st.success("Recorded center position")

    st.write("---")
    st.subheader("Recorded Positions")
    if st.session_state.measurements:
        df = pd.DataFrame(st.session_state.measurements)
        st.dataframe(df)
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "measurements.csv")
