import streamlit as st
import numpy as np
import json
import io
import zipfile
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import astropy_mpl_style, ZScaleInterval
import matplotlib.pyplot as plt

st.set_page_config(page_title="FITS → PNG+JSON Converter", layout="wide")
st.title("☀️ FITS to PNG + JSON Converter")
st.markdown(
    """
    Upload `.fit` or `.fits` files.  
    This app will:
    1. Read each FITS file, replacing NaNs  
    2. Extract **all** header cards + observation time + solar parameters  
    3. Compute basic data stats (min/max/mean/median/std/dtype)  
    4. Render a grayscale PNG via z‐scale  
    5. Bundle each PNG + metadata JSON into a ZIP for download  
    """
)

uploads = st.file_uploader(
    "Upload FITS files", type=["fit", "fits"], accept_multiple_files=True
)

if uploads and st.button("▶️ Convert all"):
    with st.spinner("Processing…"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            for u in uploads:
                name = u.name.rsplit(".", 1)[0]

                # --- Read FITS ---
                hdul = fits.open(u)
                data = hdul[0].data
                header = hdul[0].header
                hdul.close()
                data = np.nan_to_num(data)

                # --- Build metadata ---
                metadata = {}

                # 1) Dump entire header
                header_dict = {}
                for key in header.keys():
                    # Card values may be numpy types—cast to native
                    val = header[key]
                    try:
                        header_dict[key] = val.tolist() if hasattr(val, "tolist") else val
                    except:
                        header_dict[key] = str(val)
                metadata["header"] = header_dict

                # 2) Observation time
                for k in ("DATE-OBS", "DATE_OBS", "DATE", "OBSDATE"):
                    if k in header:
                        try:
                            metadata["observation_time"] = Time(header[k]).iso
                            break
                        except:
                            pass

                # 3) Solar parameters
                sun_keys = ("FNDLMBXC", "FNDLMBYC", "FNDLMBMI", "FNDLMBMA")
                if all(k in header for k in sun_keys):
                    try:
                        cx = float(header["FNDLMBXC"])
                        cy = float(header["FNDLMBYC"])
                        minor = float(header["FNDLMBMI"])
                        major = float(header["FNDLMBMA"])
                        radius = (minor + major) / 4
                        metadata["sun_params"] = {"cx": cx, "cy": cy, "radius": radius}
                    except:
                        pass
                if "sun_params" not in metadata:
                    h, w = data.shape
                    metadata["sun_params"] = {
                        "cx": w/2, "cy": h/2, "radius": min(w, h) * 0.45
                    }

                # 4) Dimensions & dtype
                h, w = data.shape
                metadata["width"] = w
                metadata["height"] = h
                metadata["dtype"] = str(data.dtype)

                # 5) Basic statistics
                metadata["data_stats"] = {
                    "min": float(np.min(data)),
                    "max": float(np.max(data)),
                    "mean": float(np.mean(data)),
                    "median": float(np.median(data)),
                    "std": float(np.std(data))
                }

                # --- Render PNG ---
                plt.style.use(astropy_mpl_style)
                fig = plt.figure(figsize=(6, 6), dpi=150)
                vmin, vmax = ZScaleInterval().get_limits(data)
                plt.imshow(data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
                plt.axis("off")
                buf = io.BytesIO()
                fig.savefig(buf, bbox_inches="tight", pad_inches=0)
                plt.close(fig)
                buf.seek(0)

                # --- Write into ZIP ---
                zipf.writestr(f"{name}.png", buf.read())
                zipf.writestr(f"{name}.json", json.dumps(metadata, indent=2).encode("utf-8"))

        zip_buffer.seek(0)
        st.success("Conversion complete!")
        st.download_button(
            "📦 Download PNG+JSON ZIP",
            zip_buffer,
            "fits_converted.zip",
            mime="application/zip"
        )

    # Inline previews & JSON
    st.header("Previews & Metadata")
    zip_data = zip_buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
        for u in uploads:
            name = u.name.rsplit(".", 1)[0]
            st.subheader(name)
            st.image(zf.read(f"{name}.png"), use_column_width=True)
            st.json(json.loads(zf.read(f"{name}.json").decode()))
