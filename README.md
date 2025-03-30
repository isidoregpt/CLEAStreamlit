# CLEAStreamlit
An app with similar functionality to CLEA

Advanced Solar Rotation Analysis
A Streamlit web application for tracking and analyzing solar features across multiple images to study solar rotation patterns.
Show Image
Overview
This tool allows astronomers, solar physicists, and enthusiasts to:

Load and analyze multiple solar images (FITS, JPG, PNG formats)
Precisely identify and track sunspots and other solar features
Calculate heliographic coordinates using SunPy transformations
Measure differential rotation rates across the solar surface
Export measurement data for further analysis

Features

Solar Image Processing:

Support for FITS astronomical data files with metadata extraction
Automatic solar disk detection with adjustable parameters
Manual fine-tuning of solar disk position and size


Interactive Analysis Tools:

Rectangle and point selection modes for feature identification
Dynamic zoom viewer for detailed feature inspection
Precise heliographic coordinate calculations using SunPy
Animation controls for sequential image analysis


Measurement & Data Collection:

Track features across multiple images
Record and organize measurements with custom labels
Export data as CSV for external analysis
Basic rotation analysis visualization



Installation
Prerequisites

Python 3.7+
pip

Required Packages
bashCopypip install streamlit streamlit-drawable-canvas streamlit-autorefresh pillow numpy pandas astropy sunpy matplotlib
For the contour detection feature, additional dependencies are required:
bashCopypip install scipy scikit-image
Usage

Launch the application:
bashCopystreamlit run solar_rotation_app.py

Upload solar images via the sidebar:

FITS files (.fits, .fit) are recommended for scientific analysis
Common image formats (.jpg, .png) are also supported


Configure disk detection settings:

Enable "Force FITS header data" to use embedded metadata (when available)
Adjust contour threshold for automatic disk detection
Fine-tune circle position with sliders if needed


Use measurement tools:

Select between rectangle or point selection modes
Click or draw on features of interest
View the zoomed region for precise placement
Add labels to track specific features across images


View and export results:

Browse recorded measurements in the data table
Filter measurements by feature label
Download data as CSV
View simple rotation analysis charts



Sun Detection Methods
The application offers several methods to identify the solar disk:

FITS Header Data (most accurate): Uses solar disk parameters from FITS file headers
Contour Detection: Automatically finds the solar disk edge using image processing
Centered Assumption: Assumes the Sun is centered in the image with a standard radius

Coordinate Systems
The app calculates two types of coordinates:

Pixel Coordinates: Raw (x,y) position in the image
Heliographic Coordinates: Solar longitude and latitude in degrees

Heliographic coordinates are calculated using:

SunPy's coordinate transformation system (primary method)
A fallback approximation for edge cases or when SunPy fails

Differential Rotation Analysis
Solar differential rotation can be studied by:

Tracking the same feature across multiple images
Labeling measurements consistently
Viewing the longitude vs. time plot generated for features with multiple measurements

Limitations

Best results obtained with properly calibrated FITS files
Accuracy decreases near the solar limb
Simple approximations are used when SunPy transformations fail

Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Streamlit for the web application framework
SunPy for solar coordinate transformations
Astropy for astronomical data handling
