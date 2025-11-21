# Fourier Epicycle Drawing from SVG Paths
<h2> Overview </h2>

This project takes an SVG image, extracts its path data, samples the path into a sequence of evenly spaced points, and then uses a Discrete Fourier Transform (DFT) to reconstruct the shape using rotating circles, also known as epicycles.

This workflow is inspired by the YouTube video“But what is a Fourier series? From heat flow to drawing with circles (DE4)”, which demonstrates how Fourier series can describe closed curves.

The result is a fully animated Fourier reconstruction of any SVG outline; here, we use a thistle illustration (A symbol for Edinburgh!)

Project Goals
- Convert an SVG path (with lines and Bézier curves) into sampled coordinate points.
- Compute the Fourier coefficients of these points.
- Animate a reconstruction using rotating vectors (epicycles).
- Provide clear Python code demonstrating:
-  - Path tokenisation
   - Bézier curve sampling
   - Arc-length parametrisation
   - DFT computation
   - Visualization
     
This forms the mathematical and computational content for the Edinburgh coursework poster.

<h2> How it works. </h2>
1. Parse the SVG file

```
  tree = ET.parse(SVG_FILE)
  root = tree.getroot()
  path_elems = root.findall('.//{http://www.w3.org/2000/svg}path') \
             + root.findall('.//path')
```

2. Tokenise SVG Path
   SVG path strings contain commands (M, L, C, etc.) mixed with numbers.
    We convert them into a clean list of tokens: 
