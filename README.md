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

   ```
   ['M', 100, 200, 'C', 120,180, 150,160, 200,200, ...]
   ```

   A regex breaks the string into commands + floats. This allows us to iterate through commands.

3. Convert Path Commands into Geometric Segments

   Example output:

   ```
   ('L', p0, p1)
   ('Q', p0, control, end)
   ('C', p0, c1, c2, end)
   ```

   Each tuple describes ONE geometric segment. For example, ('L', cur, new) means a line segment from the      current point to the new point. Complex numbers (e.g. 20 + 50j) store 2-D points conveniently.

4. Compute Length of Each Segment
   - For straight lines, length is exact.
   - For Bézier curves, we approximate by sampling them at 80 points and summing tiny distances.
   - This is required for uniform arc-length sampling.

5. Sample the Entire Path at Even Spacing
   We walk along segments and pick points spaced equally in total curve length. This ensures uniform speed     around the drawing, correct DFT behaviour, and smooth epicycle animation.

6. Compute the DFT
   A sampled points are transformed using the Discrete Fourier Transform. (Expain more in the poster, the      math behind)

7. Animate Epicycles
   Beginning at the origin, each term draws a rotating vector that adds onto all previous vectors, whose       end traces the SVG outline. This recreates the thistle using pure Fourier series.

<h2>Files in this Project</h2>

```
thistle.svg        # Input SVG file
fourier_trace.py   # Full Python source code
README.md          # This file
```

<h2>Dependencies</h2>
- Python 3
- NumPy
- Matplotlib
- xml.etree.ElementTree (built-in)

install with:

```
pip install numpy matplotlib
```

<h2>How to Run</h2>

```
python fourier_trace.py
```

This will:
- Parse the SVG
- Sample the path
- Compute Fourier coefficients
- Generate an animation window
- Plot the reconstructed drawing

<h2>References</h2>

1. YouTube: “But what is a Fourier series? From heat flow to drawing with circles (DE4)”
2. SVG Path specification — W3C
3. Notes on Bézier curves — Pomax
4. NumPy FFT Documentation
5. Matplotlib Animation Docs
6. Wolfram MathWorld — “Arc Length”
7. re (Regular Expression) Python Docs
8. Brigham, E. O. The Fast Fourier Transform and Its Applications
9. Oppenheim & Schafer — Discrete-Time Signal Processing
10. Python xml.etree.ElementTree Documentation
11. Inkscape / Illustrator SVG documentation
12. The LLM ChatGPT_o4 was used for some of the finer implementation details, such as regex's.
