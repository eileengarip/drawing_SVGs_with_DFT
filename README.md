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
   A sampled curve \( x[n] \) is transformed using the Discrete Fourier Transform:

   \[
   X_k = \sum_{n=0}^{N-1} x[n] e^{-2\pi i k n / N}.
   \]
