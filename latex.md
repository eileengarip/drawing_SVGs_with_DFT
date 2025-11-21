\section*{README --- Fourier Epicycle Drawing from SVG Paths}

\subsection*{Overview}

This project takes an \textbf{SVG image}, extracts its \textbf{path data}, samples the path into evenly spaced points, and then uses the \textbf{Discrete Fourier Transform (DFT)} to reconstruct the shape using rotating circles (epicycles).  

This workflow is inspired by the YouTube video:
\begin{quote}
\emph{``But what is a Fourier series? From heat flow to drawing with circles (DE4)''}.
\end{quote}

The result is a Fourier reconstruction of a thistle outline from its SVG path.

\subsection*{Project Goals}

\begin{itemize}
    \item Parse SVG paths containing lines and Bézier curves.
    \item Sample the entire path at uniform arc-length spacing.
    \item Compute Fourier coefficients of the sampled points.
    \item Animate the reconstruction using epicycles.
    \item Provide clear and readable Python code demonstrating each step.
\end{itemize}

\subsection*{How the Code Works}

\subsubsection*{1. Parse the SVG File}

The SVG is loaded using \texttt{xml.etree.ElementTree}.  
Path elements are extracted via:

\begin{verbatim}
path_elems = root.findall('.//{http://www.w3.org/2000/svg}path')
            + root.findall('.//path')
\end{verbatim}

This ensures both namespaced and non-namespaced SVG files are supported.

\subsubsection*{2. Tokenize the Path}

The SVG path string (the \texttt{d} attribute) mixes commands such as
\texttt{M, L, C, Q} with floating numbers.  
A regular expression separates these into an easy-to-read token list.

\subsubsection*{3. Convert Tokens to Geometric Segments}

Each SVG command is turned into a tuple describing a geometric object, for example:
\[
\texttt{('L', p_0, p_1)} \qquad \text{(line segment)}
\]
\[
\texttt{('C', p_0, c_1, c_2, p_3)} \qquad \text{(cubic Bézier)}
\]

All points are stored as complex numbers \(x + iy\).

\subsubsection*{4. Compute Segment Lengths}

To sample the curve uniformly, we need the length of each segment.

\begin{itemize}
    \item Lines: exact length using \(|p_1 - p_0|\).
    \item Bézier curves: approximate length by sampling many points and summing small distances.
\end{itemize}

\subsubsection*{5. Sample the Entire Path Evenly}

The total arc length is computed, then points are chosen at equal spacing along the curve.  
This prevents distortions in the Fourier reconstruction.

\subsubsection*{6. Compute the Discrete Fourier Transform}

The sampled points \(x[n]\) are complex numbers representing the curve.  
Their Fourier coefficients are computed using:

\[
X_k = \sum_{n=0}^{N-1} x[n] \, e^{-2\pi i k n / N}.
\]

Here,
\begin{itemize}
    \item \(|X_k|\) is the radius of the \(k\)-th epicycle,
    \item the argument of \(X_k\) is its starting phase,
    \item \(k\) is the rotation frequency.
\end{itemize}

\subsection*{Files in This Project}

\begin{itemize}
    \item \texttt{thistle.svg} --- input SVG file.
    \item \texttt{fourier\_trace.py} --- main code.
    \item \texttt{README.tex} --- this document.
\end{itemize}

\subsection*{Dependencies}

\begin{itemize}
    \item Python 3
    \item NumPy
    \item Matplotlib
    \item \texttt{xml.etree.ElementTree} (built-in)
\end{itemize}

Installation:
\begin{verbatim}
pip install numpy matplotlib
\end{verbatim}

\subsection*{How to Run}

\begin{verbatim}
python fourier_trace.py
\end{verbatim}

This will:
\begin{enumerate}
    \item Parse the SVG.
    \item Sample its path.
    \item Compute Fourier coefficients.
    \item Animate the reconstruction with rotating vectors.
\end{enumerate}

\subsection*{References}

\begin{itemize}
    \item Grant Sanderson, \emph{``But what is a Fourier series?''}, YouTube.
    \item W3C SVG Path Specification.
    \item Pomax, Notes on Bézier Curves.
\end{itemize}
