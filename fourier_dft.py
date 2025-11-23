import xml.etree.ElementTree as ET
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

SVG_FILE = "thistle_svg.svg"
OUT_DIR = Path(".") #Allows us to save all output files (PNG, CDV, etc. to current directory)
N_TOTAL = 2048  # number of sample points around the entire shape - why?: it's a power of 2, fast FFT
# 2048 is enough points to accurately represent curves

#The Idea is that the SVG path is a continuous curve abd we replace it with 2048 evenly spaced sample points.
#Each point becomes a complex number (z[k] = x[k] + y_i) <- this gets fed int fourier transform.

# --- Minimal SVG path parser (supports M, L, H, V, C, Q, Z; absolute & relative) ---
token_re = re.compile(r'([MmZzLlHhVvCcQq])|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
#This is a regular expression because of how SVG paths are stored
#They are stored like this (M 10 20 L 40 50 C 80 10 90 50 100 80)
#Commands are M, L, C, Q, Z etc, and the numbers are coordinates like 20, 1e-3
#the first part of the regex matches one SVG command letter
#M - Move
#L = line
#H = horizontal line
#V = vertical line
#C = cubic bezier
#Q = quadratic bezier
#Z = close path
#The uppercase is the absolute position of these commands and the lowercase is its relative
#The second part of the regex matches a number: handles integers, decimals, negative no and scientific notation
#An SVG path looks like this: M 100 200 C 150 250 200 100 250 200 L 300 250
#to parse this string you need to split it into: ['M', 100, 200, 'C', 150, 250, 200, 100, 250, 200, 'L', 300, 250]
#I used LLMs here to help with the regex.

#This function scans the SVG text
#finds all commands and numbers
#turns them into a clean list of tokens
#converts numbers to python floats and leaves commands as letters
def tokenize_path(d):
    tokens = token_re.findall(d)
    result = [] 
    for cmd, num in tokens:
        if cmd:
            result.append(cmd) #e.g. 'M'
        else:
            result.append(float(num)) #e.g. 10.0
    return result
#['M', 100.0, 200.0,
# 'L', 250.0, 300.0,
# 'C', 250.0, 300.0, 270.0, 310.0, 290.0, 330.0] returns something of this form

#functions defining the two bezier curves: info on bezier curves - https://javascript.info/bezier-curve, https://www.youtube.com/watch?v=pnYccz1Ha34
#returns output of curve
def cubic_bezier(p0, p1, p2, p3, t):
    return ( (1-t)**3 * p0 +
             3*(1-t)**2 * t * p1 +
             3*(1-t) * t**2 * p2 +
             t**3 * p3 )

def quad_bezier(p0, p1, p2, t):
    return ( (1-t)**2 * p0 + 2*(1-t)*t*p1 + t**2 * p2 )

#This function takes the SVG path string, and turns it into a list of segments the computer
#can understand
def path_segments_from_d(d):
    tokens = tokenize_path(d) #calling our previously defined function
    #initialise variables
    i = 0 #index into the tokens list
    cur = complex(0,0) #the current point on the path, start at 0
    start_point = complex(0,0) #start of current subpath 
    last_cmd = None #used to remember the previous command so the SVG's 'implicit commands' work
    while i < len(tokens):
        tok = tokens[i]; i += 1 #walk through svg commands one by one
        if isinstance(tok, str): #detect whether token is a command or number
            cmd = tok
        else:
            # repeat previous command if a number appears where command expected
            #why? this is because svg shortcuts mean svg presents something like this, 
            #L 10 20 30 40 but mean L 10 20 L 30 40, so if its a number repeat command.
            if last_cmd is None:
                raise ValueError("Path data starts with a number without a command.")
            cmd = last_cmd
            i -= 1
        last_cmd = cmd #used if next token is numbers

        if cmd in ('M','m'): #M is the absolute move, m is the relative move (move right 5 and up 10)
            is_rel = (cmd == 'm')
            x = tokens[i]; y = tokens[i+1]; i += 2 #take the two numbers after the command as x and y
            pt = complex(x,y) #store complex number using this x and y
            cur = cur + pt if is_rel else pt  #if the command is move relative then add the new point to it, else just return the point it says.
            start_point = cur #move pen to that point

            # extra coordinate pairs are treated as implicit 'L'
            while i+1 < len(tokens) and not isinstance(tokens[i], str):
                x = tokens[i]; y = tokens[i+1]; i += 2
                new = cur + complex(x,y) if is_rel else complex(x,y) #again, if its relative add it to the current point.
                                                                     #if its absolute make the new point that point itself.
                yield ('L', cur, new); cur = new #returning ('L', cur, new), used later, then setting the pen position to new 
                #this is so cool! its returning a line segment from start point -> end point. showing how an svg is being broken down into a list of drawable shapes.
                #this happens for quadratic and cubic curves too slightly further down.

        elif cmd in ('L','l'):
            is_rel = (cmd == 'l')
            while i+1 < len(tokens) and not isinstance(tokens[i], str):
                x = tokens[i]; y = tokens[i+1]; i += 2
                new = cur + complex(x,y) if is_rel else complex(x,y)
                yield ('L', cur, new); cur = new

        elif cmd in ('H','h'):
            is_rel = (cmd == 'h')
            while i < len(tokens) and not isinstance(tokens[i], str):
                x = tokens[i]; i += 1 #horizontal line only needs the x value - just adding x value.
                new = cur + complex(x,0) if is_rel else complex(x, cur.imag) #if absolute change the x value
                                                                             #make the y value the current imaginary y value
                yield ('L', cur, new); cur = new

        elif cmd in ('V','v'):
            is_rel = (cmd == 'v')
            while i < len(tokens) and not isinstance(tokens[i], str):
                y = tokens[i]; i += 1 #vertical lines only need the y value
                new = cur + complex(0,y) if is_rel else complex(cur.real, y)
                yield ('L', cur, new); cur = new

        elif cmd in ('C','c'):
            is_rel = (cmd == 'c')
            while i+5 < len(tokens) and not isinstance(tokens[i], str):
                x1 = tokens[i]; y1 = tokens[i+1]; x2 = tokens[i+2]; y2 = tokens[i+3]; x = tokens[i+4]; y = tokens[i+5]; i+=6
                p1 = cur + complex(x1,y1) if is_rel else complex(x1,y1)
                p2 = cur + complex(x2,y2) if is_rel else complex(x2,y2)
                p3 = cur + complex(x,y) if is_rel else complex(x,y)
                yield ('C', cur, p1, p2, p3); cur = p3 #we are returning a cubic curve tuple, used later

        elif cmd in ('Q','q'):
            is_rel = (cmd == 'q')
            while i+3 < len(tokens) and not isinstance(tokens[i], str):
                x1 = tokens[i]; y1 = tokens[i+1]; x = tokens[i+2]; y = tokens[i+3]; i+=4
                p1 = cur + complex(x1,y1) if is_rel else complex(x1,y1)
                p2 = cur + complex(x,y) if is_rel else complex(x,y)
                yield ('Q', cur, p1, p2); cur = p2 #returning quadratic curve tuple, update pen to last point on curve too

        elif cmd in ('Z','z'):
            yield ('L', cur, start_point); cur = start_point #Z/z means close up the path, so this is a final line connecting 
                                                             #the current point to the start point

        else:
            raise NotImplementedError(f"SVG command {cmd} not implemented by this script.")

# --- Parse SVG file and extract segments ---
tree = ET.parse(SVG_FILE) #ET is pythons ElementTree library, it reads XML files. (svg is xml just formatted)
root = tree.getroot() #top level <svg> element, exactly like in binary trees
# find path elements (account for namespace)
path_elems = root.findall('.//{http://www.w3.org/2000/svg}path') + root.findall('.//path') #finds all <path> elements
#SVG shapes can be <rect>, <circle> .... <path>: we only care about <path> for this
if not path_elems:
    raise RuntimeError("No <path> elements found in the SVG.")

segments = []
# will eventually contain things like 
#('L', point1, point2)
#('C', p0, p1, p2, p3)
#('Q', p0, p1, p2)

for p in path_elems:
    d = p.get('d') #path elems look like <path d="M10 20 L30 40 C 50 60 70 80 100 120"/>
    if not d:
        continue
    for seg in path_segments_from_d(d): #calls earlier function, which would break d into different line/curve segments
        segments.append(seg) #add each segment to segments

if len(segments) == 0:
    raise RuntimeError("No drawable segments extracted. The SVG may use unsupported path commands.")

# --- Estimate lengths and allocate sample counts ---
def segment_length(seg):
    if seg[0] == 'L':
        return abs(seg[2] - seg[1]) #easy: absolute of the start and end of the lines gives distance as complex (acts like vector) - as mentioned in the 3 blue 1 brown video
    elif seg[0] == 'C':
        p0,p1,p2,p3 = seg[1],seg[2],seg[3],seg[4]
        ts = np.linspace(0,1,80) #make 80 values between 0 and one -> t values
        pts = [cubic_bezier(p0,p1,p2,p3,t) for t in ts]
        return float(np.sum(np.abs(np.diff(pts))))
    elif seg[0] == 'Q':
        p0,p1,p2 = seg[1],seg[2],seg[3]
        ts = np.linspace(0,1,80)
        pts = [quad_bezier(p0,p1,p2,t) for t in ts] #for each t, compute a point on the bezier curve
        return float(np.sum(np.abs(np.diff(pts)))) #this computes the sum of the distances between each consecutive point.
        #this is the numerical arc length estimation ^^^ 
    else:
        return 0.0 #fallback for SVG commands like A (our code doesn't deal with this)

lengths = [segment_length(s) for s in segments]
total_length = sum(lengths)
samples_per_segment = [max(1, int(round(N_TOTAL * L / total_length))) for L in lengths] #the max ensures at least one sample point per seg
# adjust to match N_TOTAL exactly
while sum(samples_per_segment) != N_TOTAL:
    diff = N_TOTAL - sum(samples_per_segment) #how many more samples needed to add/take away
    for i in range(len(samples_per_segment)):
        if diff == 0:
            break
        samples_per_segment[i] += 1 if diff > 0 else -1 #add or take away based on if diff more or less than 0
        diff = N_TOTAL - sum(samples_per_segment)

# --- Sample points along segments ---
points = []
for seg, count in zip(segments, samples_per_segment): #zip makes every segment paired in a tuple with the number of samples needed in that segment
    if seg[0] == 'L':
        p0, p1 = seg[1], seg[2]
        ts = np.linspace(0, 1, count, endpoint=False) #np.linspace takes in a range from 0 to 1 and returns (count) number of evenly spaced t's (between 0 and 1)
        for t in ts:
            points.append(p0*(1-t) + p1*t) #find points by plugging the t's into the equation of the line between the two points, to get samples (points on that line)
    elif seg[0] == 'C':
        p0,p1,p2,p3 = seg[1],seg[2],seg[3],seg[4]
        ts = np.linspace(0, 1, count, endpoint=False)
        for t in ts:
            points.append(cubic_bezier(p0,p1,p2,p3,t)) #similar for cubic and quadratic curves, but here were using the cubic_bezier func to generate the points.
    elif seg[0] == 'Q':
        p0,p1,p2 = seg[1],seg[2],seg[3]
        ts = np.linspace(0, 1, count, endpoint=False)
        for t in ts:
            points.append(quad_bezier(p0,p1,p2,t))

z = np.array(points, dtype=complex)
# normalize & center 
z = z - np.mean(z) 
z = z / np.max(np.abs(z))

# --- DFT ---
N = len(z) #number of samples in the sequence
C = np.fft.fft(z) / N #numpy has a built-in function that applies the DFT to each complex number, but it doesn't normalise automatically, so divide by N
#^ gives us the Fourier coefficients
C_shifted = np.fft.fftshift(C) #center with 0 frequency at the middle
k_list = np.arange(-N//2, N//2)
mags = np.abs(C_shifted)
order = np.argsort(mags)[::-1]

def reconstruct_from_topM(M, t_samples):
    idx = order[:M]
    ks = k_list[idx]
    Cs = C_shifted[idx]
    res = np.zeros_like(t_samples, dtype=complex)
    for k, ck in zip(ks, Cs):
        res += ck * np.exp(1j * 2*np.pi * k * t_samples)
    return res, ks, Cs

t_cont = np.linspace(0,1,N,endpoint=False)

# Save reconstructions for example M values
for M in (5,25,100):
    recon, _, _ = reconstruct_from_topM(M, t_cont)
    plt.figure(figsize=(6,6))
    plt.plot(z.real, z.imag, linestyle='--', linewidth=1)
    plt.plot(recon.real, recon.imag, linewidth=1)
    plt.gca().set_aspect('equal')
    plt.title(f"Original and Reconstruction (Top {M} coefficients)")
    plt.xlabel("Real"); plt.ylabel("Imag")
    plt.tight_layout()
    out = OUT_DIR / f"reconstruction_top{M}.png"
    plt.savefig(out, bbox_inches='tight', dpi=200)
    plt.close()
    print("Saved:", out)

# Spectrum plot
plt.figure(figsize=(6,4))
plt.plot(k_list, mags)
plt.yscale('log')
plt.title("Magnitude of Fourier coefficients (log scale)")
plt.xlabel("Frequency index k"); plt.ylabel("|c_k|")
plt.tight_layout()
out = OUT_DIR / "fourier_magnitude_spectrum.png"
plt.savefig(out, bbox_inches='tight', dpi=200)
plt.close()
print("Saved:", out)

# Epicycles snapshot (top 30 coefficients) at t0
M_epic = 30
idxs = order[:M_epic]
ks = k_list[idxs]
Cs = C_shifted[idxs]
perm = np.argsort(np.abs(Cs))[::-1]
ks = ks[perm]; Cs = Cs[perm]
t0 = 0.12
positions = [0+0j]
for k, ck in zip(ks, Cs):
    positions.append(positions[-1] + ck * np.exp(1j * 2*np.pi * k * t0))

plt.figure(figsize=(6,6))
plt.plot(z.real, z.imag, linestyle='--', linewidth=1, alpha=0.6)
for i, (k, ck) in enumerate(zip(ks, Cs)):
    prev = positions[i]; curr = positions[i+1]
    r = abs(ck)
    circle_t = np.linspace(0, 2*np.pi, 200)
    plt.plot(prev.real + r*np.cos(circle_t), prev.imag + r*np.sin(circle_t), linewidth=0.8)
    plt.plot([prev.real, curr.real],[prev.imag, curr.imag], linewidth=1)
plt.scatter([positions[-1].real],[positions[-1].imag], s=30)
plt.gca().set_aspect('equal')
plt.title(f"Epicycles (top {M_epic}) at t={t0:.2f}")
plt.xlabel("Real"); plt.ylabel("Imag")
plt.tight_layout()
out = OUT_DIR / "epicycles_snapshot.png"
plt.savefig(out, bbox_inches='tight', dpi=200)
plt.close()
print("Saved:", out)

# Export top coefficients to CSV
topN = 50
top_idx = order[:topN]
top_k = k_list[top_idx]
top_C = C_shifted[top_idx]
df = pd.DataFrame({
    "k": top_k,
    "magnitude": np.abs(top_C),
    "phase": np.angle(top_C),
    "real": np.real(top_C),
    "imag": np.imag(top_C)
})
csv_out = OUT_DIR / "top_coefficients.csv"
df.to_csv(csv_out, index=False)
print("Saved:", csv_out)
print("\nPreview of top coefficients:")
print(df.head(10).to_string(index=False))
