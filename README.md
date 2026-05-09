Drone Navigation Without GPS
A working prototype. Built solo in roughly 2 months. Tested on ~970 frames of real flight data with 34.7 m median accuracy.

The Problem in One Sentence
Every drone in the world relies on GPS to know where it is — and GPS can be jammed by anyone with a $30 device off the internet.

When GPS goes down, most drones either crash, drift, or fly blindly back to their last known location. Military drones, delivery drones, search-and-rescue drones, surveying drones — they all share this single point of failure. This is one of the biggest unsolved problems in commercial and defense aviation right now, and it's getting worse: GPS jamming incidents in Europe have increased dramatically over the last two years.

This project is a working prototype that solves that problem.

What It Does
The system lets a drone figure out where it is using only a downward-pointing camera and the standard motion sensors that are already inside every drone (an accelerometer, a gyroscope, a barometer, and a compass). No GPS required.

The way it works is, in plain terms, the same way a human pilot would navigate without GPS: look down, recognize what's below you, compare it to a map you brought with you, and update your best guess about where you are.

The drone carries a pre-downloaded map of the area it will fly over (the same kind of satellite/aerial photos you see in Google Maps). Every second or so, it takes a snapshot from its camera, compares it to the map, and figures out which patch of map it's looking at. In between snapshots, it uses its motion sensors to keep track of how far and in what direction it has moved.

The two information sources — vision and motion — are combined mathematically so that each one corrects the weaknesses of the other. The motion sensors drift over time but are always available. The camera is accurate but sometimes the picture is unusable (clouds, fields with no distinctive features, sharp turns, etc.). Combining them gives you something better than either one alone.

How Well Does It Work?
Tested on a real recorded flight over Vejle, Denmark, in Microsoft Flight Simulator 2020 (a realistic simulator widely used for this kind of research because it gives you GPS ground truth to measure error against):

Metric	Result
Frames processed	969 (full in-map flight)
Median position error	34.7 meters
Mean position error	51.1 meters
Frames within 50 m of true position	61%
Frames within 100 m of true position	85%
Frames within 150 m of true position	99.6%
Processing speed	About 1 second per frame
For context: with motion sensors alone (no camera), the same drone drifts to 166 meters average error. Adding the camera-based correction cuts that error by roughly 70%.

Is 34 meters good? For a GPS-denied prototype built in two months by one person, yes — it's competitive with academic results that took research teams years. Production military systems do better, but they cost millions and take years to develop. This is a proof of concept that the underlying approach works.

Why This Is Hard
A lot of people assume "just match the camera image to a satellite photo" is easy. It's not, and here's why:

The camera looks forward and down at an angle, but the satellite map looks straight down. The two images are not the same view of the same thing. A building in a drone photo looks like a building from the side; in a satellite photo it looks like a roof. Matching these is a known hard problem in computer vision.
The drone is moving, tilting, and turning constantly. The camera angle changes every frame. The system has to compensate for the drone's pitch and roll, otherwise it thinks the ground has moved.
The camera doesn't look straight down. It points slightly forward — so the patch of ground in the picture is actually about 100 meters ahead of the drone, not directly below it. If you don't correct for this, every position estimate is wrong by 100 meters in the direction of flight. Discovering and fixing this single issue improved accuracy dramatically.
Most of the map looks the same. Forest looks like forest. Farmland looks like farmland. Sometimes the camera has nothing distinctive to match against. The system has to know when to trust the camera and when to ignore it.
The motion sensors are noisy and biased. They drift. If you just integrate them naively, the position estimate is off by hundreds of meters within a minute. Correctly fusing them with the camera updates requires careful mathematics (a 10-dimensional Extended Kalman Filter, for the technically curious).
What's Inside
The system is built from several components, each solving a different piece of the puzzle:

A neural network (SuperPoint + LightGlue, the current state of the art) finds and matches distinctive points between the camera image and the map.
A second neural network (a semantic segmentation model) labels every pixel of the camera image as water, forest, road, building, etc., and uses that to quickly narrow down which map tile to look at — like a librarian sorting books before searching the shelves.
An Extended Kalman Filter mathematically fuses the camera measurements with the motion sensors to produce a single best estimate of position.
A particle filter maintains 100 different hypotheses about where the drone might be, and uses statistical resampling to converge on the right one.
A magnetic field model (the official 2025 World Magnetic Model) corrects the compass for the fact that magnetic north and true north are different.
A live integration with Microsoft Flight Simulator 2020 so the whole pipeline can be tested in real time on simulated flights.
In total: about 6,800 lines of Python code, organized into 15 core modules, with unit tests, configuration files, and analysis tooling.

What It Looks Like to Use
Three ways to run it:

Live mode — connects to a flying simulator (or, in principle, a real drone) and tracks its position in real time.
Replay mode — feed it a recorded flight (camera frames + sensor log) and it processes the whole thing and outputs a CSV of estimated positions.
Notebook mode — interactive Jupyter notebooks for tuning parameters and visualizing what the algorithm sees on each frame.
Output is a clean CSV with one row per frame, 31 columns covering position, heading, confidence scores, timing, and quality metrics. Plug it into any GIS tool and you get a trajectory plot.

Honest About the Limitations
I'd rather you hear this from me than discover it later:

It's not real-time on every drone. It runs at about 1 frame per second on a laptop GPU. A real drone deployment would need either a dedicated AI accelerator on board, or to run the heavy computation on a ground station with a video downlink.
It needs the map pre-downloaded. You can't fly somewhere you haven't planned for. The current map covers a chunk of Denmark; extending coverage means downloading more aerial imagery and pre-processing it.
Calibration is empirical. The forward-tilt correction (110 meters) was tuned by hand for this specific simulated drone and camera. A real drone with a different camera mount needs a one-time recalibration.
It struggles in featureless or rapidly banking flight. Open ocean, identical farmland, or a sharp turn can break the visual match for a few seconds. The system gracefully falls back to motion sensors during these moments, but accuracy degrades.
It was tested in a simulator. A real-world deployment would surface new problems (camera lens distortion, lighting variation, motion blur, etc.).
The live mode contains a small auxiliary path that uses simulator GPS as a soft anchor (R = 200 m std dev) on frames where the visual gate fails. This is operational scaffolding for the live demo — it keeps the drone's search region inside the reference map when visual matching fails for several frames in a row. **It is not part of the GPS-denied method.** The headline thesis numbers should come either from file-mode replay (no fallback at all) or from live runs with that fallback disabled. See [`Pipeline_3_Rev1/docs/GPS_DENIED_INTEGRITY_AUDIT.md`](Pipeline_3_Rev1/docs/GPS_DENIED_INTEGRITY_AUDIT.md) and [`Pipeline_3_Rev1/docs/BS_CHECK.md`](Pipeline_3_Rev1/docs/BS_CHECK.md) for the full disclosure.

## Documentation Map

For readers and for future Claude Code sessions:

| Document | Purpose |
|---|---|
| [`Pipeline_3_Rev1/docs/CODEMAP.md`](Pipeline_3_Rev1/docs/CODEMAP.md) | Top-level reader index — every folder, every entry point, every key file. Start here. |
| [`Pipeline_3_Rev1/docs/GPS_DENIED_INTEGRITY_AUDIT.md`](Pipeline_3_Rev1/docs/GPS_DENIED_INTEGRITY_AUDIT.md) | Source-grounded audit: every site that touches GPS-truth lat/lon, classified A–E. |
| [`Pipeline_3_Rev1/docs/BS_CHECK.md`](Pipeline_3_Rev1/docs/BS_CHECK.md) | Brutally honest assessment of whether the GPS-denied claim is defensible. |
| [`Pipeline_3_Rev1/docs/CALL_GRAPH.md`](Pipeline_3_Rev1/docs/CALL_GRAPH.md) | Live-mode runtime call graph (line-numbered, source-grounded). |
| [`Pipeline_3_Rev1/docs/ARTEFACT_FLOW.md`](Pipeline_3_Rev1/docs/ARTEFACT_FLOW.md) | Producer → consumer table for every artefact on disk. |
| [`Pipeline_3_Rev1/docs/CURRENT_BEHAVIOUR_BASELINE.md`](Pipeline_3_Rev1/docs/CURRENT_BEHAVIOUR_BASELINE.md) | Frozen Phase-0 baseline (live_020_Odense_f1, 125 frames, 96 % gate pass). |
| [`Pipeline_3_Rev1/docs/pipeline_breakdown.tex`](Pipeline_3_Rev1/docs/pipeline_breakdown.tex) | Full LaTeX architecture document with equations. |
| [`Pipeline_3_Rev1/docs/Diagrams/`](Pipeline_3_Rev1/docs/Diagrams/) | Mermaid architecture diagrams (00–06 system level, 11–37 sub-diagrams). |
| [`Pipeline_3_Rev1/docs/FLAGS.md`](Pipeline_3_Rev1/docs/FLAGS.md) | Reference for every config flag and what it saves. |
| [`CLAUDE.md`](CLAUDE.md) | Detailed development log: bug history, design decisions. |
