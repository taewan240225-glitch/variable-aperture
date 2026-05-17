# Variable_Aperture_Design

Variable_Aperture_Design is a browser app for checking variable aperture geometry.

## Files

- `index.html`: Standalone app with input variables and two design graphs.

## Current Inputs

- `Rs = 7.6 mm`
- `Rr = 6.2 mm`
- `Pin_Num = 9`
- `Pin_Size = 0.4 mm`
- `theta_sr = 14.7 deg`
- `SL0 theta = 14 deg`
- `SL4 theta = 28 deg`
- `D_F1.47 = 7.83 mm`
- `D_F1.7 = 6.52 mm`
- `D_F4.0 = 2.78 mm`
- `D_F6.0 = 2.2 mm`
- `Leaf_max = 16.5 mm`
- `theta_1 = 20 deg`
- `theta_2 = 30 deg`
- `ex_line = 0.55 mm`
- `Pressure angle = 30 deg`

## Current Graphs

- `Variable Aperture`: stator pins from `(0, Rs)` and rotor pins offset counterclockwise by `theta_sr`.
- `Leaf`: the reference stator pin in quadrant III with the largest x-value, its corresponding rotor pin, D_min/D_F6.0/D_F1.7/D_F1.47/D_max tangent construction, points P1 through P5, L0 through L2, and SL0 through SL4.
- Both graphs support zoom controls, mouse-wheel zoom, and drag panning.
- Leaf Pins are shown only on the Leaf graph. The Variable Aperture graph remains a clean stator/rotor overview.
- The Leaf Pins visibility controls apply D-centered placement. For each D state, the pin pattern center is the D circle center, and the pattern is rotation-corrected so the reference `S4` aligns with the original stator pin.
- By default only the reference pair `S4/R4` is visible for `Pin_Num = 9`; other `S0/R0` through `S8/R8` pairs can be shown or hidden individually.
- Visible rotor pins include their D-centered drive path from `SL0 theta` to `SL4 theta`.
- `SL0` through `SL4` centers stay on the circle at distance `Rs` from the reference stator pin.
- `SL1`, `SL2`, and `SL3` use the D_F6.0, D_F1.7, and D_F1.47 centers. `SL0` uses the computed D_min center, and `SL4` uses the computed D_max center.
- D_min is tangent to the hidden infinite L1 line, not constrained to the visible L0 segment. D_max is tangent to the extended L2 line at P5.
- D_min is also constrained to stay on the D_F6.0 side of L1, with its tangent point on the left side of P1, and tangent to the D_F6.0 arc so it remains in the same line/arc sequence.
- `ex_line` only controls the visible L0 segment length; it does not set the D_min/SL0 center, D_max/SL4 center, or P5 position.
- Leaf_max circles are drawn at the D_min and D_max centers. A same-color D_F1.47 circle is also drawn at the D_min center.
- The point x-order is constrained as `P1.x > P2.x > P3.x > P4.x > P5.x`; input changes that break this order are rejected.
- `D_min > 1.00 mm` is enforced as an input condition; input changes that break this condition are rejected.
- Diameter order is enforced as `D_max > D_F1.47 > D_F1.7 > D_F4.0 > D_F6.0 > D_min`; input changes that break this condition are rejected.
- Each SL pressure angle is checked as the positive counterclockwise angle from `v` to `n`, with the required range `0 < epsilon(+) < 90 deg`.
- Constraints are listed in the left panel with checkboxes. Checked hard constraints affect solver scoring and input acceptance; unchecked constraints remain visible as measurements only.
- The `Pressure angle` value defaults to `30 deg`, and the `Pressure angle target` constraint solves `SL0`, `SL2`, and `SL4` pressure angles toward the target. It is treated as a soft target so hard geometry constraints remain intact when an exact 30 deg solution is infeasible.
- Arc1 and Arc2 center positions are checked against `SL2`; both arc centers must have x-coordinates smaller than `SL2.x`.
- Each SL point shows its center-to-point segment, the perpendicular tangent vector `n0` through `n4`, and the arc normal vector `v0` through `v4`. `v0`, `v1`, and `v4` are displayed in the reversed direction from the current clockwise-oriented reference.
- `epsilon` is reported as the pressure angle between each `v` vector and its corresponding `n` vector.
- `SL0 theta` and `SL4 theta` are fixed endpoint inputs. `SL1 = SL0 + 0.7 deg` and `SL3 = SL4 - 0.7 deg` are fixed rules, and only `SL2` is solved as a dependent value. The solver minimizes the tangency residual between the arc through `SL0-SL1-SL2` and the arc through `SL2-SL3-SL4` while keeping both arc centers on the same side of `SL2`.
- Leaf geometry is computed once per render, with SL0/SL4 centers and SL2 theta solved together for the current constraints.

## Run

Open `index.html` in a browser.

No Python runtime or package install is required.

## GitHub Pages

This repository can be published directly with GitHub Pages because the app is static HTML, CSS, and JavaScript.

## Firebase Hosting

Firebase Hosting is configured with `public = "."` and rewrites all routes to `index.html`.
