# Phase 2 worked arithmetic

## A. Constant velocity

Positions `(0,0)`, `(1,0)` at times `0,1` give `dx=1`, `dy=0`, `dt=1`.
Thus `vx=1`, `vy=0`, `speed=sqrt(1^2+0^2)=1`, and `velocity_valid=true`.

## B. Speed change without direction change

Velocities are `(1,0)` and `(2,0)` over intervals `[0,1]` and `[1,2]`.
Their sample times are `0.5` and `1.5`, so acceleration `dt=1`.
Tangential acceleration is `(2-1)/1=1`; `ax=(2-1)/1=1`, `ay=0`,
vector magnitude is `1`, and wrapped direction change is `0`.

## C. Direction change at constant speed

Velocities `(1,0)` and `(0,1)` both have speed `1`. With midpoint difference
`1`, tangential acceleration is `(1-1)/1=0`, while `ax=-1`, `ay=1`,
vector magnitude is `sqrt(2)`, and direction change is `pi/2`.

## D. Missing versus measured zero

For two consecutive valid identical velocity vectors, vector acceleration is
measured zero and `vector_acceleration_valid=true`. For the first velocity in
a unit, acceleration has no previous velocity: its numeric evidence is
unavailable and `vector_acceleration_valid=false`; a packed zero is only a
placeholder.

## E. Irregular timing

For timestamps `[0,1,3]`, velocity intervals have midpoints `0.5` and `2.0`.
Therefore acceleration `dt=2.0-0.5=1.5`, also `(1+2)/2=1.5`. Reusing the
current frame-pair duration `2` would compare velocity samples at the wrong
times.

The independent verifier records all numerators, denominators, 22 numerical
golden scenarios, six schema-negative scenarios, and the real 20-unit trace.
