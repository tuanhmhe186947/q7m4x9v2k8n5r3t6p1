# Phase 2 mathematical contract

Authority commit: `ebdc83bc942ba34dd4f820a6aba46f37233a04d6`.

Velocity is the interval-average displacement rate over `[t[i-1], t[i]]`.
It uses the bbox center in axis-normalized image coordinates:

`vx=(cx_n[i]-cx_n[i-1])/delta_t`, `vy=(cy_n[i]-cy_n[i-1])/delta_t`,
and `speed=sqrt(vx^2+vy^2)`.

The velocity sample time is the interval midpoint
`m[i]=(t[i-1]+t[i])/2`. Acceleration compares two consecutive valid
interval-average velocities using `m[i]-m[i-1]`, equivalently
`(delta_t[i]+delta_t[i-1])/2`. This is a centered interval-average discrete
acceleration convention, not instantaneous physical acceleration.

Tangential acceleration is `(speed[i]-speed[i-1])/(m[i]-m[i-1])`.
Vector components are `(vx[i]-vx[i-1])/(m[i]-m[i-1])` and
`(vy[i]-vy[i-1])/(m[i]-m[i-1])`; magnitude is `sqrt(ax^2+ay^2)`.

Direction is defined only when `velocity_valid`, speed is finite, and
`speed>0`. Exact zero speed has undefined direction. No epsilon is used.
Direction change is the signed shortest wrapped difference in `[-pi, pi)`.

Every derivative family has its own mask. Unavailable numeric values are NaN
in evidence and zero only after tensor packing with the corresponding mask
equal to zero. A valid stationary velocity or valid zero acceleration keeps
its mask true, so measured zero remains distinct from missing.

The 12D schema is fixed by `schema.pig_strenet_motion_v2`, version
`classification_v2.motion_tensor.v2`, SHA-256
`ec0c511b5f5198240492be49c0492e543c9e38eb4a4ff446259b958c2a59963b`.
All quantities are image-coordinate measurements, never physical distance,
velocity, or acceleration.
