# Gravitational Ray-Tracing with PyOpenCL

This was my first project using GPGPU computing. The code solves the ray-tracing equations
for an object plane propagating in space and is affected by the gravitational pull
of a Black Hole (a gravitational lens).

# Implementation

There are two main classes. An Image class that loads an image from disk, gathers its properties
and creates a grid containing the position and momentum of each photon (pixel).

The Device class loads the OpenCl kernel and evolves the photons in time using that kernel.

The kernel contains a 4th order Runge-Kutta integrator and a function that describes the equation system
for each photon (derived from the metric equations). Since the photons are independent, this problem is highly 
parallelizable.

# Usage

For simplicity, the user may chose the original position Z0 of the plane on the optical axis
as well as its momentum in the X, Y and Z directions. The light propagates in the positive direction
of Z. For simplicity, the image and black hole centers are aligned. It is simple to change this condition.

To chose the input file and output destination, change the respective parameters on main.py
