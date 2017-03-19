import numpy as np
import Image
import Device

if __name__ == "__main__":

    """
    This program simulates the propagation of an image on the
    object plane through the gravitational field of a Black Hole
    which acts as a gravitational lens.

    The program explores the physical ray-tracing using OpenCl
    kernels called through the PyOpenCl API.

    The image plane is represented on the XY plane and Z
    corresponds to the axis of propagation
    """

    # Path to image file
    sourcePath = "milky_way.jpg"
    outputPath = "milky_way_output.jpg"

    # Initial position of the photon on the optical axis
    Z = -10.0

    # Initial momentum of the photons in each axis
    Kx = 0.0
    Ky = 0.0
    Kz = 10.0

    # Simulation temporal window, step and saving interval
    tWidth = 2.0
    dt = 2.5e-3
    samplingInterval = 5

    # Kernel file
    KernelPath = "kernel_tracer.cl"

    ######################## SIMULATION ############################

    # Init image
    im = Image.Image(sourcePath, Z, Kx, Ky, Kz)

    # Init device
    dvc = Device.Device(KernelPath)
    dvc.initialize(im.Grid)

    # Time grid
    tGrid = np.arange(0.0, tWidth + dt, dt)

    # Photons push
    for i in range(len(tGrid)):
        t = np.float32(tGrid[i])
        dvc.push((im.Rays,), t, dt)
        if i % samplingInterval == 0:
            # Copy from device to host
            dvc.copyToHost(im.Grid)

    # Save the result
    im.saveImage("output.png")
