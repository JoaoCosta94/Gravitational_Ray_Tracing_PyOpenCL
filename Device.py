import pyopencl as cl

class Device:

    def __init__(self, KernelPath):
        self.KernelPath = KernelPath
        self.Context = cl.create_some_context()
        self.Queue = cl.CommandQueue(self.Context)
        self.MemFlags = cl.mem_flags

        self.Program = None

        self.Grid = None

    def initialize(self, grid):
        """
        Initialization function for device object. Builds kernel
        and copies array to device

        :param grid: Array to copy to device
        """

        # Read the kernel source file
        source = open(self.KernelPath, "r").read()

        # Build the kernel
        self.Program = cl.Program(self.Context, source).build()

        # Copy array state from host to device
        self.Grid = cl.Buffer(self.Context,
                              self.MemFlags.READ_WRITE | self.MemFlags.COPY_HOST_PTR,
                              hostbuf=grid)

    def push(self, shape, t, dt):
        """
        Function that pushes the grid elements a step in time

        :param shape: Shape to launch kernel
        :param t: Temporal instant
        :param dt: Integration step
        """
        # Call the OpenCl pusher
        event = self.Program.RK4Step(self.Queue,
                                     shape,
                                     None,
                                     t,
                                     dt,
                                     self.Grid)

        # Wait for thread synchronization
        event.wait()

    def copyToHost(self, hostArray):
        """
        Function that copies current state of Grid to an host array
        :param hostArray: Pointer to destination array
        """

        cl.enqueue_copy(self.Queue, hostArray, self.Grid)
