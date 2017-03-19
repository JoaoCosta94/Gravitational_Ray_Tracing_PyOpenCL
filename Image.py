from scipy import misc
import numpy as np

class Image:

    def __init__(self, Path, Z0, Kx, Ky, Kz):
        self.Path = Path
        self.Z0 = Z0
        self.Kx = Kx
        self.Ky = Ky
        self.Kz = Kz

        self.Object = None

        self.Height = None
        self.Width = None
        self.Colours = None
        self.Grid = None
        self.Rays = None

        self.initialize()

    def initialize(self):
        """
        Function for image object initialization
        """

        # Read image into scipy object
        self.Object = misc.imread(self.Path)

        # Obtain image properties
        self.Height = self.Object.shape[0]
        self.Width = self.Object.shape[1]

        """
        Grid and colors initialization
        Colours contains the information about each pixel RGB
        Grid is composed by 8 elements array that contain the information
        about the position and momentum of each photon
        """
        for pX in range(self.Height):
            X = 5 * (pX - self.Height / 2.0) / (self.Height / 2.0)
            for pY in range(self.Width):
                self.Colours.append(self.Object[pX, pY])
                Y = 5 * (pY - self.Width / 2.0) / (self.Width / 2.0)
                self.Grid.append(np.array([0.0, X, Y, self.Z0,
                                           0.0, self.Kx, self.Ky, self.Kz],
                                          dtype=np.float32))
        # Convert to numpy arrays
        self.Colours = np.array(self.Colours)
        self.Grid = np.array(self.Grid)
        self.Rays = len(self.Grid)

    def saveImage(self, outputPath, imagePlane = False, margin = 1e-2):
        """
        Function that saves current image state to an output file
        May save current state at any plane or at a selected plane

        :param outputPath:  Desired path for output image
        :param imagePlane:  Image plane (can be empty)
        :param margin:      Margin for photons to be considered in image plane (default 1e-2)
        """
        outputImage = np.zeros((self.Height, self.Width, 3))
        for i in range(self.Rays):
            x = self.Grid[i][1]
            y = self.Grid[i][2]
            x = int((x + 5.0) * (self.Height - 1) / 10.0)
            y = int((y + 5.0) * (self.Width - 1) / 10.0)
            if (x >= 0) and (x <= self.Height - 1) and (y >= 0) and (y <= self.Width - 1):
                # Only photons that stayed inside the frame go into the output
                if not(imagePlane) and (self.Grid[i][3] - imagePlane) <= margin:
                    outputImage[x, y] = self.Colours[i]
                else:
                    outputImage[x, y] = self.Colours[i]

        misc.imsave(outputPath, outputImage)