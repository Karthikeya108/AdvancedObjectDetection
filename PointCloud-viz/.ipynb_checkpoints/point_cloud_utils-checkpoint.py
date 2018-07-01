'''
Reference:
    Some of the concepts are explained in the following paper:
        + https://arxiv.org/pdf/1608.07916.pdf
    This implementation is based on the following GitHub project:
        + https://github.com/VincentCheungM/lidar_projection
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

"""
Point Cloud Visualizer for Kitti Velodyne data with default values for argumets based on Kitti dataset configuration.
"""
class PointCloudVisualizer():
    
    def __init__(self):
        self.HORIZONTAL_RANGE = (-10, 10)
        self.VERTICAL_RANGE = (-10,10)
    
    def projection_bird_view_spectral(self, points, saveto=None, figsize=None):
        """ Creates an 2D birds eye view representation of the point cloud data with spectral mapping based on height

        Args:
            points:  (np array)
                     The numpy array containing the lidar points.
                     The shape should be Nx4
                     - Where N is the number of points, and
                     - each point is specified by 4 values (x, y, z, reflectance)  
            saveto:  (str or None)(default=None)
                     Filename to save the image as.
                     If None, then it just displays the image.
            figsize: (h, w)
        """
        x_lidar = points[:, 0]
        y_lidar = points[:, 1]
        z_lidar = points[:, 2]

        # INDICES FILTER - of values within the desired rectangle
        # Note left side is positive y axis in LIDAR coordinates
        ff = np.logical_and((x_lidar > self.VERTICAL_RANGE[0]), (x_lidar < self.VERTICAL_RANGE[1]))
        ss = np.logical_and((y_lidar > -self.HORIZONTAL_RANGE[1]), (y_lidar < -self.HORIZONTAL_RANGE[0]))
        indices = np.argwhere(np.logical_and(ff, ss)).flatten()

        # POINTS TO USE FOR IMAGE
        x_img = -y_lidar[indices]       # x axis is -y in LIDAR
        y_img = x_lidar[indices]        # y axis is x in LIDAR
        pixel_values = z_lidar[indices] # Height values used for pixel intensity

        # Shift values so (0,0) is the minimum value
        x_img -= self.HORIZONTAL_RANGE[0]
        y_img -= self.VERTICAL_RANGE[0]
        # PLOT THE IMAGE
        cmap = "jet"    # Color map to use
        dpi = 100       # Image resolution
        x_max = self.HORIZONTAL_RANGE[1] - self.HORIZONTAL_RANGE[0]
        y_max = self.VERTICAL_RANGE[1] - self.VERTICAL_RANGE[0]
        if figsize:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)    
        else:
            fig, ax = plt.subplots(figsize=(600/dpi, 600/dpi), dpi=dpi)
        ax.scatter(x_img, y_img, s=1, c=pixel_values, linewidths=0, alpha=1, cmap=cmap)
        ax.set_facecolor((0, 0, 0))  # Set regions with no points to black
        ax.axis('scaled')  # {equal, scaled}
        ax.xaxis.set_visible(False)  # Do not draw axis tick marks
        ax.yaxis.set_visible(False)  # Do not draw axis tick marks
        plt.xlim([0, x_max])  # prevent drawing empty space outside of horizontal FOV
        plt.ylim([0, y_max])  # prevent drawing empty space outside of vertical FOV

        if saveto is not None:
            fig.savefig("/tmp/simple_top.jpg", dpi=dpi, bbox_inches='tight', pad_inches=0.0)

        return fig

    def scale_to_255(self, a, min, max, dtype=np.uint8):
        """ Scales an array of values from specified min, max range to 0-255
            Optionally specify the data type of the output (default is uint8)
        """
        return (((a - min) / float(max - min)) * 255).astype(dtype)
    
    def projection_birds_eye_multiple_channels(self,
                                points,
                                n_slices=8,
                                height_range=(-2.73, 1.27),
                                resolution=0.1,
                                ):
        """ Creates an array that is a birds eye view representation of the
            reflectance values in the point cloud data, separated into different
            height slices.

        Args:
            points:     (numpy array)
                        Nx4 array of the points cloud data.
                        N rows of points. Each point represented as 4 values,
                        x,y,z, reflectance
            n_slices :  (int)
                        Number of height slices to use.
            height_range: (tuple of two floats)
                        (min, max) heights (in metres) relative to the sensor.
                        The slices calculated will be within this range, plus
                        two additional slices for clipping all values below the
                        min, and all values above the max.
                        Default is set to (-2.73, 1.27), which corresponds to a
                        range of -1m to 3m above a flat road surface given the
                        configuration of the sensor in the Kitti dataset.
            resolution: (float) desired resolution in metres to use
                        Each output pixel will represent an square region res x res
                        in size along the front and side plane.
        """
        x_points = points[:, 0]
        y_points = points[:, 1]
        z_points = points[:, 2]
        r_points = points[:, 3]  # Reflectance

        # FILTER INDICES - of only the points within the desired rectangle
        # Note left side is positive y axis in LIDAR coordinates
        ff = np.logical_and((x_points > self.VERTICAL_RANGE[0]), (x_points < self.VERTICAL_RANGE[1]))
        ss = np.logical_and((y_points > -self.HORIZONTAL_RANGE[1]), (y_points < -self.HORIZONTAL_RANGE[0]))
        indices = np.argwhere(np.logical_and(ff, ss)).flatten()

        # KEEPERS - The actual points that are within the desired  rectangle
        y_points = y_points[indices]
        x_points = x_points[indices]
        z_points = z_points[indices]
        r_points = r_points[indices]

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (-y_points / resolution).astype(np.int32) # x axis is -y in LIDAR
        y_img = (x_points / resolution).astype(np.int32)  # y axis is -x in LIDAR
                                                   # direction to be inverted later
        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor used to prevent issues with -ve vals rounding upwards
        x_img -= int(np.floor(self.HORIZONTAL_RANGE[0] / resolution))
        y_img -= int(np.floor(self.VERTICAL_RANGE[0] / resolution))

        # ASSIGN EACH POINT TO A HEIGHT SLICE
        # n_slices-1 is used because values above max_height get assigned to an
        # extra index when we call np.digitize().
        bins = np.linspace(height_range[0], height_range[1], num=n_slices-1)
        slice_indices = np.digitize(z_points, bins=bins, right=False)

        # RESCALE THE REFLECTANCE VALUES - to be between the range 0-255
        pixel_values = self.scale_to_255(r_points, min=0.0, max=1.0)

        # FILL PIXEL VALUES IN IMAGE ARRAY
        # -y is used because images start from top left
        x_max = int((self.HORIZONTAL_RANGE[1] - self.HORIZONTAL_RANGE[0]) / resolution)
        y_max = int((self.VERTICAL_RANGE[1] - self.VERTICAL_RANGE[0]) / resolution)
        im = np.zeros([y_max, x_max, n_slices], dtype=np.uint8)
        im[-y_img, x_img, slice_indices] = pixel_values

        return im

    def projection_bird_view(self,
                              points,
                              resolution=0.1,
                              min_height = -2.73,
                              max_height = 1.27,
                              saveto=None):
        """ Creates an 2D birds eye view representation of the point cloud data.
            You can optionally save the image to specified filename.

        Args:
            points: (np array)
                        The numpy array containing the lidar points.
                        The shape should be Nx4
                        - Where N is the number of points, and
                        - each point is specified by 4 values (x, y, z, reflectance)
            resolution: (float) desired resolution in metres
                        Each output pixel will represent an square region `res x res`
                        in size.
            min_height: (float)(default=-2.73)
                        Used to truncate height values to this minumum height
                        relative to the sensor (in metres).
                        The default is set to -2.73, which is 1 metre below a flat
                        road surface given the configuration in the kitti dataset.
            max_height: (float)(default=1.27)
                        Used to truncate height values to this maximum height
                        relative to the sensor (in metres).
                        The default is set to 1.27, which is 3m above a flat road
                        surface given the configuration in the kitti dataset.
            saveto:     (str or None)(default=None)
                        Filename to save the image as.
                        If None, then it just displays the image.
        """
        x_lidar = points[:, 0]
        y_lidar = points[:, 1]
        z_lidar = points[:, 2]

        # INDICES FILTER - of values within the desired rectangle
        # Note left side is positive y axis in LIDAR coordinates
        ff = np.logical_and((x_lidar > self.VERTICAL_RANGE[0]), (x_lidar < self.VERTICAL_RANGE[1]))
        ss = np.logical_and((y_lidar > -self.HORIZONTAL_RANGE[1]), (y_lidar < -self.HORIZONTAL_RANGE[0]))
        indices = np.argwhere(np.logical_and(ff,ss)).flatten()

        # CONVERT TO PIXEL POSITION VALUES - Based on resolution
        x_img = (-y_lidar[indices]/resolution).astype(np.int32) # x axis is -y in LIDAR
        y_img = (x_lidar[indices]/resolution).astype(np.int32)  # y axis is x in LIDAR
                                                         # will be inverted later

        # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
        # floor used to prevent issues with -ve vals rounding upwards
        x_img -= int(np.floor(self.HORIZONTAL_RANGE[0]/resolution))
        y_img -= int(np.floor(self.VERTICAL_RANGE[0]/resolution))

        # CLIP HEIGHT VALUES - to between min and max heights
        pixel_values = np.clip(a = z_lidar[indices],
                               a_min=min_height,
                               a_max=max_height)

        # RESCALE THE HEIGHT VALUES - to be between the range 0-255
        pixel_values  = self.scale_to_255(pixel_values, min=min_height, max=max_height)

        # FILL PIXEL VALUES IN IMAGE ARRAY
        x_max = int((self.HORIZONTAL_RANGE[1] - self.HORIZONTAL_RANGE[0])/resolution)
        y_max = int((self.VERTICAL_RANGE[1] - self.VERTICAL_RANGE[0])/resolution)
        im = np.zeros([y_max, x_max], dtype=np.uint8)
        im[-y_img, x_img] = pixel_values # -y because images start from top left
        # Convert from numpy array to a PIL image
        im = Image.fromarray(im)

        # SAVE THE IMAGE
        if saveto is not None:
            im.save(saveto)

        return im


    def projection_front_view(self,
                               points,
                               v_res,
                               h_res,
                               v_fov,
                               val="depth",
                               cmap="jet",
                               y_fudge=0.0,
                               saveto=None, figsize=None
                               ):
        """ Takes points in 3D space from LIDAR data and projects them to a 2D
            "front view" image, and saves that image.

        Args:
            points: (np array)
                The numpy array containing the lidar points.
                The shape should be Nx4
                - Where N is the number of points, and
                - each point is specified by 4 values (x, y, z, reflectance)
            v_res: (float)
                vertical resolution of the lidar sensor used.
            h_res: (float)
                horizontal resolution of the lidar sensor used.
            v_fov: (tuple of two floats)
                (minimum_negative_angle, max_positive_angle)
            val: (str)
                What value to use to encode the points that get plotted.
                One of {"depth", "height", "reflectance"}
            cmap: (str)
                Color map to use to color code the `val` values.
                NOTE: Must be a value accepted by matplotlib's scatter function
                Examples: "jet", "gray"
            y_fudge: (float)
                A hacky fudge factor to use if the theoretical calculations of
                vertical range do not match the actual data.
                For a Velodyne HDL 64E, set this value to 5.
            saveto: (str or None)
                If a string is provided, it saves the image as this filename.
                If None, then it just shows the image.
            figsize: (h, w)
        """

        # validating the inputs
        assert len(v_fov) ==2, "v_fov must be list/tuple of length 2"
        assert v_fov[0] <= 0, "first element in v_fov must be 0 or negative"
        assert val in {"depth", "height", "reflectance"}, \
            'val must be one of {"depth", "height", "reflectance"}'


        x_lidar = points[:, 0]
        y_lidar = points[:, 1]
        z_lidar = points[:, 2]
        r_lidar = points[:, 3] # Reflectance
        # Distance relative to origin when looked from top
        d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2)
        # Absolute distance relative to origin
        # d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

        v_fov_total = -v_fov[0] + v_fov[1]

        # Convert to Radians
        v_res_rad = v_res * (np.pi/180)
        h_res_rad = h_res * (np.pi/180)

        # PROJECT INTO IMAGE COORDINATES
        x_img = np.arctan2(-y_lidar, x_lidar)/ h_res_rad
        y_img = np.arctan2(z_lidar, d_lidar)/ v_res_rad

        # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
        x_min = -360.0 / h_res / 2  # Theoretical min x value based on sensor specs
        x_img -= x_min              # Shift
        x_max = 360.0 / h_res       # Theoretical max x value after shifting

        y_min = v_fov[0] / v_res    # theoretical min y value based on sensor specs
        y_img -= y_min              # Shift
        y_max = v_fov_total / v_res # Theoretical max x value after shifting

        y_max += y_fudge            # Fudge factor if the calculations based on
                                    # spec sheet do not match the range of
                                    # angles collected by in the data.

        # WHAT DATA TO USE TO ENCODE THE VALUE FOR EACH PIXEL
        if val == "reflectance":
            pixel_values = r_lidar
        elif val == "height":
            pixel_values = z_lidar
        else:
            pixel_values = -d_lidar

        # PLOT THE IMAGE
        cmap = "jet"            # Color map to use
        dpi = 100               # Image resolution
        if figsize:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)    
        else:
            fig, ax = plt.subplots(figsize=(x_max/dpi, y_max/dpi), dpi=dpi)
        ax.scatter(x_img,y_img, s=1, c=pixel_values, linewidths=0, alpha=1, cmap=cmap)
        ax.set_facecolor((0, 0, 0)) # Set regions with no points to black
        ax.axis('scaled')              # {equal, scaled}
        ax.xaxis.set_visible(False)    # Do not draw axis tick marks
        ax.yaxis.set_visible(False)    # Do not draw axis tick marks
        plt.xlim([0, x_max])   # prevent drawing empty space outside of horizontal FOV
        plt.ylim([0, y_max])   # prevent drawing empty space outside of vertical FOV

        if saveto is not None:
            fig.savefig(saveto, dpi=dpi, bbox_inches='tight', pad_inches=0.0)

        return fig