## AdvancedObjectDetection -  *An ongoing quest for real time 3D object detection*

 
 Velodyne point cloud bird view projection      |  2D object detection (YOLOv2)
:-------------------------:|:-------------------------:
![](./images/vlp-viz.gif)  |  ![](./images/2d-obj-detect.gif)
 

**Velodyne point cloud front view reflectance projectioin**

<img src="./images/vlp-viz-pro-gif.gif" height="200" width="1000">

### **Dataset description**

The dataset used here to explore the possibities of 3D object detection is from the KITTI Vision Benchmark Suite.

The dataset has multiple sources of data - various sensors and cameras. As per thier website, following are the equipments used to collect the data.

    1 Inertial Navigation System (GPS/IMU): OXTS RT 3003
    2 Laserscanner: Velodyne HDL-64E
    3 Grayscale cameras, 1.4 Megapixels: Point Grey Flea 2 (FL2-14S3M-C)
    4 Color cameras, 1.4 Megapixels: Point Grey Flea 2 (FL2-14S3C-C)
    5 Varifocal lenses, 4-8 mm: Edmund Optics NT59-917

The set up of all these sensors and cameras on Volkswagen Passat B6 is as follows:

<img src="./images/KittiSensorSettings.png">

*Image source:* http://www.cvlibs.net/datasets/kitti/setup.php

The raw data can be download from the following link:

http://www.cvlibs.net/datasets/kitti/raw_data.php

The processed datasets can be downloaded from the following link:

http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

The dataset can be used to detect various objects in the street while driving in a city. Namely, other vehicles, pedestrains, cyclists, traffic light etc.

### **Camera image processing and 2D object detection**

For 2D object detection on images, we can use the images as it is without much preprocessing. The images can be fed into a convolutional neural networks in order train a object detection model. Here I make use of a pre-trained model/weights (transfer learning) to detect objects in the images. For this purpose, I use the images from the left (color) camera. 

    Total number of images for training:  7481
    Total number of images for testing:  7518
    Image resolution:  (375, 1242, 3)

The technique used to implement this model is called YOLOv2 object detection technique and is described in the following paper:

    + https://pjreddie.com/media/files/papers/YOLO9000.pdf
    
YOLOv2 technique is a state of the art technique for 2D object detection and can applied in real time. A resulting output after applying this technique is shown below. 

<img src="./images/obj-detect.png">

The `gif` file added at the begining of this document is also a result of this model. As you can see, the object detector not only detects other `cars`, but also detects a `person`, `traffic light`, `bicycle` etc.

My next goal is to build a 3D object detection model to this dataset. For this I also need the depth information which is not available in the images. 

### **Feature extraction and visualization of lidar point clouds**


![](./images/bird_view_multichannel.png)

![](./images/birdview.png)

![](./images/frontview_depth.png)

![](./images/frontview_height.png)

![](./images/frontview_reflectance.png)








