# Model-based B-splines for microbial single-cell analysis

## How does it work?
We propose a hybrid approach for multi-object microbial cell segmentation. The approach combines an ML-based detection with a geometry-aware variational-based segmentation using B-splines that are parametrized based on a geometric model. The detection is done first using YOLOv5. In a second step, each detected cell is segmented individually.

![alt-text](https://github.com/kruzaeva/model_spline_seg/blob/master/test_data_weights__500_500_.gif)

## Detection step

We do not provide the code for the detection (to be updated). The proposed approach was tested with the YOLOV5 detection framework, therefore we recommend using it.
Follow the instructions for the installation and training:
https://github.com/ultralytics/yolov5 

The training data you can find [here](https://github.com/kruzaeva/model_spline_seg/tree/master/yolo_data/training%20data)



## Segmentation step
The segmentation requires the image and the .txt file with bounding boxes (Yolo format).

For the test data segmentation, we provide [the bounding boxes](https://github.com/kruzaeva/model_spline_seg/tree/master/data/gtframes) for every image in [image sequence](https://github.com/kruzaeva/model_spline_seg/blob/master/data/name1.tif)
We also provide the anaconda environment file "environment.yml" with all of the dependencies.

Currently, we offer only the rod-model segmentation (to be updated).
To run the test segmentation use
    $  python splines.py