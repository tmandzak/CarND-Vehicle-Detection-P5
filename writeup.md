# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Implement a sliding-window technique and use a trained classifier to search for vehicles in images.
* Run the pipeline on a video stream project_video.mp4 and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/hog.png
[image2]: ./output_images/sliding.png
[image3]: ./output_images/pipeline.png
[image4]: ./examples/labels_map.png
[image5]: ./examples/labels_map.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/labels_map.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

The whole solution is implemented in a form of a class `VehicleDetection` defined in the file `VehicleDetection.py` and an IPython notebook `P5_Mandzak.ipynb` responsible for instantiating the class, running the pipline, drawing illustrations and generating video output mostly by calling corresponding methods. Names of methods responsible for drawing start with `draw_`. All input paramenters of the pipeline are passed to constructor, parameters used for the final result are specified as default values of constructor's paramenters. All lines of code referenced below correspond to `VehicleDetection.py` and cell numbers to `P5_Mandzak.ipynb`.

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

 
The code for this step is contained in `get_hog_features()` method (**lines 232 - 245**). The method is implemented in such a way
that number of returned HOG layers corresponds the number of input image layers.
When `ravel` parameter of the method equals `True` HOG features will be returned as a single vector, otherwise as a list of vectors per each of the layers.

`get_hog_features()` method is called in **line 292** of a `single_img_features()` method  responsible for extracting features from a single image (**lines 272 - 297**). Before an image is passed to `get_hog_features()` method it's converted to a specified color space by `convert_color()` method (**line 278**) defined in **lines 109 - 154**.

To choose the color space I've trained the classifier on the `vehicle` and `non-vehicle` images provided for the project applying various conversions and computing test accuracies (**cells 2 - 4**) along with feature vector length. For this experiment I've set
the input parameters to values met in lessons: 

```
orient = 9,  # HOG orientations
pix_per_cell = 8, # HOG pixels per cell
cell_per_block = 2, # HOG cells per block
spatial_size = (32, 32), # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
```

| Color space | Accuracy | # of Features |
| --- | --- | --- |
| RGB |	0.9718	| 8460 |
| HSV |	0.9842 |	8460 |
| LUV |	0.9887 |	8460 |
| HLS |	0.9769 |	8460 |
| YUV |	0.9876 |	8460 |
| YCrCb |	0.9904 |	8460 |
| H(LS) |	0.9645 |	5640 |
| H(S)V |	0.9398 |	2820 |
| (L)UV |	0.9437 |	2820 |




![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
