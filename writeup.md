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

#### 2. Explain how you settled on your final choice of HOG parameters.

To choose the color space I've trained the classifier on the `vehicle` and `non-vehicle` images provided for the project applying various conversions and computing test accuracies (**cells 2 - 4**) along with feature vector length. Aa a starting point I've set
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
| H(LS)* |	0.9645 |	5640 |
| H(S)V* |	0.9398 |	2820 |
| (L)UV* |	0.9437 |	2820 |

\*H(LS) means L and S layers of HLS, H(S)V means S layer of HSV and so on.

Since YCrCb let achieve the highest accuracy I've chosen it for image conversion and the next step was to tweak input parameters just mentioned based on the same approach:

| #	| pix_per_cell	| cell_per_block	| orient	| spatial_size	| hist_bins	| Accuracy | # of Features |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1	| 8	| 2	| 9	| 32	| 32	| 0.9904	| 8460 |
| 2	| 16	| 2	| 9	| 32	| 32	| 0.9887	| 4140 |
| 3	| 8	| 3	| 9	| 32	| 32	| 0.9916	| 11916 |
| 4	| 8	| 2	| 8	| 32	| 32	| 0.9932	| 7872 |
| 5	| 8	| 2	| 7	| 32	| 32	| 0.991	| 7284 |
| 6	| 8	| 2	| 10	| 32	| 32	| 0.9887	| 9048 |
| 7	| 16	| 2	| 8	| 32	| 32	| 0.9882	| 4032 |
| 8	| 16	| 2	| 10	| 32	| 32	| 0.987	| 4248 |
| 9	| 16	| 2	| 9	| 16	| 32	| 0.9876	| 1836 |
| 10	| 16	| 2	| 9	| 8	| 32	| 0.9859	| 1260 |
| 11	| 16	| 2	| 9	| 8	| 16	| 0.9831	| 1212 |

As it can be seen from experiments 1 and 10 increasing `pix_per_cell` to 16 and decreasing `spatial_size` to 8 let reduce length of the feature vector from 8460 to 1260, while the accuracy only reduces from 0.9904 to 0.9859.

Finally set of input parameters looks like below (see **lines 23 - 28**):

```
color_space = 'YCrCb', # Color space base for features
spatial_size = (8, 8), # Spatial binning dimensions
hist_bins = 32,    # Number of histogram bins
orient = 9,  # HOG orientations
pix_per_cell = 16, # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
 ```

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image1]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM calling ```train()``` method (**cell 4**) defined in **lines 397 - 453**. Following constructor parameters are responsible for using or not spatial, histogram and HOG sets of features:

```
spatial_feat = True, # Spatial features on or off
hist_feat = True, # Histogram features on or off
hog_feat = True # HOG features on or off
```
Lines **409 - 421** are responsible for feature extraction by calling the ```extract_features()``` method defined in lines **300 - 318**. This method was refactored to reuse code of ```single_img_features()``` (**lines 272 - 297**). In order to overcome PNG / JPEG scaling issue, folowing line of code was used to always read images in 0..255 scale (**line 309**):
```
image = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
```
Data scaling, suffling and train / test splitting take place in **lines 424 - 437**.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding Window Search is implemented in a ```detect_cars()``` method (**lines 457 - 554**) that is kind of a hybrid of ```windows_search()``` and ```find_cars()``` methods from the lessons. ```detect_cars()``` is as flexible as ```windows_search()``` and reuses HOG features through scaling the whole image as ```find_cars()``` does.
I decided to use three sizes of windows: 64, 92 and 128, where smaller windows are used for detection of more distant cars.
Such a multiscale windows search is implemented in a ```detect_cars_multiscale()``` (**lines 557 - 570**) by calling ```detect_cars()``` 
method for each windows size.

Windows sizes, areas to be occupied by windows as well as overlapping are defined by these parameters of the counstructor (**lines 32 - 37**):
```
overlap = 48/64, # Sliding windows overlap
x_start_stop= [400, None], # Min and max in x to search in slide_window()
y_start_stop = [400, 656], # Min and max in y to search in slide_window()
win_sizes = [[64, 500, 512],
             [96, 500, 568],
             [128, 400, 656]], # Sizes and margins for sliding windows [[win_size, xstart, ystop], ...]
```

Windows of mentioned scales defined above are shown on the image below. Scales and overlaps were tweaked to get good final video output.

![alt text][image2]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image3]

As mentioned above in order to optimize performance of the classifier, I've used YCrCb colospace, increased ```pix_per_cell``` parameter to 16 and decreased ```spatial_size``` to 8, spatial, histogram and HOG features were used together.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
The video was generated in **cell 8** using ```pipeline()``` method defined in **lines 657 - 678**
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video (**line 660**).  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions (method ```get_bboxes()``` **lines 574 - 623**, called in **line 666** ).  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap (** lines 602 - 615** 
og ```get_bboxes()```).  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected (**lines 618 - 623**).  

An example result showing the heatmap from a series of frames of video plese find on the last image.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

