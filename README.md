**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./project_images/car_not_car.png
[image21]: ./project_images/HOG_example1.png
[image22]: ./project_images/HOG_example2.png
[image23]: ./project_images/HOG_example3.png
[image24]: ./project_images/HOG_example4.png
[image25]: ./project_images/HOG_example5.png
[image26]: ./project_images/HOG_example6.png

[image3]: ./project_images/sliding_window.png
[image4]: ./examples/sliding_window.jpg
[image5]: ./project_images/bboxes_and_heat.png
[image6]: ./project_images/labels_map.png
[image7]: ./project_images/output_bboxes.png
[video1]: ./project_video_out.mp4

## 1. File Details
* Vehicle Detection.ipynb - Python notebook containing both classifier training pipeline and video annotations code.
* utils.py - Contains most of the helper functions. Most of this code was from lecture exercises. I modified some of those functions.
* test_video_out.mp4 - Test video output
* project_video_out.mp4 - Project video output
* project_images - Directory containing all the report images

## 2. Training for classifier

Code for classifier training pipeline is split into following sections

1. Data Loading - Block 2
2. Feature Parameters - Block 3
3. Feature Visualization - Block 4
4. Feature Extraction - Block 5
5. Classifier Trainng - Block 6

In data loading step images for both 'vehicle' and 'non-vehicles' were loaded.
Below is an example of one of each of the `vehicle` and `non-vehicle` classes:
![alt text][image1]

I then explored different color spaces and different hog parameters specifically - `orientations`, `pixels_per_cell`, and `cells_per_block`.  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `LUV` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` and 'hog_channels = ALL':

![alt text][image21]
![alt text][image22]
![alt text][image23]
![alt text][image24]
![alt text][image25]
![alt text][image26]

####2. Explain how you settled on your final choice of HOG parameters.

I explored following parameters and ran experiments with the SVM classifier to arrive at the final choice. Though I did spent lot of time getting these values, these might not be the best values for HOG parameters. However they performed reasonably well for this setup

  My empirical values for HOG parameters are

* color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
* orient = 32 # HOG orientations
* pix_per_cell = 16 # HOG pixels per cell
* hog_channel = "ALL"

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM Classifier with hinge loss function. For training my classifier, I used randomly selected balanced car and non car data. In my classifer I utilize spatial, color and hog features. Detailed feature generation methods are in utils.py file. My train and test split was 80% and 20% respectively.
On this dataset, my classifier achieved accuracy of 99.44%.

## 3. Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Sliding search is implemented in block 7 to 9 of the python notebook right after classifier prediction.
I tested sliding window generation with two methods. I found Hog Subsampling window search to me much faster though the other method was easier to understand and experiment with. For normal sliding window method I utilized two window sizes (64,64) for farther objects and (128,128) for nearby objects. I tested the result on 6 test images. In each image, car was captured by the sliding windows. As can be seen from the illustration below, there were some false positives.


![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.
In order to optimize the performance, I use Hog subsampling method. Another optimization I wanted to try was to search for larger windows nearer to the car, and smaller windows farther away. This would have significantly reduced number of windows in the frame. Results from the pipeline running on test images are shown below.

---

## 3. Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
Heatmap code is located in block 10 of the notebook. For removing out false positives and smoothening of the results, I record  bounding boxes for each frame . I create a heatmap of last 20 frames and then threshold the map to identify vehicle positions.
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

In order to reduce false positives, SVC prediction score with threshold was used. I also averaged heatmaps of last 10 frames to get smoother bounding box. 

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

## 4. Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

For training my classifier, I used a SVM classifier with hinge loss. However I did not tune my hyperparams for the model. This could potentially result in better performance. Also I could increase robhustness of the classifier by adding more data and thresholding the score.

I used hog subsampling for faster processing, however due to my lack of proper understanding of that function, I could not tune it better. Also read that cv2 hog function performs much better but did not try it yet.
