# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/classes_vis.png "Classes visualization"
[image2]: ./images/class_vis.png "Diffrent images for one class"
[image3]: ./images/dataset_dist.png "Distribution of images across classes"
[image4]: ./images/jittered.png "Jittered image example"
[image5]: ./images/normalised.png "Normalised image"
[image6]: ./images/testsigns.png "New Test Signs"
[image7]: ./images/predictions.png "Predictions"
[image8]: ./images/softmax_prob.png "Softmax probabilities"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3 (RGB)
* The number of unique classes/labels in the data set is 43

See code in Step 1 (**Traffic_Sign_ClassifierFinal.ipynb**).

#### 2. Include an exploratory visualization of the dataset.

The below are exploratory visualizations of the dataset. 

![alt text][image1]
![alt text][image2]

* Many images have different perspective, sign size, brightness and background. This is especially visible in the images for a particular class (here it is the 100 km/h speed limit). 
* All the images are in the same size of 32 x 32 px (RGB).

* The next bar charts illustrate a distribution of images across datasets. It is clear from the histograms that some classes are underrepresented. This might results in a lower performance of the ANN for these signs. Following the recommendation from [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), a fake images (jittered) should be added to a dataset. 
![alt text][image3]

See code in Step 1 (**Traffic_Sign_ClassifierFinal.ipynb**).

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The most of improvements were inspired from the [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and from a classroom exercise with the LeNet implementation.

* Conversion to grayscale - Not much improvement recorded. However, it seems that training time is shorter than for a 3-channels image. This can be useful if large datasets are used e.g. when jittered images are added. 

* Standardisation to between -1 and 1 - Easy to implement and worked very well. However, it is important to have a result as float, otherwise the effect is quite opposite; a bid drop in the accuracy. 

* Conversion to YUV and normalisation of Y channel - A noticeable improvement with respect to initial 89% but in the later stage of the project this technique was dismissed in favour of the global normalisation and standardisation.

* Global normalisation - Implemented with use of cv2.equalizeHist(). This provided a significant boost to the training and validation accuracy. Very likely the increased contrast (see figure below) helps the network to extract more features.
![alt text][image5]
* Additional data - The histograms of the original dataset indicated that the classes were not equally represented by the number of images. Also, the mentioned article suggested to add few additional augmented images per each original image. The suggested augmentations were: random translation by [-2:2] pixels, random rotation within [-15: +15] degrees and random scaling within [-10:10] %. Here is an example of an original image and an augmented image:
![alt text][image4]
* The original dataset has 34799 images whereas the augmented dataset contains 208794 images. The augmentation introduced another significant boost, allowing to consistently achieving the validation accuracy above 96%.

Final pre-processing involves:
Step 1: Global normalisation
Step 2: Standardisation
Step 3: Generate additional jittered images

See code in Step 2:Pre-process the Data Set  (**Traffic_Sign_ClassifierFinal.ipynb**).

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The LeNet network, as suggested, was chosen as a base for my model. The mentioned paper suggested further modifications but once a dropout was added after each layer the model started to perform very well and focus was placed on tuning the hyperparameters and adding fake data.   

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5    	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Dropout	         	| 0.9				             |
| Max pooling	      	| 2x2 stride, same padding, outputs 14x14x6		|
| Dropout	         	| 0.9				             |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|	 				     					| 
| Dropout	         	  | 0.9				             |
| Max pooling	      	| 2x2 stride, same padding, outputs 5x5x16		|
| Dropout	         	  | 0.9		|
| Flatten				| outputs 400									|
| Fully connected       | outputs 120 |
| RELU					|	 				     					| 
| Dropout	         	  | 0.9				             |
| Fully connected       | outputs 84 |
| RELU					|	 				     					| 
| Dropout	         	  | 0.9				             |
| Fully connected       | outputs 43|

See code in Step 2: Architecture (**Traffic_Sign_ClassifierFinal.ipynb**).

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* The Adam Optimiser as a default optimiser performed very well. The Gradient Descent Optimizer was also tried few times but its performance was worse than Adam's. 
* The hyperparameters were adjust by trial and error. The final settings were Epochs=30, Batch=128, learning rate=0.0005, dropout probability 0.9 (10%), mu=0, sigma=0.1,
* Training time was quite fast and took approximately just several minutes on the NVIDA GTX 980. However, making TensorFlow to work on a GPU required quite some time.    

See code in Step 2:Train, Validate and Test the Model  (**Traffic_Sign_ClassifierFinal.ipynb**).

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


My final model results were:
* training set accuracy of 100%
* validation set accuracy of 98.5%
* test set accuracy of 96.6%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The adopted LeNet network from the classroom's handwriting exercise produced good initial results i.e. 89% hence it was used.
* What were some problems with the initial architecture?
A correct adoption in terms of layers sizes and selection of a correct sigma parameter. Once sigma was changed from 0.01 to 0.1 much better results were obtained.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
The attempt to replicate the suggested architecture as in the [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) failed. Instead a dropout of 10% was introduced after each layer to address the overfitting. Different level of dropout was tried but 10% seem to work the best for my model. 
* Which parameters were tuned? How were they adjusted and why?
The batch size and epochs were set fixed for most of trials. The major tuning parameter was the learning rate. It was initially set to 0.001 but gradually it was reduced to 0.0008 and then to 0.0005.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
It was noticed that the model was overfitting. To address this a dropout was added. The introduction of dropout was one of the major boosts to the model's accuracy.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
The convolutional networks used in the LeNet network and the mentioned paper demonstrated very good performance in symbols recognition. And as stated in the paper they can work in real time on embedded systems makes them an excellent tool for autonomous vehicles.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 96.6% out of 12630 images were predicted correctly which is strong enough evidence that model works well. However, 4410 images for a validation set seem to be not enough to accurately monitor training progres. Further work would increase this amount to approx. 20% of a training dataset size.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 9 German traffic signs that I found on the web:

![alt text][image6] 

The images are different in quality and size; they were cropped out from the original images. 

The yield, no vehicles and no entry signs should be easy to recognise due to their unique features. Next 4 signs are triangular with a red border and black symbol. It expected these would be a bigger challenge for a model. The final two sings for speed limits also share some similar features so it would be interesting to see if the model can distinguish them correctly.

The images were resized into 32x32 RGB px with use of cv2.resize(, , interpolation=cv2.INTER_AREA). The interpolation can play significant role so it was chosen such way the imported images resemble the quality of images from the training set.

See code in Step 3: Load and Output the Images  (**Traffic_Sign_ClassifierFinal.ipynb**).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:
![alt text][image7] 

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield					| Yield											|
| No vehicles      		| **Speed limit 80km/h**				 				|
| No entry  			| No entry           							|
| General caution		| General caution								|
| Dangerous curve to the right	| Dangerous curve to the right			|
| Road work 			| Road work          							|
| Wild animals			| Wild animals									|
| Speed limit 60km/h   	| Speed limit 60km/h   			 				|
| Speed limit 80km/h   	| **Speed limit 120km/h**		 				|


The model was able to correctly guess 7 of the 9 traffic signs, which gives an accuracy of 77.78%. 

See code in Step 3: Predict the Sign Type for Each Image (**Traffic_Sign_ClassifierFinal.ipynb**).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The bar charts below visualise the softmax probabilities for each image found on the web.

![alt text][image8] 


The final trained network was very confident in its guesses for the correctly recognised 7 images. 

The four triangular shaped signs were predicted correctly with the top probabilities reaching 100%, however it worth to notice that the remaining four probabilities are mostly also for triangular shaped signs. This shape shared by these signs was recognised by the network very well. 

The predictions for the speed limit 60km/h and 80 km/h and no vehicles signs contain also different speed limit signs, what suggested that network have no trouble in recognition of unique features of these signs; a white circle with a red border. Very likely because of this the no vehicles sign was misclassified as a speed limit 80km/h. A similar case is for the speed limit 80km/h sign, which was recognised as the speed limit 120km/h. However, one would expect a misclassification as different double digits speed limit rather than three digits. It is worth to highlight that prior this final run the network had not trouble to recognise all the signs correctly with almost 100% confidence. As the all hyperparameters were fixed the reason for this difference is very likely due to randomness of the generated additional images. 

There is still a room for the improvement as demonstrated in  [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) with a network capable achieving 99.17% test accuracy for this dataset. But even with such high accuracy it would interesting exercise to apply these models to videos, especially from city centres with many ads and other obstructions, which would make the traffic sign recognition much more challenging.



See code in Step 3: Softmax Probabilities For Each Image Found on the Web(**Traffic_Sign_ClassifierFinal.ipynb**).

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


