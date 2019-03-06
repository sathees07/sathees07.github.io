---
layout: post
title: "Animal Recognition"
date: 2019-03-06
excerpt: "how to train tensorflow for Animal Recogniton."
tags: [Machine Learning, Tensorflow]
comments: false
---

#INTRODUCTION

* The purpose of this tutorial is to explain how to train your own convolutional neural network object detection classifier for multiple objects, starting from scratch. At the end of this tutorial, you will have a program that can identify and draw boxes around specific objects in pictures, videos, or in a webcam feed.


This wiki describes every step required to get going with your own object detection classifier: 
1. [Installing TensorFlow-GPU](https://github.com/sathees07/Animal_recognition/wiki/tensorflow_gpu)
2. [Setting up the Object Detection directory structure and Anaconda Virtual Environment](https://github.com/sathees07/Animal_recognition/wiki/Setting-up)
3. [Gathering and labeling pictures](https://github.com/sathees07/Animal_recognition/wiki/Gathering_and_labeling_pictures)
4. [Generating training data](https://github.com/sathees07/Animal_recognition/wiki/generating_tfrecords)
5. [Creating a label map and configuring training](https://github.com/sathees07/Animal_recognition/wiki/labelmap)
6. [Training](https://github.com/sathees07/Animal_recognition/wiki/training)
7. [Exporting the inference graph](https://github.com/sathees07/Animal_recognition/wiki/Exporting_the_inference_graph)
8. [Testing and using your newly trained object detection classifier](https://github.com/sathees07/Animal_recognition/wiki/testing)


### 1. Install TensorFlow-GPU 1.5 (skip this step if TensorFlow-GPU 1.5 is already installed)
Install TensorFlow-GPU by following the instructions in [this YouTube Video by Mark Jay](https://www.youtube.com/watch?v=RplXYjxgZbw).

The video is made for TensorFlow-GPU v1.4, but the “pip install --upgrade tensorflow-gpu” command will automatically download version 1.5. Download and install CUDA v9.0 and cuDNN v7.0 (rather than CUDA v8.0 and cuDNN v6.0 as instructed in the video), because they are supported by TensorFlow-GPU v1.5. As future versions of TensorFlow are released, you will likely need to continue updating the CUDA and cuDNN versions to the latest supported version.

Be sure to install Anaconda with Python 3.6 as instructed in the video, as the Anaconda virtual environment will be used for the rest of this tutorial.

Visit [TensorFlow's website](https://www.tensorflow.org/install/install_windows) for further installation details, including how to install it on other operating systems (like Linux). The [object detection repository](https://github.com/tensorflow/models/tree/master/research/object_detection) itself also has [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).


### 2. Set up TensorFlow Directory and Anaconda Virtual Environment
The TensorFlow Object Detection API requires using the specific directory structure provided in its GitHub repository. It also requires several additional Python packages, specific additions to the PATH and PYTHONPATH variables, and a few extra setup commands to get everything set up to run or train an object detection model. 

This portion of the tutorial goes over the full set up required. It is fairly meticulous, but follow the instructions closely, because improper setup can cause unwieldy errors down the road.

#### 2a. Download TensorFlow Object Detection API repository from GitHub
Create a folder and name it “tensorflow1”. This working directory will contain the full TensorFlow object detection framework, as well as your training images, training data, trained classifier, configuration files, and everything else needed for the object detection classifier.

Download the full TensorFlow object detection repository located at https://github.com/tensorflow/models by clicking the “Clone or Download” button and downloading the zip file. Open the downloaded zip file and extract the “models-master” folder directly into the C:\tensorflow1 directory you just created. Rename “models-master” to just “models”.
(Note, this tutorial was done using this [GitHub commit](https://github.com/tensorflow/models/tree/079d67d9a0b3407e8d074a200780f3835413ef99) of the TensorFlow Object Detection API. If portions of this tutorial do not work, it may be necessary to download and use this exact commit rather than the most up-to-date version.)

#### 2b. Download the ssd_mobilenet_v1_coco_2017_11_17 model from TensorFlow's model zoo
TensorFlow provides several object detection models (pre-trained classifiers with specific neural network architectures) in its [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md). Some models (such as the SSD-MobileNet model) have an architecture that allows for faster detection but with less accuracy, while some models (such as the Faster-RCNN model) give slower detection but with more accuracy.

You can choose which model to train your objection detection classifier on. If you are planning on using the object detector on a device with low computational power (such as a smart phone or Raspberry Pi), use the SDD-MobileNet model. If you will be running your detector on a decently powered laptop or desktop PC, use one of the RCNN models. 

This tutorial will use the ssd_mobilenet_v1_coco_2017_11_17 model. [Download the model here.](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz) Open the downloaded ssd_mobilenet_v1_coco_2017_11_17.tar.gz file with a file archiver  and extract the ssd_mobilenet_v1_coco_2017_11_17 folder to the C:\tensorflow1\models\research\object_detection folder. (Note: The model date and version will likely change in the future, but it should still work with this tutorial.)

#### 2c. Set up new Anaconda virtual environment

In the command terminal that pops up, create a new virtual environment called “tensorflow1” by issuing the following command:
```
conda create -n tensorflow1 pip python=3.5
```
Then, activate the environment by issuing:
```
activate tensorflow1
```
Install tensorflow-gpu in this environment by issuing:
```
(tensorflow1) pip install --ignore-installed --upgrade tensorflow-gpu
```
Install the other necessary packages by issuing the following commands:
```
(tensorflow1) conda install -c anaconda protobuf
(tensorflow1) pip install pillow
(tensorflow1) pip install lxml
(tensorflow1) pip install Cython
(tensorflow1) pip install jupyter
(tensorflow1) pip install matplotlib
(tensorflow1) pip install pandas
(tensorflow1) pip install opencv-python
```
(Note: The ‘pandas’ and ‘opencv-python’ packages are not needed by TensorFlow, but they are used in the Python scripts to generate TFRecords and to work with images, videos, and webcam feeds.)

#### 2d. Configure PYTHONPATH environment variable
A PYTHONPATH variable must be created that points to the \models, \models\research, and \models\research\slim directories. Do this by issuing the following commands (from any directory):
```
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

```
(Note: Every time the "tensorflow1" virtual environment is exited, the PYTHONPATH variable is reset and needs to be set up again.)

#### 2e. Compile Protobufs and run setup.py
Next, compile the Protobuf files, which are used by TensorFlow to configure model and training parameters. Unfortunately, the short protoc compilation command posted on TensorFlow’s Object Detection API [installation page](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) does not work on Windows. Every  .proto file in the \object_detection\protos directory must be called out individually by the command.

In the Anaconda Command Prompt, change directories to the \models\research directory and copy and paste the following command into the command line and press Enter:
```
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
```
This creates a name_pb2.py file from every name.proto file in the \object_detection\protos folder.

**(Note: TensorFlow occassionally adds new .proto files to the \protos folder. If you get an error saying ImportError: cannot import name 'something_something_pb2' , you may need to update the protoc command to include the new .proto files.)**

Finally, run the following commands from the C:\tensorflow1\models\research directory:
```
(tensorflow1) :\tensorflow1\models\research> python setup.py build
(tensorflow1) :\tensorflow1\models\research> python setup.py install
```

#### 2f. Test TensorFlow setup to verify it works
The TensorFlow Object Detection API is now all set up to use pre-trained models for object detection, or to train a new one. You can test it out and verify your installation is working by launching the object_detection_tutorial.ipynb script with Jupyter. From the \object_detection directory, issue this command:
```
(tensorflow1) :\tensorflow1\models\research\object_detection> jupyter notebook object_detection_tutorial.ipynb
```
This opens the script in your default web browser and allows you to step through the code one section at a time. You can step through each section by clicking the “Run” button in the upper toolbar. The section is done running when the “In [ * ]” text next to the section populates with a number (e.g. “In [1]”). 

(Note: part of the script downloads the ssd_mobilenet_v1 model from GitHub, which is about 74MB. This means it will take some time to complete the section, so be patient.)

![jupyter](https://github.com/sathees07/Animal_recognition/blob/master/jupyter_notebook_dogs.jpg)

Once you have stepped all the way through the script, you should see two labeled images at the bottom section the page. If you see this, then everything is working properly! If not, the bottom section will report any errors encountered. See the [Appendix](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10#appendix-common-errors) for a list of errors I encountered while setting this up.


### 3. Gather and Label Pictures
* Now that the TensorFlow Object Detection API is all set up and ready to go, we need to provide the images it will use to train a new detection classifier.

#### 3a. Gather Pictures
* TensorFlow needs hundreds of images of an object to train a good detection classifier. To train a robust classifier, the training images should have random objects in the image along with the desired objects, and should have a variety of backgrounds and lighting conditions. There should be some images where the desired object is partially obscured, overlapped with something else, or only halfway in the picture. 

* You can use your phone to take pictures of the objects or download images of the objects from Google Image Search. I recommend having at least 200 pictures overall. 
* Make sure the images aren’t too large. They should be less than 200KB each, and their resolution shouldn’t be more than 720x1280. The larger the images are, the longer it will take to train the classifier. You can use the resizer.py script in this repository to reduce the size of the images.

* After you have all the pictures you need, move 20% of them images\test directory, and 80% of them to the images\train directory. Make sure there are a variety of pictures in both the \test and \train directories.

#### 3b. Label Pictures
* Here comes the fun part! With all the pictures gathered, it’s time to label the desired objects in every picture. LabelImg is a great tool for labeling images, and its GitHub page has very clear instructions on how to install and use it.

* [LabelImg GitHub link](https://github.com/tzutalin/labelImg)

![labelimg](https://raw.githubusercontent.com/tzutalin/labelImg/master/demo/demo3.jpg)
* Download and install LabelImg, point it to your \images\train directory, and then draw a box around each object in each image. Repeat the process for all the images in the \images\test directory. This will take a while! 

* LabelImg saves a .xml file containing the label data for each image. These .xml files will be used to generate TFRecords, which are one of the inputs to the TensorFlow trainer. Once you have labeled and saved each image, there will be one .xml file for each image in the \test and \train directories.


### 4. Generate Training Data
With the images labeled, it’s time to generate the TFRecords that serve as input data to the TensorFlow training model. This tutorial uses the xml_to_csv.py and generate_tfrecord.py scripts from [Dat Tran’s Raccoon Detector dataset](https://github.com/datitran/raccoon_dataset), with some slight modifications to work with our directory structure.

First, the image .xml data will be used to create .csv files containing all the data for the train and test images. From the \object_detection folder, issue the following command in the Anaconda command prompt:
```
(base) $ \tensorflow1\models\research\object_detection> python xml_to_csv.py
```
This creates a train_labels.csv and test_labels.csv file in the \object_detection\data folder. 

Next, open the generate_tfrecord.py file in a text editor. Replace the label map starting at line 31 with your own label map, where each object is assigned an ID number. This same number assignment will be used when configuring the labelmap.pbtxt file in Step 5b. 

For example, say you are training a classifier to detect basketballs, shirts, and shoes. You will replace the following code in generate_tfrecord.py:
```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'cat':
        return 1
    elif row_label == 'dog':
        return 2
    elif row_label == 'frog':
        return 3
    elif row_label == 'lizards':
        return 4
    elif row_label == 'monkey':
        return 5
    elif row_label == 'snake':
        return 6
    else:
        return None
```
With this:
```
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'basketball':
        return 1
    elif row_label == 'shirt':
        return 2
    elif row_label == 'shoe':
        return 3
    else:
        return None
```

Then, generate the TFRecord files by issuing these commands from the \object_detection folder:
```
python generate_tfrecord.py --csv_input=images\train_labels.csv --image_dir=images\train --output_path=train.record
python generate_tfrecord.py --csv_input=images\test_labels.csv --image_dir=images\test --output_path=test.record
```
These generate a train.record and a test.record file in \object_detection. These will be used to train the new object detection classifier.



### 5. Create Label Map and Configure Training
The last thing to do before training is to create a label map and edit the training configuration file.

#### 5a. Label map
The label map tells the trainer what each object is by defining a mapping of class names to class ID numbers. Use a text editor to create a new file and save it as labelmap.pbtxt in the \tensorflow1\models\research\object_detection\training folder. (Make sure the file type is .pbtxt, not .txt !) In the text editor, copy or type in the label map in the format below (the example below is the label map for my Animal Detector):
```
item {
  id: 1
  name: 'cat'
}

item {
  id: 2
  name: 'dog'
}

item {
  id: 3
  name: 'frog'
}

item {
  id: 4
  name: 'lizards'
}

item {
  id: 5
  name: 'monkey'
}

item {
  id: 6
  name: 'snake'
}
```
The label map ID numbers should be the same as what is defined in the generate_tfrecord.py file. For the basketball, shirt, and shoe detector example mentioned in Step 4, the labelmap.pbtxt file will look like:
```
item {
  id: 1
  name: 'basketball'
}

item {
  id: 2
  name: 'shirt'
}

item {
  id: 3
  name: 'shoe'
}
```

#### 5b. Configure training
Finally, the object detection training pipeline must be configured. It defines which model and what parameters will be used for training. This is the last step before running training!

Navigate to C:\tensorflow1\models\research\object_detection\samples\configs and copy the ssd_mobilenet_v1_coco.config file into the \object_detection\training directory. Then, open the file with a text editor. There are several changes to make to the .config file, mainly changing the number of classes and examples, and adding the file paths to the training data.

Make the following changes to the ssd_mobilenet_v1_coco.config file. Note: The paths must be entered with single forward slashes (NOT backslashes), or TensorFlow will give a file path error when trying to train the model! Also, the paths must be in double quotation marks ( " ), not single quotation marks ( ' ).

- Line 9. Change num_classes to the number of different objects you want the classifier to detect. For the above basketball, shirt, and shoe detector, it would be num_classes : 3

- Lines 126 and 128. In the train_input_reader section, change input_path and label_map_path to:
  - input_path : "/tensorflow1/models/research/object_detection/train.record"
  - label_map_path: "/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

- Line 132. Change num_examples to the number of images you have in the \images\test directory.

- Lines 140 and 142. In the eval_input_reader section, change input_path and label_map_path to:
  - input_path : "/tensorflow1/models/research/object_detection/test.record"
  - label_map_path: "/tensorflow1/models/research/object_detection/training/labelmap.pbtxt"

Save the file after the changes have been made. That’s it! The training job is all configured and ready to go!

### 6. Run the Training
**UPDATE 9/26/18:** 
*As of version 1.9, TensorFlow has deprecated the "train.py" file and replaced it with "model_main.py" file. I haven't been able to get model_main.py to work correctly yet (I run in to errors related to pycocotools). Fortunately, the train.py file is still available in the /object_detection/legacy folder. Simply move train.py from /object_detection/legacy into the /object_detection folder and then continue following the steps below.*

Here we go! From the \object_detection directory, issue the following command to begin training:
```
python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config	
```
If everything has been set up correctly, TensorFlow will initialize the training. The initialization can take up to 30 seconds before the actual training begins. 

Each step of training reports the loss. It will start high and get lower and lower as training progresses.  I recommend allowing your model to train until the loss consistently drops below 0.05, which will take about 40,000 steps, or about 2 hours (depending on how powerful your CPU and GPU are). Note: The loss numbers will be different if a different model is used. MobileNet-SSD starts with a loss of about 20, and should be trained until the loss is consistently under 2.
![training](https://github.com/sathees07/Animal_recognition/blob/master/training.png)

You can view the progress of the training job by using TensorBoard. To do this, open a new instance of Anaconda Prompt, activate the base virtual environment, change to the \tensorflow1\models\research\object_detection directory, and issue the following command:
```
(base) \tensorflow1\models\research\object_detection>tensorboard --logdir=training
```
This will create a webpage on your local machine at YourPCName:6006, which can be viewed through a web browser. The TensorBoard page provides information and graphs that show how the training is progressing. One important graph is the Loss graph, which shows the overall loss of the classifier over time.

![tensor](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10/raw/master/doc/loss_graph.JPG)

The training routine periodically saves checkpoints about every five minutes. You can terminate the training by pressing Ctrl+C while in the terminal window. I typically wait until just after a checkpoint has been saved to terminate the training. You can terminate training and start it later, and it will restart from the last saved checkpoint. The checkpoint at the highest number of steps will be used to generate the frozen inference graph.


### 7. Export Inference Graph
Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the \object_detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder:
```
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_mobilenet_v1_pets.config	 --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph
```
This creates a frozen_inference_graph.pb file in the \object_detection\inference_graph folder. The .pb file contains the object detection classifier.


### 8. Use Your Newly Trained Object Detection Classifier!
The object detection classifier is all ready to go! I’ve written Python scripts to test it out on an image, video, or webcam feed.

Before running the Python scripts, you need to modify the NUM_CLASSES variable in the script to equal the number of classes you want to detect. (For my animal Detector, there are six animal I want to detect, so NUM_CLASSES = 6.)

Results:

![result1](https://github.com/sathees07/Animal_recognition/blob/master/result1.png)

![result2](https://github.com/sathees07/Animal_recognition/blob/master/result21.png)

To test your object detector, move a picture of the object or objects into the \object_detection folder, and change the IMAGE_NAME variable in the Object_detection_image.py to match the file name of the picture. Alternatively, you can use a video of the objects (using Object_detection_video.py), or just plug in a USB webcam and point it at the objects (using Object_detection_webcam.py).




