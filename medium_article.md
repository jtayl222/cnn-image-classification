# Using Convolutional Neural Networks for Classification: From Training to Deployment

If you’ve been keeping an eye on the rapidly evolving field of deep learning, you’ve undoubtedly heard about Convolutional Neural Networks (CNNs) and their near-magical ability to classify images. With a powerful CNN under the hood, you can build applications that recognize dog breeds, diagnose medical scans, or even detect diseases in plants. In this article, we’ll dive into the details of CNN-based classification—covering data loading, training, validation, inference, and even cloud deployment—so you can get started on your own image classification journey.

But first, a little background on where all of this comes from.

---

## Introduction: Why the AI Programming with Python Nanodegree?

Everything we explore here is rooted in the *AI Programming with Python Nanodegree* from [Udacity](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089). This program is a well-rounded introduction to essential AI concepts, covering Python, NumPy, pandas, Matplotlib, PyTorch, and more. By the end of it, you not only gain foundational knowledge in building AI applications—you also develop hands-on projects that give you the confidence to tackle more advanced topics, such as CNNs.

If you’re serious about a career in AI, **I highly recommend signing up** for this Nanodegree. It’s structured, self-paced (with deadlines to keep you motivated), and packed with real-world applications. CNN-based image classification is just one of many projects you’ll master, giving you the perfect springboard into more specialized machine learning and deep learning roles.

---

Below is **all** the information you need to get started on building and deploying a CNN-based image classification project. The instructions and explanations below come straight from a working example of an Image Classifier project.

The text below includes:

1. A detailed walkthrough of the project’s structure and objectives.  
2. The exact steps for deployment—both locally and via AWS Sagemaker.

Everything here is provided under the same open-source license as the original repository. For full license details, see:  
[https://github.com/jtayl222/aipnd-project/blob/master/LICENSE](https://github.com/jtayl222/aipnd-project/blob/master/LICENSE)

---

## The Image Classification Project

**Image Classifier Project**  

In this project, you’ll train an image classifier to recognize different species of flowers. You can then use the trained classifier to make predictions of flower name from new images. You’ll also be given the option to export your model for use in a “Command Line Application” or an “Amazon SageMaker Application.”  

We’ll be using this project to practice using PyTorch, your newly acquired knowledge of the command line, and more advanced Python concepts like object-oriented programming and file I/O. Also, it is used to measure your newly acquired knowledge in the field of deep learning and implementing them using python. The project can be broken down into multiple steps. They are:

1. Load and preprocess the image dataset  
2. Train the image classification model  
3. Use the model to make predictions  
4. Export the model to run in a different environment.

We’ll begin by importing packages and writing the code to load and preprocess the data. Then we’ll proceed to model training and validation. Next, we use the model to make predictions and measure the accuracy. Finally, we provide instructions for how to export your model.

We’ll also go through an optional part of the project: how to deploy the model to AWS Sagemaker. We’ll cover how to prepare your model for deployment, how to set up AWS Sagemaker, how to create an endpoint, and how to make predictions from new images using the endpoint.

Let’s get started!

**The dataset**  
The dataset we will be using is from the Oxford Flower dataset. This dataset contains images of 102 flower categories, most of which are common in the United Kingdom. The dataset is split into three parts: training, validation, and testing. The images are not necessarily of the same size or orientation, so we’ll need to process them to be uniform in size. Our image classifier will learn to identify each class of flower in the dataset. We’ll measure performance by seeing how well our model does on the test dataset.

**Project structure**  
The project has the following structure:

- `README.md`: A description of the project and instructions  
- `cat_to_name.json`: A mapping of flower category labels to names  
- `predict.py`: A command line application to predict flower type  
- `train.py`: A command line application to train the image classifier  
- `workspace_utils.py`: Utility functions to help with training in the Udacity Workspace  
- `notebooks`: Contains Jupyter notebooks for experimenting  
- `images`: Contains sample images for testing  
- `checkpoint.pth`: A saved model checkpoint  

The Jupyter Notebooks or `train.py` can be used to build and train the model. The final model can be saved to `checkpoint.pth`. Then you can use `predict.py` to run predictions on new images.

**Technical details**  
We’ll be using PyTorch for building the neural networks. We’ll use a pretrained model like VGG or Densenet. Then, we’ll define a new feed-forward network as a classifier, attach it to the pretrained network, and train only the classifier parameters. We’ll use the Adam optimizer and the `NLLLoss` (negative log likelihood) for classification. We’ll also use techniques like validation to measure overfitting or underfitting during training, as well as transformations like random resizing and flips to augment the training data. In the final stage, we measure the test accuracy to see how well the model performs on unseen data.

**Training**  
To train the model, we do the following steps:

1. Load the data  
2. Create data transforms for training, validation, and testing  
3. Load the datasets with `ImageFolder`  
4. Using the image datasets and the transforms, define the dataloaders  
5. Build and train a new feed-forward network as a classifier using a pretrained model  
6. Use the validation data to measure the accuracy during training  
7. Save the model parameters to `checkpoint.pth`

**Prediction**  
We can load the saved model and use it to predict flower categories for new images. We’ll process the input image with the same transforms we used for validation. Then feed it to the model and get the probabilities for each category. We’ll return the top K categories with the highest probabilities. Using `cat_to_name.json`, we can map the category labels to the actual flower names.

**Command line applications**  
The `train.py` script trains a new network on a dataset and saves the model as a checkpoint. It takes arguments such as data directory, architecture, hyperparameters like learning rate, hidden units, training epochs, GPU, etc. The `predict.py` script predicts the flower name from an image using a trained network. It also has optional arguments like the top K classes, the category names file, and whether to use GPU.

**AWS deployment**  
We’ll optionally show how to deploy the model to AWS Sagemaker. We’ll create a model endpoint that can receive requests with the image data and return the predictions for the flower class. We’ll discuss the steps to configure AWS IAM roles, create a Docker container with the model and dependencies, push it to ECR, and create a Sagemaker endpoint. We’ll then show how to send a request to the endpoint with an image and receive the classification result.


---

## Deployment Guide

**Deployment Guide**  

This guide walks you through deploying your image classifier project to various environments. We’ll cover local deployment, command line interface usage, AWS Sagemaker deployment, and further suggestions.

**Local environment**  
1. **Requirements**:  
   - Python 3.x  
   - PyTorch  
   - torchvision  
   - numpy  
   - PIL  
   - command line tools  
2. Clone or download the repository.  
3. Install the requirements via pip:

   ```bash
   pip install -r requirements.txt
   ```

4. Run `python train.py` to train a new network or `python predict.py` to predict from an existing checkpoint.

**Command line interface**  

**train.py usage**:

```bash
python train.py data_dir --save_dir SAVE_DIRECTORY --arch ARCH --learning_rate LEARNING_RATE --hidden_units HIDDEN_UNITS --epochs EPOCHS --gpu
```

**example**:

```bash
python train.py flowers --save_dir checkpoints --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 3 --gpu
```

**predict.py usage**:

```bash
python predict.py image_path checkpoint --top_k TOP_K --category_names CATEGORY_NAMES --gpu
```

**example**:

```bash
python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 3 --category_names cat_to_name.json --gpu
```

**AWS Sagemaker**  
To deploy the model to AWS Sagemaker, follow these steps:

1. Create an AWS account and configure IAM roles with Sagemaker and ECR permissions.  
2. Build a Docker image that has your model and inference code. Include the Python environment and dependencies.  
3. Tag and push the Docker image to ECR (Elastic Container Registry).  
4. In the Sagemaker console, create a new model by pointing to the ECR image.  
5. Create an endpoint configuration and an endpoint using the model.  
6. Test the endpoint by sending an image payload and receiving the predicted class.

**Details**:
- The inference script should load the trained model checkpoint and accept image data.  
- Convert the incoming image into the same format used during training (transforms).  
- Run a forward pass in the model to get probabilities.  
- Return the top class(es) in JSON format.

**Tips and further suggestions**  
- Monitor GPU usage and memory usage while training large models.  
- Experiment with different architectures (e.g. `vgg16`, `densenet121`, `resnet50`) to see which performs best.  
- Adjust hyperparameters, data augmentations, and training durations to improve results.  
- If deploying to the cloud, be mindful of cost and resource usage.  
- Consider building a simple front-end to allow users to upload images and see predictions in real-time.

We hope this guide helps you successfully deploy your image classifier project! If you run into any issues, check the forums or documentation for troubleshooting. Good luck!

---

## Conclusion

Convolutional Neural Networks have revolutionized the field of image classification by making it easier than ever to achieve high accuracy on complex datasets. By walking through the entire process—loading data, configuring a pretrained model, training, validation, and finally deployment to AWS—this project shows the power and flexibility of modern deep learning techniques.

If you’re hungry for more, remember that this entire tutorial is grounded in skills taught in [Udacity’s AI Programming with Python Nanodegree](https://www.udacity.com/course/ai-programming-python-nanodegree--nd089). It’s a fantastic program for learning the foundations of deep learning with PyTorch and for building practical, resume-boosting projects like this Image Classifier.

So why wait? **Sign up** for the Nanodegree, roll up your sleeves, and start building! There’s never been a better time to harness the power of AI in your own projects. By mastering CNNs, you’ll be poised to push the boundaries of computer vision and bring cutting-edge applications to life.

---

For license details covering this content, please see:  
[https://github.com/jtayl222/aipnd-project/blob/master/LICENSE](https://github.com/jtayl222/aipnd-project/blob/master/LICENSE)

Happy learning—and here’s to building something amazing!