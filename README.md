# Diagnose Pneumonia with Convolution Neural Network

## Background

Inspired by the paper we found named “Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning” published earlier this year on Cell about the breakthroughs of utilizing Machine Learning on certain medical field, our team decided to do classification task on the chest X-ray to be and most interesting and most relevant to what we’ve learn in the class since will have the opportunity to practice optimization of training convolutional network on real life dataset.

### Dataset

The dataset contains around 5000 Chest X-ray images of pediatric patients from one to five years old. The images are classified into two classes: the ones with Pneumonia and the normal.  They were pre-selected for quality control and all the images are human readable (for doctors) and variable sizes.

## Steps

### Choosing Pretrained CNN for Feature Extraction

Our first step was finding a CNN model that suits the task. For every CNN, there are two groups of layers, the feature extraction layers and the linear classification layers that based on the extracted features. The feature extraction layers are often the more complex ones in a CNN compare to the linear classification layers. At the same time, the feature extraction layers are much more reusable than the linear classification layers since most of the image recognition tasks depends on similar lower level features. Therefore, we decided to start with using a pretrained version of renowned convolutional network for feature extraction. There were four popular CNNs in our consideration: AlexNet, DenseNet, Inception V3, and ResNet. These networks all have strong feature extraction capabilities. Since we need to choose one from them we need to do some testing to make comparison of their performance on the dataset. However, one issue we encountered is that these network have very different linear classification layers. The Inception V3 and Resnet both have only 1 layer of linear classifier while the our two have multiple ones.  In order to do a fair comparison of the networks on feature extraction, we decided to replaced the original classifier layers of all four CNNs with our own classifier layers:

```python
model.classifier = nn.Sequential(
    nn.Linear(in_features, 1024, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(1024, 512, bias=True),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.3),
    nn.Linear(512, NUM_CLASSES, bias=True)
)
```

We carefully choose the layers to do the linear classification with extra dropout layers to reduce potential overfitting. We freezed the convolution layers of these networks in training that they will stay on their pretrian values and only trained the classifier layers. With learning rate 1e-5 and 30 epochs, we got the following results:

AlexNet:

![AlexNet](https://github.com/liangyuRain/chest-xray-pneumonia/blob/master/images/ALEX_30_1e-05_result.png)

DenseNet:

![DenseNet](https://github.com/liangyuRain/chest-xray-pneumonia/blob/master/images/DENSE_30_1e-05_result.png)

Inception V3:

![Inception V3](https://github.com/liangyuRain/chest-xray-pneumonia/blob/master/images/INCEPT_30_1e-05_result.png)

ResNet:

![ResNet](https://github.com/liangyuRain/chest-xray-pneumonia/blob/master/images/RES_30_1e-05_result.png)

As we can see from the results, all networks have similar performance in terms of test accuracy. 
However, the test loss of Inception v3 is very stable compare to the other models and keeps on decrease continuously as we ran through the Epochs. The increment of test loss of all the other models is a clear sign that these networks were overfitting. This made Inception v3 standout as we saw its best potential in avoiding overfitting. After these results, we decide to continue our optimization of CNN with Inception V3 network.

### First Attempt to Unfreeze Convolution Layers

After choosing the network to continue with, we had to decide if we want to keep all convolutional layers freezed in training. According to what we’ve learned in class, lower layers of convolution network are recognizing edges and shapes of the image, and only the higher layers of the network are recognizing the overall contents of the image. Because our dataset have several key differences from the imagenet dataset which the pretrained networks were trained to adapt (that our dataset does not have color and the images in the two categories have many similarities), we may require different higher layers. Note that the lower layers of the pretrained network are still useful, because the edge and shape recognition should be the same across different images. Therefore, we now try to gradually unfreeze higher layers of the Inception network and compare the results.

![Inception V3](https://github.com/liangyuRain/chest-xray-pneumonia/blob/master/images/INCEPT_30_1e-05_unfrozen.png)

We have unfreeze the last three layers of the Inception network. Unfortunately, the network overfits quickly during the training without significant improvement in test accuracy.

### Data Augmentation

In order to combat the overfitting problem, we decide to do some data augmentation on our dataset. Because our training data and target data are all chest X-ray images, it is impossible for target images to be too different from training images in terms of certain characteristics. For example, the color of the images must be grayscale, and the position of the lungs are roughly the same. Therefore, we need to be careful when choosing data augmentation techniques. Some techniques like random vertical flip and color Jitter are not suitable in our case. We end up with the following transforms:

```python
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(size=input_size),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(1, 1.2), shear=0.1),
    transforms.RandomCrop(size=input_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

```

Since human lungs are generally symmetric, we use horizontal flip in the data augmentation. Besides the horizontal flip, we have also used random affine with relatively small parameters, because we do not want to change the shape and direction of the X-ray images too much.

With data augmentation, we retrained Inception without unfreeze pretrained layers. The result ends up with a slight improvement on test accuracy. Without data augmentation, test accuracy becomes stable around 83%; with data augmentation, test accuracy becomes stable around 85%.

### Second Attempt to Unfreeze Pretrained Layers

With data augmentation, our network becomes more robust against overfitting. Then we continue to work on our idea of unfreezing higher level pretrained layers. This time the improvement is significant. Based on our training result before, we found that the networks generally converge within 5 epochs. This is also what we expected with pretrained network. We gradually unfreeze more and more pretrained layers with one module of layers as a step. 

3 modules unfrozen:

![3 modules unfrozen](https://github.com/liangyuRain/chest-xray-pneumonia/blob/master/images/INCEPT_5_1e-05_4_unfrozen.png)

9 modules unfrozen:

![9 modules unfrozen](https://github.com/liangyuRain/chest-xray-pneumonia/blob/master/images/INCEPT_5_1e-05_10_unfrozen.png)

12 modules unfrozen:

![12 modules unfrozen](https://github.com/liangyuRain/chest-xray-pneumonia/blob/master/images/INCEPT_5_1e-05_13_unfrozen.png)

Based on our experiments, our best performance ends up with 9 modules of the Inception V3 network unfrozen. The accuracy is around 91%, significantly higher than our previous performance.

### Future Work: Balance Classes

In our dataset, we have two classes: normal and pneumonia. However, the sizes of two classes are imbalanced with pneumonia having 3875 images and normal having only 1341 images. In our exploration, we have only tried simple oversampling the normal data to balance the classes; however, the result is not ideal. The network overfits quickly, which is a clear disadvantage of oversampling technique. 

There are many other more advanced balancing techniques existing, such as anomaly detection algorithms. Here, we do not have enough time to do experiment on them, but in our opinions, balancing classes has a great potential in further improving the performance of our network.

### Conclusion

During our exploration for a optimal network to diagnose pneumonia, we have tried numerous approaches to improve our pretrained network. We firstly replaced the original classifier part of pretrained CNNs with our own classifier layers. Then we perform training on different pretrained networks and choose Inception V3 network based on the performance. Finally, we used data augmentation and unfroze higher layers of the CNN to obtain an accuracy over 90% on our test dataset. Because we use pretrained network, the network easily overfit in our training. Our exploration is a continuous combat versus overfitting. If better techniques to prevent overfitting are used, we believe the network has great potential to have better performance.
