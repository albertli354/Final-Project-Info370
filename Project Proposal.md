# Project Proposal

##### INFO 370 Core Methods in Data Science

##### Group Member: Katie (Ziqi) Chen, Albert (Xiang) Li, Emily Ding, Kenneth Long

## Project Overview

A digital recognizer for handwriting is sophisticated in many areas. However, it’s still hard to identify different handwriting style for different people. The recent developments in online handwriting recognition allow for such input modalities but contain lots of problems. [Online Handwriting Recognition Problem: Issues and Techniques](https://pdfs.semanticscholar.org/f957/3acd8405b5c594314491dedfbeb3bf40750f.pdf) Meanwhile, the different handwriting style may cause misunderstanding, and reduce the accuracy of digital technology. Therefore, our research problem is, how to differentiate the handwriting style by different people? Through the ML and statistical model comparison, we hope to find a good model to predict the owner of a handwritten image. For future work, we may establish our own model theoretically to achieve a high accuracy prediction.

## Project Description

#### The Purpose of the Study:

The overarching purpose of this research is to test whether we can use ML to differentiate different persons based on their handwriting images. We think it is important in such a way that once we can identify different persons, we can train the algorithm to adapt to that particular writing style to increase recognition accuracy. This process is especially important in the area of online handwriting recognition. 

Currently, most handwritten recognition requires the adherence to standard order, stoke, appearance, etc. Such restriction makes it hard for people from different cultures to use the technology since writing style is hugely different across cultures. 

As we found there are research consolidated with some natural ML tool to predict the actual character based on a handwritten image, however, using a model to distinguish the handwriting style is still a big challenge. Our project focuses on the prediction for different writers instead of the exact character by using ML and statistical model. The goal is to establish a model for high accuracy prediction and test whether if it does matter for the difference by different handwriting style. The broader problem domain is, how to differentiate the handwriting style by different people? If we can pick a model to increase the accuracy of prediction for people, we can facilitate the current related research or studies as a sort of reference.

Forensic handwriting analysts have previously individually performed handwriting analysis for writer identification in a legal context, and legal professionals in the justice system have relied on these skills and abilities to aid in investigative and prosecutorial capacities [Individuality of Handwriting](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.3775&rep=rep1&type=pdf). Using these techniques, the confidence with which statements can be made and the number of documents that can be processed can increase. Making these systems language independent can offer a wide variety of advances in ML itself, as well as specific handwriting applications [End-to-End Online Writer Identification With Recurrent Neural Network](https://ieeexplore.ieee.org/abstract/document/7801018). These types of technologies can also be utilized in the financial industry and government for signature validation, allowing early detection to suggest investigating possible fraud or additional verification measures [Learning strategies and classification methods for off-line signature verification](https://ieeexplore.ieee.org/abstract/document/1363904).

One of the target audiences for this project is law enforcement. When they perform forensic handwriting analysts, they face the exact same task as our project - identifying the owner of a certain handwriting image. By using our model, they can use existing suspects’ handwriting images to train our model and let it predict whether the handwriting belongs to the suspect. Law enforcement can also use the accuracy score as a reference for whether to adopt forensic handwriting as strong evidence. We know there are no perfect forensic handwriting analysts and there is some dispute regarding whether handwriting analysis is sufficiently reliable to be admissible. (https://www.frostbrowntodd.com/resources-is-your-handwriting-expert-testimony-admissible.html) By using our model, judges have a complementary reference to decide whether to adopt the result of forensic handwriting analysts

A potential question for them is the reliability of the prediction. If we want law enforcement to utilize the model, the accuracy should be high enough to avoid potential false blames. We want to incorporate multiple machine learning methods to improve accuracy. 


#### The Hypotheses:

Null hypothesis: There is no significant difference between different persons in terms of handwriting style and we cannot differentiate persons based on handwritten images.

Alternative hypothesis: there is a significant difference between individuals in terms of handwriting style and we can differentiate persons based on handwritten images.

#### The Dataset:

The dataset we are using is [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19). It consists of 5 zipped files with a total of 3,992,357 images. The dataset completed by Bethesda High school students and Census Bureau employees in Suitland, Maryland. Every participant asked to write all alphanumeric characters in a case-sensitive manner. We will use the dataset classified by author (the by_write organization of the dataset per the description in https://s3.amazonaws.com/nist-srd/SD19/sd19_users_guide_edition_2.pdf), meaning we have the labels that tell us the image belong to different authors. However, all personal identifying information was removed in the dataset. 

## Technical Description


For ML model, we are considering about support vector machines (**SVM**), Convolutional Neural Networks (**CNN**), Recurrent Neural Networks (**RNN**) and Deep Neural Networks (**DNN**) because these are popular ML models for Computer Vision and Classification. 2D Convolution is commonly used in Image processing, it is like simple 1D convolution commonly covered in Differential Equations when discussing Unit Impulse Response, the Heaviside function, and Dirac delta. A visual explanation that greatly simplifies the concept in this context can be found [here](http://setosa.io/ev/image-kernels/).  Especially, CNN, RNN, and DNN are able to perform tasks like classification, prediction, decision-making and visualization in an efficient way to increase the performance. DNN is used in the popularized model [DeepDream](https://ai.googleblog.com/2015/07/deepdream-code-example-for-visualizing.html).

For the statistical models, we may use logistic regression: given the image with a known writer, we want to predict if another image belongs to the same writer (1) or not (0). If we obtain the estimated probability to be greater than 0.5, we can conclude that these two images belong to the same writer. Classifiers often group samples features together through clustering in some vector subspace. Metrics like KNN “distance” are used in some literature we have cited. As previously mentioned, we will evaluate Chi-Squared Goodness of Fit, ANOVA variations and possibly methods that are new to us like the Friedman Test to evaluate the probabilities between different features and writers.

Our final resource will present as sorts of HTML pages based on a GitHub repository. The process of this project can be broken down into three general steps: preprocessing data, feature engineering, and classification. 

For preprocessing, since all images are stored in PNG format, we plan to use the `Pillow` module to convert the image to `numpy` array in a Pandas DataFrame.


```#example loading .png files into Pandas DataFrames
from PIL import Image
import numpy as np
import PIL

df = pd.DataFrame(columns=["image", "label", "writer"])

#this can be replaced with os.listdir() or some similar way to iterate dir
#then the path will be parsed for the writer and files in the writer directory (each directory contains multiple samples of each character, digit et c.
for writer in ["45"]:
    for i in range(1):
        #load image using pillow
        im = Image.open("../by_write/hsf_2/f1000_45/d1000_45/d1000_"+writer+"_000"+"{:02d}".format(i)+".png")
	#convert the image to 2bit color
        bit2 = im.convert("P", palette=Image.ADAPTIVE, colors=4)
	#calculate lower res size to use less memory
	#alternatively we may use numpy.memmap to lazy load images
        x = int(im.width/2)  # 128
        y = int(im.height/2) # 128
        #resize to smaller resolution
        bit2xy = bit2.resize((x, y), resample=PIL.Image.LANCZOS)
	#create a numpy array using the pixel data
        arr = np.array(bit2xy.getdata())
	#tell numpy to reshape this to a 2d array
        arr = arr.reshape((x, y))
	#for demonstration purposes, show the individual pixel data
        display(arr)
	#demo, we can show the image
        display(plt.imshow(arr))
	#create a new row with the pixel data, digit label in this case, and the writer ID
        row = {"image":arr, "label":i%9, "writer":writer}
	#demo what the row looks like
        display(row)
	#add the row to the pandas DataFrame
        df = df.append(row, ignore_index=True)

#show what the DataFrame looks like
df
```

Each value in the `numpy` array represents the gray-scale of the particular pixel on the image. The other important step in preprocessing is the categorization of writers. Since different writers’ images are stored in different folders and there are no labels in the dataset indicating any personal identifying information, we need to add writer label for each row of the data frame. We will parse the filename paths and iterate over directories using python’s built-in os module listdir() function or some other similar built-in method. Ideally, when we’ve completed preprocessing, our dataset’s individual observation contains a writer id label, an array of values indicating the pixels in the corresponding image, and other features we will generate based on analysis of labeled data (e.g. the digit or character they wrote) and the image data itself using ML techniques. 

We then want to perform exploratory data analysis for the dataset. We want to address the distribution of the same letter across different writers to see whether there is a significant difference between writers. We also want to use the scatter matrix to explore the correlation between different features to understand which features contribute most to the label.

The next stage is feature engineering. We want to create features indicating whether the writer is a student or an employee at the Census Bureau. We also need to create a label for each individual observations. We then want to use feature selection techniques to filter relevant pixel information into our model. 

We will then test classification. In this step, various models are used to map the selected features to different classes and thus differentiate the writers. Based on accuracy score and other metrics we’ve previously discussed, we can evaluate the best model for writer identification predictions.

Finally, we will discuss our results as part of the evaluation of the test set predictions and state conclusions based on statistical test metrics and other results. For future work, we may build up our own model to test the accuracy and facilitate these industries which authenticate and identify the users who use their system or product. It’s also meaningful for the social application. For example, a system can separate different group of age (e.g., teenagers or adults) using a scanner or digital recognizer based on the writing style.






