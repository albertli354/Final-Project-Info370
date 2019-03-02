# Project Proposal

##### INFO 370 Core Methods in Data Science

##### Member info: Member: Katie (Ziqi) Chen, Abert (Xiang) Li, Emily Ding, Kenneth Long

## Project Overview

A digital recognizer for handwriting is sophisticated in many areas. However, it’s still hard to identify different handwriting style for different people. The recent developments in online handwriting recognition allow for such input modalities but contain lots of problems. [Online Handwriting Recognition Problem: Issues and Techniques](https://pdfs.semanticscholar.org/f957/3acd8405b5c594314491dedfbeb3bf40750f.pdf) Meanwhile, the different handwriting style may cause misunderstanding, and reduce the accuracy of digital technology. Therefore, our research problem is, how to differentiate the handwriting style by different people? Through the ML and statistical model comparison, we hope to find a good model to predict the owner of a handwritten image. For future work, we may establish our own model theoretically to achieve a high accuracy prediction.

## Project Description

#### The Purpose of the Study:

The overarching purpose of this research is to test whether we can use ML to differentiate different persons based on their handwriting images. We think it is important in such a way that once we can identify different persons, we can train the algorithm to adapt to that particular writing style to increase recognition accuracy. This process is especially important in the area of online handwriting recognition. 

Currently, most handwritten recognition requires the adherence to standard order, stoke, appearance, etc. Such restriction makes it hard for people from different cultures to use the technology since writing style is hugely different across cultures. 

As we found there are researches consolidated with some natural ML tool to predict the actual character based on a handwritten image, however, using a model to distinguish the handwriting style is still a big challenge. Our project focuses on the prediction for different writers instead of the exact character by using ML and statistical model. The goal is to establish a model for high accuracy prediction and test whether if it does matter for the difference by different handwriting style. The broader problem domain is, how to differentiate the handwriting style by different people? If we can pick a model to increase the accuracy of prediction for people, we can facilitate the current related research or studies as a sort of reference.

Forensic handwriting analysts have previously individually performed handwriting analysis for writer identification in a legal context, and legal professionals in the justice system have relied on these skills and abilities to aid in investigative and prosecutorial capacities [Individuality of Handwriting](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.121.3775&rep=rep1&type=pdf). Using these techniques, the confidence with which statements can be made and the number of documents that can be processed can increase. Making these systems language independent can offer a wide variety of advances in ML itself, as well as specific handwriting applications [End-to-End Online Writer Identification With Recurrent Neural Network](https://ieeexplore.ieee.org/abstract/document/7801018). These types of technologies can also be utilized in the financial industry and government for signature validation, allowing early detection to suggest investigating possible fraud or additional verification measures [Learning strategies and classification methods for off-line signature verification](https://ieeexplore.ieee.org/abstract/document/1363904).

One of the target audience for the project is law enforcement. When they perform forensic handwriting analysts, they face the exact same task as our project - identify the owner of certain handwriting image. They may want to use existing suspects’ handwriting images to train our data and let the model to predict whether the evidence contains handwriting belong to the suspect. A potential question for them is the reliability of the prediction.

#### The Hypotheses:

Null hypothesis: There is no significant difference between different persons in terms of handwriting style and we cannot differentiate persons based on handwritten images.

Alternative hypothesis: there is a significant difference between individuals in terms of handwriting style and we can differentiate persons based on handwritten images.

#### The Dataset:

The dataset we are using is [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19). It consists of 5 zipped files with a total of 3,992,357 images. The dataset completed by Bethesda High school students and Census Bureau employees in Suitland, Maryland. Every participant asked to write all alphanumeric characters in a case-sensitive manner. We will use the dataset classified by author (the by_write organization of the dataset per the description in https://s3.amazonaws.com/nist-srd/SD19/sd19_users_guide_edition_2.pdf), meaning we have the labels that tell us the image belong to different authors. However, all personal identifying information was removed in the dataset. 

## Technical Description

Our final resource will present as sorts of HTML pages based on a GitHub repository.The process of this project can be broken down into three general steps: preprocessing data, feature engineering, and classification. 

For preprocessing, since all images are stored in PNG format, we want to use the `inx2numpy_array` library to convert the image to `numpy array`. Each value in the numpy array represents the gray-scale of the particular pixel on the image. The other important step in preprocessing is the categorization of writers. Since different writers’ images are store in different folders and there are no labels in the dataset indicating any personal identifying information, we need to add writer label for each image. Ideally, when completing preprocessing, our dataset’s individual observation is writer’s label with a array of values indicating the pixel on the corresponding image. 

The next stage is feature engineering. 

