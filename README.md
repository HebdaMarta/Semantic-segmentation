# Semantic-segmentation

A simple model for semantic segmentation - that is, the classification of images based on pixels. The repository contains the code, training charts and the result in a sample photo (selection of 3 classes: person, road, car).

A dataset consists of 5000 photos with annotated annotations. There are 3475 photos in the training and validation set, and 1525 photos in the test set. The collection has 30 classes. The set includes normal, unprocessed rgb photos and for each photo:

- a photo with already applied pixel segmentation for all classes,
- a photo with segmentation only of people and cars,
- a grayscale photo with the segmentation of all classes,
- json file with segmentation annotations. This file is in the form of a Python dictionary and consists of a set of points which are the corners of polygons containing the given objects.
