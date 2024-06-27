# Computer Vision I

This repository contains some of my solutions for the Computer Vision course at the University of Bonn.

Tech stack: Python, OpenCV.

## ActiveContour
**Task description**

Read the images ball.png and coffee.png and segment the object in both images using snakes. Initialize the snake by a circle around the object and optimize it using dynamic programming. The elastic term should be used as pairwise cost, penalizing deviation from the average distance between pairs of nodes. Visualize for both images how the snake converges to the boundary of the object.

**Solution notes**

I utilized OOP principles and organized the code into classes. This approach greatly improves code readability and re-usability.

## K-means
**Task description**

Implement the function myKmeans and then use it to segment the image flower.png based on:
- (a) Intensity,
- (b) Color,
- (c) Intensity and (properly scaled) image position,
- (d) Other property that you choose as the feature space.

Visualize the results for all the cases with k = 2, 4, 6. Analyze your results.
