# Snakes: Active Contour Models

My implementation of an active contour model for a task in Computer Vision course at the University of Bonn.

## Task
Read the images ball.png and coffee.png and segment the object in both images using snakes. Initialize the snake by a circle around the object and optimize it using dynamic programming. The elastic term should be used as pairwise cost, penalizing deviation from the average distance between pairs of nodes. Visualize for both images how the snake converges to the boundary of the object.

## Results
<div>
    <div>
        <img src="output/1_initial.png" alt="initial ball" width="250" height="auto">
        <img src="output/1_result.png" alt="initial ball" width="300" height="auto">
    </div>
    <div>
        <img src="output/2_initial.png" alt="initial ball" width="300" height="auto">
        <img src="output/2_result.png" alt="initial ball" width="300" height="auto">
    </div>
</div>