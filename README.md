# Video Correction
This codebase contains a correction method for corrupted videos. The implementation mainly uses RMSE and Mean Absolute Distance for obtaining the pairwise distance between frames, cleaning the outliers, and ordering the frames.

### Building the Docker Image
To create the docker image, run the following command inside the repository
```
sudo docker build -t meero-image .
```

### Checking the Docker Image
Check if the docker image has been created properly. The image name meero-image should be displayed.
```
sudo docker images

```

### Run the Docker Image
Enable display
```
xhost +
```
Use the following command to run the generated docker image as a container
```
sudo docker run -i -t -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v shuffled_19_new.mp4:/meero/shuffled_19_new.mp4 meero-image:latest /bin/bash
```

### Run Video Correction
Run the video correction script inside the docker container. The input video and output directory can be configured using the --input-video and --output-dir flags respectively.
```
python3 solution1.py --input-video shuffled_19.mp4 --output-dir output
```

### Advantages
Image registration solves the issue of getting a small distance when using MSE on frames with similar appearance but distant in timeline. This is also implemented with phase correlation to reduce the computational cost using Fourier space. RMSE is then calculated directly at the point of maximum correlation between frames so the actual image registration or the transforming of images into one coordinate system is not performed. However, registration RMSE error can underestimate the difference between very dissimilar frames since it tries to align images for the maximum correlation. To make the implementation more robust to outliers, MAD is done because it provides large diffrences berween dissimilar frames. It is combined with the single linkage property of the clustering step as well to better identify outleirs.

### Limitations
A limitation of this method is that it doesn't identify whether the video has any outliers before it performs the cleaning of the frames. The dendogram might contain taller branches for frames that are still relevant to the context of some shots in the vides; thus, these frames should not be removed. One way to solve this issue is to show the user some sample frames of the potential outliers to determine whether they should be removed or not which. THis is similar to how the direction of the video is determined which improves the performance of the method.
