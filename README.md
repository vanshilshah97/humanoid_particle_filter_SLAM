# Monte Carlo Localization Based SLAM
The data consists of the lidar scans from a THOR-OP Humanoid Robot. The lidar data is transoformed into the map co-ordinates by applying suitable transformations and the ground plane points are filtered. Suitable number of particles are initialized which indicate the pose of the robot. 
Based on the paricle filter approach the best particle with maximum correlation is chosen and the log odds of the map is updated.


## Maps built for 3 datasets
<img src="https://github.com/Nagarakshith1/MCL_SLAM/blob/master/images/processing_SLAM_map_train_0.jpg?raw=true" width="300" height="300"> <img src="https://github.com/Nagarakshith1/MCL_SLAM/blob/master/images/processing_SLAM_map_train_1.jpg?raw=true" width="300" height="300"> 
>
<img src="https://github.com/Nagarakshith1/MCL_SLAM/blob/master/images/processing_SLAM_map_train_2.jpg?raw=true" width="300" height="300"> <img src="https://github.com/Nagarakshith1/MCL_SLAM/blob/master/images/processing_SLAM_map_train_3.jpg?raw=true" width="300" height="300">
