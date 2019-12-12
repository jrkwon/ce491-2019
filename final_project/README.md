# CE491/691 Final Project

## Gazebo version
We must use Gazebo9. First remove Gazebo7 if you use Ubuntu 16.04.

```
$ sudo apt remove ros-kinetic-gazebo*
$ sudo apt remove libgazebo*
$ sudo apt remove gazebo*
```

Install Gazebo9.
```
$ curl -sSL http://get.gazebosim.org | sh
$ sudo apt install ros-kinetic-gazebo9-ros 
```
## Joystick (Logitech G920)

### Install ros-kinetic-joystick-drivers

```
$ sudo apt install ros-kinetic-joystick-drivers
```
### Setup Logitech G920
```
$ cd catkin_ws
$ bash src/car_demo/car_demo/nodes/setup_driving_simulator.sh
```

## Prereqs

Create conda environments. The first one is for ROS.

```
$ conda env create -f ros.yaml
```

Then, create an environment for Keras with Tensorflow backend.

```
$ conda env create -f neural_net.yaml
```

## How to build

Start `ros` environmnet.
```
$ conda activate ros
```

Go to `catkin_ws` and build it.
```
(ros) $ cd catkin_ws
(ros) $ catkin_make
```

## How to start 

If you have not activated `ros` environment, do so with the following command.
```
$ conda activate ros
```

First enable your workspace environment.
```
(ros) $ . devel/setup.bash
```
Then, you will be able to start *rviz* and *Gazebo*.
```
(ros) $ roslaunch car_demo track.launch
```

## How to collect data

If you have not activated `ros` environment, do so with the following command.
```
$ conda activate ros
```

Open a new terminal and run the following commands.
```
(ros) $ . devel/setup.bash
(ros) $ cd catkin_ws
(ros) $ rosrun data_collection data_collection.py your_data_name
```
Your data will be saved at `data/your_data_name/year_month_date_time/*.jpg`. All image file names with corresponding steering angle and throttle value will be saved in the same folder.

## Remove bad training images

From the data collection step, you may have many of images that are not necessarily helpful for training. You can delete those images from the data folder. Then, you need to change .csv file accordingly. 

Here is a tool that can rebuild .csv file based on image files in your collected data folder.

Activate `neural_net` environment if you have not done yet.

```
$ conda activate neural_net
```

Go to `neural_net` folder. 

```
(neural_net) $ python rebuild_csv.py ../data/your_data_name/year_month_date_time/
```

This will recreated .csv file using the same name. The original csv file is saved as the same name with the `.bak` extension.

## How to train

Activate `neural_net` environment.

```
$ conda activate neural_net
```

Go to `neural_net` folder.

```
(neural_net) $ python train.py ../data/your_data_name/year_month_date_time/
```

After the training is done, you will have .h5 and .json file in the `../data/your_data_name/` folder.

## How to run the trained ANN controller

Activate `neural_net` environment if you haven't yet.

```
$ conda activate neural_net
```

Assuming the `track.launch` is already started, you can run the trained artificial neural network with following commands.

```
(neural_net) $ cd catkin_ws
(neural_net) $ rosrun run_neural run_neural.py ../data/your_data_name/year_month_date_time_n7
```
Then it will load the trained weight `your_data_name/year_month_date_time_n7.h5` and run it to generate the steering angle.
