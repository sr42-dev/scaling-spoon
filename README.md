# scaling-spoon

A master repository containing all the SME issued material for the Integrated Robotics and Applied Computer Vision course of PESU IO Slot 14 (2022).

![Cover Image](./resources/graphics/courseCover.jpeg)

## day 1

### Topics covered :

- Introduction to Robotics and Computer Vision
- Introduction to Git, creating and cloning a repository
- Introduction to OpenCV
- Introduction to ROS
- Software installations

### Resources :

- ./day1
- OpenCV documentation : https://docs.opencv.org/4.x/
- ROS documentation : http://wiki.ros.org/Documentation
- Git cheat sheet : https://education.github.com/git-cheat-sheet-education.pdf

### Tasks : 

1. For your first task, you'll be required to follow the attached youtube video word for word to dual boot your computers to run Ubuntu 20.04. Be thorough, take backups if you want and do not screw this up because it will cost you more than just points.

Video link : https://www.youtube.com/watch?v=lGR_VNwUfzk&t=807s
Torrent download link for the ubuntu-20.04.5-desktop-amd64.iso disk image : https://releases.ubuntu.com/20.04/ubuntu-20.04.5-desktop-amd64.iso.torrent

For once you have your Ubuntu system up and running, I've written a shell executable that will -

Install ROS
Initialize your catkin workspace
Install and initialize the ROS packages you'll need later in the course
Install all of your python dependencies
Install VSCode on your systems
and update & reboot your computer.

Do not modify this executable. I've tested it more than once and can assure you that it will definitely work on a properly installed Ubuntu 20.04 system.

Connect your system to your home internet and follow these instructions to the dot to execute the script -


- Download and save the file under /home/your username/
- Navigate to /home/your username/ on the terminal
- Execute 'sudo chmod 777 IRACV-IO-setup' on the terminal
- Execute 'ls' on the terminal
- Verify that 'IRACV-IO-setup' appears in green on your terminal. Otherwise, redo step 3.
- Execute './IRACV-IO-setup >> IRACV-IO-setup-output.txt'
- Be on the lookout for user prompts during execution


Mail your 'IRACV-IO-setup-output.txt' (located under /home/<your username>/) as an attachment to samuelthomas1049@gmail.com and sriram.radhakrishna42@gmail.com by 14:30, November 1st, 2022.

This text file will tell us exactly what installation errors you guys have run into and will determine whether or not we will require an online session to resolve them.

This assignment will take anywhere between 20 to 45 minutes depending on how carefully you're following the instructions.

The ability to read thoroughly and debug system errors is an essential skill in robotics. You will not survive without it. Give every error it's due time for resolution. In case you run into them, post them on the group and hep each other out.

Good luck :)


2. For your second task, you'll be required to create a repository on your github account and clone it to your local machine. 

You'll then be required to make a commit (any) and push it to your remote repository. 

There's no deadline or mandatory requirement for this task. You can do it anytime you want. Show us your work in class for extra points.


Code :

- ./day1/IRACV-IO-setup : The shell script for the first task
- ./day1/opencvFeedLoading.py : A python script that loads a video feed from your webcam and displays it on your screen

## day 2

### Topics covered :

- Kernels and their usage in dilation & erosion.
- Grayscaling, gaussian blur & the reasons to use them. [HANDS ON]
- Matrix rotation transform demonstrated using warp perspective.
- Inserting shapes & text (to emphasize the visual attractiveness of computer vision software) [HANDS ON]
- Matrix stretch transform demonstrated by image slicing & resizing
- HSV color space & cylindrical representation demonstrated by color detection & masking.
- Laplacian operator in HSV space demonstrated by Canny edge detection.
- Contour detection
- ROS topics overview, publisher and subscriber 
