#include <ros/ros.h>
#include <random>
#include <string>
#include <random>
#include <vector>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Pose2D.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Path.h>
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "math.h"

using namespace std;

cv::Vec3f rotationMatrixToEulerAngles(cv::Mat &R)
{
 
    //assert(isRotationMatrix(R));
     
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
 
    bool singular = sy < 1e-6; // If
 
    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return cv::Vec3f(x, y, z);
     
}

geometry_msgs::Quaternion toQuaternion(double pitch, double roll, double yaw)
{
    geometry_msgs::Quaternion q;
        // Abbreviations for the various angular functions
    double cy = cos(yaw * 0.5);
    double sy = sin(yaw * 0.5);
    double cr = cos(roll * 0.5);
    double sr = sin(roll * 0.5);
    double cp = cos(pitch * 0.5);
    double sp = sin(pitch * 0.5);

    q.w = cy * cr * cp + sy * sr * sp;
    q.x = cy * sr * cp - sy * cr * sp;
    q.y = cy * cr * sp + sy * sr * cp;
    q.z = sy * cr * cp - cy * sr * sp;
    return q;
}

geometry_msgs::Quaternion CalculateRotation( cv::Mat a ) {
    geometry_msgs::Quaternion q;
    float trace = a.at<double>(0,0) + a.at<double>(1,1) + a.at<double>(2,2); // I removed + 1.0f; see discussion with Ethan
    if( trace > 0 ) {// I changed M_EPSILON to 0
        float s = 0.5f / sqrtf(trace+ 1.0f);
        q.w = 0.25f / s;
        q.x = ( a.at<double>(2,1) - a.at<double>(1,2) ) * s;
        q.y = ( a.at<double>(0,2) - a.at<double>(2,0) ) * s;
        q.z = ( a.at<double>(1,0) - a.at<double>(0,1) ) * s;
    } else {
    if ( a.at<double>(0,0) > a.at<double>(1,1) && a.at<double>(0,0) > a.at<double>(2,2) ) {
      float s = 2.0f * sqrtf( 1.0f + a.at<double>(0,0) - a.at<double>(1,1) - a.at<double>(2,2));
      q.w = (a.at<double>(2,1) - a.at<double>(1,2) ) / s;
      q.x = 0.25f * s;
      q.y = (a.at<double>(0,1) + a.at<double>(1,0) ) / s;
      q.z = (a.at<double>(0,2) + a.at<double>(2,0) ) / s;
    } else if (a.at<double>(1,1) > a.at<double>(2,2)) {
      float s = 2.0f * sqrtf( 1.0f + a.at<double>(1,1) - a.at<double>(0,0) - a.at<double>(2,2));
      q.w = (a.at<double>(0,2) - a.at<double>(2,0) ) / s;
      q.x = (a.at<double>(0,1) + a.at<double>(1,0) ) / s;
      q.y = 0.25f * s;
      q.z = (a.at<double>(1,2) + a.at<double>(2,1) ) / s;
    } else {
      float s = 2.0f * sqrtf( 1.0f + a.at<double>(2,2) - a.at<double>(0,0) - a.at<double>(1,1) );
      q.w = (a.at<double>(1,0) - a.at<double>(0,1) ) / s;
      q.x = (a.at<double>(0,2) + a.at<double>(2,0) ) / s;
      q.y = (a.at<double>(1,2) + a.at<double>(2,1) ) / s;
      q.z = 0.25f * s;
    }
  }
  return q;
}

void publish_pose (cv::Mat frame_pose) {

	static int setup_done = 0;
    static ros::NodeHandle *n = NULL;
    static ros::Publisher *pPublisherObject = NULL, *pPublisherObject2 = NULL ;
    static geometry_msgs::PoseStamped stereo_pose;
    static nav_msgs::Path stereo_path;

    if (0 == setup_done) {
        setup_done = 1;

		int argc = 0;
		ros::init(argc, NULL, "my_ros_pose_publisher");
		ros::start();

		n = new ros::NodeHandle;

		pPublisherObject = new ros::Publisher;
		*pPublisherObject = n->advertise<geometry_msgs::PoseStamped> ("/stereo_pose", 10);
		pPublisherObject2 = new ros::Publisher;
		*pPublisherObject2 = n->advertise<nav_msgs::Path> ("/stereo_path", 10);
        stereo_pose.header.frame_id = "/pose";
	stereo_path.header.frame_id = "/pose";
    }

    stereo_pose.pose.position.x = frame_pose.at<double>(0,3);
    stereo_pose.pose.position.y = frame_pose.at<double>(2,3);
    stereo_pose.pose.position.z = frame_pose.at<double>(1,3);
    stereo_pose.pose.orientation = CalculateRotation(frame_pose);
    
    stereo_path.poses.push_back(stereo_pose);

    pPublisherObject->publish(stereo_pose);
    pPublisherObject2->publish(stereo_path);

    return;
}
