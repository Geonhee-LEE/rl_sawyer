#include <ros/ros.h>
#include <visualization_msgs/Marker.h>
#include <std_srvs/Empty.h>
#include <cstdlib> // for std::rand() and std::srand()
#include <ctime>


bool done_flg = false;
bool start_flg = true;

bool done_cb(std_srvs::Empty::Request  &req,
         std_srvs::Empty::Response &res)
{
  done_flg = true;
  ROS_INFO("done");
  return true;
}

int main( int argc, char** argv )
{
  ros::init(argc, argv, "create_mearker");
  ros::NodeHandle n;
  ros::Rate r(1);
  ros::Publisher marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1);
  ros::ServiceServer service = n.advertiseService("done", done_cb);

  // Set our initial shape type to be a cube
  uint32_t shape = visualization_msgs::Marker::CUBE;
  std::srand((unsigned int)time(0));

  visualization_msgs::Marker marker;

  while (ros::ok())
  {
    if(done_flg == false && start_flg == false)
    {
      ;
    }
    else
    {        
      done_flg = false;
      start_flg = false;

      // Set the frame ID and timestamp.  See the TF tutorials for information on these.
      marker.header.frame_id = "/base";
      marker.header.stamp = ros::Time::now();

      // Set the namespace and id for this marker.  This serves to create a unique ID
      // Any marker sent with the same namespace and id will overwrite the old one
      marker.ns = "basic_shapes";
      marker.id = 0;

      // Set the marker type.  Initially this is CUBE, and cycles between that and SPHERE, ARROW, and CYLINDER
      marker.type = shape;

      // Set the marker action.  Options are ADD, DELETE, and new in ROS Indigo: 3 (DELETEALL)
      marker.action = visualization_msgs::Marker::ADD;

      // Set the pose of the marker.  This is a full 6DOF pose relative to the frame/time specified in the header
      marker.pose.position.x = (std::rand()%200)*0.005 + 0.1;
      marker.pose.position.y = (std::rand()%250)*0.004 - 0.125;
      marker.pose.position.z = (std::rand()%200)*0.005 + 0.1;
      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;

      // Set the scale of the marker -- 1x1x1 here means 1m on a side
      marker.scale.x = 0.10;
      marker.scale.y = 0.10;
      marker.scale.z = 0.10;

      // Set the color -- be sure to set alpha to something non-zero!
      marker.color.r = 0.0f;
      marker.color.g = 1.0f;
      marker.color.b = 0.0f;
      marker.color.a = 1.0;

      marker.lifetime = ros::Duration();


      // Cycle between different shapes
      switch (shape)
      {
      case visualization_msgs::Marker::CUBE:
        //shape = visualization_msgs::Marker::SPHERE;
        break;
      case visualization_msgs::Marker::SPHERE:
        shape = visualization_msgs::Marker::ARROW;
        break;
      case visualization_msgs::Marker::ARROW:
        shape = visualization_msgs::Marker::CYLINDER;
        break;
      case visualization_msgs::Marker::CYLINDER:
        shape = visualization_msgs::Marker::CUBE;
        break;
      }
    }

    // Publish the marker
    while (marker_pub.getNumSubscribers() < 1)
    {
      if (!ros::ok())
      {
        return 0;
      }
      ROS_WARN_ONCE("Please create a subscriber to the marker");
      sleep(0.1);
    }
    marker_pub.publish(marker);
    
    ros::spinOnce();
    r.sleep();    
  }
}
