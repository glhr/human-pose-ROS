#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker, MarkerArray

rospy.init_node('fake_pose_visualizer')

skel_pub = rospy.Publisher('fake_markers', Marker, queue_size=100)

skel_centroid = [0.5,1,1.2]
centroid_marker = Marker()
centroid_marker.header.frame_id = "/world"
centroid_marker.type = centroid_marker.SPHERE
centroid_marker.action = centroid_marker.ADD
centroid_marker.scale.x, centroid_marker.scale.y, centroid_marker.scale.z = 0.1, 0.1, 0.1
centroid_marker.color.a = 1.0
centroid_marker.color.r, centroid_marker.color.g, centroid_marker.color.b = (1.0,0.0,0.0)
centroid_marker.pose.orientation.w = 1.0
centroid_marker.pose.position.x = skel_centroid[0]
centroid_marker.pose.position.y = skel_centroid[1]
centroid_marker.pose.position.z = skel_centroid[2]
centroid_marker.id = 1
centroid_marker.header.stamp = rospy.get_rostime()

cube = [0,0,1.85799]
cube_marker = Marker()
cube_marker.header.frame_id = "/world"
cube_marker.type = cube_marker.CUBE
cube_marker.action = cube_marker.ADD
cube_marker.scale.x, cube_marker.scale.y, cube_marker.scale.z = 2.5, 2.5, 0.001
cube_marker.color.a = 0.3
cube_marker.color.r, cube_marker.color.g, cube_marker.color.b = (0.0,0.0,0.0)
cube_marker.pose.orientation.w = 1.0
cube_marker.pose.position.x = cube[0]
cube_marker.pose.position.y = cube[1]
cube_marker.pose.position.z = cube[2]
cube_marker.id = 2
cube_marker.header.stamp = rospy.get_rostime()


while not rospy.is_shutdown():
    skel_pub.publish(centroid_marker)
    skel_pub.publish(cube_marker)
    rospy.sleep(1)
