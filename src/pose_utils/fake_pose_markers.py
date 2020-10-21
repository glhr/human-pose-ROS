#!/usr/bin/env python3
import rospy
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

rospy.init_node('fake_pose_visualizer')

skel_pub = rospy.Publisher('fake_markers', Marker, queue_size=100)

camera_point = [-0.0334744,-0.20912,1.85799]

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

cube = camera_point
cube_marker = Marker()
cube_marker.header.frame_id = "/world"
cube_marker.type = cube_marker.CUBE
cube_marker.action = cube_marker.ADD
cube_marker.scale.x, cube_marker.scale.y, cube_marker.scale.z = 0.001, 2.5, 2.5
cube_marker.color.a = 0.3
cube_marker.color.r, cube_marker.color.g, cube_marker.color.b = (0.0,0.0,0.0)
cube_marker.pose.position.x = cube[0]
cube_marker.pose.position.y = cube[1]
cube_marker.pose.position.z = cube[2]
cube_marker.id = 2
cube_marker.header.stamp = rospy.get_rostime()

arrow_marker = Marker()
arrow_marker.header.frame_id = "/world"
arrow_marker.type = arrow_marker.ARROW
arrow_marker.action = arrow_marker.ADD
arrow_marker.scale.x, arrow_marker.scale.y, arrow_marker.scale.z = 0.03, 0.05, 0.1
arrow_marker.color.a = 1.0
arrow_marker.color.r, arrow_marker.color.g, arrow_marker.color.b = (0.0,0.0,0.0)
start = Point()
start.x = skel_centroid[0]
start.y = skel_centroid[1]
start.z = skel_centroid[2]
end = Point()
end.x = camera_point[0]
end.y = camera_point[1]
end.z = camera_point[2]
arrow_marker.points.append(end)
arrow_marker.points.append(start)
arrow_marker.id = 3
arrow_marker.header.stamp = rospy.get_rostime()

arrow_marker2 = Marker()
arrow_marker2.header.frame_id = "/world"
arrow_marker2.type = arrow_marker2.ARROW
arrow_marker2.action = arrow_marker2.ADD
arrow_marker2.scale.x, arrow_marker2.scale.y, arrow_marker2.scale.z = 0.03, 0.05, 0.1
arrow_marker2.color.a = 1.0
arrow_marker2.color.r, arrow_marker2.color.g, arrow_marker2.color.b = (1.0,0.0,0.0)
start = Point()
start.x = camera_point[0]
start.y = camera_point[1]+1.25
start.z = camera_point[2]
end = Point()
end.x = camera_point[0]
end.y = camera_point[1]
end.z = camera_point[2]
arrow_marker2.points.append(end)
arrow_marker2.points.append(start)
arrow_marker2.id = 4
arrow_marker2.header.stamp = rospy.get_rostime()



while not rospy.is_shutdown():
    skel_pub.publish(centroid_marker)
    skel_pub.publish(arrow_marker)
    skel_pub.publish(arrow_marker2)
    skel_pub.publish(cube_marker)
    rospy.sleep(1)
