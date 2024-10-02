import rclpy
from rclpy.node import Node
import math
import numpy as np
from scipy import linalg

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from nav_msgs.msg import Odometry

#Laser offset
l1_off_x = 1.7
l1_off_z = 1.01
l2_off_x = -0.5
l2_off_z = 1.01
Katt = 10
radius = 3
KObstacles = 300
kgoal = 1.5
maxSteps = 200
maxVel = 0.5


class MyNode(Node):
    def __init__(self):
        super().__init__("my_node")

        #### Variables ####
        self.lin_vel = None
        self.ang_vel = None

        self.laser1 = None
        self.laser2 = None

        self.goal_pose = PoseStamped().pose
        self.robot_pose = Odometry().pose.pose.position

        self.laser1_subs = self.create_subscription(LaserScan, topic='/laser1', callback=self.laser1_callback, qos_profile=10)
        self.laser2_subs = self.create_subscription(LaserScan, topic='/laser2', callback=self.laser2_callback, qos_profile=10)
        self.rob_pos_subs = self.create_subscription(Odometry, topic='/base_pose_ground_truth', callback=self.rob_pos_callback, qos_profile=10)
        self.goal_subs = self.create_subscription(PoseStamped, topic='/goal_pose', callback=self.goal_pos_callback, qos_profile=10)

        self.mov_pub = self.create_publisher(Twist, topic='/cmd_vel', qos_profile=10)
        self.publish_timer = self.create_timer(0.1, self.publish_command)

    def laser1_callback(self, laser1_info: LaserScan):
        self.laser1 = laser1_info
    
    def laser2_callback(self, laser2_info: LaserScan):
        self.laser2 = laser2_info
    
    def rob_pos_callback(self, rob_pose: Odometry):
        self.lin_vel = rob_pose.twist.twist.linear
        self.ang_vel = rob_pose.twist.twist.angular
        self.robot_pose = rob_pose.pose.pose.position

    def goal_pos_callback(self, goal_pose: PoseStamped):
        self.goal_pose = goal_pose.pose

    def publish_command(self):
        msg = Twist()
        msg.linear.x = 0.1

        dgoal = self.goalDist()

        Map = self.objMap()
        xRobot = np.vstack([self.robot_pose.x, self.robot_pose.y])
        

        FRep = self.repulsive_force(xRobot, Map, radius, KObstacles)
        FAtt = self.attractive_force(kgoal, np.vstack(dgoal))

        FTotal = FAtt + FRep

        Theta = np.arctan2(FTotal[1,0], FTotal[0,0])

        if math.isnan(Theta):
            Theta = 0

        msg.angular.z = 0.07*Theta

        goalDist = sum(dgoal)

        if goalDist != 0.0:
            msg.linear.x = msg.linear.x * goalDist * 10
            if msg.linear.x > maxVel:
                msg.linear.x = maxVel
        else:
            msg.linear.x = 0.0

        self.mov_pub.publish(msg)

    def goalDist(self):
        return [abs(self.robot_pose.x - self.goal_pose.position.x), abs(self.robot_pose.y - self.goal_pose.position.y)]
    
    def objMap(self):
        rows = 2
        col1 = len(self.laser1.ranges)
        col2 = len(self.laser2.ranges)

        Map = np.zeros((rows, col1+col2))
        
        for j in range(col1+col2):
            if j < col1:
                Map[:, j] = self.objCoord(self.laser1.ranges[j],1)
            else:
                Map[:, j] = self.objCoord(self.laser2.ranges[j-col1],2)

        return Map
    
    def objCoord(self, dist, laser):
        if(laser == 1):
            idx = self.laser1.ranges.index(dist)
            angle = self.laser1.angle_min + self.laser1.angle_increment*idx
            xdist = dist * math.sin(math.degrees(angle)) - l1_off_x
            ydist = dist * math.cos(math.degrees(angle))
            return np.array([xdist, ydist])
        else:
            idx = self.laser2.ranges.index(dist)
            angle = self.laser2.angle_min + self.laser2.angle_increment*idx
            xdist = dist * math.sin(math.degrees(angle)) - l2_off_x
            ydist = dist * math.cos(math.degrees(angle))
            return np.array([xdist, ydist])
    
    def repulsive_force(self, xRobot, Map, RadiusOfInfluence, KObstacles):
  
        p_to_object = xRobot - Map
        d = np.sqrt(np.sum(p_to_object**2, axis=0))
        iInfluential = np.where(d <= RadiusOfInfluence)[0]
        
        if iInfluential.shape[0] > 0:
            p_to_object = p_to_object[:, iInfluential]
            d = d[iInfluential]
            FRep = KObstacles*np.vstack(np.sum((1/d - 1/RadiusOfInfluence)*(1/d**2)*(p_to_object/d), axis=1))

        else:
            FRep = 0
        
        return FRep
    
    def attractive_force(self, KGoal, GoalError):
       
        FAtt = -KGoal*GoalError
        FAtt /= linalg.norm(GoalError)
        
        return FAtt


def main(args=None):
    rclpy.init(args=args)
    node = MyNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
