#!/usr/bin/env python3
import numpy as np
np.random.seed(42)
import rospy 
import random
random.seed(42)
from std_msgs.msg import Float64MultiArray
from collections import deque

## Topic Parameters ## 
QUEUE_SIZE = 10
NODE_FREQUENCY = 20.0  # [Hz] # node publishing frequency 
SOROSIM_TAG = "/sorosimpp" # initial part of the topic name to publish the actuation messages
CONTROL_BOX_TOPIC = "control_box_topic"
PRIMITIVE = "motor_babbling" #options are : motor_babbling, lines_random

## Primitive Parameters ##
N_ACT = 3  # Number of actuators
Pmin = 0  # Min actuator pressure 
Pmax = 18  # Max actuator pressure 
MAX_STEP = 0.1 * Pmax  # Max step size for gaussian motor babbling
INITIAL_ANGLE = 0 # starting the angular direction for "lines_complete" primitive
PHASES = [0.0, (2.0/3.0)*np.pi, (4.0/3.0)*np.pi] # phases for the circle primitive sinusoidal input  

### Primitive Controller Class ###
class Primitive_Controller:
    def __init__(self, n_act : int, step_size : int, primitive : str, start_angle: int):

        #Instance attributes
        self.n_act = n_act 
        self.step_size = step_size 
        self.primitive = primitive 
        self.start_angle = start_angle # initial angle for linear primitive
        self.angle_dif_calc = False
        self.angle_pool = deque(np.random.permutation(360).tolist())
        self.trajectory_count = 0
        self.t0 = rospy.get_time()


        # ROS Publisher
        self.pub_obj = rospy.Publisher(SOROSIM_TAG + "/actuators", Float64MultiArray, queue_size=QUEUE_SIZE)
        #self.pub_obj = rospy.Publisher(CONTROL_BOX_TOPIC, Float64MultiArray, queue_size=QUEUE_SIZE)


        # Initialize ROS message
        self.init_actMsg()

        # Main loop timer
        self.timer_obj = rospy.Timer(rospy.Duration(1.0 / NODE_FREQUENCY), self.main_loop)

    #Initial Actuation Message
    def init_actMsg(self):
        self.act_msg = Float64MultiArray()
        start_time = rospy.get_time()
        rate = rospy.Rate(3)  # 3 Hz publishing frequency 

        while rospy.get_time() - start_time < 2.0:  # running for 2 second  
            self.act_msg.data = [0.0] * self.n_act
            self.pub_obj.publish(self.act_msg)
            rate.sleep()

        self.t0 = rospy.get_time()
            
    ## Interpolation Functions ##
    def smoothstep(self, t_now, t_start, t_end): 
        if t_now < t_start : 
            return 0 
        elif t_now > t_end :
            return 1 
        else : 
            x = (t_now - t_start) / (t_end - t_start) 
            return x * x * (3 - 2 * x)

    #Updating the Actuator Message
    def update_actMsg(self):
        # Time since start
        t = rospy.Time.now().to_sec() - self.t0

        if self.primitive == "lines_random" : 
            if t < 3.5 :  
                if not self.angle_dif_calc : 
                    self.trajectory_count += 1
                    rospy.loginfo(f"Trajectory Number: {self.trajectory_count}")
                    if not self.angle_pool : 
                        self.angle_pool = deque(np.random.permutation(360).tolist())
                    self.start_angle = self.angle_pool.popleft()
                    self.act_ang_dif = np.array([self.start_angle, 120 - self.start_angle, 240 - self.start_angle])
                    self.act_contrib = np.array([((1 - np.cos(np.radians(self.act_ang_dif[i])))/2) for i in range(self.n_act)]) 
                    self.unmod_act = self.act_contrib * Pmax
                    self.angle_dif_calc = True
                self.act_msg.data = self.unmod_act * self.smoothstep(t, 0, 2.5) 
            elif t > 3.5 and t < 7 :  
                self.act_msg.data = self.unmod_act * (1 - self.smoothstep(t, 3.5, 6))    
            else :  
                self.angle_dif_calc = False
                self.t0 = rospy.Time.now().to_sec()            

        ### MOTOR BABBLING ###
        elif self.primitive == "motor_babbling" : 
            self.delta_u = np.random.normal(loc= 0, scale= 0.33, size= 3) * self.step_size
            self.act_msg.data = self.act_msg.data + self.delta_u
            self.act_msg.data = np.clip(self.act_msg.data, Pmin, Pmax) 

        else : 
            rospy.loginfo('No Primitive Selected')

            
    #Main Loop called by the timer every defined freq
    def main_loop(self, event):
        # Update and publish message
        self.update_actMsg()
        self.pub_obj.publish(self.act_msg)


### Main ###
def main():
    rospy.init_node("primitive_controller", anonymous=True)
    Primitive_Controller(N_ACT, MAX_STEP, PRIMITIVE, INITIAL_ANGLE) 
    rospy.spin()


# Execute
if __name__ == '__main__':
    main()    