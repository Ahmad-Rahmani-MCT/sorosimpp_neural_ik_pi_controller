#!/usr/bin/env python3
# %%
import numpy as np 
import torch 
import random 
import os 
import pickle
import rospy
from std_msgs.msg import Float64MultiArray 
from tf2_msgs.msg import TFMessage
from sklearn.preprocessing import MinMaxScaler
import math

# setting seeds
def set_all_seeds(seed: int = 42): 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
set_all_seeds(42) 

# configuration and model parameters
lag_state = 3 
lag_input = 0
input_flat_size = 21 
output_size = 3 
num_hidden_layers = 1 
hidden_units = 30 
QUEUE_SIZE = 10
NODE_FREQUENCY = 20.0 
DT = 1.0 / NODE_FREQUENCY # time step for the integral
SOROSIM_TAG = "/sorosimpp" 
N_ACT = 3 
REQUIRED_FRAMES = ["cs19", "tip"] 

# control configuration and limits
ACTUATION_LIMIT_MIN = 0.0
ACTUATION_LIMIT_MAX = 18.0
ACTUATION_RATE_LIMIT = 0.6 
TARGET_TOLERANCE = 0.01 # within 1 cm

# --- PI Fine Tuning Gains ---
# CRITICAL: Negative Gains because robot bends AWAY from actuator
PID_KP = -20  
PID_KI = -20   

# actuator angles
ACT_ANGLES = [math.radians(90), math.radians(330), math.radians(210)]

# model definition 
class MLP_model(torch.nn.Module): 
    def __init__(self, input_flat_size: int, hidden_units: int, output_size: int, num_hidden_layers: int):
        super().__init__()
        layers = [] 
        in_dimension = input_flat_size 
        self.input_layer = torch.nn.Linear(in_features=in_dimension, out_features=hidden_units) 
        for i in range(num_hidden_layers): 
            layers.append(torch.nn.Linear(in_features=hidden_units, out_features=hidden_units)) 
            layers.append(torch.nn.ReLU()) 
        self.backbone = torch.nn.Sequential(*layers) 
        self.output_layer = torch.nn.Linear(in_features=hidden_units, out_features=output_size) 
        self.relu = torch.nn.ReLU()
    def forward(self, x): 
        out = self.input_layer(x) 
        out = self.relu(out)
        out = self.output_layer(out) 
        return out

# loading resources 
script_path = os.path.abspath(__file__) 
script_dir = os.path.dirname(script_path) 
model_directory = os.path.join(script_dir, "ik_model_lines_data")
scaler_directory = model_directory

inverse_model = MLP_model(input_flat_size, hidden_units, output_size, num_hidden_layers)  
inverse_model.load_state_dict(torch.load(os.path.join(model_directory, "IK_MLP_lines.pth"), map_location=torch.device('cpu'))) 
inverse_model.eval() 

with open(os.path.join(scaler_directory, "input_scaler_lines.pkl"), 'rb') as file: 
    input_scaler = pickle.load(file) 
with open(os.path.join(scaler_directory, "state_scaler_lines.pkl"), 'rb') as file:
    state_scaler = pickle.load(file)  
with open(os.path.join(scaler_directory, "ee_xy_scaler.pkl"), 'rb') as file:
    ee_xy_scaler = pickle.load(file) 

# path generator
def lin_path_gen(x_des, y_des, x_init, y_init, num_points):  
     x = np.linspace(start=x_init, stop=x_des, num=num_points) 
     y = np.linspace(start=y_init, stop=y_des, num=num_points) 
     return np.column_stack((x, y)) 

# control cystem class 
class IK_CTRL_SYS:
    def __init__(self, n_act, req_frames, state_scaler, input_scaler, IK_model, des_trajectory):
        self.n_act = n_act 
        self.required_frames = req_frames 
        self.state_scaler = state_scaler 
        self.input_scaler = input_scaler 
        self.ik_model_nn = IK_model
        
        self.latched = False
        self.latched_u_base = None 
        self.integral_error = np.zeros(2) 
        
        self.raw_des_trajectory = des_trajectory
        self.final_target = self.raw_des_trajectory[-1] 
        self.scaled_des_trajectory = self.scale_trajectory(des_trajectory)

        # Limits
        physical_limits = np.array([[ACTUATION_LIMIT_MIN]*self.n_act, [ACTUATION_LIMIT_MAX]*self.n_act])
        scaled_limits = self.input_scaler.transform(physical_limits)
        self.u_min_scaled = np.min(scaled_limits, axis=0)
        self.u_max_scaled = np.max(scaled_limits, axis=0)
        self.u_rate_limit_scaled = ACTUATION_RATE_LIMIT * self.input_scaler.scale_ 
        
        self.latest_tf = None 
        self.pose_buffer = [] 
        self.init_pose_buf_filled = False
        self.init_u = np.zeros(self.n_act) 
        self.current_u = self.init_u
        self.counter = 0
        
        self.pub_obj = rospy.Publisher(SOROSIM_TAG + "/actuators", Float64MultiArray, queue_size=QUEUE_SIZE) 
        self.sub_obj = rospy.Subscriber("/tf", TFMessage, self.tf_callback)
        
        self.init_act_msg() 
        self.timer_obj = rospy.Timer(rospy.Duration(DT), self.main_loop)

    def tf_callback(self, msg): 
        self.latest_tf = msg 
    
    def init_act_msg(self): 
        self.act_msg = Float64MultiArray()
        start_time = rospy.get_time()
        rate = rospy.Rate(10) 
        while rospy.get_time() - start_time < 2.0:   
            self.act_msg.data = [0.0] * self.n_act
            self.pub_obj.publish(self.act_msg)
            rate.sleep() 
        
    def filter_tf(self, tf_msg): 
        filtered_tf = []
        for tf_data in tf_msg.transforms: 
            if tf_data.child_frame_id in self.required_frames: 
                filtered_tf.append(tf_data)
        return filtered_tf   
    
    def latest_poses(self, filtered_tf): 
        cs19_pose = None
        ee_pose = None
        for tf_data in filtered_tf: 
            if tf_data.child_frame_id == self.required_frames[0]: 
                cs19_pose = tf_data.transform.translation 
            elif tf_data.child_frame_id == self.required_frames[1]: 
                ee_pose = tf_data.transform.translation 
        if cs19_pose and ee_pose:
            return [cs19_pose.x, cs19_pose.y, ee_pose.x, ee_pose.y]
        return None
    
    def scale_pose(self, latest_pose): 
        pose_np = np.array(latest_pose)
        scaled_pose = self.state_scaler.transform(pose_np.reshape(1, -1)) 
        return scaled_pose
    
    def scale_trajectory(self, raw_traj):
        tip_scale = self.state_scaler.scale_[2:4] 
        tip_min = self.state_scaler.min_[2:4]
        scaled_traj = raw_traj * tip_scale + tip_min
        return scaled_traj

    def init_pose(self, current_pose):  
        self.pose_buffer.append(current_pose)

    def check_tolerance(self, current_pose_raw):
        tip_current = np.array(current_pose_raw[2:4])
        distance = np.linalg.norm(tip_current - self.final_target)
        return distance <= TARGET_TOLERANCE

    def calculate_fine_tuning(self, current_pose_raw):
        # error calculation as Error = Target - Current Pose 
        tip_current = np.array(current_pose_raw[2:4])
        error_vector = self.final_target - tip_current 
        
        # updating the integral (clamped)
        self.integral_error += error_vector * DT
        self.integral_error = np.clip(self.integral_error, -0.5, 0.5) 
        
        # geometric projection of error along actuator angles (positions)
        delta_u_phys = np.zeros(self.n_act)
        for i in range(self.n_act):
            angle = ACT_ANGLES[i]
            
            # projecting P and I into the actuator axis 
            proj_p = error_vector[0] * math.cos(angle) + error_vector[1] * math.sin(angle)
            proj_i = self.integral_error[0] * math.cos(angle) + self.integral_error[1] * math.sin(angle)
            
            # calculating the force note: gains are NEGATIVE) [we consider PID output as force ]
            force = (PID_KP * proj_p) + (PID_KI * proj_i)
            delta_u_phys[i] = force

        # scaling
        delta_u_scaled = delta_u_phys * self.input_scaler.scale_ 
        return delta_u_scaled

    def prepare_features(self, current_pose, current_input, des_ee_trajectory): 
        if self.counter <= des_ee_trajectory.shape[0] - 1: 
            future_pose_des = des_ee_trajectory[self.counter, :] 
            self.counter += 1 
        else: 
            future_pose_des = des_ee_trajectory[-1, :] 
        prev_pose_buffer_flat = np.array(self.pose_buffer).flatten() 
        current_pose_flat = current_pose.flatten()
        feature_prepped = np.concatenate((future_pose_des, current_pose_flat, prev_pose_buffer_flat, current_input), axis=0) 
        return feature_prepped
            
    def main_loop(self, event): 
        if self.latest_tf is None: return
        filtered_tf = self.filter_tf(self.latest_tf) 
        if len(filtered_tf) < 2: return 
        raw_pose = self.latest_poses(filtered_tf)
        if raw_pose is None: return
        latest_poses_scaled = self.scale_pose(raw_pose) 

        if not self.init_pose_buf_filled: 
            if len(self.pose_buffer) < lag_state:
                self.init_pose(latest_poses_scaled) 
            else:
                self.init_pose_buf_filled = True   
        else: 
            if not self.latched:
                # PHASE 1: Neural Network
                if self.check_tolerance(raw_pose):
                    rospy.loginfo("--- TARGET REACHED (<1cm). SWITCHING TO PI CONTROL ---")
                    self.latched = True
                    self.latched_u_base = self.current_u.copy() 
                    return 

                in_feature = self.prepare_features(current_pose=latest_poses_scaled, 
                                                current_input=self.current_u, 
                                                des_ee_trajectory=self.scaled_des_trajectory) 
                tensor_in = torch.tensor(in_feature, dtype=torch.float32).unsqueeze(0)
                network_out = self.ik_model_nn(tensor_in) 
                raw_output = network_out.detach().squeeze(0).numpy()

                delta_u = raw_output - self.current_u
                delta_u_clamped = np.clip(delta_u, -self.u_rate_limit_scaled, self.u_rate_limit_scaled)
                self.current_u = np.clip(self.current_u + delta_u_clamped, self.u_min_scaled, self.u_max_scaled)

            else:
                # PHASE 2: PI Control
                delta_pi_scaled = self.calculate_fine_tuning(raw_pose)
                target_u = self.latched_u_base + delta_pi_scaled
                
                # applying rate limit to PI changes too 
                delta_u = target_u - self.current_u
                delta_u_clamped = np.clip(delta_u, -self.u_rate_limit_scaled, self.u_rate_limit_scaled)
                self.current_u = np.clip(self.current_u + delta_u_clamped, self.u_min_scaled, self.u_max_scaled)
            
            self.pose_buffer = self.pose_buffer[1:] + [latest_poses_scaled] 
            real_actuation = self.input_scaler.inverse_transform(self.current_u.reshape(1, -1))
            self.act_msg.data = real_actuation.flatten().tolist()
            self.pub_obj.publish(self.act_msg)  

# execution 
def main(): 
    rospy.init_node("IK_cntrl_sys", anonymous=True) 
    raw_des_trajectory = lin_path_gen(0.05, 0.05, 0, 0, 40) 
    IK_CTRL_SYS(n_act=N_ACT, req_frames=REQUIRED_FRAMES, state_scaler=state_scaler, input_scaler=input_scaler, IK_model=inverse_model,
                des_trajectory=raw_des_trajectory)
    rospy.spin() 

if __name__ == '__main__': 
    main()