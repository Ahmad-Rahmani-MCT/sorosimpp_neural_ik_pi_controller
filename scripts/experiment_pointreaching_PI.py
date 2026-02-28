#!/usr/bin/env python3
# %%
import numpy as np 
import torch 
import random 
import os 
import pickle
import rospy
import csv
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

# experiment durations in steps (assuming 20Hz)
HOME_STEPS = 100    # 5 seconds
SETTLE_STEPS = 60   # 3 seconds
PI_TIMEOUT = 400    # 20 seconds max for PI to prevent infinite loops
PI_DWELL_STEPS = 20 # 1 second of continuous stability required (<2mm)

# PI configuration
TARGET_TOLERANCE = 0.002 # 2 mm
PID_KP = -20  
PID_KI = -20   
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
    def __init__(self, n_act, req_frames, state_scaler, input_scaler, IK_model, targets_list):
        self.n_act = n_act 
        self.required_frames = req_frames 
        self.state_scaler = state_scaler 
        self.input_scaler = input_scaler 
        self.ik_model_nn = IK_model
        
        # Experiment state variables
        self.targets_list = targets_list
        self.current_target_idx = 0
        self.experiment_results = [] # To store [target_x, target_y, final_x, final_y, error, nn_steps, total_steps]
        
        # State Machine counters
        self.state = 0 
        self.state_step_counter = 0
        self.active_steps = 0
        self.nn_active_steps = 0
        self.pi_dwell_counter = 0 # New counter to track stability
        
        # PI variables
        self.latched_u_base = None 
        self.integral_error = np.zeros(2)
        
        # Trajectory variables (populated dynamically)
        self.raw_des_trajectory = None
        self.final_target = None 
        self.scaled_des_trajectory = None

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
        self.act_msg.data = [0.0] * self.n_act
        self.pub_obj.publish(self.act_msg)
        
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
            proj_p = error_vector[0] * math.cos(angle) + error_vector[1] * math.sin(angle)
            proj_i = self.integral_error[0] * math.cos(angle) + self.integral_error[1] * math.sin(angle)
            force = (PID_KP * proj_p) + (PID_KI * proj_i)
            delta_u_phys[i] = force

        # scaling
        delta_u_scaled = delta_u_phys * self.input_scaler.scale_ 
        return delta_u_scaled

    def generate_new_trajectory(self):
        target = self.targets_list[self.current_target_idx]
        x_des, y_des = target[0], target[1]
        x_init, y_init = 0.0, 0.0
        average_step_size = 0.0015
        
        total_distance = math.sqrt((x_des - x_init)**2 + (y_des - y_init)**2)
        calculated_points = max(2, int(round(total_distance / average_step_size)))
        
        self.raw_des_trajectory = lin_path_gen(x_des, y_des, x_init, y_init, calculated_points)
        self.final_target = self.raw_des_trajectory[-1] 
        self.scaled_des_trajectory = self.scale_trajectory(self.raw_des_trajectory)
        
        self.counter = 0
        self.integral_error = np.zeros(2) # Reset PI integral for new target
        self.pi_dwell_counter = 0 # Reset stabilization counter
        
        rospy.loginfo(f"--- Starting Target {self.current_target_idx + 1}/{len(self.targets_list)}: X={x_des:.4f}, Y={y_des:.4f} ---")

    def main_loop(self, event): 
        if self.latest_tf is None: return
        filtered_tf = self.filter_tf(self.latest_tf) 
        if len(filtered_tf) < 2: return 
        raw_pose = self.latest_poses(filtered_tf)
        if raw_pose is None: return
        latest_poses_scaled = self.scale_pose(raw_pose) 

        # STATE 0: HOMING
        if self.state == 0:
            self.current_u = self.init_u # Command zeros
            self.act_msg.data = [0.0] * self.n_act
            self.pub_obj.publish(self.act_msg)
            
            self.state_step_counter += 1
            if self.state_step_counter >= HOME_STEPS:
                self.generate_new_trajectory()
                self.pose_buffer.clear()
                self.init_pose_buf_filled = False
                self.active_steps = 0
                self.nn_active_steps = 0
                self.state_step_counter = 0
                self.state = 1 # Switch to NN moving
            return

        # Buffer filling logic (only happens at start of STATE 1)
        if not self.init_pose_buf_filled: 
            if len(self.pose_buffer) < lag_state:
                self.init_pose(latest_poses_scaled) 
            else:
                self.init_pose_buf_filled = True   
            return

        # STATE 1: MOVING (Neural Network)
        if self.state == 1:
            if self.counter >= self.scaled_des_trajectory.shape[0]:
                # Trajectory finished - Transition to PI Control
                self.nn_active_steps = self.active_steps # Save NN effort
                self.latched_u_base = self.current_u.copy() 
                rospy.loginfo(f"Path finished in {self.nn_active_steps} steps. Activating PI Controller...")
                self.state = 2 # Switch to PI
                return 

            self.active_steps += 1
            
            in_feature = self.prepare_features(current_pose=latest_poses_scaled, 
                                            current_input=self.current_u, 
                                            des_ee_trajectory=self.scaled_des_trajectory) 
            tensor_in = torch.tensor(in_feature, dtype=torch.float32).unsqueeze(0)
            network_out = self.ik_model_nn(tensor_in) 
            raw_output = network_out.detach().squeeze(0).numpy()

            delta_u = raw_output - self.current_u
            delta_u_clamped = np.clip(delta_u, -self.u_rate_limit_scaled, self.u_rate_limit_scaled)
            self.current_u = np.clip(self.current_u + delta_u_clamped, self.u_min_scaled, self.u_max_scaled)
            
            self.pose_buffer = self.pose_buffer[1:] + [latest_poses_scaled] 
            real_actuation = self.input_scaler.inverse_transform(self.current_u.reshape(1, -1))
            self.act_msg.data = real_actuation.flatten().tolist()
            self.pub_obj.publish(self.act_msg)

        # STATE 2: MOVING (PI Fine-Tuning)
        elif self.state == 2:
            self.active_steps += 1
            
            # Check current error
            tip_current = np.array(raw_pose[2:4])
            error = np.linalg.norm(tip_current - self.final_target)
            
            # Increment stability counter if within tolerance, else reset it
            if error <= TARGET_TOLERANCE:
                self.pi_dwell_counter += 1
            else:
                self.pi_dwell_counter = 0
            
            # Stop condition: Held within 2mm for full dwell time OR Timeout reached
            if self.pi_dwell_counter >= PI_DWELL_STEPS or (self.active_steps - self.nn_active_steps) >= PI_TIMEOUT:
                if self.pi_dwell_counter >= PI_DWELL_STEPS:
                    rospy.loginfo(f"Target stabilized (<2mm for 1s) in {self.active_steps} total steps. Settling...")
                else:
                    rospy.logwarn("PI Timeout reached. Settling...")
                    
                self.state_step_counter = 0
                self.state = 3 # Switch to settling
                return
            
            # Apply PI Control (Actively fights oscillations while dwelling)
            delta_pi_scaled = self.calculate_fine_tuning(raw_pose)
            target_u = self.latched_u_base + delta_pi_scaled
            
            delta_u = target_u - self.current_u
            delta_u_clamped = np.clip(delta_u, -self.u_rate_limit_scaled, self.u_rate_limit_scaled)
            self.current_u = np.clip(self.current_u + delta_u_clamped, self.u_min_scaled, self.u_max_scaled)
            
            self.pose_buffer = self.pose_buffer[1:] + [latest_poses_scaled] 
            real_actuation = self.input_scaler.inverse_transform(self.current_u.reshape(1, -1))
            self.act_msg.data = real_actuation.flatten().tolist()
            self.pub_obj.publish(self.act_msg)

        # STATE 3: SETTLING
        elif self.state == 3:
            self.pub_obj.publish(self.act_msg) # Keep holding last PI command
            self.state_step_counter += 1
            if self.state_step_counter >= SETTLE_STEPS:
                self.state = 4 # Switch to recording
            return

        # STATE 4: RECORDING
        elif self.state == 4:
            tip_current = np.array(raw_pose[2:4])
            error = np.linalg.norm(tip_current - self.final_target)
            
            # Save data row
            self.experiment_results.append([
                self.final_target[0], self.final_target[1], 
                tip_current[0], tip_current[1], 
                error, self.nn_active_steps, self.active_steps
            ])
            rospy.loginfo(f"Target {self.current_target_idx + 1} Final Error: {error:.4f}m. Saved.")
            
            self.current_target_idx += 1
            if self.current_target_idx < len(self.targets_list):
                rospy.loginfo(f"Resetting to [0,0,0] for {HOME_STEPS} steps...")
                self.state_step_counter = 0
                self.state = 0 # Loop back to home
            else:
                self.state = 5 # Finished
        
        # STATE 5: FINISHED
        elif self.state == 5:
            self.save_results()
            rospy.loginfo("--- ALL 100 TARGETS COMPLETED ---")
            rospy.signal_shutdown("Experiment finished cleanly.")

    def save_results(self):
        file_path = os.path.join(script_dir, "experiment_results_NN_PI.csv")
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Target_X', 'Target_Y', 'Final_X', 'Final_Y', 'Error_m', 'NN_Steps', 'Total_Steps'])
            writer.writerows(self.experiment_results)
        rospy.loginfo(f"Results saved to {file_path}")

def load_targets(filename="workspace_targets.csv"):
    file_path = os.path.join(script_dir, filename)
    targets = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader) # Skip header
        for row in reader:
            targets.append([float(row[0]), float(row[1])])
    return targets

# execution 
def main(): 
    rospy.init_node("IK_cntrl_sys_experiment_PI", anonymous=True) 
    
    # Load the 100 targets
    try:
        targets_list = load_targets("workspace_targets.csv")
        rospy.loginfo(f"Successfully loaded {len(targets_list)} targets.")
    except Exception as e:
        rospy.logerr(f"Could not load workspace_targets.csv: {e}")
        return

    IK_CTRL_SYS(n_act=N_ACT, req_frames=REQUIRED_FRAMES, state_scaler=state_scaler, input_scaler=input_scaler, IK_model=inverse_model,
                targets_list=targets_list)
    rospy.spin() 

if __name__ == '__main__': 
    main()