#!/usr/bin/env python3
import rospy
import csv 
import os
from std_msgs.msg import Float64MultiArray #message type for the actuators input
from tf2_msgs.msg import TFMessage #message type for the frames' pose


# Generating the list of required frames: base, cs1 to cs38, and tip
REQUIRED_FRAMES = ["base"] + [f"cs{i}" for i in range(1, 39)] + ["tip"]


class Data_Logger: 
    def __init__(self, required_frames) : 
        self.required_frames = required_frames 
        # getting directory where the current script is (complete + the script name)
        abs_path = os.path.abspath(__file__) 
        # extracting just the directory path and disregarding the filename
        script_dir = os.path.dirname(abs_path) 
        # defining the folder name relative to script directory 
        log_folder = os.path.join(script_dir, "logged_data_csv") 
        # create the directory if it doesn't exist
        if not os.path.exists(log_folder) : 
            os.makedirs(log_folder, exist_ok=True) 
            rospy.loginfo(f"created logging directory: {log_folder}") 
        else : 
            rospy.loginfo(f"logging directory already exists: {log_folder}") 
        # defining the full path 
        file_path = os.path.join(log_folder, "ros_data_logged.csv")
        
        self.csv_file = open(file_path, mode="w") #write mode, created if it doesn't exist and overwritten if it exists, self.csv_file is now an object you can write to
        self.csv_writer = csv.writer(self.csv_file) #writer object to handle csv formatting 

        # Constructing the header dynamically for all frames
        header_row = ["time", "actuator_1_input", "actuator_2_input", "actuator_3_input"]
        for frame in self.required_frames:
            header_row.extend([f"{frame}_pos_x", f"{frame}_pos_y", f"{frame}_pos_z"])

        self.csv_writer.writerow(header_row) 
        
        self.latest_pose = None
        self.latest_act_in = None 

        rospy.Subscriber("/sorosimpp/actuators", Float64MultiArray, self.actuator_callback) 
        rospy.Subscriber("/tf", TFMessage, self.tf_callback)
        
    #Callback Functions 
    def actuator_callback(self, msg) : 
        self.latest_act_in = msg.data 
        self.log_data() 
    
    def tf_callback (self, msg) : 
        filtered_tf_data = [] 
        #Selecting only the tf data of required frames
        for tf_data in msg.transforms : 
            if tf_data.child_frame_id in self.required_frames : 
                filtered_tf_data.append(tf_data)  
        self.latest_pose = filtered_tf_data   

    #Logging Function
    def log_data(self): 
        if self.latest_pose is not None and self.latest_act_in is not None: 
            t_now = rospy.Time.now().to_sec()  
            act_in = self.latest_act_in 
            
            # Helper dict to map frame names to their current translation for easy access
            current_transforms = {}
            for tf_data in self.latest_pose :
                current_transforms[tf_data.child_frame_id] = tf_data.transform.translation

            data_to_be_logged = [t_now, act_in[0], act_in[1], act_in[2]]
            
            # Loop through all required frames to append their X, Y, Z coordinates
            for frame in self.required_frames:
                if frame in current_transforms:
                    pos = current_transforms[frame]
                    data_to_be_logged.extend([pos.x, pos.y, pos.z])
                else:
                    # Failsafe if a specific frame is missing in the latest packet
                    data_to_be_logged.extend([0.0, 0.0, 0.0])
            
            #rospy.loginfo("%s",data_to_be_logged)

            self.csv_writer.writerow(data_to_be_logged) 
            self.csv_file.flush() 

    def close(self) : 
        self.csv_file.close()    


def main() : 
    rospy.init_node("data_logger", anonymous=True) 
    logger = Data_Logger(REQUIRED_FRAMES) 
    rospy.on_shutdown(logger.close)
    rospy.spin()


if __name__ == '__main__' :
    main()