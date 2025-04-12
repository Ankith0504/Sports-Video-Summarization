import tensorflow as tf
from keras.models import load_model

from collections import defaultdict
from keras.applications.resnet50 import preprocess_input
from glob import glob
import os
import json
print("here")
# Load the model
#model = load_model('facefeatures_new_model_resnet1.h5')
# Read the excel sheet into a Pandas DataFrame
#excel_sheet = pd.read_excel('D:/capstone/phase-2/france_ligue-1-c/1/segmented_frames_1_720p/frame_info.xlsx')
# Create a defaultdict to store the folder names
# folders_dict = defaultdict(lambda: 'D:/capstone/phase-2/')
# for folder in glob('D:/capstone/phase-2/france_ligue-1-c/*'):
#     print(folder)
#     folders_dict[folder.split('/')[-1]] = folder #1
#     print(folders_dict)

# Create a dictionary to store the mapping of frame names to feature maps
frame_name_to_feature_map = {"labels":[]}
cur={}
folder_name = 'output_folder'
folder_name_2='output_folder_1'
# List all image files in the folder
frame_files = [f for f in os.listdir(folder_name) if f.endswith('.jpg') or f.endswith('.png')]
frame_files_2 = [f for f in os.listdir(folder_name_2) if f.endswith('.jpg') or f.endswith('.png')]
def label_finder(label_number):
    classes = { 0:'Ball out of play',1:'Clearance',2:'Corner',3:'Direct free-kick',4:'Foul',5:'Goal',6:'Indirect free-kick',7:'Kick-off',8:'Offside',9:'Penalty', 
                10:'Red card',11:'Shots off target',12:'Shots on target',13:'Substitution',14:'Throw-in',15:'Yellow card',16:"I don't know"}
    return classes[label_number]
for frame_name in frame_files:#for loop for 1st half 
    # Get the folder name for the frame name
    cur={"half":1}
    if folder_name is not None:
        # Get the frame image path
        frame_image_path = os.path.join(folder_name, f"{frame_name}")
        #print(frame_image_path)
        frame_feature_map=0
        # Check if the image file exists before loading
        if os.path.exists(frame_image_path):
            # Load the frame image
            frame_image = tf.io.read_file(frame_image_path)
            frame_image = tf.image.decode_jpeg(frame_image)

            # Preprocess the image
            frame_image = tf.image.resize(frame_image, size=(224, 224))
            frame_image = tf.image.convert_image_dtype(frame_image, dtype=tf.float32)
            frame_image = preprocess_input(frame_image)

            # Extract the feature map
            #frame_feature_map = model.predict(tf.expand_dims(frame_image, axis=0))  # Add axis 0 for batch dimension

            # Map the feature map to the frame name
            frame_name_to_feature_map[frame_name] = frame_feature_map
            timestamp=int((frame_name.split("_")[1]).split(".")[0])
            cur["gameTime"]=timestamp
            cur["label"]=label_finder(frame_feature_map)
            print("Processed Frame",cur["gameTime"])
            if cur["label"]!="I don't know": #if the label is Unknown no need to add to dictionary
                frame_name_to_feature_map["labels"].append(cur)
        else:
            # Handle missing image file here, e.g., print a message or add a placeholder feature map
            #print(f"Image file not found for frame {frame_name}.")
            frame_name_to_feature_map[frame_name] = None
    else:
        # Handle missing folder here, e.g., print a message or add a placeholder feature map
        print(f"Folder not found for frame {frame_name}.")
        frame_name_to_feature_map[frame_name] = None


for frame_name in frame_files_2:#for loop for second half
    # Get the folder name for the frame name
    cur={"half":2}
    if folder_name is not None:
        # Get the frame image path
        frame_image_path = os.path.join(folder_name, f"{frame_name}")
        #print(frame_image_path)

        # Check if the image file exists before loading
        if os.path.exists(frame_image_path):
            # Load the frame image
            frame_image = tf.io.read_file(frame_image_path)
            frame_image = tf.image.decode_jpeg(frame_image)

            # Preprocess the image
            frame_image = tf.image.resize(frame_image, size=(224, 224))
            frame_image = tf.image.convert_image_dtype(frame_image, dtype=tf.float32)
            frame_image = preprocess_input(frame_image)

            # Extract the feature map
            #frame_feature_map = model.predict(tf.expand_dims(frame_image, axis=0))  # Add axis 0 for batch dimension

            # Map the feature map to the frame name
            frame_name_to_feature_map[frame_name] = frame_feature_map
            timestamp=int((frame_name.split("_")[1]).split(".")[0])
            cur["gameTime"]=timestamp
            cur["label"]=frame_feature_map
            print("Processed Frame",cur["gameTime"])
            if cur["label"]!="I don't know": #if the label is Unknown no need to add to dictionary
                frame_name_to_feature_map["labels"].append(cur)
        else:
            # Handle missing image file here, e.g., print a message or add a placeholder feature map
            #print(f"Image file not found for frame {frame_name}.")
            frame_name_to_feature_map[frame_name] = None
    else:
        # Handle missing folder here, e.g., print a message or add a placeholder feature map
        print(f"Folder not found for frame {frame_name}.")
        frame_name_to_feature_map[frame_name] = None


'''
# Map the feature maps to the frame names in the DataFrame
excel_sheet['FeatureMap'] = excel_sheet['FrameName'].map(frame_name_to_feature_map)

# Save the updated excel sheet
excel_sheet.to_excel('D:/capstone/phase-2/france_ligue-1-c/1/segmented_frames_1_720p/frame_info.xlsx', index=False)
'''

print("Outputed features.json file")