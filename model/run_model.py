import torch
import os

#!/usr/bin/env python

from train_model import GraspingPoseEval  # Make sure train_model.py is in your Python path

# Define model parameters
input_size = 10
hidden_size = 10
output_size = 1

# Get user's home directory
home_dir = os.path.expanduser("~")

current_dir = os.path.dirname(__file__) 

# Initialize the model
model = GraspingPoseEval(input_size, hidden_size, output_size)

# Build the model path
model_path = os.path.join(home_dir, "dev", "storage", "model.pt")

# Load the model state dictionary
model.load_state_dict(torch.load(model_path))

# Define the scene file path and output path
scene_folder = os.path.join(current_dir, "scenes")
output_folder = os.path.join(home_dir, "dev", "storage", "outputs1")
os.makedirs(output_folder, exist_ok=True)

# Define the number of scenes
num_scenes = 10

# Loop through each scene
for i in range(1, num_scenes + 1):
    # Load the test data for the current scene
    new_poses = []
    test_data_path = os.path.join(scene_folder, f"/home/yanwenli/ros/noetic/system/src/test_franka_gmp_common_bringup/src/test_data/combined_output{i}.txt")
    with open(test_data_path, 'r') as f:
        for line in f:
            pose = [float(j) for j in line.split()]
            new_poses.append(pose)

    # Model prediction
    model.eval()
    with torch.no_grad():
        new_poses_tensor = torch.FloatTensor(new_poses)
        predictions = model(new_poses_tensor)

    # Rank the predictions
    ml_ranking = torch.argsort(predictions.squeeze(), descending=True)

    # Save the ranking to a file
    output_path = os.path.join(output_folder, f"ranking_scene_{i}.txt")
    ml_array = ml_ranking.numpy()
    ml_list = list(ml_array)
    ml_list_formatted = [str(item) for item in ml_list]
    
    with open(output_path, 'w') as f:
        f.write("\n".join(ml_list_formatted))

    print(f"Ranking for scene {i} saved to {output_path}")

    # Evaluate the ranking
    score_file = os.path.join(scene_folder, f"/home/yanwenli/ros/noetic/system/src/test_franka_gmp_common_bringup/src/test_data/result{i}.txt")
    eval_output_file = os.path.join(output_folder, f"eval_rank_scene_{i}.txt")
    
    def eval_ranking(ranking, score_file, eval_output_file):
        no_of_ones = 0
        scores = []
        right = 0
        wrong = 0

        # Open the file and read the scores
        with open(score_file, 'r') as f:
            for line in f:
                score = int(line.strip())
                scores.append(score)
                if score == 1:
                    no_of_ones += 1

        # Evaluate the ranking
        for idx, item in enumerate(ranking):
            if idx < no_of_ones:
                if scores[item] == 1:
                    right += 1
                else:
                    wrong += 1
            else:
                if scores[item] == 0:
                    right += 1
                else:
                    wrong += 1

        

        # Output the results and save them
        output = f'Scene {i}: Right: {right} Wrong: {wrong}'
        with open(eval_output_file, 'w') as f:
            f.write(output + '\n')

        print(f"Evaluation results for scene {i} saved to {eval_output_file}")

    # Read the ranking and evaluate it
    with open(output_path, 'r') as f:
        ranking = [int(line.strip()) for line in f]
    
    eval_ranking(ranking, score_file, eval_output_file)
