import os
import shutil
from difflib import SequenceMatcher
import csv

# Define the paths for the training images and masks
train_images_path = "data_road/training/image_2/"
train_masks_path = "data_road/training/gt_image_2/"

# Define the new folder for renamed files
new_folder_path = "data_road/renamed_data/"
new_images_path = os.path.join(new_folder_path, "images/")
new_masks_path = os.path.join(new_folder_path, "masks/")

# Create the new folders if they do not exist
os.makedirs(new_images_path, exist_ok=True)
os.makedirs(new_masks_path, exist_ok=True)

# Function to find the best match file in the mask folder
def find_best_match(image_file, mask_files):
    best_match = None
    highest_similarity = 0
    for mask_file in mask_files:
        similarity = SequenceMatcher(None, image_file, mask_file).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = mask_file
    return best_match

# List of all image files and mask files
train_images_files = os.listdir(train_images_path)
train_masks_files = os.listdir(train_masks_path)

# Ensure that the lists are sorted for consistency
train_images_files.sort()
train_masks_files.sort()

# Open a CSV file to save the mapping information
with open(os.path.join(new_folder_path, "file_mappings.csv"), "w", newline='') as csvfile:
    fieldnames = ['Image Name', 'Mask Name', 'New Image Name', 'New Mask Name']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    # Iterate over image files and find the best match in mask files
    for i, image_file in enumerate(train_images_files):
        best_match_mask = find_best_match(image_file, train_masks_files)
        new_file_name = f"{i:06}.png"  # Format: 000001.png, 000002.png, etc.
        
        # Copy and rename image file
        src_image_path = os.path.join(train_images_path, image_file)
        dest_image_path = os.path.join(new_images_path, new_file_name)
        shutil.copyfile(src_image_path, dest_image_path)
        
        if best_match_mask:
            # Copy and rename mask file
            src_mask_path = os.path.join(train_masks_path, best_match_mask)
            dest_mask_path = os.path.join(new_masks_path, new_file_name)
            shutil.copyfile(src_mask_path, dest_mask_path)
            
            # Write the mapping information to the CSV file
            writer.writerow({
                'Image Name': image_file,
                'Mask Name': best_match_mask,
                'New Image Name': new_file_name,
                'New Mask Name': new_file_name
            })
            
            # Remove the matched mask file from the list to avoid re-matching
            train_masks_files.remove(best_match_mask)
        else:
            # Write the mapping information indicating no mask found
            writer.writerow({
                'Image Name': image_file,
                'Mask Name': "No corresponding mask found",
                'New Image Name': new_file_name,
                'New Mask Name': "N/A"
            })

print(f"Renaming and copying completed. Files are saved in {new_folder_path}")
