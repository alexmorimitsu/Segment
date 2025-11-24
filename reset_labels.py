import os
from PIL import Image
import numpy as np

labels_dir = "/home/io/IO2025/Code/MexilhaoQt/Anotacoes"

def process_image(image_path):
    """
    Read an image and convert all non-white pixels to black.
    Save the result as {image_name}_original.png
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Convert to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to numpy array for easier processing
        img_array = np.array(img)
        
        # Create a mask for non-white pixels (pixels that are not [255, 255, 255])
        # A pixel is considered white if all RGB values are 255
        non_white_mask = ~np.all(img_array == [255, 255, 255], axis=2)
        
        # Set non-white pixels to black [0, 0, 0]
        img_array[non_white_mask] = [0, 0, 0]
        
        # Convert back to PIL Image
        processed_img = Image.fromarray(img_array)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(labels_dir, f"{base_name}_original.png")
        
        # Save the processed image
        processed_img.save(output_path)
        print(f"Processed: {os.path.basename(image_path)} -> {os.path.basename(output_path)}")
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main():
    """
    Process all images in the labels directory
    """
    # Get all image files in the directory
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    for filename in os.listdir(labels_dir):
        file_path = os.path.join(labels_dir, filename)
        
        # Check if it's a file and has an image extension
        if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
            # Skip files that already have _original in the name
            if '_original' not in filename:
                process_image(file_path)
            else:
                print(f"Skipping already processed file: {filename}")

if __name__ == "__main__":
    print("Starting image processing...")
    main()
    print("Processing complete!")

