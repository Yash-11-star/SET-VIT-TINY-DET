"""
Create sample data for testing the model
"""

import os
import numpy as np
from PIL import Image, ImageDraw

def create_sample_image(width=512, height=512, boxes=None, file_path=None):
    # Create a random background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Add some random noise
    noise = np.random.randint(200, 256, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(noise)
    draw = ImageDraw.Draw(img)
    
    # Draw boxes if provided
    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = box
            # Draw slightly darker rectangle
            draw.rectangle([x1, y1, x2, y2], 
                         fill=(180, 180, 180),
                         outline=(100, 100, 100),
                         width=2)
    
    # Save the image
    if file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        img.save(file_path)

if __name__ == '__main__':
    # Create training images
    create_sample_image(
        boxes=[(10, 10, 30, 30), (100, 100, 120, 120)],
        file_path='data/train/sample001.jpg'
    )
    create_sample_image(
        boxes=[(50, 50, 60, 60)],
        file_path='data/train/sample002.jpg'
    )
    create_sample_image(
        file_path='data/train/sample003.jpg'
    )
    
    # Create validation images
    create_sample_image(
        boxes=[(10, 10, 30, 30)],
        file_path='data/val/sample001.jpg'
    )
    create_sample_image(
        file_path='data/val/sample002.jpg'
    )
    
    print("✓ Sample images created")