'''
python program opens a PSD with image layers in it, exports them to a 
1024 x 1024 image atlas and generates a .tscn that can be opened in godotv4

you may to run this line in your python console to install dependancies

python -m pip install opencv-python numpy Pillow psd-tools rectpack scipy tk

most the code was generated using ChatGPT
-mariokart64n (Feb14,2024)

'''

import os
import cv2
import numpy as np
from tkinter import Tk, filedialog
from psd_tools import PSDImage
from PIL import Image
from rectpack import newPacker
from scipy.spatial import Delaunay, ConvexHull

padding = 20

def preprocess_alpha_channel(image, blur_radius=1, downsample_size=None):
    """Apply preprocessing steps to simplify the alpha channel."""
    alpha_channel = image.split()[-1]
    if downsample_size is not None:
        alpha_channel = alpha_channel.resize(downsample_size, Image.ANTIALIAS)
    alpha_array = np.array(alpha_channel)
    # Ensure blur_radius is odd and greater than 0
    blur_radius = max(1, blur_radius | 1)  # | 1 sets the least significant bit, ensuring oddness
    alpha_array = cv2.GaussianBlur(alpha_array, (blur_radius, blur_radius), 0)
    _, thresh = cv2.threshold(alpha_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if downsample_size is not None:
        thresh = cv2.resize(thresh, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
    return thresh


def extract_alpha_contours(image, blur_radius=2, downsample_size=None):
    """Extract contours from the alpha channel of an image with preprocessing."""
    thresh = preprocess_alpha_channel(image, blur_radius, downsample_size)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def generate_mesh_from_alpha(image, useCV=True):
    '''
        using open cv we can use image filtering to get a better fit on the alpha channel
    '''
    points = []
    
    if useCV:
        # Flatten all contours to a single array
        alpha = extract_alpha_contours(image)
        points = np.vstack(alpha).squeeze()
    else:
        # Find all pixels that are not completely transparent
        alpha = np.array(image)[:, :, 3]
        # Get non-zero (non-transparent) pixel coordinates
        points = np.argwhere(alpha > 0)
        # Swap x and y coordinates to match expected format
        points[:, [0, 1]] = points[:, [1, 0]]
    
    if len(points) < 3:
        return [], []  # Not enough points for Delaunay triangulation
    
    # Calculate the convex hull to get the outline
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    
    # Triangulate the convex hull points for a detailed mesh
    delaunay = Delaunay(hull_points)
    
    # Extract vertices and indices from the triangulation
    vertices = delaunay.points
    indices = delaunay.simplices
    
    return vertices, indices

def find_optimal_bin_size(layers):
    """Determine the smallest bin size that fits all layers."""
    total_area = sum((l.width + padding) * (l.height + padding) for l in layers)
    size = 8
    while size ** 2 < total_area:
        size *= 2
    return size

def select_psd_file():
    """Prompt user to select a PSD file and return its path."""
    Tk().withdraw()  # Hide the root window
    return filedialog.askopenfilename(title="Select a PSD file", filetypes=[("PSD Files", "*.psd")])


def main(createHull=True):
    psd_path = select_psd_file()
    
    if not psd_path:
        print("No file selected.")
        return
    
    psd_name = os.path.basename(psd_path).replace('.psd', '')
    output_atlas_path = os.path.splitext(psd_path)[0] + '_atlas.png'
    output_scene_path = os.path.splitext(psd_path)[0] + '.tscn'
    
    # Load the PSD file
    psd = PSDImage.open(psd_path)
    canvas_width, canvas_height = psd.width, psd.height
    
    # Initialize the packer with rotation enabled
    packer = newPacker(rotation=True)
    
    # Collect layers for bin size calculation
    visible_layers = [layer for layer in psd if not layer.is_group() and layer.is_visible()]
    
    # Calculate optimal bin size based on layers
    bin_size = find_optimal_bin_size(visible_layers)
    packer.add_bin(bin_size, bin_size)
    
    images = []
    rid_to_idx = {}
    layer_details = {}
    
    # Process each layer
    for idx, layer in enumerate(visible_layers):
        image = layer.composite().convert("RGBA")
        current_rid = len(images)
        packer.add_rect(image.width + padding, image.height + padding, rid=current_rid)
        images.append((image, layer.name, layer.bbox))
        rid_to_idx[current_rid] = idx
        # Correctly access the bounding box coordinates with tuple indexing
        layer_details[current_rid] = {'original_position': (layer.bbox[0], layer.bbox[1])}
    
    
    packer.pack()
    
    bin = packer[0] 
    
    # Initialize maximum extents
    max_x, max_y = 0, 0
    
    # Calculate the actual used extents of the bin
    for rect in bin:
        max_x = max(max_x, rect.x + rect.width)
        max_y = max(max_y, rect.y + rect.height)
    
    # Adjust extents to be multiples of 8
    adjusted_width = ((max_x + 7) // 8) * 8
    adjusted_height = ((max_y + 7) // 8) * 8
    
    # Create a new image atlas with the adjusted dimensions
    atlas_image = Image.new('RGBA', (adjusted_width, adjusted_height), (0, 0, 0, 0))

    
    scene_content = '[gd_scene load_steps=2 format=2]\n\n'
    scene_content += '[ext_resource path="res://' + os.path.relpath(output_atlas_path).replace("\\", "/") + '" type="Texture" id=1]\n\n'
    scene_content += '[node name="' + psd_name + '" type="Node2D"]\n\n'
    scene_content += '[node name="Meshes" type="Node2D" parent="."]\n\n'
    
    sorted_rects = sorted(packer[0], key=lambda rect: rid_to_idx.get(rect.rid, -1))
    
    for rect in sorted_rects:
        x, y, w, h, rid = rect.x, rect.y, rect.width, rect.height, rect.rid
        if rid is not None and rid < len(images):
            image, name, bbox = images[rid]
            original_width, original_height = image.width, image.height
            
            if not createHull:
                vertices = [(0, 0), (0, original_height), (original_width, original_height), (original_width, 0)]
                triangles = [[0, 1, 2, 3]]
            else:
                vertices, triangles = generate_mesh_from_alpha(image)
            
            
            rotated = w - padding == original_height and h - padding == original_width
            layer_pos = layer_details[rid]['original_position']
            
            # Calculate inverse position for the GDScript
            gd_position_x = layer_pos[0] - x  # Use the corrected original_position
            gd_position_y = layer_pos[1] - y  # Use the corrected original_position
            
            if rotated:
                image = image.rotate(90, expand=True)
                rotation = 1.5708
                gd_position_x = layer_pos[0] + (y + original_width + padding)  # Use the corrected original_position
                gd_position_y = layer_pos[1] - x  # Use the corrected original_position
            else:
                rotation = None
            
            paste_x, paste_y = x + int(padding / 2), y + int(padding / 2)
            atlas_image.paste(image, (paste_x, paste_y), image)
            
            original_position = layer_details[rid]['original_position']
            gd_position_x, gd_position_y = original_position
            
            # Mesh hull vertices are now adjusted for use in .tscn, not affecting packing
            
            adjusted_vertices = [(vx - bbox[0] + gd_position_x, vy - bbox[1] + gd_position_y) for vx, vy in vertices]
            vertices_str = 'PackedVector2Array(' + ', '.join(f'{v[0]}, {v[1]}' for v in adjusted_vertices) + ')'
            
            if rotated:
                adjusted_tvertices = [(paste_x + vy - (padding / 2) + (padding / 2), paste_y + original_width - vx - (padding / 2) + (padding / 2)) for vx, vy in adjusted_vertices]
            else:
                adjusted_tvertices = [(vx + paste_x, vy + paste_y) for vx, vy in adjusted_vertices]
            
            tvertices_str = 'PackedVector2Array(' + ', '.join(f'{v[0]}, {v[1]}' for v in adjusted_tvertices) + ')'
            
            indices_str_list = [f'PackedInt32Array({", ".join(map(str, triangles[i]))})' for i in range(len(triangles))]
            indices_str = '[' + ', '.join(indices_str_list) + ']'
            
            scene_content += f'[node name="{name}" type="Polygon2D" parent="Meshes"]\n'
            scene_content += f'texture = ExtResource( 1 )\n'
            scene_content += f'position = Vector2( {gd_position_x}, {gd_position_y} )\n'
            scene_content += f'polygon = {vertices_str}\n'
            scene_content += f'uv = {tvertices_str}\n'
            scene_content += f'polygons = {indices_str}\n\n'
            
    
    atlas_image.save(output_atlas_path)
    print(f"Texture atlas saved: {output_atlas_path}")
    
    with open(output_scene_path, 'w') as scene_file:
        scene_file.write(scene_content)
    print(f"Scene file saved: {output_scene_path}")

if __name__ == "__main__":
    main()