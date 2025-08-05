import replicate
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import torch
from segment_anything import sam_model_registry, SamPredictor
from ultralytics import YOLO
import requests
from io import BytesIO
import json
import os
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# CONFIG - Update these for your production environment
REPLICATE_API_TOKEN = os.getenv('REPLICATE_API_KEY')
IMAGE_PATH = "room.jpg"
SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"
YOLO_MODEL_PATH = "yolov8n.pt"  # Will download automatically if not present
OUTPUT_PATH = "redesigned_room.jpg"
EMPTY_ROOM_PATH = "empty_room.jpg"

# Initialize Replicate client
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

class ComprehensiveRoomTransformer:
    """
    A complete interior design platform that transforms room photos into 
    beautiful redesigned spaces through comprehensive object detection,
    complete room clearing, and intelligent style-based regeneration.
    
    This system follows your platform's four-stage workflow:
    1. Detect ALL objects in the room (comprehensive detection)
    2. Clear the room completely to create an empty space
    3. Apply user-selected design style
    4. Generate complete new interior design
    """
    
    def __init__(self):
        # Comprehensive object detection categories
        # This covers everything that should be removed to create an empty room
        self.comprehensive_objects = {
            # Major Furniture
            56: 'chair', 57: 'couch', 58: 'potted_plant', 59: 'bed', 
            60: 'dining_table', 61: 'toilet', 72: 'tv', 
            
            # Electronics and Appliances
            63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
            67: 'cell_phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
            71: 'sink', 73: 'refrigerator',
            
            # Decorative and Personal Items
            74: 'book', 75: 'clock', 76: 'vase', 77: 'scissors',
            78: 'teddy_bear', 79: 'hair_drier', 80: 'toothbrush',
            
            # Textiles and Soft Furnishings
            # Note: YOLO doesn't detect curtains/rugs directly, 
            # we'll handle these through additional detection methods
        }
        
        # Style templates that define how different design styles should look
        # This is the key to translating user selections into AI-generated designs
        self.style_templates = {
            'modern': {
                'description': 'clean lines, neutral colors, minimal ornamentation, functional furniture',
                'furniture_types': 'sleek sofas, glass tables, metal fixtures, geometric shapes',
                'color_palette': 'whites, grays, blacks with occasional bold accent colors',
                'materials': 'glass, steel, polished wood, leather',
                'lighting': 'recessed lighting, modern fixtures, natural light emphasis',
                'accessories': 'minimal, functional art pieces, geometric patterns'
            },
            'minimalist': {
                'description': 'extremely clean aesthetic, maximum functionality, minimum visual clutter',
                'furniture_types': 'essential furniture only, multi-functional pieces',
                'color_palette': 'monochromatic whites, beiges, soft grays',
                'materials': 'natural wood, white surfaces, simple textiles',
                'lighting': 'soft, even lighting, natural light maximized',
                'accessories': 'almost none, one or two carefully chosen pieces'
            },
            'scandinavian': {
                'description': 'cozy functionality, natural materials, light colors, hygge feeling',
                'furniture_types': 'light wood furniture, comfortable seating, simple forms',
                'color_palette': 'whites, light woods, soft pastels, muted blues',
                'materials': 'light oak, pine, wool textiles, natural fibers',
                'lighting': 'warm lighting, candles, natural light celebration',
                'accessories': 'cozy textiles, plants, simple ceramics'
            },
            'industrial': {
                'description': 'raw materials, exposed elements, urban aesthetic',
                'furniture_types': 'metal and wood combinations, vintage pieces, functional design',
                'color_palette': 'grays, blacks, browns, metallic accents',
                'materials': 'exposed brick, steel, reclaimed wood, concrete',
                'lighting': 'exposed bulbs, metal fixtures, track lighting',
                'accessories': 'vintage industrial pieces, metal art, urban elements'
            },
            'bohemian': {
                'description': 'eclectic mix, rich textures, warm colors, artistic expression',
                'furniture_types': 'vintage pieces, floor cushions, layered textiles',
                'color_palette': 'warm earth tones, jewel tones, rich patterns',
                'materials': 'natural fibers, vintage woods, metals, ceramics',
                'lighting': 'warm ambient lighting, lanterns, string lights',
                'accessories': 'abundant textiles, art, plants, cultural artifacts'
            }
        }
        
        self.sam_predictor = None
        self.yolo_model = None
        
    def load_models(self):
        """
        Load the AI models required for comprehensive room transformation.
        
        We need both detection (to find objects) and segmentation (to remove them precisely).
        Think of YOLO as our "room scanner" and SAM as our "precision eraser."
        """
        print("Loading AI models for comprehensive room transformation...")
        
        # Load YOLO for comprehensive object detection
        # This model will find every piece of furniture and object in the room
        print("- Loading YOLO for comprehensive object detection...")
        self.yolo_model = YOLO(YOLO_MODEL_PATH)
        
        # Load SAM for precise object segmentation
        # This model creates perfect masks around each detected object
        print("- Loading Segment Anything Model (SAM) for precision masking...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT_PATH).to(device)
        self.sam_predictor = SamPredictor(sam)
        
        print("✓ All models loaded successfully!")
    
    def detect_all_room_objects(self, image_np: np.ndarray) -> List[Dict]:
        """
        Perform comprehensive object detection to find EVERYTHING in the room.
        
        This is different from selective furniture detection - we want to find
        every single item that should be removed to create an empty room.
        Think of this as creating a complete inventory of the room's contents.
        """
        print("Performing comprehensive room object detection...")
        
        # Run YOLO detection with high sensitivity to catch everything
        results = self.yolo_model(image_np, conf=0.3, verbose=False)  # Lower confidence to catch more objects
        
        all_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Include objects that should be removed from the room
                    if class_id in self.comprehensive_objects and confidence > 0.3:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        all_detections.append({
                            'class_id': class_id,
                            'class_name': self.comprehensive_objects[class_id],
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'center_point': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                            'area': int((x2 - x1) * (y2 - y1))
                        })
        
        # Sort by area (largest objects first) for better processing
        all_detections.sort(key=lambda x: x['area'], reverse=True)
        
        print(f"✓ Detected {len(all_detections)} objects for removal:")
        for detection in all_detections[:10]:  # Show first 10
            print(f"  - {detection['class_name']} (confidence: {detection['confidence']:.2f}, area: {detection['area']}px²)")
        if len(all_detections) > 10:
            print(f"  ... and {len(all_detections) - 10} more objects")
        
        return all_detections
    
    def detect_textiles_and_soft_furnishings(self, image_np: np.ndarray) -> List[Dict]:
        """
        Use additional methods to detect curtains, rugs, and other textiles
        that YOLO might miss. This is crucial for complete room clearing.
        
        We use computer vision techniques to identify fabric textures and patterns
        that indicate curtains, rugs, or decorative textiles.
        """
        print("Detecting textiles and soft furnishings...")
        
        # Convert to HSV for better color and texture analysis
        hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
        
        # Look for potential curtain areas (typically vertical fabric panels)
        # This is a simplified approach - in production, you might use specialized models
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours that might represent curtains or large textile areas
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        textile_detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Only consider large areas
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = h / w
                
                # Curtains tend to be tall and narrow
                if aspect_ratio > 1.5 and area > 10000:
                    textile_detections.append({
                        'class_id': 999,  # Custom ID for textiles
                        'class_name': 'curtain_or_textile',
                        'confidence': 0.6,  # Moderate confidence since this is heuristic
                        'bbox': [x, y, x + w, y + h],
                        'center_point': [x + w // 2, y + h // 2],
                        'area': area
                    })
        
        if textile_detections:
            print(f"✓ Found {len(textile_detections)} potential textile areas")
        
        return textile_detections
    
    def create_comprehensive_room_mask(self, image_np: np.ndarray, all_detections: List[Dict]) -> np.ndarray:
        """
        Create a single comprehensive mask that covers ALL objects to be removed.
        
        This is the heart of the room clearing process. We're creating a mask that
        covers every single item that needs to be erased to create an empty room.
        The quality of this mask directly determines how clean our empty room will be.
        """
        print("Creating comprehensive room clearing mask...")
        
        self.sam_predictor.set_image(image_np)
        
        # Start with an empty mask
        height, width = image_np.shape[:2]
        comprehensive_mask = np.zeros((height, width), dtype=np.uint8)
        
        successful_masks = 0
        
        # Process each detected object to create precise masks
        for i, detection in enumerate(all_detections):
            try:
                # Use the center point of each object to generate a precise mask
                center_point = np.array([detection['center_point']])
                input_label = np.array([1])  # 1 means "include this object"
                
                # Generate masks for this object
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=center_point,
                    point_labels=input_label,
                    multimask_output=True
                )
                
                # Choose the best mask and add it to our comprehensive mask
                best_mask = masks[np.argmax(scores)]
                comprehensive_mask = np.logical_or(comprehensive_mask, best_mask).astype(np.uint8)
                successful_masks += 1
                
            except Exception as e:
                print(f"  Warning: Could not mask {detection['class_name']} - {str(e)}")
                continue
        
        print(f"✓ Successfully created masks for {successful_masks}/{len(all_detections)} objects")
        
        # Apply morphological operations to clean up the mask
        # This removes small holes and smooths edges for better inpainting
        kernel = np.ones((5, 5), np.uint8)
        comprehensive_mask = cv2.morphologyEx(comprehensive_mask, cv2.MORPH_CLOSE, kernel)
        comprehensive_mask = cv2.dilate(comprehensive_mask, kernel, iterations=2)  # Expand slightly to ensure complete coverage
        
        # Convert to 0-255 range for image processing
        comprehensive_mask = (comprehensive_mask * 255).astype(np.uint8)
        
        return comprehensive_mask
    
    def create_empty_room(self, original_image: np.ndarray, comprehensive_mask: np.ndarray) -> str:
        """
        Transform the room into a completely empty space by removing all objects
        and intelligently reconstructing the hidden architectural elements.
        
        This is the most challenging part of the process because the AI needs to
        "imagine" what the room looks like behind all the furniture and objects.
        """
        print("Creating completely empty room...")
        
        # Convert to PIL for Replicate API
        original_pil = Image.fromarray(original_image)
        mask_pil = Image.fromarray(comprehensive_mask).convert("RGB")
        mask_pil = mask_pil.resize(original_pil.size)
        
        # Save temporary files
        original_pil.save("temp_original.png")
        mask_pil.save("temp_comprehensive_mask.png")
        
        try:
            # Craft a detailed prompt for complete room clearing
            # This prompt is crucial - it guides the AI to understand architectural reconstruction
            empty_room_prompt = """
            Remove all furniture, objects, decorations, and personal items from this room.
            Create a completely empty room showing only the architectural structure:
            - Keep walls, windows, doors, and their original colors/textures
            - Preserve flooring patterns and materials
            - Maintain ceiling details and architectural features
            - Reconstruct wall areas hidden behind furniture
            - Keep original lighting and room proportions
            - Result should be a clean, empty room ready for new furniture
            - No furniture, no decorations, no personal items whatsoever
            """
            
            # Use high-quality settings for architectural reconstruction
            empty_room_url = replicate_client.run(
                "stability-ai/stable-diffusion-inpainting:4f24abdf14a7211f7d0545f6e50d7aa299f8b956e77875f8b10ccdb4b1b8b6e4",
                input={
                    "prompt": empty_room_prompt.strip(),
                    "image": open("temp_original.png", "rb"),
                    "mask": open("temp_comprehensive_mask.png", "rb"),
                    "num_inference_steps": 75,  # Higher quality for architectural reconstruction
                    "guidance_scale": 8.0,      # Strong adherence to prompt
                }
            )
            
            return empty_room_url
            
        finally:
            # Clean up temporary files
            for temp_file in ["temp_original.png", "temp_comprehensive_mask.png"]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    def generate_styled_interior(self, empty_room_url: str, selected_style: str) -> str:
        """
        Generate a complete interior design in the selected style.
        
        This is where we transform the empty room into a beautifully designed space
        that matches the user's style preferences. We use our style templates to
        create detailed prompts that guide the AI to generate appropriate furniture,
        colors, and arrangements.
        """
        print(f"Generating complete {selected_style} interior design...")
        
        # Get the detailed style template
        if selected_style not in self.style_templates:
            selected_style = 'modern'  # Default fallback
            
        style_template = self.style_templates[selected_style]
        
        # Create a comprehensive design prompt based on the style template
        design_prompt = f"""
        Transform this empty room into a beautiful {selected_style} interior design:
        
        Style: {style_template['description']}
        Furniture: {style_template['furniture_types']}
        Colors: {style_template['color_palette']}
        Materials: {style_template['materials']}
        Lighting: {style_template['lighting']}
        Accessories: {style_template['accessories']}
        
        Requirements:
        - Furniture should fit naturally in the space with proper proportions
        - Maintain the room's original lighting and perspective
        - Create functional furniture arrangements with good traffic flow
        - Include appropriate accessories and decorative elements for the style
        - Ensure all elements work together cohesively
        - Make it look professionally designed and photo-realistic
        """
        
        # Generate the complete interior design
        styled_room_url = replicate_client.run(
            "stability-ai/stable-diffusion:db21e45e1c40d4d5cbdd0c14fbe938f4f9f4c2b355c2a2f3f37747c1e295e7d4",
            input={
                "prompt": design_prompt.strip(),
                "image": empty_room_url,
                "num_inference_steps": 50,
                "guidance_scale": 7.5,
                "strength": 0.8  # Allow significant transformation while keeping room structure
            }
        )
        
        return styled_room_url
    
    def download_and_enhance_result(self, image_url: str, original_image: np.ndarray, save_path: str) -> Image.Image:
        """
        Download the final result and apply quality enhancements.
        
        This step ensures the final image maintains professional quality and
        lighting consistency with the original room's characteristics.
        """
        print("Downloading and enhancing final result...")
        
        # Download the generated image
        response = requests.get(image_url)
        final_image = Image.open(BytesIO(response.content))
        
        # Apply lighting consistency (match original room's lighting characteristics)
        original_pil = Image.fromarray(original_image)
        enhanced_image = self.match_lighting_characteristics(original_pil, final_image)
        
        # Save the final result
        enhanced_image.save(save_path, quality=95, optimize=True)
        
        return enhanced_image
    
    def match_lighting_characteristics(self, original: Image.Image, generated: Image.Image) -> Image.Image:
        """
        Ensure the generated design matches the lighting characteristics of the original room.
        
        This prevents the "pasted-in" look by making sure the new furniture appears
        to exist in the same lighting environment as the original room.
        """
        # Analyze lighting characteristics of the original room
        original_np = np.array(original)
        generated_np = np.array(generated)
        
        # Calculate brightness and color temperature metrics
        original_brightness = np.mean(original_np)
        generated_brightness = np.mean(generated_np)
        
        original_warmth = np.mean(original_np[:, :, 0]) / np.mean(original_np[:, :, 2])  # Red/Blue ratio
        generated_warmth = np.mean(generated_np[:, :, 0]) / np.mean(generated_np[:, :, 2])
        
        # Apply adjustments to match original lighting
        brightness_factor = min(original_brightness / generated_brightness, 1.3)  # Cap adjustment
        warmth_factor = min(original_warmth / generated_warmth, 1.2)
        
        # Enhance the generated image to match
        brightness_enhancer = ImageEnhance.Brightness(generated)
        enhanced = brightness_enhancer.enhance(brightness_factor)
        
        color_enhancer = ImageEnhance.Color(enhanced)
        final_enhanced = color_enhancer.enhance(warmth_factor)
        
        return final_enhanced
    
    def transform_room(self, image_path: str, selected_style: str = 'modern') -> Tuple[Image.Image, Image.Image]:
        """
        Complete room transformation pipeline that follows your platform's workflow:
        1. Detect all objects in the room
        2. Clear the room completely
        3. Apply selected design style
        4. Generate beautiful new interior
        
        Returns both the empty room and the final designed room for your platform.
        """
        print("🏠 Starting complete room transformation...")
        print(f"Input: {image_path}")
        print(f"Style: {selected_style}")
        print("=" * 60)
        
        # Load the original image
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Stage 1: Load models if not already loaded
        if self.sam_predictor is None or self.yolo_model is None:
            self.load_models()
        
        # Stage 2: Detect ALL objects in the room (comprehensive detection)
        object_detections = self.detect_all_room_objects(image_rgb)
        textile_detections = self.detect_textiles_and_soft_furnishings(image_rgb)
        all_detections = object_detections + textile_detections
        
        if not all_detections:
            print("⚠️  No objects detected for removal. This might be an empty room already.")
            # In this case, skip to style generation with the original image
            empty_room_pil = Image.fromarray(image_rgb)
        else:
            # Stage 3: Create comprehensive mask and clear the room completely
            comprehensive_mask = self.create_comprehensive_room_mask(image_rgb, all_detections)
            
            # Stage 4: Generate completely empty room
            empty_room_url = self.create_empty_room(image_rgb, comprehensive_mask)
            empty_room_pil = self.download_and_enhance_result(empty_room_url, image_rgb, EMPTY_ROOM_PATH)
            
            print(f"✓ Empty room saved to: {EMPTY_ROOM_PATH}")
        
        # Stage 5: Generate styled interior design from the empty room
        if isinstance(empty_room_pil, Image.Image):
            # Convert PIL back to URL format for the style generation
            empty_room_pil.save("temp_empty_for_style.png")
            empty_room_url = "temp_empty_for_style.png"  # In production, upload to cloud storage
        
        styled_room_url = self.generate_styled_interior(empty_room_url, selected_style)
        
        # Stage 6: Download and enhance the final styled room
        final_room_pil = self.download_and_enhance_result(styled_room_url, image_rgb, OUTPUT_PATH)
        
        # Clean up temporary files
        if os.path.exists("temp_empty_for_style.png"):
            os.remove("temp_empty_for_style.png")
        
        print("=" * 60)
        print(f"✅ Room transformation complete!")
        print(f"Empty room: {EMPTY_ROOM_PATH}")
        print(f"Final design: {OUTPUT_PATH}")
        
        return empty_room_pil, final_room_pil

def main():
    """
    Demonstration of the complete room transformation platform.
    This shows how your users will interact with the system.
    """
    
    # Initialize the room transformer
    transformer = ComprehensiveRoomTransformer()
    
    # Available styles for your platform users
    available_styles = list(transformer.style_templates.keys())
    print("Available design styles:")
    for style in available_styles:
        print(f"  - {style}: {transformer.style_templates[style]['description']}")
    
    # User selects their preferred style (this would come from your UI)
    selected_style = 'scandinavian'  # Example user selection
    
    try:
        # Transform the room according to user's style preference
        empty_room, final_room = transformer.transform_room(
            image_path=IMAGE_PATH,
            selected_style=selected_style
        )
        
        print(f"\n🎉 Success! Room transformed in {selected_style} style.")
        print("Your platform now has both the empty room and final design ready for display.")
        
    except Exception as e:
        print(f"\n❌ Transformation failed: {str(e)}")
        print("Check your API credentials, file paths, and internet connection.")

if __name__ == "__main__":
    main()