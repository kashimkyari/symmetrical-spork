# AI-Powered Interior Design Platform

Transform any room photo into a beautifully redesigned space in minutes using advanced artificial intelligence. This system combines cutting-edge computer vision, object detection, and generative AI to create professional-quality interior designs automatically.

## 🏠 What This System Does

Our AI interior design platform follows a sophisticated four-stage transformation process that mimics how professional interior designers approach room redesigns:

**Stage 1: Comprehensive Room Analysis**
The system uses advanced object detection to identify every piece of furniture, decoration, and personal item in your room photo. Think of this as having an AI assistant that can instantly catalog everything in a space, from major furniture pieces like sofas and tables to smaller items like books, plants, and electronics.

**Stage 2: Complete Room Clearing**
Rather than simply replacing individual items, our system completely clears the room to create an empty architectural shell. This involves sophisticated AI inpainting that reconstructs walls, floors, and architectural features that were hidden behind furniture. The result is a clean, empty room that preserves the original space's proportions and character while removing all existing furnishings.

**Stage 3: Style-Guided Design Generation**
Users select from professionally curated design styles like Modern, Scandinavian, Industrial, or Bohemian. Each style includes detailed parameters for furniture types, color palettes, materials, and accessories. The AI uses this comprehensive style guide to generate cohesive interior designs that follow established design principles.

**Stage 4: Complete Room Regeneration**
The empty room becomes a canvas for generating an entirely new interior design that matches the selected style. The AI understands spatial relationships, proper furniture scaling, and functional room layouts to create realistic, livable spaces that look professionally designed.

## 🚀 Quick Start Guide

### Prerequisites

Before you begin, ensure you have the following requirements ready:

**System Requirements:**
- Python 3.9 or 3.10 (Python 3.11+ may have compatibility issues with some AI libraries)
- At least 8GB of RAM (16GB recommended for optimal performance)
- GPU with CUDA support recommended but not required (the system works on CPU, though slower)
- Stable internet connection for cloud AI model access

**Required Accounts:**
- Replicate account with API access (sign up at https://replicate.com)
- At least 5GB of free disk space for AI models and temporary files

### Installation Process

The installation involves setting up a Python environment and downloading several large AI models. Here's how to do it step by step:

**Step 1: Create and Activate Virtual Environment**
```bash
# Create a new virtual environment to avoid conflicts with other projects
python -m venv interior_design_env

# Activate the environment (Windows)
interior_design_env\Scripts\activate

# Activate the environment (macOS/Linux)
source interior_design_env/bin/activate
```

**Step 2: Install Dependencies**
```bash
# Install all required packages from requirements.txt
pip install -r requirements.txt

# For GPU acceleration (optional but recommended), install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Step 3: Download Required AI Models**
The system requires two large AI models that will be downloaded automatically on first run:

- **Segment Anything Model (SAM)**: Download `sam_vit_h_4b8939.pth` from https://github.com/facebookresearch/segment-anything#model-checkpoints
- **YOLOv8**: Will download automatically when first used (approximately 6MB)

Place the SAM model file in the same directory as your Python script.

### Configuration Setup

**Step 1: Get Your Replicate API Token**
1. Visit https://replicate.com and create an account
2. Go to your account settings and generate an API token
3. Copy the token (it starts with 'r8_')

**Step 2: Update Configuration**
Open the main Python file and update these configuration variables:
```python
REPLICATE_API_TOKEN = "your_actual_token_here"  # Replace with your Replicate token
IMAGE_PATH = "your_room_photo.jpg"              # Path to your room photo
SAM_CHECKPOINT_PATH = "sam_vit_h_4b8939.pth"   # Path to downloaded SAM model
```

## 📸 Usage Instructions

### Basic Room Transformation

The simplest way to transform a room is to run the main script with your configuration:

```python
from room_transformer import ComprehensiveRoomTransformer

# Initialize the transformer
transformer = ComprehensiveRoomTransformer()

# Transform a room with your preferred style
empty_room, final_room = transformer.transform_room(
    image_path="my_living_room.jpg",
    selected_style="scandinavian"
)

print("Transformation complete! Check the output files.")
```

### Available Design Styles

The system includes professionally curated design styles, each with specific characteristics:

**Modern Style**
- Clean geometric lines with minimal ornamentation
- Neutral color palette with occasional bold accents
- Materials: glass, steel, polished wood, leather
- Perfect for contemporary urban spaces

**Minimalist Style** 
- Extremely clean aesthetic with maximum functionality
- Monochromatic whites, beiges, and soft grays
- Only essential furniture pieces with multi-functional design
- Ideal for small spaces or clutter-free living

**Scandinavian Style**
- Cozy functionality with natural materials and light colors
- Light wood furniture with comfortable, simple forms
- Emphasizes natural light and warm, inviting atmospheres
- Great for creating hygge-inspired living spaces

**Industrial Style**
- Raw materials and exposed architectural elements
- Metal and wood combinations with urban aesthetics  
- Grays, blacks, browns with metallic accent pieces
- Perfect for loft-style or urban apartments

**Bohemian Style**
- Eclectic mix with rich textures and artistic expression
- Warm earth tones and jewel colors with abundant patterns
- Vintage pieces combined with cultural artifacts and textiles
- Ideal for creative, expressive living environments

### Understanding the Output Files

The transformation process generates several important files:

**empty_room.jpg**: This shows your room completely cleared of all furniture and objects while preserving the architectural structure. This intermediate result helps you understand how the AI interpreted your room's basic layout and proportions.

**redesigned_room.jpg**: The final transformed room featuring your selected design style. This image should look professionally designed and photo-realistic, with furniture and accessories that fit naturally in your space.

## 🔧 Technical Architecture

### How Object Detection Works

The system uses YOLOv8 (You Only Look Once version 8), one of the most advanced real-time object detection models available. YOLO examines your room photo and identifies objects by analyzing visual patterns it learned from millions of training images.

When YOLO processes your room, it creates bounding boxes around every object it recognizes and assigns confidence scores. The system is configured to detect a comprehensive range of items including major furniture (sofas, tables, beds), electronics (TVs, laptops), decorative items (plants, artwork), and personal belongings (books, remote controls).

The detection process operates with adjusted confidence thresholds to ensure comprehensive object identification. Lower thresholds mean the system errs on the side of detecting more objects rather than missing items that should be removed.

### How Precise Segmentation Works

While YOLO tells us where objects are located, the Segment Anything Model (SAM) creates precise pixel-level masks around each detected object. Think of SAM as a digital cutting tool that can trace around the exact edges of furniture and objects with surgical precision.

SAM uses the center points of YOLO's detected objects as guidance inputs. For each detected sofa, table, or lamp, SAM generates multiple possible masks and selects the highest-quality option. This two-step process (detection then segmentation) ensures both comprehensive coverage and precise boundaries.

The system combines all individual object masks into a single comprehensive mask that covers everything to be removed. Additional morphological operations smooth mask edges and fill small gaps to prevent inpainting artifacts.

### How Room Clearing Works

The room clearing process represents the most technically sophisticated aspect of the system. When the AI removes all objects from your room, it must intelligently reconstruct architectural elements that were hidden behind furniture.

For example, if a bookshelf was against a wall, the system needs to "imagine" what that wall section looks like based on visible wall patterns elsewhere in the room. This requires the AI to understand architectural consistency, lighting patterns, and material textures.

The inpainting model receives detailed prompts that guide it toward architectural reconstruction rather than random pattern generation. These prompts specifically instruct the AI to preserve flooring patterns, maintain wall textures, and keep ceiling details while removing only moveable objects.

### How Style Generation Works

The style generation system transforms abstract design concepts into specific technical parameters for AI generation. Each design style includes detailed templates that specify furniture types, color palettes, material preferences, and spatial arrangement principles.

When you select "Scandinavian" style, the system doesn't just add random Scandinavian furniture. Instead, it applies a comprehensive design framework that includes light wood tones, cozy textiles, functional furniture arrangements, and accessories that work together to create an authentic Scandinavian atmosphere.

The AI generation process receives prompts that include both the style template details and spatial constraints from your specific room. This ensures that generated furniture not only matches the chosen aesthetic but also fits properly within your room's proportions and lighting conditions.

## 🎯 Performance Optimization

### Processing Time Expectations

Room transformation involves several computationally intensive steps, and understanding timing helps set proper expectations:

**Model Loading (First Run Only)**: 30-60 seconds to load YOLO and SAM into memory
**Object Detection**: 5-15 seconds depending on image size and complexity  
**Mask Generation**: 10-30 seconds depending on number of detected objects
**Room Clearing**: 60-120 seconds for cloud-based inpainting processing
**Style Generation**: 45-90 seconds for complete interior design generation
**Total Processing Time**: Typically 3-6 minutes per room transformation

### Hardware Recommendations

**For Development and Testing:**
- CPU-only processing is adequate but slower (add 50-100% to processing times)
- 8GB RAM minimum, though the system may swap to disk during peak usage
- Standard broadband internet connection sufficient for API calls

**For Production Deployment:**  
- GPU with 6GB+ VRAM dramatically improves SAM processing speed
- 16GB+ RAM prevents memory bottlenecks during concurrent processing
- High-speed internet connection reduces cloud API latency

### Memory Management

The system automatically manages GPU and CPU memory, but understanding resource usage helps with optimization:

- SAM requires approximately 2.5GB of GPU/CPU memory when loaded
- YOLOv8 uses about 500MB during inference
- Large room images (4K+) may require additional memory for processing
- The system releases model memory between transformations to prevent accumulation

## 🛠 Troubleshooting Guide

### Common Installation Issues

**"Could not find a version that satisfies torch"**
This typically indicates Python version incompatibility. Ensure you're using Python 3.9 or 3.10, as newer versions may not have compatible PyTorch builds available.

**"CUDA out of memory" errors**
Your GPU doesn't have enough memory for the AI models. Either use CPU-only processing by setting the device manually, or reduce your input image resolution before processing.

**"SAM checkpoint file not found"**  
Download the SAM model file from the official repository and ensure it's in the correct location with the exact filename specified in your configuration.

### API and Processing Issues

**"Rate limit exceeded" from Replicate**
Replicate has usage limits on free accounts. Either wait for the limit to reset, upgrade to a paid plan, or implement retry logic with exponential backoff.

**"No objects detected" warnings**
This can occur with very clean or minimalist rooms, or with unusual camera angles. Try adjusting YOLO confidence thresholds or manually verify that your room photo contains recognizable furniture.

**Poor quality empty room generation**
Complex rooms with many overlapping objects can be challenging for inpainting. Consider pre-processing images to improve lighting and contrast, or manually mask areas that require special attention.

### Quality Improvement Tips

**Input Image Optimization:**
- Use well-lit photos with even lighting across the room
- Ensure the camera is roughly at human eye level for natural perspective
- Avoid extreme wide-angle lenses that distort room proportions
- Higher resolution images (1080p+) generally produce better results

**Style Selection Guidance:**
- Choose styles that complement your room's architectural features
- Modern and minimalist styles typically work best for challenging room layouts
- Bohemian and industrial styles require more complex object arrangements

## 📋 Development and Deployment

### Extending the System

The modular architecture makes it easy to add new capabilities:

**Adding New Design Styles:**
Create new entries in the `style_templates` dictionary with detailed descriptions for furniture types, color palettes, materials, and spatial arrangements.

**Custom Object Detection:**
Train custom YOLO models for detecting specific furniture types or room elements not covered by standard models.

**Advanced Post-Processing:**
Implement additional image enhancement algorithms for lighting correction, color grading, or style consistency improvements.

### Production Deployment Considerations

**Scaling for Multiple Users:**
- Implement task queuing systems (like Celery with Redis) to handle concurrent room transformations
- Consider local GPU infrastructure to reduce cloud API dependencies and improve processing speed
- Add database storage for user uploads, transformation history, and style preferences

**Error Handling and Monitoring:**
- Implement comprehensive logging for debugging failed transformations
- Add health checks for model availability and API connectivity
- Create fallback mechanisms for when cloud services are unavailable

**Security and Privacy:**
- Validate and sanitize all uploaded images to prevent security vulnerabilities
- Implement user authentication and authorization for accessing transformation features
- Consider temporary file cleanup policies to protect user privacy

## 🤝 Contributing and Support

This AI interior design system represents a foundation that can be extended and improved in many directions. Whether you're adding new design styles, improving the object detection accuracy, or optimizing the generation quality, the modular architecture supports experimentation and enhancement.

For technical questions about implementation details, model optimization, or deployment strategies, the code includes comprehensive documentation and examples that demonstrate best practices for each component of the transformation pipeline.

The system serves as both a complete interior design solution and a learning platform for understanding how modern AI vision systems work together to solve complex real-world problems. Each component from object detection through style generation can be studied and modified independently while maintaining compatibility with the overall system architecture.