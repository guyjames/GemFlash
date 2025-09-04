# Feature: Model Selection with Aspect Ratio Control

## Overview

This update adds support for switching between different AI models in GemFlash, with proper aspect ratio control for models that support it.

## New Features

### 1. Model Selection Dropdown
- **Location**: Header area of the application
- **Options**: 
  - **Gemini 2.5 Flash Image**: Fast, conversational image generation (generates square images)
  - **Imagen 4.0**: High-quality, specialized image generation with precise aspect ratio control

### 2. Conditional Aspect Ratio Controls
- **Imagen Model**: Shows full aspect ratio dropdown with options:
  - `1:1` - Square - Social Media Profile
  - `16:9` - Widescreen - Desktop/Video
  - `9:16` - Portrait - Mobile/Stories
  - `4:3` - Standard - Photo Landscape
  - `3:4` - Portrait - Social Posts

- **Gemini Model**: Hides aspect ratio dropdown and shows informational note about square image generation

### 3. Visual Indicators
- **Model Badges**: Each generated image shows which model was used
- **Icons**: 
  - 🤖 Bot icon for Gemini 2.5 Flash
  - 🎨 Brush icon for Imagen 4.0

### 4. Smart Model Usage
- **Image Generation**: Uses selected model (Gemini or Imagen)
- **Image Editing**: Always uses Gemini for multimodal editing capabilities
- **Image Composition**: Always uses Gemini for multimodal composition capabilities

## Technical Implementation

### Backend Changes (`main.py`)

#### Updated Request Models
```python
class ImageGenerationRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "1:1"
    model: str = "gemini"  # "gemini" or "imagen"
```

#### Dual API Support
- **Gemini API**: Uses `client.models.generate_content()` with prompt-based aspect ratio instructions
- **Imagen API**: Uses `client.models.generate_images()` with proper `aspect_ratio` parameter

#### Response Processing
- Separate functions for handling Gemini and Imagen responses
- Consistent base64 image data extraction

### Frontend Changes (`App.jsx`)

#### Model Selection State
```javascript
const [selectedModel, setSelectedModel] = useState("gemini")
```

#### Conditional UI Rendering
- Aspect ratio dropdown only shows for Imagen
- Informational notes for each model
- Model-specific descriptions in tab headers

#### Enhanced Image Metadata
- Tracks which model generated each image
- Displays model badges on images
- Preserves model information through edit/compose workflows

## User Experience

### Default Behavior
- App starts with Gemini 2.5 Flash selected
- No aspect ratio controls shown initially
- Clear indication of what each model offers

### When Switching to Imagen
- Aspect ratio dropdown appears
- User can select from 5 standard aspect ratios
- Generated images respect the selected ratio precisely

### When Switching to Gemini
- Aspect ratio dropdown disappears
- Helpful note explains that Gemini generates square images
- Option to switch to Imagen for aspect ratio control

### Visual Feedback
- Each image shows which model created it
- Consistent icons and naming throughout the interface
- Clear notes about model capabilities in editing/composition

## API Endpoints Updated

### `/api/generate_image`
- Added `model` parameter to request body
- Routes to appropriate API based on model selection
- Returns model information in response

### `/api/edit_image` & `/api/compose_images`
- Accept `model` parameter but use Gemini for multimodal operations
- Include helpful logging about model usage

## Benefits

1. **Flexibility**: Users can choose the best model for their needs
2. **Precision**: Imagen provides exact aspect ratio control
3. **Performance**: Gemini offers faster generation
4. **Transparency**: Clear indication of which model was used
5. **Backward Compatibility**: Existing functionality remains unchanged

## Usage Recommendations

- **Use Gemini** for:
  - Quick image generation
  - Conversational editing and composition
  - When square images are acceptable

- **Use Imagen** for:
  - Professional content requiring specific aspect ratios
  - Social media posts with platform-specific dimensions
  - High-quality standalone image generation

## Future Enhancements

- Add Imagen support for image editing (when API supports it)
- Include generation time and cost information
- Add model-specific quality settings
- Support for additional aspect ratios in Imagen
