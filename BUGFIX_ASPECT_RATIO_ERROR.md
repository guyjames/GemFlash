# Fix: Pydantic ValidationError for aspect_ratio parameter

## Issue Summary

When attempting to generate images, the application was throwing a Pydantic validation error:

```
1 validation error for GenerateContentConfig
aspect_ratio
  Extra inputs are not permitted [type=extra_forbidden, input_value='1:1', input_type=str]
For further information visit https://errors.pydantic.dev/2.11/v/extra_forbidden
```

## Root Cause

The error occurred because the `aspect_ratio` parameter was being passed to the Google Gemini API's `GenerateContentConfig` object, but this parameter is not supported by the official Gemini API. According to the [official Gemini API documentation](https://ai.google.dev/gemini-api/docs/image-generation), the `GenerateContentConfig` only accepts specific parameters like `response_modalities`, but not `aspect_ratio`.

## Files Affected

- `backend/main.py` - Main backend API file containing the image generation endpoints

## Solution Applied

### 1. Removed Invalid Parameter from API Calls

**Before:**
```python
response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents={"parts": [{"text": detailed_prompt}]},
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        aspect_ratio=request.aspect_ratio  # ❌ This parameter is invalid
    )
)
```

**After:**
```python
response = client.models.generate_content(
    model="gemini-2.5-flash-image-preview",
    contents={"parts": [{"text": detailed_prompt}]},
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"]  # ✅ Only valid parameters
    )
)
```

### 2. Incorporated Aspect Ratio into Prompt Instructions

Instead of passing the aspect ratio to the API configuration, the fix incorporates aspect ratio preferences directly into the text prompts:

```python
# Create detailed prompt for image generation, incorporating aspect ratio in the prompt text
aspect_ratio_instruction = ""
if request.aspect_ratio == "16:9":
    aspect_ratio_instruction = " Create the image in a wide, landscape format (16:9 aspect ratio)."
elif request.aspect_ratio == "9:16":
    aspect_ratio_instruction = " Create the image in a tall, portrait format (9:16 aspect ratio)."
elif request.aspect_ratio == "4:3":
    aspect_ratio_instruction = " Create the image in a standard 4:3 aspect ratio format."
else:  # Default to 1:1
    aspect_ratio_instruction = " Create the image in a square format (1:1 aspect ratio)."

detailed_prompt = f"""You are an expert AI image generator. Create a high-quality, photorealistic image based on the user's request.

User Request: "{request.prompt}"

Generation Guidelines:
- Create a detailed, high-quality image that matches the description
- Ensure the image is photorealistic and well-composed
- Pay attention to lighting, colors, and overall aesthetics{aspect_ratio_instruction}

Output: Return ONLY the final generated image. Do not return text."""
```

### 3. Updated All Affected Endpoints

The fix was applied consistently across all image generation endpoints:
- `/api/generate_image` - Text-to-image generation
- `/api/edit_image` - Image editing functionality  
- `/api/compose_images` - Multi-image composition

### 4. Maintained UI Compatibility

The `aspect_ratio` parameter was kept in the Pydantic request models to maintain backward compatibility with the frontend:

```python
class ImageGenerationRequest(BaseModel):
    prompt: str
    aspect_ratio: str = "1:1"  # Keep this for UI purposes, but don't pass to API
```

## Testing Performed

1. ✅ **Container Setup**: Fixed missing port mapping in `docker-compose.yml`
2. ✅ **Code Fix**: Updated backend to remove invalid API parameters
3. ✅ **Container Rebuild**: Successfully rebuilt and restarted the application
4. ✅ **Service Verification**: Confirmed the application is running and accessible

## Additional Infrastructure Fix

During troubleshooting, also discovered and fixed a separate issue where the container was not accessible due to missing port mapping in `docker-compose.yml`:

**Before:**
```yaml
services:
  gemflash:
    build: .
    env_file:
      - .env
```

**After:**
```yaml
services:
  gemflash:
    build: .
    ports:
      - "8999:8999"  # Added missing port mapping
    env_file:
      - .env
```

## How to Apply This Fix

1. **Backup the current file:**
   ```bash
   cp backend/main.py backend/main.py.backup
   ```

2. **Update the `backend/main.py` file** with the corrected code that removes `aspect_ratio` from all `GenerateContentConfig` calls and incorporates aspect ratio instructions into the prompts instead.

3. **Ensure port mapping exists in `docker-compose.yml`:**
   ```yaml
   ports:
     - "8999:8999"
   ```

4. **Rebuild and restart the container:**
   ```bash
   docker compose down
   docker compose up -d --build
   ```

5. **Verify the fix:**
   - Test image generation functionality
   - Confirm no more Pydantic validation errors occur

## Expected Outcome

- ✅ No more `aspect_ratio Extra inputs are not permitted` errors
- ✅ Image generation works with all aspect ratio options
- ✅ Aspect ratios are respected through natural language instructions to the AI model
- ✅ Full backward compatibility maintained with existing frontend code
- ✅ Application accessible at the configured port (8999)

## References

- [Google Gemini API Image Generation Documentation](https://ai.google.dev/gemini-api/docs/image-generation)
- [Pydantic Validation Error Documentation](https://docs.pydantic.dev/2.11/errors/validation_errors/#extra_forbidden)
