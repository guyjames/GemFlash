import { useState, useRef, useEffect } from 'react'
import { Button } from "./components/ui/button"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "./components/ui/select"
import { Textarea } from "./components/ui/textarea"
import { Input } from "./components/ui/input"
import { Label } from "./components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs"
import { Checkbox } from "./components/ui/checkbox"
import { Bot, Brush, Download, Edit3, Image as ImageIcon, Layers, Monitor, Moon, Send, Sparkles, Sun, Upload } from 'lucide-react'
import { PaintBrushOverlay, CameraCaptureOverlay, MergeImagesOverlay } from "./components/CreativeProcessingOverlays";

function App() {
  // Tab state
  const [activeTab, setActiveTab] = useState("generate")
  
  // Model selection state
  const [selectedModel, setSelectedModel] = useState("gemini") // "gemini" or "imagen"
  
  // Generate tab state
  const [resolution, setResolution] = useState("1:1")
  const [generatePrompt, setGeneratePrompt] = useState("")
  const [generatedImages, setGeneratedImages] = useState([])
  
  // Edit tab state  
  const [editPrompt, setEditPrompt] = useState("")
  const [editImage, setEditImage] = useState(null)
  const [editImageUrl, setEditImageUrl] = useState("")
  const [editedImages, setEditedImages] = useState([])
  const [selectedImageForEdit, setSelectedImageForEdit] = useState(null)
  const [editModelImages, setEditModelImages] = useState([])
  
  // Compose tab state
  const [composePrompt, setComposePrompt] = useState("")
  const [composeImages, setComposeImages] = useState([])
  const [selectedForCompose, setSelectedForCompose] = useState(new Set())
  const [composedImages, setComposedImages] = useState([])
  
  // UI state
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [isGenerating, setIsGenerating] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [isComposing, setIsComposing] = useState(false);
  // Theme (system | light | dark) with auto-switching
  const getSystemIsDark = () => (
    window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
  );
  const [themeMode, setThemeMode] = useState(() => {
    try { return localStorage.getItem('theme') || 'system'; } catch { return 'system'; }
  });
  // Apply theme and subscribe to system changes when in 'system' mode
  useEffect(() => {
    const root = document.documentElement;
    const media = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : { matches: false };
    const apply = (isDark) => { if (isDark) root.classList.add('dark'); else root.classList.remove('dark'); };
    if (themeMode === 'system') { apply(media.matches); } else { apply(themeMode === 'dark'); }
    const handler = (e) => { if (themeMode === 'system') apply(e.matches); };
    if (media.addEventListener) media.addEventListener('change', handler); else if (media.addListener) media.addListener(handler);
    return () => { if (media.removeEventListener) media.removeEventListener('change', handler); else if (media.removeListener) media.removeListener(handler); };
  }, [themeMode]);
  // Persist selection
  useEffect(() => { try { localStorage.setItem('theme', themeMode); } catch {} }, [themeMode]);

  // File input refs
  const editFileRef = useRef(null)
  const composeFileRef = useRef(null)

  const aspectRatios = [
    { value: "1:1", label: "1:1 Square - Social Media Profile" },
    { value: "16:9", label: "16:9 Widescreen - Desktop/Video" },
    { value: "9:16", label: "9:16 Portrait - Mobile/Stories" },
    { value: "4:3", label: "4:3 Standard - Photo Landscape" },
    { value: "3:4", label: "3:4 Portrait - Social Posts" },
  ]

  const models = [
    { 
      value: "gemini", 
      label: "Gemini 2.5 Flash Image", 
      description: "Fast, conversational image generation",
      icon: Bot
    },
    { 
      value: "imagen", 
      label: "Imagen 4.0", 
      description: "High-quality, specialized image generation with aspect ratio control",
      icon: Brush
    },
  ]

  // Generate image handler
  const handleGenerateImage = async () => {
    if (!generatePrompt.trim()) {
      setError("Please enter a prompt")
      return
    }

    setLoading(true)
    setIsGenerating(true)
    setError("")
    
    try {
      const response = await fetch('/api/generate_image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          prompt: generatePrompt, 
          aspect_ratio: resolution,
          model: selectedModel 
        }),
      })
      
      const data = await response.json()
      
      if (data.error) {
        setError(data.error)
      } else if (data.image) {
        const newImage = {
          id: Date.now(),
          src: `data:image/png;base64,${data.image}`,
          prompt: generatePrompt,
          aspect_ratio: resolution,
          model: selectedModel,
          type: 'generated'
        }
        setGeneratedImages(prev => [newImage, ...prev])
        setGeneratePrompt("")
      } else {
        setError("No image generated")
      }
    } catch (err) {
      setError("Failed to generate image: " + err.message)
    } finally {
      setLoading(false)
      setIsGenerating(false)
    }
  }

  // Edit image handler
  const handleEditImage = async () => {
    if (!editPrompt.trim()) {
      setError("Please enter an edit prompt")
      return
    }

    if (!selectedImageForEdit && !editImage && !editImageUrl.trim()) {
      setError("Please select an image to edit")
      return
    }

    setLoading(true)
    setIsEditing(true)
    setError("")
    
    try {
      const formData = new FormData()
      formData.append('prompt', editPrompt)
      formData.append('aspect_ratio', resolution)
      formData.append('model', selectedModel)
      
      // Use selected image if available
      if (selectedImageForEdit && editImage) {
        formData.append('image_file', editImage)
      } else if (editImageUrl.trim()) {
        formData.append('image_urls', editImageUrl)
      } else if (editImage) {
        formData.append('image_file', editImage)
      }
      
      const response = await fetch('/api/edit_image', {
        method: 'POST',
        body: formData,
      })
      
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.error || `HTTP error! status: ${response.status}`)
      }
      
      if (data.error) {
        setError(data.error)
      } else if (data.image) {
        const newImage = {
          id: Date.now(),
          src: `data:image/png;base64,${data.image}`,
          prompt: editPrompt,
          model: selectedModel,
          type: 'edited',
          originalImage: selectedImageForEdit
        }
        setEditedImages(prev => [newImage, ...prev])
        // Add new edited image to model cards and select it for potential further editing
        const newModelImage = {
          id: Date.now() + 1,
          src: `data:image/png;base64,${data.image}`,
          prompt: editPrompt,
          model: selectedModel,
          type: 'edited'
        }
        setEditModelImages(prev => [newModelImage, ...prev])
        setSelectedImageForEdit(newModelImage)
        setEditPrompt("")
        // Convert new image to file for next edit
        fetch(`data:image/png;base64,${data.image}`)
          .then(res => res.blob())
          .then(blob => {
            const file = new File([blob], 'edited-image.png', { type: 'image/png' })
            setEditImage(file)
          })
      } else {
        setError("No edited image returned")
      }
    } catch (err) {
      console.error("Edit image error:", err)
      setError("Failed to edit image: " + err.message)
    } finally {
      setLoading(false)
      setIsEditing(false)
    }
  }

  // Compose images handler
  const handleComposeImages = async () => {
    if (!composePrompt.trim()) {
      setError("Please enter a composition prompt")
      return
    }

    const selectedImages = composeImages.filter(img => selectedForCompose.has(img.id))
    if (selectedImages.length === 0) {
      setError("Please select at least one image for composition")
      return
    }

    setLoading(true)
    setIsComposing(true)
    setError("")
    
    try {
      const formData = new FormData()
      formData.append('prompt', composePrompt)
      formData.append('model', selectedModel)
      
      // Add selected images
      selectedImages.forEach((img, index) => {
        if (img.file) {
          formData.append('image_files', img.file)
        }
      })
      
      const response = await fetch('/api/compose_images', {
        method: 'POST',
        body: formData,
      })
      
      const data = await response.json()
      
      if (data.error) {
        setError(data.error)
      } else if (data.image) {
        const newImage = {
          id: Date.now(),
          src: `data:image/png;base64,${data.image}`,
          prompt: composePrompt,
          model: selectedModel,
          type: 'composed'
        }
        setComposedImages(prev => [newImage, ...prev])
        setComposePrompt("")
        setSelectedForCompose(new Set())
      } else {
        setError("No composed image returned")
      }
    } catch (err) {
      setError("Failed to compose images: " + err.message)
    } finally {
      setLoading(false)
      setIsComposing(false)
    }
  }

  // Send image to edit tab
  const sendToEdit = (image) => {
    setActiveTab("edit")
    // Add image to edit model cards and select it
    const editImage = {
      id: Date.now(),
      src: image.src,
      prompt: image.prompt,
      model: image.model,
      type: 'model'
    }
    setEditModelImages(prev => [editImage, ...prev])
    setSelectedImageForEdit(editImage)
    // Also set as file for API call
    fetch(image.src)
      .then(res => res.blob())
      .then(blob => {
        const file = new File([blob], 'image.png', { type: 'image/png' })
        setEditImage(file)
      })
  }

  // Send image to compose tab
  const sendToCompose = (image) => {
    setActiveTab("compose")
    fetch(image.src)
      .then(res => res.blob())
      .then(blob => {
        const file = new File([blob], 'image.png', { type: 'image/png' })
        const newComposeImage = {
          id: Date.now(),
          src: image.src,
          file: file,
          name: `image-${Date.now()}.png`
        }
        setComposeImages(prev => [newComposeImage, ...prev])
      })
  }

  // Download image
  const downloadImage = (image) => {
    const link = document.createElement('a')
    link.href = image.src
    link.download = `image-${image.id}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Send image to edit tab
  const sendToEditTab = async (image) => {
    try {
      // Convert base64 to blob
      const response = await fetch(image.src)
      const blob = await response.blob()
      
      // Create a File object from the blob
      const file = new File([blob], `composed-image-${image.id}.png`, { type: 'image/png' })
      
      // Add to edit model images and select it
      const newModelImage = {
        id: Date.now() + Math.random(),
        src: image.src,
        prompt: image.prompt || 'Composed image',
        model: image.model,
        type: 'transferred'
      }
      
      // Set the edit file and add to model images
      setEditImage(file)
      setEditImageUrl("") // Clear URL field when transferring composed images
      setEditModelImages(prev => [newModelImage, ...prev])
      setSelectedImageForEdit(newModelImage)
      setActiveTab('edit')
    } catch (error) {
      console.error('Error sending image to edit tab:', error)
    }
  }

  // Add image to composition selection
  const addToComposition = async (image) => {
    try {
      // Convert base64 to blob
      const response = await fetch(image.src)
      const blob = await response.blob()
      
      // Create a File object from the blob
      const file = new File([blob], `composed-image-${image.id}.png`, { type: 'image/png' })
      
      // Create image object for composition
      const newImage = {
        id: Date.now() + Math.random(),
        src: image.src,
        file: file,
        name: `composed-image-${image.id}.png`
      }
      
      // Add to composition images and switch to compose tab
      setComposeImages(prev => [...prev, newImage])
      setActiveTab('compose')
    } catch (error) {
      console.error('Error adding image to composition:', error)
    }
  }

  // Handle file uploads for compose
  const handleComposeFileUpload = (e) => {
    const files = Array.from(e.target.files)
    files.forEach(file => {
      const reader = new FileReader()
      reader.onload = (e) => {
        const newImage = {
          id: Date.now() + Math.random(),
          src: e.target.result,
          file: file,
          name: file.name
        }
        setComposeImages(prev => [newImage, ...prev])
      }
      reader.readAsDataURL(file)
    })
  }

  // Handle compose URL upload
  const handleComposeUrlAdd = () => {
    const url = prompt("Enter image URL:")
    if (url && url.trim()) {
      const newImage = {
        id: Date.now(),
        src: url.trim(),
        name: `image-${Date.now()}`
      }
      setComposeImages(prev => [newImage, ...prev])
    }
  }

  // Toggle compose selection
  const toggleComposeSelection = (imageId) => {
    setSelectedForCompose(prev => {
      const newSet = new Set(prev)
      if (newSet.has(imageId)) {
        newSet.delete(imageId)
      } else {
        newSet.add(imageId)
      }
      return newSet
    })
  }

  // Remove image from compose selection
  const removeComposeImage = (imageId) => {
    setComposeImages(prev => prev.filter(img => img.id !== imageId))
    setSelectedForCompose(prev => {
      const newSet = new Set(prev)
      newSet.delete(imageId)
      return newSet
    })
  }

  // Select image for editing
  const selectImageForEdit = (image) => {
    setSelectedImageForEdit(image)
    // Convert to file for API
    fetch(image.src)
      .then(res => res.blob())
      .then(blob => {
        const file = new File([blob], 'selected-image.png', { type: 'image/png' })
        setEditImage(file)
      })
  }

  // Remove image from edit model cards
  const removeImageFromEditModel = (imageId) => {
    setEditModelImages(prev => prev.filter(img => img.id !== imageId))
    if (selectedImageForEdit && selectedImageForEdit.id === imageId) {
      setSelectedImageForEdit(null)
      setEditImage(null)
    }
  }

  // Get model badge for images
  const getModelBadge = (model) => {
    const modelInfo = models.find(m => m.value === model)
    if (!modelInfo) return null
    
    const Icon = modelInfo.icon
    return (
      <div className="absolute top-2 left-2 bg-black/70 text-white px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1">
        <Icon className="w-3 h-3" />
        {modelInfo.label}
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <PaintBrushOverlay open={isGenerating} />
      <CameraCaptureOverlay open={isEditing} />
      <MergeImagesOverlay open={isComposing} />
      {/* Header */}
      <div className="bg-card border-b border-border">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary to-accent rounded-xl flex items-center justify-center">
                <Sparkles className="w-5 h-5 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-foreground">GemFlash</h1>
                <p className="text-sm text-muted-foreground">AI-powered image generation and editing</p>
              </div>
            </div>
            
            <div className="flex items-center gap-3">
              {/* Model Selection */}
              <div className="min-w-[200px]">
              <Label htmlFor="model-select" className="text-sm font-medium">AI Model</Label>
              <Select value={selectedModel} onValueChange={setSelectedModel}>
                <SelectTrigger id="model-select">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {models.map((model) => {
                    const Icon = model.icon
                    return (
                      <SelectItem key={model.value} value={model.value}>
                        <div className="flex items-center gap-2">
                          <Icon className="w-4 h-4" />
                          <div>
                            <div className="font-medium">{model.label}</div>
                            <div className="text-xs text-muted-foreground">{model.description}</div>
                          </div>
                        </div>
                      </SelectItem>
                    )
                  })}
                </SelectContent>
              </Select>
            </div>
              <div className="min-w-[180px]">
                <Label className="text-sm font-medium">Theme</Label>
                <Select value={themeMode} onValueChange={setThemeMode}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="system">
                      <div className="flex items-center gap-2"><Monitor className="w-4 h-4" /> System</div>
                    </SelectItem>
                    <SelectItem value="light">
                      <div className="flex items-center gap-2"><Sun className="w-4 h-4" /> Light</div>
                    </SelectItem>
                    <SelectItem value="dark">
                      <div className="flex items-center gap-2"><Moon className="w-4 h-4" /> Dark</div>
                    </SelectItem>
                  </SelectContent>
                </Select>
                <div className="text-xs text-muted-foreground mt-1">System follows your device theme.</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        {error && (
          <div className="mb-6 p-4 bg-destructive/10 border border-destructive/20 rounded-lg text-destructive">
            {error}
          </div>
        )}

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3 mb-6">
            <TabsTrigger value="generate" className="flex items-center gap-2">
              <Sparkles className="w-4 h-4" />
              Generate
            </TabsTrigger>
            <TabsTrigger value="edit" className="flex items-center gap-2">
              <Edit3 className="w-4 h-4" />
              Edit
            </TabsTrigger>
            <TabsTrigger value="compose" className="flex items-center gap-2">
              <Layers className="w-4 h-4" />
              Compose
            </TabsTrigger>
          </TabsList>

          {/* Generate Tab */}
          <TabsContent value="generate" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Generate Image</CardTitle>
                <CardDescription>
                  Create new images from text prompts using {models.find(m => m.value === selectedModel)?.label}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Conditionally show aspect ratio only for Imagen */}
                {selectedModel === "imagen" && (
                  <div>
                    <Label htmlFor="resolution">Aspect Ratio</Label>
                    <Select value={resolution} onValueChange={setResolution}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {aspectRatios.map((ratio) => (
                          <SelectItem key={ratio.value} value={ratio.value}>
                            {ratio.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                {/* Show note for Gemini */}
                {selectedModel === "gemini" && (
                  <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
                    <p className="text-sm text-blue-800">
                      <Bot className="w-4 h-4 inline mr-1" />
                      <strong>Gemini 2.5 Flash:</strong> Generates square (1:1) images by default. 
                      For precise aspect ratio control, switch to Imagen 4.0.
                    </p>
                  </div>
                )}
                
                <div>
                  <Label htmlFor="prompt">Prompt</Label>
                  <Textarea
                    id="prompt"
                    placeholder="Describe the image you want to generate..."
                    value={generatePrompt}
                    onChange={(e) => setGeneratePrompt(e.target.value)}
                    className="min-h-[100px]"
                  />
                </div>
                
                <Button 
                  onClick={handleGenerateImage} 
                  disabled={loading || !generatePrompt.trim()}
                  className="w-full"
                >
                  {loading ? "Generating..." : "Generate Image"}
                </Button>
              </CardContent>
            </Card>

            {/* Generated Images Gallery */}
            {generatedImages.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Generated Images</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {generatedImages.map((image) => (
                      <Card key={image.id} className="overflow-hidden">
                        <div className="relative bg-gray-100">
                          <img 
                            src={image.src} 
                            alt={image.prompt}
                            className="w-full h-auto object-contain"
                          />
                          {getModelBadge(image.model)}
                        </div>
                        <CardContent className="p-3">
                          <p className="text-sm text-gray-600 mb-3 line-clamp-2">{image.prompt}</p>
                          <div className="flex gap-1 flex-wrap">
                            <Button size="sm" variant="outline" onClick={() => sendToEdit(image)}>
                              <Edit3 className="w-3 h-3 mr-1" /> Edit
                            </Button>
                            <Button size="sm" variant="outline" onClick={() => sendToCompose(image)}>
                              <Layers className="w-3 h-3 mr-1" /> Compose
                            </Button>
                            <Button size="sm" variant="outline" onClick={() => downloadImage(image)}>
                              <Download className="w-3 h-3 mr-1" /> Download
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Edit Tab */}
          <TabsContent value="edit" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Edit Image</CardTitle>
                <CardDescription>
                  Modify existing images with AI using {models.find(m => m.value === selectedModel)?.label}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Show note for image editing */}
                <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                  <p className="text-sm text-amber-800">
                    <Edit3 className="w-4 h-4 inline mr-1" />
                    <strong>Note:</strong> Image editing uses Gemini's multimodal capabilities regardless of selected model.
                  </p>
                </div>

                {/* Show selected image preview */}
                {selectedImageForEdit && (
                  <div className="border rounded-lg p-4 bg-gray-50">
                    <Label className="text-sm font-medium text-gray-700 mb-2 block">Selected Image:</Label>
                    <div className="flex items-center gap-3">
                      <img 
                        src={selectedImageForEdit.src} 
                        alt="Selected" 
                        className="w-16 h-16 object-cover rounded-lg border"
                      />
                      <div className="flex-1">
                        <p className="text-sm text-gray-600 line-clamp-2">
                          {selectedImageForEdit.prompt || 'Ready for editing'}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Upload from Computer</Label>
                    <Input
                      ref={editFileRef}
                      type="file"
                      accept="image/*"
                      onChange={(e) => {
                        const file = e.target.files[0]
                        if (file) {
                          setEditImage(file)
                          // Add to model cards
                          const reader = new FileReader()
                          reader.onload = (e) => {
                            const newImage = {
                              id: Date.now(),
                              src: e.target.result,
                              prompt: file.name,
                              model: selectedModel,
                              type: 'uploaded'
                            }
                            setEditModelImages(prev => [newImage, ...prev])
                            setSelectedImageForEdit(newImage)
                          }
                          reader.readAsDataURL(file)
                        }
                      }}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label htmlFor="edit-url">Or Enter Image URL</Label>
                    <Input
                      id="edit-url"
                      placeholder="https://example.com/image.jpg"
                      value={editImageUrl}
                      onChange={(e) => setEditImageUrl(e.target.value)}
                    />
                  </div>
                </div>

                <div>
                  <Label htmlFor="edit-prompt">Edit Prompt</Label>
                  <Textarea
                    id="edit-prompt"
                    placeholder="Describe how you want to modify the selected image..."
                    value={editPrompt}
                    onChange={(e) => setEditPrompt(e.target.value)}
                    className="min-h-[100px]"
                  />
                </div>

                <Button 
                  onClick={handleEditImage} 
                  disabled={loading || !editPrompt.trim() || (!selectedImageForEdit && !editImage && !editImageUrl.trim())}
                  className="w-full"
                >
                  {loading ? "Editing..." : "Edit Image"}
                </Button>
              </CardContent>
            </Card>

            {/* Image Model Cards - moved to bottom for consistency */}
            {editModelImages.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Available Images</CardTitle>
                  <CardDescription>Select an image to edit</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {editModelImages.map((image) => (
                      <Card 
                        key={image.id} 
                        className={`overflow-hidden cursor-pointer transition-all ${
                          selectedImageForEdit?.id === image.id 
                            ? 'ring-2 ring-blue-500 bg-blue-50' 
                            : 'hover:shadow-md'
                        }`}
                        onClick={() => selectImageForEdit(image)}
                      >
                        <div className="relative bg-gray-100">
                          <img 
                            src={image.src} 
                            alt={image.prompt || 'Image'}
                            className="w-full h-auto object-contain"
                          />
                          {image.model && getModelBadge(image.model)}
                          {selectedImageForEdit?.id === image.id && (
                            <div className="absolute inset-0 bg-blue-500 bg-opacity-20 flex items-center justify-center">
                              <div className="bg-blue-500 text-white px-2 py-1 rounded-full text-xs font-medium">
                                Selected
                              </div>
                            </div>
                          )}
                        </div>
                        <CardContent className="p-3">
                          <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                            {image.prompt || 'Transferred image'}
                          </p>
                          <div className="flex gap-1 flex-wrap">
                            <Button size="sm" variant="outline" onClick={(e) => {
                              e.stopPropagation()
                              selectImageForEdit(image)
                            }}>
                              <Edit3 className="w-3 h-3 mr-1" /> Select
                            </Button>
                            <Button size="sm" variant="outline" onClick={(e) => {
                              e.stopPropagation()
                              downloadImage(image)
                            }}>
                              <Download className="w-3 h-3 mr-1" /> Download
                            </Button>
                            <Button size="sm" variant="destructive" onClick={(e) => {
                              e.stopPropagation()
                              removeImageFromEditModel(image.id)
                            }}>
                              ×
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Edited Images Gallery */}
            {editedImages.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Edited Images</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {editedImages.map((image) => (
                      <Card key={image.id} className="overflow-hidden">
                        <div className="relative bg-gray-100">
                          <img 
                            src={image.src} 
                            alt={image.prompt}
                            className="w-full h-auto object-contain"
                          />
                          {image.model && getModelBadge(image.model)}
                        </div>
                        <CardContent className="p-3">
                          <p className="text-sm text-gray-600 mb-3 line-clamp-2">{image.prompt}</p>
                          <div className="flex gap-1 flex-wrap">
                            <Button size="sm" variant="outline" onClick={() => sendToCompose(image)}>
                              <Layers className="w-3 h-3 mr-1" /> Compose
                            </Button>
                            <Button size="sm" variant="outline" onClick={() => downloadImage(image)}>
                              <Download className="w-3 h-3 mr-1" /> Download
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          {/* Compose Tab */}
          <TabsContent value="compose" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Compose Images</CardTitle>
                <CardDescription>
                  Combine multiple images into one composition using {models.find(m => m.value === selectedModel)?.label}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Show note for image composition */}
                <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
                  <p className="text-sm text-amber-800">
                    <Layers className="w-4 h-4 inline mr-1" />
                    <strong>Note:</strong> Image composition uses Gemini's multimodal capabilities regardless of selected model.
                  </p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label>Upload Images</Label>
                    <Input
                      ref={composeFileRef}
                      type="file"
                      accept="image/*"
                      multiple
                      onChange={handleComposeFileUpload}
                      className="mt-1"
                    />
                  </div>
                  <div>
                    <Label>Add from URL</Label>
                    <Button variant="outline" onClick={handleComposeUrlAdd} className="w-full mt-1">
                      <Upload className="w-4 h-4 mr-2" /> Add Image URL
                    </Button>
                  </div>
                </div>

                <div>
                  <Label htmlFor="compose-prompt">Composition Prompt</Label>
                  <Textarea
                    id="compose-prompt"
                    placeholder="Describe how you want to combine the selected images..."
                    value={composePrompt}
                    onChange={(e) => setComposePrompt(e.target.value)}
                    className="min-h-[100px]"
                  />
                </div>

                <Button 
                  onClick={handleComposeImages} 
                  disabled={loading || !composePrompt.trim() || selectedForCompose.size === 0}
                  className="w-full"
                >
                  {loading ? "Composing..." : `Compose Selected Images (${selectedForCompose.size})`}
                </Button>
              </CardContent>
            </Card>

            {/* Available Images for Composition */}
            {composeImages.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Select Images for Composition</CardTitle>
                  <CardDescription>
                    {selectedForCompose.size === 0 
                      ? "Click on images to select them for composition. You must select at least one image before composing." 
                      : `${selectedForCompose.size} image${selectedForCompose.size > 1 ? 's' : ''} selected`
                    }
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                    {composeImages.map((image) => (
                      <Card 
                        key={image.id} 
                        className={`overflow-hidden cursor-pointer transition-all border-2 ${
                          selectedForCompose.has(image.id) 
                            ? 'border-blue-500 ring-2 ring-blue-200 shadow-lg' 
                            : 'border-transparent hover:border-gray-300'
                        }`} 
                        onClick={() => toggleComposeSelection(image.id)}
                      >
                        <div className="relative bg-gray-100">
                          <img 
                            src={image.src} 
                            alt={image.name}
                            className="w-full h-auto object-contain"
                          />
                          <div className="absolute top-2 left-2 bg-white rounded-full p-1 shadow-lg">
                            <Checkbox 
                              checked={selectedForCompose.has(image.id)}
                              onChange={() => toggleComposeSelection(image.id)}
                              className="w-5 h-5"
                            />
                          </div>
                          <div className="absolute top-2 right-2">
                            <Button 
                              size="sm" 
                              variant="destructive"
                              onClick={(e) => {
                                e.stopPropagation()
                                removeComposeImage(image.id)
                              }}
                              className="w-6 h-6 p-0 rounded-full"
                            >
                              ×
                            </Button>
                          </div>
                          {selectedForCompose.has(image.id) && (
                            <div className="absolute inset-0 bg-blue-500 bg-opacity-20 flex items-center justify-center">
                              <div className="bg-blue-500 text-white px-3 py-1 rounded-full text-sm font-medium">
                                Selected
                              </div>
                            </div>
                          )}
                        </div>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}

            {/* Composed Images Gallery */}
            {composedImages.length > 0 && (
              <Card>
                <CardHeader>
                  <CardTitle>Composed Images</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {composedImages.map((image) => (
                      <Card key={image.id} className="overflow-hidden">
                        <div className="relative bg-gray-100">
                          <img 
                            src={image.src} 
                            alt={image.prompt}
                            className="w-full h-auto object-contain"
                          />
                          {image.model && getModelBadge(image.model)}
                        </div>
                        <CardContent className="p-3">
                          <p className="text-sm text-gray-600 mb-3 line-clamp-2">{image.prompt}</p>
                          <div className="flex gap-1">
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => sendToEditTab(image)}
                              className="flex-1"
                            >
                              <Edit3 className="w-3 h-3 mr-1" /> Edit
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => addToComposition(image)}
                              className="flex-1"
                            >
                              <Layers className="w-3 h-3 mr-1" /> Compose
                            </Button>
                            <Button
                              size="sm"
                              variant="outline"
                              onClick={() => downloadImage(image)}
                              className="flex-1"
                            >
                              <Download className="w-3 h-3 mr-1" /> Download
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

export default App
