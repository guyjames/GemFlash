# GemFlash - AI Image Editor

A containerized web application that uses Google Gemini 2.5 Flash Image API to generate and edit images with a modern React frontend and FastAPI backend.

## 🚀 Features

- **Image Generation**: Create images from text prompts using Google Gemini 2.5 Flash
- **Image Editing**: Edit existing images with AI-powered modifications
- **Image Composition**: Combine multiple images into new compositions
- **Reuse Workflow**: Generated and composed images can be reused for further editing
- **Modern UI**: Built with React, ShadCN UI components, and Tailwind CSS
- **Docker Ready**: Fully containerized for easy deployment

## 🔧 Setup & Installation

### Prerequisites
- Docker and Docker Compose
- Google Gemini API key (get one at [ai.studio/apps](https://ai.studio/apps))

### 1. Environment Configuration

**IMPORTANT**: Never commit API keys to the repository.

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Google API key
GOOGLE_API_KEY="your_actual_api_key_here"
```

### 2. Build & Run

```bash
# Build the application
docker-compose build

# Start the application
docker-compose up

# Or run in background
docker-compose up -d
```

### 3. Access the Application

- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/api/docs

## 📁 Project Structure

```
GemFlash/
├── frontend/          # React frontend with ShadCN UI
├── backend/           # FastAPI backend
├── .env.example       # Environment template
├── Dockerfile         # Multi-stage Docker build
├── docker-compose.yml # Container orchestration
├── CONDUCTOR.md       # Security & compliance guidelines
└── ARCHITECTURE.md    # System architecture documentation
```

## 🔒 Security

This project follows security best practices outlined in `CONDUCTOR.md`:

- **No API keys in repository**: Use `.env` files (gitignored)
- **Container security**: Non-root user, minimal base images
- **TLS enforcement**: All external communications use HTTPS
- **Ephemeral data**: No persistent storage of sensitive data

## 📚 Documentation

- [GOALS.md](./GOALS.md) - Project requirements and objectives
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture and design
- [BUILD.md](./BUILD.md) - Detailed build instructions
- [CONDUCTOR.md](./CONDUCTOR.md) - Security and operational guidelines
- [GEMINI.md](./GEMINI.md) - Google Gemini API integration details

## 🛠 Development

The application includes comprehensive development tools:
- Claude Flow framework for AI-assisted development
- SPARC methodology for systematic development
- Multiple specialized AI agents for different aspects of development

## 🚀 Deployment

The application is designed for deployment behind Nginx Proxy Manager on a Docker network called `shared_net`. See `CONDUCTOR.md` for production deployment guidelines.

## 📄 License

This project is developed following enterprise security standards and is intended for internal use as specified in the security documentation.
