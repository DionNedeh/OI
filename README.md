OI: A Local Multimodal AI Companion
OI is a powerful, fully offline AI desktop application built with Python, llama-cpp-python, and Tkinter. It provides a clean, responsive interface to run Large Language Models (LLMs) locally with GPU acceleration. Designed for privacy and performance, it supports text, code, images, and voice interaction—all processed entirely on your local hardware.
🚀 Key Features
100% Offline & Private: No data ever leaves your machine. Your chats, documents, and voice stay strictly on your local disk.
CUDA Acceleration: Utilizes your NVIDIA GPU for lightning-fast inference via llama.cpp.
Multimodal Capabilities:
Vision: Process and analyze images using state-of-the-art vision models (e.g., Gemma, Llava, Qwen, MiniCPM).
File Context: Attach text, code, or log files for the AI to analyze directly within the conversation.
Native Voice Interaction: Seamlessly record and transcribe audio using faster-whisper. Optimized for Gemma 4 (E2B/E4B) architectures.
Persistent Memory: Automatically saves chat history in JSON format, allowing you to resume conversations later.
🛠 Prerequisites
1. Hardware Requirements
GPU: An NVIDIA GPU with sufficient VRAM for the models you intend to run is highly recommended.
OS: Windows (recommended for the UI styling), Linux, or macOS.
2. Software Requirements
Python 3.10+ installed.
CUDA Toolkit: Ensure your NVIDIA drivers and the CUDA toolkit are installed if you want to use GPU acceleration.
Models: You will need .gguf model files.
Place them in the folder: ~/.lmstudio/models/ (or create this path).
Note: The UI will automatically scan this directory on launch.
📥 Installation
Clone the Repository:
code
Bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
Install Dependencies:
It is recommended to use a virtual environment:
code
Bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

Optional (UI Theme):
The application uses the azure.tcl theme for a clean Windows 11-style UI. Ensure azure.tcl is in the same directory as OI2.py.
📖 How to Use
Loading a Model
Launch the application: python OI2.py.
Select your desired .gguf model from the "Model Selector" dropdown.
Configure your Context Window and GPU Layers. (Setting GPU Layers to -1 will attempt to offload all layers to your GPU).
Click "Initialize Engine".
Chatting
Standard Chat: Type in the bottom input bar and press Enter.
Attachments: Click the + button to add images or code files.
Text files: The AI will read the content directly.
Images: If your model supports vision, the AI will "see" the image.
Voice Input: Click the * button to start recording. Click again to stop and trigger transcription. (Supported on specific Gemma 4 models).
Saving and Loading
Use the "Save Chat" and "Load Chat" buttons to store your sessions in ~/OI_chats/.
⚙️ Configuration & Notes
Vision Compatibility: The app automatically detects if a mmproj (multimodal projector) file exists in your model folder to enable vision support for compatible models.
Voice Runtime: Ensure you have your microphone permissions enabled. Voice features require sounddevice and faster-whisper.
Troubleshooting:
If the app fails to see your GPU, verify your llama-cpp-python installation is compiled with CUDA support (CMAKE_ARGS="-DLLAMA_CUDA=on").
Ensure your .gguf files are not corrupted; test them in other tools like LM Studio if they fail to load.
📜 License
This project is open-source and available under the MIT License.
🤝 Contributing
Contributions are welcome! Please feel free to open a Pull Request or create an Issue if you encounter bugs or have suggestions for new features.
