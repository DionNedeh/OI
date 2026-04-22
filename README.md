# OI

# Local LLM GUI (OI)

A custom local model UI built with Python, `llama-cpp-python`, and Tkinter. It provides a seamless interface for running LLMs offline with GPU acceleration via CUDA drivers.

## Features
- **Local Execution:** 100% offline, private AI interaction.
- **Hardware Acceleration:** Uses `llama.cpp` with CUDA support for fast inference.
- **Multimodal Support:** 
  - **Vision:** Handles image uploads for supported models (Gemma, Llava, Qwen, etc.).
  - **File Analysis:** Upload text/code files for context-aware chatting.
- **Voice Input:** Native audio recording and transcription using `faster-whisper`, specifically optimized for Gemma 4 (E2B/E4B) models.
- **History Management:** Save and load chat sessions locally.

## Setup
1. **Requirements:** Ensure you have Python 3.10+ and the necessary CUDA drivers installed for your GPU.
2. **Install Dependencies:**
   ```bash
   pip install llama-cpp-python faster-whisper sounddevice soundfile numpy pillow

   1. Run:
 python OI2.py

 License - MIT
