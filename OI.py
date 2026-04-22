import gc
import inspect
import json
import os
import tempfile
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

from llama_cpp import Llama

try:
    import numpy as np
except ImportError:
    np = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None


class LocalLLMGUI:
    DEFAULT_SYSTEM_PROMPT = (
        "You are a highly capable and responsive AI companion."
    )
    DEFAULT_CTX = 2048
    DEFAULT_GPU_LAYERS = -1
    MAX_RESPONSE_TOKENS = 1024

    MAX_ATTACHMENTS = 8
    MAX_TEXT_FILE_BYTES = 250_000
    MAX_TEXT_CHARS = 12_000
    MAX_INLINE_FILE_SECTIONS = 4

    VOICE_MAX_SECONDS = 90
    VOICE_MIN_SECONDS = 0.35
    ASR_MODEL_NAME = "tiny.en"
    ASR_LANGUAGE = "en"

    MESSAGE_TOKEN_OVERHEAD = 6
    IMAGE_TOKEN_ESTIMATE = 256

    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".gif",
    }
    TEXT_EXTENSIONS = {
        ".txt",
        ".md",
        ".markdown",
        ".py",
        ".json",
        ".csv",
        ".tsv",
        ".yaml",
        ".yml",
        ".ini",
        ".toml",
        ".log",
        ".xml",
        ".html",
        ".css",
        ".js",
        ".ts",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".rs",
        ".go",
        ".sh",
        ".bat",
        ".ps1",
        ".sql",
    }

    def __init__(self, root):
        self.root = root
        self.root.title("OI")
        self.root.geometry("1000x800")

        self.llm = None
        self.chat_history = []
        self.model_dir = Path.home() / ".lmstudio" / "models"
        self.history_dir = Path.home() / "OI_chats"
        self.available_models = []

        self.loading_model = False
        self.generating_response = False
        self.stop_event = threading.Event()

        self.pending_attachments = []

        self.supports_vision = False
        self.model_supports_voice_input = False
        self.audio_runtime_available = self._can_use_voice_runtime()
        self.asr_model = None

        self.voice_recording = False
        self.voice_transcribing = False
        self.voice_stream = None
        self.voice_frames = []
        self.voice_sample_rate = 16000
        self.voice_auto_stop_job = None

        self.active_model_name = ""
        self.active_model_path = ""

        for directory in [self.model_dir, self.history_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        self.create_widgets()
        self.scan_for_models()
        self._refresh_voice_state()
        self._refresh_voice_button_state()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def create_widgets(self):
        settings_frame = ttk.LabelFrame(self.root, text="System Controls", padding=10)
        settings_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Label(settings_frame, text="Model Selector:").pack(anchor=tk.W, pady=(0, 5))
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(settings_frame, textvariable=self.model_var, state="readonly", width=55)
        self.model_dropdown.pack(anchor=tk.W, pady=(0, 15))

        ttk.Label(settings_frame, text="Context Window (n_ctx):").pack(anchor=tk.W)
        self.ctx_var = tk.IntVar(value=self.DEFAULT_CTX)
        ttk.Entry(settings_frame, textvariable=self.ctx_var, width=15).pack(anchor=tk.W, pady=(0, 15))

        ttk.Label(settings_frame, text="GPU Layers (-1 for max):").pack(anchor=tk.W)
        self.gpu_var = tk.IntVar(value=self.DEFAULT_GPU_LAYERS)
        ttk.Entry(settings_frame, textvariable=self.gpu_var, width=15).pack(anchor=tk.W, pady=(0, 15))

        ttk.Label(settings_frame, text="System Prompt:").pack(anchor=tk.W)
        self.sys_prompt_text = scrolledtext.ScrolledText(settings_frame, height=8, width=35)
        self.sys_prompt_text.pack(anchor=tk.W, pady=(0, 15))
        self.sys_prompt_text.insert(tk.END, self.DEFAULT_SYSTEM_PROMPT)

        self.load_btn = ttk.Button(settings_frame, text="Initialize Engine", command=self.load_model_thread)
        self.load_btn.pack(fill=tk.X, pady=5)

        self.deload_btn = ttk.Button(settings_frame, text="Deload Engine", command=self.deload_model, state=tk.DISABLED)
        self.deload_btn.pack(fill=tk.X, pady=5)

        ttk.Separator(settings_frame, orient="horizontal").pack(fill=tk.X, pady=15)

        self.new_chat_btn = ttk.Button(settings_frame, text="New Chat", command=self.new_chat)
        self.new_chat_btn.pack(fill=tk.X, pady=5)

        self.save_btn = ttk.Button(settings_frame, text="Save Chat", command=self.save_chat)
        self.save_btn.pack(fill=tk.X, pady=5)

        self.load_chat_btn = ttk.Button(settings_frame, text="Load Chat", command=self.load_chat)
        self.load_chat_btn.pack(fill=tk.X, pady=5)

        self.status_var = tk.StringVar(value="Status: Awaiting model load...")
        ttk.Label(settings_frame, textvariable=self.status_var, wraplength=250).pack(anchor=tk.W, pady=(20, 4))

        self.vision_var = tk.StringVar(value="Vision: OFF")
        ttk.Label(settings_frame, textvariable=self.vision_var, wraplength=250).pack(anchor=tk.W, pady=(0, 4))

        self.voice_var = tk.StringVar(value="Voice: OFF (load model)")
        ttk.Label(settings_frame, textvariable=self.voice_var, wraplength=250).pack(anchor=tk.W, pady=(0, 4))

        self.token_stats_var = tk.StringVar(
            value="Usage (Est.): prompt 0 | completion 0 | total 0 | ctx 0.0%"
        )
        ttk.Label(settings_frame, textvariable=self.token_stats_var, wraplength=250).pack(anchor=tk.W)

        chat_frame = ttk.Frame(self.root, padding=10)
        chat_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.chat_display = scrolledtext.ScrolledText(chat_frame, state="disabled", wrap=tk.WORD, font=("Segoe UI", 11))
        self.chat_display.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill=tk.X)

        self.attach_btn = ttk.Button(input_frame, text="+", width=3, command=self.pick_attachments)
        self.attach_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.voice_btn = ttk.Button(input_frame, text="*", width=3, command=self.toggle_voice_input, state=tk.DISABLED)
        self.voice_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.user_input = ttk.Entry(input_frame, font=("Segoe UI", 12))
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.user_input.bind("<Return>", lambda event: self.send_message())

        self.send_btn = ttk.Button(input_frame, text="Transmit", command=self.send_message, state=tk.DISABLED)
        self.send_btn.pack(side=tk.RIGHT)

        self.stop_btn = ttk.Button(input_frame, text="Stop", command=self.stop_generation, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, padx=(0, 8))

        attachments_frame = ttk.Frame(chat_frame)
        attachments_frame.pack(fill=tk.X, pady=(8, 0))

        self.attachment_var = tk.StringVar(value="Attachments: none")
        self.attachment_label = ttk.Label(attachments_frame, textvariable=self.attachment_var, wraplength=700)
        self.attachment_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.clear_attachments_btn = ttk.Button(
            attachments_frame,
            text="Clear",
            command=self.clear_attachments,
            state=tk.DISABLED,
        )
        self.clear_attachments_btn.pack(side=tk.RIGHT)

    def scan_for_models(self):
        self.available_models = sorted(
            [
                path
                for path in self.model_dir.rglob("*.gguf")
                if path.is_file() and "mmproj" not in path.name.lower()
            ],
            key=lambda p: str(p).lower(),
        )

        if self.available_models:
            display_names = [str(model.relative_to(self.model_dir)) for model in self.available_models]
            self.model_dropdown["values"] = display_names
            self.model_dropdown.current(0)
        else:
            self.model_dropdown.set("No .gguf models found")
            self.model_dropdown["values"] = ()

    def load_model_thread(self):
        if self.loading_model:
            return
        if self.voice_recording or self.voice_transcribing:
            messagebox.showinfo("Busy", "Please wait for voice input to finish before loading a model.")
            return
        if not self.available_models:
            messagebox.showerror("File Error", f"Please place .gguf files in '{self.model_dir}'.")
            return

        selected_index = self.model_dropdown.current()
        if selected_index < 0:
            messagebox.showerror("Selection Error", "Please select a model first.")
            return

        n_ctx, n_gpu_layers = self._read_runtime_settings()
        if n_ctx is None or n_gpu_layers is None:
            return

        model_path = self.available_models[selected_index]
        if not self.chat_history:
            sys_prompt = self.sys_prompt_text.get("1.0", tk.END).strip()
            self.chat_history = [{"role": "system", "content": sys_prompt}]

        self.loading_model = True
        self.status_var.set("Status: LINKING...")
        self.load_btn.config(state=tk.DISABLED)
        self.deload_btn.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self.model_supports_voice_input = False
        self._refresh_voice_state()
        self._refresh_voice_button_state()
        self.update_chat_display("System: Allocating memory... please wait.\n\n")

        threading.Thread(
            target=self.load_model,
            args=(str(model_path), n_ctx, n_gpu_layers),
            daemon=True,
        ).start()

    def _read_runtime_settings(self):
        try:
            n_ctx = int(self.ctx_var.get())
            n_gpu_layers = int(self.gpu_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("Input Error", "n_ctx and GPU layers must be valid integers.")
            return None, None

        if n_ctx <= 0:
            messagebox.showerror("Input Error", "n_ctx must be greater than 0.")
            return None, None

        if n_gpu_layers < -1:
            messagebox.showerror("Input Error", "GPU layers must be -1 or greater.")
            return None, None

        return n_ctx, n_gpu_layers

    @staticmethod
    def _can_use_voice_runtime():
        return np is not None and sd is not None and sf is not None and WhisperModel is not None

    @staticmethod
    def _is_small_gemma_model(model_path_or_name):
        name = str(model_path_or_name).lower()
        return "gemma" in name and ("e2b" in name or "e4b" in name)

    def _refresh_voice_state(self):
        if self.voice_recording:
            self.voice_var.set("Voice: Recording... click * to stop")
            return

        if self.voice_transcribing:
            self.voice_var.set("Voice: Transcribing...")
            return

        if not self.audio_runtime_available:
            self.voice_var.set("Voice: OFF (missing sounddevice/soundfile/faster-whisper)")
            return

        if not self.llm:
            self.voice_var.set("Voice: OFF (load model)")
            return

        if not self.model_supports_voice_input:
            self.voice_var.set("Voice: OFF (Gemma 4 E2B/E4B only)")
            return

        self.voice_var.set("Voice: READY (* to speak)")

    def _refresh_voice_button_state(self):
        if not hasattr(self, "voice_btn"):
            return

        if self.voice_recording:
            self.voice_btn.config(state=tk.NORMAL)
            return

        if self.voice_transcribing:
            self.voice_btn.config(state=tk.DISABLED)
            return

        ready = (
            self.llm is not None
            and not self.loading_model
            and not self.generating_response
            and self.audio_runtime_available
            and self.model_supports_voice_input
        )
        self.voice_btn.config(state=tk.NORMAL if ready else tk.DISABLED)

    def _send_text_message(self, text):
        clean_text = text.strip()
        if not clean_text:
            return
        self.user_input.delete(0, tk.END)
        self.user_input.insert(0, clean_text)
        self.send_message()

    def _find_mmproj_for_model(self, model_path: Path):
        candidates = sorted(
            [
                path
                for path in model_path.parent.glob("*.gguf")
                if path.is_file() and "mmproj" in path.name.lower()
            ],
            key=lambda p: p.name.lower(),
        )
        if candidates:
            return candidates[0]
        return None

    def _instantiate_optional_handler(self, handler_cls, mmproj_path: Path, model_name: str):
        kwargs = {
            "clip_model_path": str(mmproj_path),
            "verbose": False,
        }
        try:
            init_signature = inspect.signature(handler_cls.__init__)
            if "enable_thinking" in init_signature.parameters:
                # Gemma-4 E2B/E4B variants generally require thinking enabled.
                kwargs["enable_thinking"] = ("e2b" in model_name or "e4b" in model_name)
        except (TypeError, ValueError):
            pass

        return handler_cls(**kwargs)

    @staticmethod
    def _build_gemma_fallback_handler(base_handler_cls):
        class GemmaVisionFallbackChatHandler(base_handler_cls):
            DEFAULT_SYSTEM_MESSAGE = None

            CHAT_FORMAT = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' and message['content'] is not none %}"
                "<start_of_turn>system\n{{ message['content'] }}<end_of_turn>\n"
                "{% endif %}"
                "{% if message['role'] == 'user' %}"
                "<start_of_turn>user\n"
                "{% if message['content'] is string %}"
                "{{ message['content'] }}"
                "{% endif %}"
                "{% if message['content'] is iterable and message['content'] is not string %}"
                "{% for content in message['content'] %}"
                "{% if content.type == 'image_url' and content.image_url is string %}"
                "{{ content.image_url }}"
                "{% endif %}"
                "{% if content.type == 'image_url' and content.image_url is mapping %}"
                "{{ content.image_url.url }}"
                "{% endif %}"
                "{% endfor %}"
                "{% for content in message['content'] %}"
                "{% if content.type == 'text' %}{{ content.text }}{% endif %}"
                "{% endfor %}"
                "{% endif %}"
                "<end_of_turn>\n"
                "{% endif %}"
                "{% if message['role'] == 'assistant' and message['content'] is not none %}"
                "<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
            )

        return GemmaVisionFallbackChatHandler

    def _create_chat_handler_for_model(self, model_path: Path):
        mmproj_path = self._find_mmproj_for_model(model_path)
        if mmproj_path is None:
            return None, "Vision: OFF (no mmproj in model folder)"

        model_name = model_path.name.lower()

        try:
            from llama_cpp.llama_chat_format import (
                Llama3VisionAlphaChatHandler,
                Llava15ChatHandler,
                MiniCPMv26ChatHandler,
                Qwen25VLChatHandler,
            )
            import llama_cpp.llama_chat_format as chat_format_module
        except Exception:
            return None, "Vision: OFF (vision handlers unavailable)"

        Gemma4ChatHandler = getattr(chat_format_module, "Gemma4ChatHandler", None)
        Gemma3ChatHandler = getattr(chat_format_module, "Gemma3ChatHandler", None)

        try:
            if "gemma" in model_name:
                if Gemma4ChatHandler is not None and ("gemma-4" in model_name or "gemma4" in model_name):
                    handler = self._instantiate_optional_handler(Gemma4ChatHandler, mmproj_path, model_name)
                    return handler, f"Vision: ON (Gemma4 handler, {mmproj_path.name})"

                if Gemma3ChatHandler is not None:
                    handler = self._instantiate_optional_handler(Gemma3ChatHandler, mmproj_path, model_name)
                    return handler, f"Vision: ON (Gemma handler, {mmproj_path.name})"

                fallback_cls = self._build_gemma_fallback_handler(Llava15ChatHandler)
                handler = fallback_cls(clip_model_path=str(mmproj_path), verbose=False)
                return handler, f"Vision: ON (Gemma fallback handler, {mmproj_path.name})"
            if "qwen" in model_name:
                handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path), verbose=False)
            elif "llava" in model_name:
                handler = Llava15ChatHandler(clip_model_path=str(mmproj_path), verbose=False)
            elif "minicpm" in model_name:
                handler = MiniCPMv26ChatHandler(clip_model_path=str(mmproj_path), verbose=False)
            elif "llama" in model_name and "vision" in model_name:
                handler = Llama3VisionAlphaChatHandler(clip_model_path=str(mmproj_path), verbose=False)
            else:
                return None, "Vision: OFF (no matching vision handler)"
        except Exception:
            return None, "Vision: OFF (vision handler initialization failed)"

        return handler, f"Vision: ON ({mmproj_path.name})"

    def load_model(self, model_path, n_ctx, n_gpu_layers):
        old_model = self.llm

        try:
            if old_model is not None:
                self.llm = None
                del old_model
                gc.collect()

            model_path_obj = Path(model_path)
            base_kwargs = {
                "model_path": model_path,
                "n_gpu_layers": n_gpu_layers,
                "n_ctx": n_ctx,
                "verbose": False,
            }

            chat_handler, vision_status = self._create_chat_handler_for_model(model_path_obj)
            supports_vision = False

            if chat_handler is not None:
                try:
                    llm = Llama(**base_kwargs, chat_handler=chat_handler)
                    supports_vision = True
                except Exception as vision_error:
                    llm = Llama(**base_kwargs)
                    vision_status = f"Vision: OFF (fallback after {vision_error.__class__.__name__})"
            else:
                llm = Llama(**base_kwargs)

            self.llm = llm
            self.supports_vision = supports_vision
            self.root.after(0, self._on_model_loaded, model_path_obj.name, str(model_path_obj), vision_status)
        except Exception as e:
            self.llm = None
            self.supports_vision = False
            self.root.after(0, self._on_model_load_failure, str(e))

    def _on_model_loaded(self, model_name, model_path, vision_status):
        self.loading_model = False
        self.active_model_name = model_name
        self.active_model_path = model_path
        self.model_supports_voice_input = self._is_small_gemma_model(model_path)
        self.status_var.set(f"Status: ONLINE ({model_name})")
        self.vision_var.set(vision_status)
        self.send_btn.config(state=tk.NORMAL)
        self.load_btn.config(state=tk.NORMAL)
        self.deload_btn.config(state=tk.NORMAL)
        self._refresh_voice_state()
        self._refresh_voice_button_state()
        self.update_chat_display("System: Engine Online. Ready for transmission.\n\n")

    def _on_model_load_failure(self, error_message):
        self.loading_model = False
        self.active_model_name = ""
        self.active_model_path = ""
        self.model_supports_voice_input = False
        self.status_var.set("Status: Load Failure.")
        self.vision_var.set("Vision: OFF")
        self._refresh_voice_state()
        self.update_chat_display(f"Critical Error: {error_message}\n\n")
        self.load_btn.config(state=tk.NORMAL)
        self.deload_btn.config(state=tk.DISABLED)
        self.send_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        self._refresh_voice_button_state()

    def deload_model(self):
        if self.loading_model:
            messagebox.showinfo("Busy", "Please wait for model loading to finish.")
            return

        if self.generating_response:
            messagebox.showinfo("Busy", "Please stop or wait for the current response to complete.")
            return
        if self.voice_recording or self.voice_transcribing:
            messagebox.showinfo("Busy", "Please wait for voice input/transcription to finish.")
            return

        if self.llm:
            del self.llm
            self.llm = None
            gc.collect()
            self.supports_vision = False
            self.model_supports_voice_input = False
            self.active_model_name = ""
            self.active_model_path = ""
            self.status_var.set("Status: OFFLINE (Engine Deloaded)")
            self.vision_var.set("Vision: OFF")
            self._update_usage_stats(0, 0)
            self._refresh_voice_state()
            self.update_chat_display("System: Memory purged. VRAM released.\n\n")
            self.send_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.load_btn.config(state=tk.NORMAL)
            self.deload_btn.config(state=tk.DISABLED)
            self._refresh_voice_button_state()

    def new_chat(self):
        if self.generating_response:
            messagebox.showinfo("Busy", "Please stop or wait for the current response to complete.")
            return
        if self.voice_recording or self.voice_transcribing:
            messagebox.showinfo("Busy", "Please wait for voice input/transcription to finish.")
            return

        sys_prompt = self.sys_prompt_text.get("1.0", tk.END).strip()
        self.chat_history = [{"role": "system", "content": sys_prompt}]

        self.chat_display.config(state="normal")
        self.chat_display.delete("1.0", tk.END)
        self.chat_display.config(state="disabled")

        self.clear_attachments()
        self._update_usage_stats(0, 0)
        self.update_chat_display("System: Chat cleared.\n\n")

    def save_chat(self):
        filepath = filedialog.asksaveasfilename(
            initialdir=str(self.history_dir),
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json")],
            title="Save Chat History",
        )
        if filepath:
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(self.chat_history, f, indent=4, ensure_ascii=False)
                self.update_chat_display(f"System: Chat saved to {Path(filepath).name}\n\n")
            except OSError as e:
                messagebox.showerror("Save Error", f"Could not save chat: {e}")

    def load_chat(self):
        if self.generating_response:
            messagebox.showinfo("Busy", "Please stop or wait for the current response to complete.")
            return
        if self.voice_recording or self.voice_transcribing:
            messagebox.showinfo("Busy", "Please wait for voice input/transcription to finish.")
            return

        filepath = filedialog.askopenfilename(
            initialdir=str(self.history_dir),
            filetypes=[("JSON Files", "*.json")],
            title="Load Chat History",
        )
        if filepath:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    loaded_history = json.load(f)

                if not self._is_valid_chat_history(loaded_history):
                    raise ValueError("Invalid chat history format.")
            except (OSError, json.JSONDecodeError, ValueError) as e:
                messagebox.showerror("Load Error", f"Could not load chat: {e}")
                return

            self.chat_history = loaded_history
            self.clear_attachments()

            self.chat_display.config(state="normal")
            self.chat_display.delete("1.0", tk.END)
            self.chat_display.config(state="disabled")

            for msg in self.chat_history:
                role = "S" if msg["role"] == "user" else "OI" if msg["role"] == "assistant" else "System"
                display_content = self._content_to_display_text(msg.get("content", ""))
                self.update_chat_display(f"{role}: {display_content}\n\n")

            self.update_chat_display("System: Chat history loaded successfully.\n\n")

    @staticmethod
    def _is_valid_message_content(content):
        if isinstance(content, str):
            return True

        if not isinstance(content, list):
            return False

        for part in content:
            if not isinstance(part, dict):
                return False

            part_type = part.get("type")
            if part_type == "text":
                if not isinstance(part.get("text"), str):
                    return False
            elif part_type == "image_url":
                image_url = part.get("image_url")
                if isinstance(image_url, str):
                    continue
                if isinstance(image_url, dict) and isinstance(image_url.get("url"), str):
                    continue
                return False
            else:
                return False

        return True

    def _is_valid_chat_history(self, history):
        if not isinstance(history, list):
            return False
        for msg in history:
            if not isinstance(msg, dict):
                return False
            if msg.get("role") not in {"system", "user", "assistant"}:
                return False
            if not self._is_valid_message_content(msg.get("content")):
                return False
        return True

    def _content_to_display_text(self, content):
        if isinstance(content, str):
            return content

        if not isinstance(content, list):
            return str(content)

        parts = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text" and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif item_type == "image_url":
                parts.append("[Image attachment]")

        return "\n".join(part for part in parts if part).strip()

    def _format_bytes(self, num_bytes):
        units = ["B", "KB", "MB", "GB"]
        size = float(num_bytes)
        unit = units[0]
        for next_unit in units:
            unit = next_unit
            if size < 1024.0 or unit == units[-1]:
                break
            size /= 1024.0
        return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"

    def pick_attachments(self):
        if self.generating_response or self.voice_recording or self.voice_transcribing:
            messagebox.showinfo("Busy", "Please wait for the current response to finish.")
            return

        selected_paths = filedialog.askopenfilenames(
            title="Select files or images",
            filetypes=[
                ("All Supported", "*.txt *.md *.py *.json *.csv *.log *.yaml *.yml *.xml *.html *.css *.js *.ts *.png *.jpg *.jpeg *.webp *.bmp *.gif"),
                ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.gif"),
                ("Text and Code", "*.txt *.md *.py *.json *.csv *.log *.yaml *.yml *.xml *.html *.css *.js *.ts"),
                ("All Files", "*.*"),
            ],
        )

        if not selected_paths:
            return

        added_count = 0
        for path_str in selected_paths:
            if self._add_attachment(path_str):
                added_count += 1

        self._refresh_attachment_ui()

        skipped_count = len(selected_paths) - added_count
        if skipped_count > 0:
            self.update_chat_display(
                f"System: Added {added_count} attachment(s), skipped {skipped_count}.\n\n"
            )

    def _add_attachment(self, path_str):
        path = Path(path_str)
        if not path.exists() or not path.is_file():
            return False

        if len(self.pending_attachments) >= self.MAX_ATTACHMENTS:
            return False

        normalized = str(path.resolve()).lower()
        if any(item.get("normalized") == normalized for item in self.pending_attachments):
            return False

        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        suffix = path.suffix.lower()
        kind = "image" if suffix in self.IMAGE_EXTENSIONS else "file"

        self.pending_attachments.append(
            {
                "path": str(path.resolve()),
                "name": path.name,
                "size": size,
                "kind": kind,
                "normalized": normalized,
            }
        )
        return True

    def clear_attachments(self):
        self.pending_attachments.clear()
        self._refresh_attachment_ui()

    def _refresh_attachment_ui(self):
        if not self.pending_attachments:
            self.attachment_var.set("Attachments: none")
            self.clear_attachments_btn.config(state=tk.DISABLED)
            return

        preview_items = [item["name"] for item in self.pending_attachments[:3]]
        preview = ", ".join(preview_items)
        remainder = len(self.pending_attachments) - len(preview_items)
        if remainder > 0:
            preview = f"{preview} (+{remainder} more)"

        self.attachment_var.set(f"Attachments ({len(self.pending_attachments)}): {preview}")
        self.clear_attachments_btn.config(state=tk.NORMAL)

    def _image_dimensions(self, image_path: Path):
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                return img.size
        except Exception:
            return None

    def _extract_text_from_file(self, file_path: Path):
        if file_path.suffix.lower() not in self.TEXT_EXTENSIONS:
            return "", "unsupported/binary format (metadata only)"

        try:
            with open(file_path, "rb") as f:
                raw = f.read(self.MAX_TEXT_FILE_BYTES + 1)
        except OSError as e:
            return "", f"read failed ({e})"

        truncated = len(raw) > self.MAX_TEXT_FILE_BYTES
        raw = raw[: self.MAX_TEXT_FILE_BYTES]

        text = raw.decode("utf-8", errors="replace").replace("\x00", "")
        text = text.strip()

        if len(text) > self.MAX_TEXT_CHARS:
            text = text[: self.MAX_TEXT_CHARS]
            truncated = True

        if not text:
            return "", "file appears empty or unreadable as text"

        note = "text truncated for prompt limits" if truncated else ""
        return text, note

    def _build_user_message_content(self, message_text, attachments):
        base_text = message_text.strip()
        display_lines = []
        notices = []
        attachment_lines = []
        inline_file_sections = []
        image_parts = []

        included_file_sections = 0

        for attachment in attachments:
            path = Path(attachment["path"])
            size_label = self._format_bytes(attachment.get("size", 0))

            if attachment["kind"] == "image":
                dims = self._image_dimensions(path)
                dims_label = f", {dims[0]}x{dims[1]}" if dims else ""
                display_lines.append(f"[Image] {path.name} ({size_label}{dims_label})")
                attachment_lines.append(f"- image: {path.name} ({size_label}{dims_label})")

                if self.supports_vision:
                    try:
                        image_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": path.resolve().as_uri()},
                            }
                        )
                    except ValueError:
                        notices.append(f"{path.name}: could not convert to image URI.")
                else:
                    notices.append(
                        f"{path.name}: current model is text-only, so image pixels are not analyzed."
                    )
            else:
                display_lines.append(f"[File] {path.name} ({size_label})")
                attachment_lines.append(f"- file: {path.name} ({size_label})")

                extracted_text, extraction_note = self._extract_text_from_file(path)
                if extraction_note:
                    notices.append(f"{path.name}: {extraction_note}.")

                if extracted_text:
                    if included_file_sections < self.MAX_INLINE_FILE_SECTIONS:
                        inline_file_sections.append(f"[File: {path.name}]\n{extracted_text}")
                        included_file_sections += 1
                    else:
                        notices.append(
                            f"{path.name}: skipped inline content (max {self.MAX_INLINE_FILE_SECTIONS} files)."
                        )

        prompt_sections = []
        if base_text:
            prompt_sections.append(base_text)
        elif attachments:
            prompt_sections.append("Please analyze the uploaded attachments.")

        if attachment_lines:
            prompt_sections.append("Uploaded attachments:\n" + "\n".join(attachment_lines))

        if inline_file_sections:
            prompt_sections.append("Extracted file content:\n\n" + "\n\n".join(inline_file_sections))

        if attachments and any(item["kind"] == "image" for item in attachments) and not self.supports_vision:
            prompt_sections.append(
                "Image note: one or more images were uploaded, but this model is text-only and receives metadata only."
            )

        payload_text = "\n\n".join(section for section in prompt_sections if section).strip()

        if image_parts:
            content = [{"type": "text", "text": payload_text or "Please analyze the uploaded images."}] + image_parts
        else:
            content = payload_text

        return content, display_lines, notices

    def _cancel_voice_auto_stop(self):
        if self.voice_auto_stop_job is not None:
            try:
                self.root.after_cancel(self.voice_auto_stop_job)
            except tk.TclError:
                pass
            self.voice_auto_stop_job = None

    def _safe_stop_voice_stream(self):
        stream = self.voice_stream
        self.voice_stream = None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass

    def _ensure_asr_model_loaded(self):
        if self.asr_model is not None:
            return
        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed.")

        self.asr_model = WhisperModel(
            self.ASR_MODEL_NAME,
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        )

    def toggle_voice_input(self):
        if self.voice_recording:
            self._stop_voice_recording_and_transcribe()
            return

        self._start_voice_recording()

    def _voice_callback(self, indata, frames, time_info, status):
        if not self.voice_recording:
            return
        if status:
            # Status can indicate overflow/underflow but recording can continue.
            pass
        if np is None:
            return
        self.voice_frames.append(np.copy(indata))

    def _start_voice_recording(self):
        if self.loading_model or self.generating_response or self.voice_transcribing:
            return
        if not self.llm:
            return
        if not self.model_supports_voice_input:
            messagebox.showinfo("Voice Input", "Voice input is enabled only for Gemma 4 E2B/E4B models.")
            return
        if not self.audio_runtime_available:
            messagebox.showerror(
                "Voice Runtime Missing",
                "Voice input needs sounddevice, soundfile, numpy, and faster-whisper.",
            )
            return
        if sd is None:
            return

        try:
            device_info = sd.query_devices(None, "input")
            sample_rate = int(device_info["default_samplerate"]) if device_info else 16000
            channels = 1

            self.voice_sample_rate = sample_rate
            self.voice_frames = []
            self.voice_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
                callback=self._voice_callback,
            )
            self.voice_stream.start()
        except Exception as e:
            self.voice_stream = None
            messagebox.showerror("Microphone Error", f"Could not start recording: {e}")
            return

        self.voice_recording = True
        self.send_btn.config(state=tk.DISABLED)
        self.attach_btn.config(state=tk.DISABLED)
        self._refresh_voice_state()
        self._refresh_voice_button_state()
        self._cancel_voice_auto_stop()
        self.voice_auto_stop_job = self.root.after(
            self.VOICE_MAX_SECONDS * 1000,
            self._stop_voice_recording_and_transcribe,
        )

    def _stop_voice_recording_and_transcribe(self):
        if not self.voice_recording:
            return
        if np is None:
            return

        self.voice_recording = False
        self._cancel_voice_auto_stop()
        self._safe_stop_voice_stream()

        if not self.voice_frames:
            self._refresh_voice_state()
            self._refresh_voice_button_state()
            if self.llm and not self.loading_model and not self.generating_response:
                self.send_btn.config(state=tk.NORMAL)
                self.attach_btn.config(state=tk.NORMAL)
            return

        try:
            audio = np.concatenate(self.voice_frames, axis=0)
        except Exception:
            self.voice_frames = []
            self._refresh_voice_state()
            self._refresh_voice_button_state()
            if self.llm and not self.loading_model and not self.generating_response:
                self.send_btn.config(state=tk.NORMAL)
                self.attach_btn.config(state=tk.NORMAL)
            return

        self.voice_frames = []

        if audio.ndim > 1:
            audio = audio[:, 0]

        duration = len(audio) / float(self.voice_sample_rate) if self.voice_sample_rate > 0 else 0.0
        if duration < self.VOICE_MIN_SECONDS:
            self.update_chat_display("System: Voice input too short. Try speaking a little longer.\n\n")
            self._refresh_voice_state()
            self._refresh_voice_button_state()
            if self.llm and not self.loading_model and not self.generating_response:
                self.send_btn.config(state=tk.NORMAL)
                self.attach_btn.config(state=tk.NORMAL)
            return

        self.voice_transcribing = True
        self._refresh_voice_state()
        self._refresh_voice_button_state()

        threading.Thread(
            target=self._transcribe_audio_and_send,
            args=(audio, self.voice_sample_rate),
            daemon=True,
        ).start()

    def _transcribe_audio_and_send(self, audio, sample_rate):
        temp_path = None
        try:
            self._ensure_asr_model_loaded()
            if sf is None:
                raise RuntimeError("soundfile is not installed.")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                temp_path = tmp.name

            sf.write(temp_path, audio, sample_rate, subtype="PCM_16")

            segments, _ = self.asr_model.transcribe(
                temp_path,
                language=self.ASR_LANGUAGE,
                task="transcribe",
                beam_size=1,
                best_of=1,
                vad_filter=True,
                without_timestamps=True,
            )
            transcript = " ".join(segment.text.strip() for segment in segments if segment.text.strip()).strip()
            self.root.after(0, self._on_voice_transcription_ready, transcript)
        except Exception as e:
            self.root.after(0, self._on_voice_transcription_failed, str(e))
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _on_voice_transcription_ready(self, transcript):
        self.voice_transcribing = False
        self._refresh_voice_state()

        if not transcript:
            self.update_chat_display("System: Could not detect speech from audio.\n\n")
            self._refresh_voice_button_state()
            if self.llm and not self.loading_model and not self.generating_response:
                self.send_btn.config(state=tk.NORMAL)
                self.attach_btn.config(state=tk.NORMAL)
            return

        self.update_chat_display(f"System: Voice transcript captured.\n\n")
        self._refresh_voice_button_state()
        if not self.llm or self.loading_model or self.generating_response:
            self.user_input.delete(0, tk.END)
            self.user_input.insert(0, transcript)
            if self.llm and not self.loading_model:
                self.send_btn.config(state=tk.NORMAL)
            self.attach_btn.config(state=tk.NORMAL)
            return
        self._send_text_message(transcript)

    def _on_voice_transcription_failed(self, error_message):
        self.voice_transcribing = False
        self._refresh_voice_state()
        self._refresh_voice_button_state()
        if self.llm and not self.loading_model and not self.generating_response:
            self.send_btn.config(state=tk.NORMAL)
            self.attach_btn.config(state=tk.NORMAL)
        self.update_chat_display(f"System: Voice transcription failed: {error_message}\n\n")

    def stop_generation(self):
        if not self.generating_response:
            return

        self.stop_event.set()
        self.stop_btn.config(state=tk.DISABLED)
        self.status_var.set("Status: STOPPING...")

    def send_message(self):
        if (
            not self.llm
            or self.loading_model
            or self.generating_response
            or self.voice_recording
            or self.voice_transcribing
        ):
            return

        message = self.user_input.get().strip()
        attachments_for_turn = [dict(item) for item in self.pending_attachments]

        if not message and not attachments_for_turn:
            return

        self.user_input.delete(0, tk.END)
        self.clear_attachments()

        user_content, display_lines, notices = self._build_user_message_content(message, attachments_for_turn)

        display_message = message if message else "[Attachment message]"
        if display_lines:
            display_message = f"{display_message}\n" + "\n".join(display_lines)

        self.generating_response = True
        self.stop_event.clear()
        self.send_btn.config(state=tk.DISABLED)
        self.attach_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self._refresh_voice_button_state()

        self.update_chat_display(f"S: {display_message}\n\n")

        if notices:
            for note in notices:
                self.update_chat_display(f"System: {note}\n")
            self.update_chat_display("\n")

        self.chat_history.append({"role": "user", "content": user_content})

        prompt_token_estimate = self._estimate_messages_tokens(self.chat_history)

        self.update_chat_display("OI: ")
        threading.Thread(
            target=self.generate_response,
            args=(prompt_token_estimate,),
            daemon=True,
        ).start()

    def generate_response(self, prompt_token_estimate):
        stream = None
        full_reply = ""
        stopped = False

        try:
            stream = self.llm.create_chat_completion(
                messages=self.chat_history,
                max_tokens=self.MAX_RESPONSE_TOKENS,
                stream=True,
            )

            for chunk in stream:
                if self.stop_event.is_set():
                    stopped = True
                    break

                text_chunk = self._extract_stream_text(chunk)
                if text_chunk:
                    full_reply += text_chunk
                    self.root.after(0, self.update_chat_display, text_chunk)

            if stopped:
                if full_reply:
                    self.root.after(0, self.update_chat_display, "\n[Generation stopped]\n\n")
                else:
                    full_reply = "[Generation stopped by user.]"
                    self.root.after(0, self.update_chat_display, full_reply + "\n\n")
            else:
                self.root.after(0, self.update_chat_display, "\n\n")

            if full_reply:
                self.chat_history.append({"role": "assistant", "content": full_reply})

            completion_tokens = self._estimate_text_tokens(full_reply)
            self.root.after(0, self._update_usage_stats, prompt_token_estimate, completion_tokens)
        except Exception as e:
            self.root.after(0, self.update_chat_display, f"\n[Error: {str(e)}]\n\n")
        finally:
            if stream is not None and hasattr(stream, "close"):
                try:
                    stream.close()
                except Exception:
                    pass
            self.root.after(0, self._unlock_after_response)

    @staticmethod
    def _extract_stream_text(chunk):
        if not isinstance(chunk, dict):
            return ""

        choices = chunk.get("choices")
        if not choices or not isinstance(choices, list):
            return ""

        first_choice: Any = choices[0]
        if not isinstance(first_choice, dict):
            return ""

        delta = first_choice.get("delta")
        if not isinstance(delta, dict):
            return ""

        content = delta.get("content")
        return content if isinstance(content, str) else ""

    def _estimate_text_tokens(self, text):
        if not text:
            return 0
        if self.llm is None:
            return max(1, len(text) // 4)

        try:
            return len(self.llm.tokenize(text.encode("utf-8"), add_bos=False, special=True))
        except Exception:
            return max(1, len(text) // 4)

    def _estimate_content_tokens(self, content):
        if isinstance(content, str):
            return self._estimate_text_tokens(content)

        if isinstance(content, list):
            total = 0
            for part in content:
                if not isinstance(part, dict):
                    continue
                part_type = part.get("type")
                if part_type == "text":
                    total += self._estimate_text_tokens(part.get("text", ""))
                elif part_type == "image_url":
                    total += self.IMAGE_TOKEN_ESTIMATE
            return total

        return self._estimate_text_tokens(str(content))

    def _estimate_messages_tokens(self, messages):
        total = 0
        for msg in messages:
            total += self.MESSAGE_TOKEN_OVERHEAD
            total += self._estimate_content_tokens(msg.get("content", ""))
        return total + 3

    def _update_usage_stats(self, prompt_tokens, completion_tokens):
        total_tokens = max(0, prompt_tokens) + max(0, completion_tokens)

        context_size = 0
        if self.llm is not None:
            try:
                context_size = int(self.llm.n_ctx())
            except Exception:
                context_size = 0

        if context_size <= 0:
            try:
                context_size = int(self.ctx_var.get())
            except (ValueError, tk.TclError):
                context_size = 0

        context_pct = (total_tokens / context_size * 100.0) if context_size > 0 else 0.0
        self.token_stats_var.set(
            f"Usage (Est.): prompt {prompt_tokens} | completion {completion_tokens} | total {total_tokens} | ctx {context_pct:.1f}%"
        )

    def _unlock_after_response(self):
        self.generating_response = False
        self.stop_event.clear()
        self.stop_btn.config(state=tk.DISABLED)
        self.attach_btn.config(state=tk.NORMAL)
        self._refresh_voice_state()
        self._refresh_voice_button_state()

        if self.llm and not self.loading_model:
            self.send_btn.config(state=tk.NORMAL)
            if self.active_model_name:
                self.status_var.set(f"Status: ONLINE ({self.active_model_name})")

    def _on_close(self):
        self.voice_recording = False
        self.voice_transcribing = False
        self._cancel_voice_auto_stop()
        self._safe_stop_voice_stream()
        self.root.destroy()

    def update_chat_display(self, text):
        self.chat_display.config(state="normal")
        self.chat_display.insert(tk.END, text)
        self.chat_display.see(tk.END)
        self.chat_display.config(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.tk.call("source", "azure.tcl")
        root.tk.call("set_theme", "dark")
    except tk.TclError:
        pass
    app = LocalLLMGUI(root)
    root.mainloop()
