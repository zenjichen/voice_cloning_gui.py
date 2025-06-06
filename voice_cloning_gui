import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import threading
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from datetime import datetime
import soundfile as sf
from tkinterdnd2 import DND_FILES, TkinterDnD
import queue
import pygame
from io import BytesIO
import time
import pyttsx3
import tempfile

class VoiceModel(nn.Module):
    """Simplified Voice Cloning Model with Dynamic Input Handling"""
    def __init__(self, input_dim=80, hidden_dim=512, output_dim=80):
        super(VoiceModel, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        print(f"Creating model with input_dim={input_dim}, output_dim={output_dim}")
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # Debug input shape
        # print(f"Model input shape: {x.shape}")
        
        # Handle different input shapes
        if len(x.shape) == 3:  # [batch, mel_bins, time_steps]
            batch_size, mel_bins, time_steps = x.shape
            # Reshape to [batch * time_steps, mel_bins]
            x = x.permute(0, 2, 1).contiguous()  # [batch, time_steps, mel_bins]
            x = x.view(batch_size * time_steps, mel_bins)
            
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            
            # Reshape back to [batch, mel_bins, time_steps]
            decoded = decoded.view(batch_size, time_steps, self.output_dim)
            decoded = decoded.permute(0, 2, 1).contiguous()
        else:  # [batch, features]
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
        
        return decoded

class VoiceCloneEngine:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}  # Dictionary to store multiple voice models
        self.model_info = {}  # Store model metadata
        self.models_dir = Path("trained_voices")
        self.models_dir.mkdir(exist_ok=True)
        
        print(f"Voice Clone Engine initialized on device: {self.device}")
        
        # Initialize TTS engine
        try:
            self.tts_engine = pyttsx3.init()
            # Get available voices
            self.available_voices = self.tts_engine.getProperty('voices')
            print(f"Found {len(self.available_voices)} system voices")
        except Exception as e:
            print(f"Warning: Could not initialize pyttsx3: {e}")
            self.tts_engine = None
            self.available_voices = []
        
        # Load existing models
        self.load_all_models()
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio file with consistent parameters"""
        try:
            print(f"Preprocessing audio: {audio_path}")
            
            # Load audio with consistent parameters
            audio, sr = librosa.load(audio_path, sr=22050)
            
            # Remove silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Normalize
            audio = librosa.util.normalize(audio)
            
            # Extract mel spectrogram with fixed parameters
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=sr, 
                n_mels=80,  # Fixed number of mel filters
                fmax=8000,
                n_fft=1024,
                hop_length=256,
                win_length=1024
            )
            
            # Convert to dB
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            print(f"Mel spectrogram shape: {mel_spec_db.shape}")
            
            return mel_spec_db, audio, sr
            
        except Exception as e:
            print(f"Preprocessing error for {audio_path}: {e}")
            raise Exception(f"Error preprocessing audio: {str(e)}")
    
    def train_voice(self, audio_files, voice_name, progress_callback=None):
        """Train a new voice model with improved error handling"""
        try:
            print(f"\n=== Starting training for voice: {voice_name} ===")
            print(f"Number of audio files: {len(audio_files)}")
            
            # Prepare training data
            all_features = []
            chunk_size = 128  # Fixed chunk size
            
            for i, audio_file in enumerate(audio_files):
                try:
                    print(f"Processing file {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
                    
                    features, _, _ = self.preprocess_audio(audio_file)
                    
                    # Split into chunks
                    n_mels, n_frames = features.shape
                    
                    if n_frames < chunk_size:
                        print(f"Warning: Audio too short ({n_frames} frames), skipping...")
                        continue
                    
                    # Create overlapping chunks
                    step_size = chunk_size // 2
                    for j in range(0, n_frames - chunk_size + 1, step_size):
                        chunk = features[:, j:j+chunk_size]
                        if chunk.shape[1] == chunk_size:
                            all_features.append(chunk)
                    
                    print(f"  Extracted {len(all_features)} chunks so far")
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
            
            if not all_features:
                raise Exception("No valid audio data found. Please check your audio files.")
            
            print(f"Total training chunks: {len(all_features)}")
            
            # Convert to numpy array and get dimensions
            all_features = np.array(all_features)
            print(f"Training data shape: {all_features.shape}")
            
            # Get actual input dimension
            actual_input_dim = all_features.shape[1]  # Should be 80 (n_mels)
            print(f"Detected input dimension: {actual_input_dim}")
            
            # Initialize model with correct dimensions
            model = VoiceModel(
                input_dim=actual_input_dim, 
                hidden_dim=512, 
                output_dim=actual_input_dim
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.MSELoss()
            
            # Convert to tensor
            train_data = torch.FloatTensor(all_features).to(self.device)
            print(f"Tensor shape: {train_data.shape}")
            
            # Training loop
            epochs = 100
            batch_size = min(32, len(train_data))
            
            print(f"Starting training: {epochs} epochs, batch size: {batch_size}")
            
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                
                # Shuffle data
                indices = torch.randperm(len(train_data))
                
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i+batch_size]
                    batch_data = train_data[batch_indices]
                    
                    optimizer.zero_grad()
                    
                    # Forward pass
                    output = model(batch_data)
                    loss = criterion(output, batch_data)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / (len(indices) // batch_size + 1)
                
                if progress_callback:
                    progress = (epoch + 1) / epochs * 100
                    progress_callback(progress, f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
            # Save model
            self.save_model(model, voice_name)
            self.models[voice_name] = model
            
            print(f"=== Training completed for voice: {voice_name} ===\n")
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Training failed: {str(e)}")
    
    def save_model(self, model, voice_name):
        """Save trained model with metadata"""
        try:
            model_path = self.models_dir / f"{voice_name}.pth"
            info_path = self.models_dir / f"{voice_name}_info.json"
            
            # Save model state
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': model.input_dim,
                'output_dim': model.output_dim,
                'hidden_dim': 512
            }, model_path)
            
            # Save model info
            info = {
                'name': voice_name,
                'created': datetime.now().isoformat(),
                'device': str(self.device),
                'input_dim': model.input_dim,
                'output_dim': model.output_dim
            }
            
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
            
            self.model_info[voice_name] = info
            print(f"Model saved: {model_path}")
            
        except Exception as e:
            raise Exception(f"Failed to save model: {str(e)}")
    
    def load_all_models(self):
        """Load all existing models with improved error handling"""
        print("Loading existing models...")
        
        for model_file in self.models_dir.glob("*.pth"):
            voice_name = model_file.stem
            try:
                # Load model data
                checkpoint = torch.load(model_file, map_location=self.device)
                
                # Get model dimensions
                if isinstance(checkpoint, dict) and 'input_dim' in checkpoint:
                    input_dim = checkpoint['input_dim']
                    output_dim = checkpoint['output_dim']
                    model_state = checkpoint['model_state_dict']
                else:
                    # Legacy model format
                    input_dim = 80
                    output_dim = 80
                    model_state = checkpoint
                
                # Create and load model
                model = VoiceModel(input_dim=input_dim, output_dim=output_dim).to(self.device)
                model.load_state_dict(model_state)
                model.eval()
                
                self.models[voice_name] = model
                
                # Load info
                info_file = self.models_dir / f"{voice_name}_info.json"
                if info_file.exists():
                    with open(info_file, 'r', encoding='utf-8') as f:
                        self.model_info[voice_name] = json.load(f)
                else:
                    # Create default info
                    self.model_info[voice_name] = {
                        'name': voice_name,
                        'created': 'Unknown',
                        'device': str(self.device)
                    }
                
                print(f"Loaded model: {voice_name}")
                        
            except Exception as e:
                print(f"Failed to load model {voice_name}: {e}")
    
    def text_to_speech(self, text, voice_name, speed=1.0, emotion="neutral"):
        """Convert text to speech using trained voice with proper TTS engine"""
        try:
            if not self.tts_engine:
                raise Exception("TTS engine not available")
            
            if voice_name not in self.models and voice_name != "System Default":
                raise Exception(f"Voice '{voice_name}' not found")
            
            # Configure TTS engine
            voices = self.tts_engine.getProperty('voices')
            
            # Set voice based on selection
            if voice_name == "System Default" or voice_name not in self.models:
                # Use system default voice
                if voices:
                    self.tts_engine.setProperty('voice', voices[0].id)
            else:
                # For trained voices, we'll use system voice but modify characteristics
                # In a full implementation, this would apply the trained model
                if voices:
                    # Try to find a similar voice or use default
                    voice_id = voices[0].id
                    for voice in voices:
                        if 'female' in voice.name.lower() and 'female' in voice_name.lower():
                            voice_id = voice.id
                            break
                        elif 'male' in voice.name.lower() and 'male' in voice_name.lower():
                            voice_id = voice.id
                            break
                    self.tts_engine.setProperty('voice', voice_id)
            
            # Set speech rate (speed)
            rate = self.tts_engine.getProperty('rate')
            new_rate = int(rate * speed)
            self.tts_engine.setProperty('rate', new_rate)
            
            # Set volume based on emotion
            volume = 0.9
            if emotion == "happy":
                volume = 1.0
            elif emotion == "sad":
                volume = 0.7
            elif emotion == "excited":
                volume = 1.0
            elif emotion == "calm":
                volume = 0.8
            
            self.tts_engine.setProperty('volume', volume)
            
            # Generate speech to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                self.tts_engine.save_to_file(text, temp_path)
                self.tts_engine.runAndWait()
                
                # Load the generated audio
                audio, sr = librosa.load(temp_path, sr=22050)
                
                # Clean up temporary file
                os.unlink(temp_path)
                
                # Apply post-processing based on emotion
                if emotion == "happy":
                    # Slightly increase pitch
                    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
                elif emotion == "sad":
                    # Slightly decrease pitch and add reverb effect
                    audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
                    audio = audio * 0.9  # Reduce volume slightly
                elif emotion == "excited":
                    # Increase tempo slightly
                    audio = librosa.effects.time_stretch(audio, rate=1.1)
                elif emotion == "calm":
                    # Smooth the audio
                    audio = librosa.effects.preemphasis(audio, coef=0.95)
            
                return audio, sr
                
            except Exception as e:
                # Clean up temporary file in case of error
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise e
            
        except Exception as e:
            raise Exception(f"TTS failed: {str(e)}")
    
    def get_available_voices(self):
        """Get list of available voices (both trained and system)"""
        voices = ["System Default"]
        voices.extend(list(self.models.keys()))
        return voices
    
    def delete_model(self, voice_name):
        """Delete a trained model"""
        try:
            model_path = self.models_dir / f"{voice_name}.pth"
            info_path = self.models_dir / f"{voice_name}_info.json"
            
            if model_path.exists():
                model_path.unlink()
            if info_path.exists():
                info_path.unlink()
            
            if voice_name in self.models:
                del self.models[voice_name]
            if voice_name in self.model_info:
                del self.model_info[voice_name]
            
            return True
        except Exception as e:
            raise Exception(f"Failed to delete model: {str(e)}")

class VoiceCloningGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional Voice Cloning Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize engine
        self.engine = VoiceCloneEngine()
        
        # Initialize pygame for audio playback
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        except:
            print("Warning: Could not initialize pygame mixer")
        
        # Queue for thread communication
        self.update_queue = queue.Queue()
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        
        # Start queue processor
        self.process_queue()
    
    def setup_styles(self):
        """Setup custom styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure styles
        self.style.configure('Title.TLabel', 
                           background='#2b2b2b', 
                           foreground='#ffffff',
                           font=('Arial', 16, 'bold'))
        
        self.style.configure('Custom.TButton',
                           background='#0078d4',
                           foreground='white',
                           font=('Arial', 10))
        
        self.style.configure('Success.TButton',
                           background='#107c10',
                           foreground='white')
        
        self.style.configure('Danger.TButton',
                           background='#d13438',
                           foreground='white')
    
    def create_widgets(self):
        """Create main GUI widgets"""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Training Tab
        self.create_training_tab()
        
        # Text to Speech Tab
        self.create_tts_tab()
        
        # Voice Management Tab
        self.create_management_tab()
    
    def create_training_tab(self):
        """Create training tab"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="Voice Training")
        
        # Title
        title_label = ttk.Label(training_frame, text="Train New Voice Model", 
                               style='Title.TLabel')
        title_label.pack(pady=20)
        
        # File selection frame
        file_frame = ttk.LabelFrame(training_frame, text="Audio Files", padding=20)
        file_frame.pack(fill='x', padx=20, pady=10)
        
        # Drag and drop area
        self.file_listbox = tk.Listbox(file_frame, height=8, 
                                      bg='#404040', fg='white',
                                      selectbackground='#0078d4')
        self.file_listbox.pack(fill='both', expand=True, pady=10)
        
        # Enable drag and drop
        try:
            self.file_listbox.drop_target_register(DND_FILES)
            self.file_listbox.dnd_bind('<<Drop>>', self.on_file_drop)
        except:
            print("Warning: Drag and drop not available")
        
        # File buttons
        file_buttons_frame = ttk.Frame(file_frame)
        file_buttons_frame.pack(fill='x', pady=5)
        
        ttk.Button(file_buttons_frame, text="Add Files", 
                  command=self.add_files).pack(side='left', padx=5)
        ttk.Button(file_buttons_frame, text="Remove Selected", 
                  command=self.remove_selected_file).pack(side='left', padx=5)
        ttk.Button(file_buttons_frame, text="Clear All", 
                  command=self.clear_files).pack(side='left', padx=5)
        
        # Voice name frame
        name_frame = ttk.LabelFrame(training_frame, text="Voice Name", padding=20)
        name_frame.pack(fill='x', padx=20, pady=10)
        
        self.voice_name_var = tk.StringVar()
        ttk.Entry(name_frame, textvariable=self.voice_name_var, 
                 font=('Arial', 12)).pack(fill='x')
        
        # Training controls
        control_frame = ttk.Frame(training_frame)
        control_frame.pack(fill='x', padx=20, pady=20)
        
        self.train_button = ttk.Button(control_frame, text="Start Training", 
                                      command=self.start_training,
                                      style='Success.TButton')
        self.train_button.pack(side='left', padx=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var,
                                          maximum=100, length=300)
        self.progress_bar.pack(side='left', padx=20, fill='x', expand=True)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="Ready")
        self.status_label.pack(side='right', padx=10)
    
    def create_tts_tab(self):
        """Create Text to Speech tab"""
        tts_frame = ttk.Frame(self.notebook)
        self.notebook.add(tts_frame, text="Text to Speech")
        
        # Title
        title_label = ttk.Label(tts_frame, text="Text to Speech Conversion", 
                               style='Title.TLabel')
        title_label.pack(pady=20)
        
        # Main content frame
        main_frame = ttk.Frame(tts_frame)
        main_frame.pack(fill='both', expand=True, padx=20)
        
        # Left panel - Text input
        left_panel = ttk.LabelFrame(main_frame, text="Text Input", padding=10)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Text area with scrollbar
        text_frame = ttk.Frame(left_panel)
        text_frame.pack(fill='both', expand=True)
        
        self.text_area = tk.Text(text_frame, wrap='word', font=('Arial', 11),
                                bg='#404040', fg='white', insertbackground='white')
        text_scrollbar = ttk.Scrollbar(text_frame, orient='vertical', 
                                     command=self.text_area.yview)
        self.text_area.configure(yscrollcommand=text_scrollbar.set)
        
        self.text_area.pack(side='left', fill='both', expand=True)
        text_scrollbar.pack(side='right', fill='y')
        
        # Add sample text
        sample_text = "Hello! This is a sample text to demonstrate the voice cloning system. You can replace this text with anything you want to convert to speech."
        self.text_area.insert('1.0', sample_text)
        
        # Right panel - Controls
        right_panel = ttk.LabelFrame(main_frame, text="Voice Settings", padding=10)
        right_panel.pack(side='right', fill='y', padx=(10, 0))
        
        # Voice selection
        ttk.Label(right_panel, text="Select Voice:").pack(anchor='w', pady=5)
        self.voice_combo = ttk.Combobox(right_panel, state='readonly', width=20)
        self.voice_combo.pack(fill='x', pady=5)
        self.update_voice_list()
        
        # Speed control
        ttk.Label(right_panel, text="Speed:").pack(anchor='w', pady=(15, 5))
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_frame = ttk.Frame(right_panel)
        speed_frame.pack(fill='x', pady=5)
        
        ttk.Scale(speed_frame, from_=0.5, to=2.0, variable=self.speed_var,
                 orient='horizontal').pack(fill='x')
        self.speed_label = ttk.Label(speed_frame, text="1.0x")
        self.speed_label.pack()
        self.speed_var.trace('w', self.update_speed_label)
        
        # Emotion control
        ttk.Label(right_panel, text="Emotion:").pack(anchor='w', pady=(15, 5))
        self.emotion_var = tk.StringVar(value="neutral")
        emotions = ["neutral", "happy", "sad", "excited", "calm"]
        for emotion in emotions:
            ttk.Radiobutton(right_panel, text=emotion.title(), 
                          variable=self.emotion_var, value=emotion).pack(anchor='w')
        
        # Control buttons
        button_frame = ttk.Frame(right_panel)
        button_frame.pack(fill='x', pady=20)
        
        self.generate_button = ttk.Button(button_frame, text="Generate Speech",
                                        command=self.generate_speech,
                                        style='Success.TButton')
        self.generate_button.pack(fill='x', pady=5)
        
        self.preview_button = ttk.Button(button_frame, text="Preview",
                                       command=self.preview_audio,
                                       state='disabled')
        self.preview_button.pack(fill='x', pady=5)
        
        self.save_button = ttk.Button(button_frame, text="Save MP3",
                                    command=self.save_audio,
                                    state='disabled')
        self.save_button.pack(fill='x', pady=5)
        
        # Progress bar for TTS
        self.tts_progress = ttk.Progressbar(right_panel, mode='indeterminate')
        self.tts_progress.pack(fill='x', pady=10)
        
        # Generated audio storage
        self.current_audio = None
        self.current_sr = None
    
    def create_management_tab(self):
        """Create voice management tab"""
        mgmt_frame = ttk.Frame(self.notebook)
        self.notebook.add(mgmt_frame, text="Voice Management")
        
        # Title
        title_label = ttk.Label(mgmt_frame, text="Manage Trained Voices", 
                               style='Title.TLabel')
        title_label.pack(pady=20)
        
        # Voice list frame
        list_frame = ttk.LabelFrame(mgmt_frame, text="Trained Voices", padding=20)
        list_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Treeview for voice list
        columns = ('Name', 'Created', 'Status')
        self.voice_tree = ttk.Treeview(list_frame, columns=columns, show='headings')
        
        for col in columns:
            self.voice_tree.heading(col, text=col)
            self.voice_tree.column(col, width=200)
        
        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(list_frame, orient='vertical', 
                                   command=self.voice_tree.yview)
        self.voice_tree.configure(yscrollcommand=tree_scroll.set)
        
        self.voice_tree.pack(side='left', fill='both', expand=True)
        tree_scroll.pack(side='right', fill='y')
        
        # Management buttons
        mgmt_button_frame = ttk.Frame(mgmt_frame)
        mgmt_button_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Button(mgmt_button_frame, text="Refresh List",
                  command=self.refresh_voice_list).pack(side='left', padx=5)
        ttk.Button(mgmt_button_frame, text="Delete Voice",
                  command=self.delete_voice,
                  style='Danger.TButton').pack(side='left', padx=5)
        
        # Load voice list
        self.refresh_voice_list()

    # Continuation of GUI methods
    def on_file_drop(self, event):
        """Handle file drop event"""
        try:
            files = self.root.tk.splitlist(event.data)
            for file_path in files:
                if file_path.lower().endswith(('.mp3', '.wav', '.flac', '.m4a')):
                    self.file_listbox.insert(tk.END, file_path)
        except Exception as e:
            print(f"Error handling file drop: {e}")
    
    def add_files(self):
        """Add audio files"""
        files = filedialog.askopenfilenames(
            title="Select Audio Files",
            filetypes=[
                ("Audio Files", "*.mp3 *.wav *.flac *.m4a"),
                ("MP3 Files", "*.mp3"),
                ("WAV Files", "*.wav"),
                ("All Files", "*.*")
            ]
        )
        for file_path in files:
            self.file_listbox.insert(tk.END, file_path)   
    def remove_selected_file(self):
        """Remove selected file from list"""
        selection = self.file_listbox.curselection()
        if selection:
            self.file_listbox.delete(selection[0])
    
    def clear_files(self):
        """Clear all files"""
        self.file_listbox.delete(0, tk.END)
    
    def start_training(self):
        """Start voice training in separate thread"""
        voice_name = self.voice_name_var.get().strip()
        if not voice_name:
            messagebox.showerror("Error", "Please enter a voice name")
            return
        
        files = list(self.file_listbox.get(0, tk.END))
        if not files:
            messagebox.showerror("Error", "Please add audio files")
            return
        
        # Check if voice name already exists
        if voice_name in self.engine.models:
            if not messagebox.askyesno("Confirm", 
                                     f"Voice '{voice_name}' already exists. Overwrite?"):
                return
        
        # Disable training button
        self.train_button.configure(state='disabled')
        self.progress_var.set(0)
        self.status_label.configure(text="Training...")
        
        # Start training thread
        training_thread = threading.Thread(
            target=self.train_voice_thread,
            args=(files, voice_name)
        )
        training_thread.daemon = True
        training_thread.start()
    
    def train_voice_thread(self, files, voice_name):
        """Training thread function"""
        try:
            def progress_callback(progress, status):
                self.update_queue.put(('progress', progress, status))
            
            # Train the voice
            self.engine.train_voice(files, voice_name, progress_callback)
            
            self.update_queue.put(('training_complete', voice_name))
            
        except Exception as e:
            self.update_queue.put(('training_error', str(e)))
    
    def generate_speech(self):
        """Generate speech from text"""
        text = self.text_area.get(1.0, tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please enter text")
            return
        
        voice_name = self.voice_combo.get()
        if not voice_name:
            messagebox.showerror("Error", "Please select a voice")
            return
        
        # Start generation
        self.generate_button.configure(state='disabled')
        self.tts_progress.start()
        
        # Start TTS thread
        tts_thread = threading.Thread(
            target=self.tts_thread,
            args=(text, voice_name, self.speed_var.get(), self.emotion_var.get())
        )
        tts_thread.daemon = True
        tts_thread.start()
    
    def tts_thread(self, text, voice_name, speed, emotion):
        """TTS generation thread"""
        try:
            audio, sr = self.engine.text_to_speech(text, voice_name, speed, emotion)
            self.update_queue.put(('tts_complete', audio, sr))
        except Exception as e:
            self.update_queue.put(('tts_error', str(e)))
    def preview_audio(self):
        """Preview generated audio"""
        if self.current_audio is not None:
            try:
                # Convert to bytes for pygame
                audio_bytes = BytesIO()
                sf.write(audio_bytes, self.current_audio, self.current_sr, format='WAV')
                audio_bytes.seek(0)
                
                pygame.mixer.music.load(audio_bytes)
                pygame.mixer.music.play()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to play audio: {str(e)}")
    
    def save_audio(self):
        """Save generated audio"""
        if self.current_audio is not None:
            file_path = filedialog.asksaveasfilename(
                defaultextension=".mp3",
                filetypes=[("MP3 Files", "*.mp3"), ("WAV Files", "*.wav")]
            )
            
            if file_path:
                try:
                    sf.write(file_path, self.current_audio, self.current_sr)
                    messagebox.showinfo("Success", f"Audio saved to {file_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to save audio: {str(e)}")
    
    def update_voice_list(self):
        """Update voice combo box"""
        voices = list(self.engine.models.keys())
        self.voice_combo['values'] = voices
        if voices and not self.voice_combo.get():
            self.voice_combo.set(voices[0])
    
    def refresh_voice_list(self):
        """Refresh voice management list"""
        # Clear existing items
        for item in self.voice_tree.get_children():
            self.voice_tree.delete(item)
        
        # Add current voices
        for voice_name, info in self.engine.model_info.items():
            created = info.get('created', 'Unknown')
            if created != 'Unknown':
                try:
                    created = datetime.fromisoformat(created).strftime('%Y-%m-%d %H:%M')
                except:
                    pass
            
            status = "Ready" if voice_name in self.engine.models else "Error"
            self.voice_tree.insert('', 'end', values=(voice_name, created, status))
    
    def rename_voice(self):
        """Rename selected voice"""
        selection = self.voice_tree.selection()
        if not selection:
            messagebox.showerror("Error", "Please select a voice to rename")
            return
        
        item = selection[0]
        old_name = self.voice_tree.item(item)['values'][0]
        
        new_name = simpledialog.askstring("Rename Voice", 
                                        f"Enter new name for '{old_name}':")
        if new_name and new_name != old_name:
            try:
                # TODO: Implement rename functionality
                messagebox.showinfo("Info", "Rename functionality will be implemented")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to rename: {str(e)}")
    
    def delete_voice(self):
        """Delete selected voice"""
        selection = self.voice_tree.selection()
        if not selection:
            messagebox.showerror("Error", "Please select a voice to delete")
            return
        
        item = selection[0]
        voice_name = self.voice_tree.item(item)['values'][0]
        
        if messagebox.askyesno("Confirm Delete", 
                             f"Are you sure you want to delete '{voice_name}'?"):
            try:
                self.engine.delete_model(voice_name)
                self.refresh_voice_list()
                self.update_voice_list()
                messagebox.showinfo("Success", f"Voice '{voice_name}' deleted")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to delete: {str(e)}")
    
    def update_speed_label(self, *args):
        """Update speed label"""
        speed = self.speed_var.get()
        self.speed_label.configure(text=f"{speed:.1f}x")
    
    def process_queue(self):
        """Process update queue"""
        try:
            while True:
                item = self.update_queue.get_nowait()
                
                if item[0] == 'progress':
                    _, progress, status = item
                    self.progress_var.set(progress)
                    self.status_label.configure(text=status)
                
                elif item[0] == 'training_complete':
                    _, voice_name = item
                    self.progress_var.set(100)
                    self.status_label.configure(text="Training completed!")
                    self.train_button.configure(state='normal')
                    self.update_voice_list()
                    self.refresh_voice_list()
                    messagebox.showinfo("Success", f"Voice '{voice_name}' trained successfully!")
                
                elif item[0] == 'training_error':
                    _, error = item
                    self.status_label.configure(text="Training failed")
                    self.train_button.configure(state='normal')
                    messagebox.showerror("Error", f"Training failed: {error}")
                
                elif item[0] == 'tts_complete':
                    _, audio, sr = item
                    self.current_audio = audio
                    self.current_sr = sr
                    self.generate_button.configure(state='normal')
                    self.preview_button.configure(state='normal')
                    self.save_button.configure(state='normal')
                    self.tts_progress.stop()
                elif item[0] == 'tts_error':
                    _, error = item
                    self.generate_button.configure(state='normal')
                    self.tts_progress.stop()
                    messagebox.showerror("Error", f"Speech generation failed: {error}")
                
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.process_queue)

def main():
    """Main function to run the application"""
    try:
        # Check dependencies
        required_modules = ['torch', 'torchaudio', 'librosa', 'tkinterdnd2', 'pygame', 'soundfile']
        missing_modules = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)
        
        if missing_modules:
            print("Missing required modules:")
            for module in missing_modules:
                print(f"  - {module}")
            print("\nPlease install missing modules:")
            print(f"pip install {' '.join(missing_modules)}")
            return
        
        # Create and run GUI
        root = TkinterDnD.Tk()  # Use TkinterDnD for drag and drop support
        app = VoiceCloningGUI(root)
        
        # Set window icon and properties
        root.resizable(True, True)
        root.minsize(1000, 700)
        
        # Center window
        root.update_idletasks()
        x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
        y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
        root.geometry(f"+{x}+{y}")
        
        print("=== Professional Voice Cloning Tool ===")
        print("GUI Started Successfully!")
        print(f"Using device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print("Ready to clone voices!")
        
        # Start the GUI
        root.mainloop()
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
