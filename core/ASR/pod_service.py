#!/usr/bin/env python3
"""
ASR Microservice for Kubernetes Pod
Watches input directory for audio files and writes transcriptions to output directory
"""
import os
import sys
import time
from pathlib import Path
import torch
import fairseq
import librosa
import torch.nn.functional as F
import torchaudio.sox_effects as ta_sox
import json
from datetime import datetime
import fairseq.data.dictionary

torch.serialization.add_safe_globals([fairseq.data.dictionary.Dictionary])


class AudioToText:
    """Wav2Vec2 based Hindi speech-to-text transcriber"""
    
    DEFAULT_SAMPLING_RATE = int(os.getenv("DEFAULT_SAMPLING_RATE", "16000"))

    def __init__(self, model_path="hindi.pt", warmup_iterations=1, sample_audio_path="./samples/hindi.wav"):
        """Initialize the ASR model"""
        print(f"[{datetime.now()}] Loading model from {model_path}...")
        self.model, self.cfg, self.task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
        self.model = self.model[0]
        self.dtype = torch.float32
        self.model.to(self.dtype)
        self.model.eval()
        print(f"[{datetime.now()}] Model loaded successfully")

        self.effects = [["gain", "-n"]]
        self.token = self.task.target_dictionary
        self.warmup_audio_path = sample_audio_path
        
        print(f"[{datetime.now()}] Running {warmup_iterations} warmup iteration(s)...")
        self.warmup(warmup_iterations)
        print(f"[{datetime.now()}] Model ready for inference!")
    
    def warmup(self, warmup_iters: int):
        """Run warmup iterations"""
        for i in range(warmup_iters):
            with torch.no_grad():
                _ = self.transcribe(self.warmup_audio_path)
            print(f"[{datetime.now()}] Warmup {i+1}/{warmup_iters} complete")

    def transcribe(self, path, sample_rate=None):
        """
        Transcribe an audio file to text.
        
        Returns:
            transcriptions: List of transcribed text
            total_time: Time taken for inference in seconds
        """
        if sample_rate is None:
            sample_rate = self.DEFAULT_SAMPLING_RATE
            
        # Load audio
        audio, sr = librosa.load(path, sr=sample_rate)
        
        # Start timing
        st = time.perf_counter()
        
        # Apply effects
        input_sample, rate = ta_sox.apply_effects_tensor(
            torch.tensor(audio).unsqueeze(0), sample_rate, self.effects)
        input_sample = input_sample.to(self.dtype)
        
        # Normalize
        with torch.no_grad():
            input_sample = F.layer_norm(input_sample, input_sample.shape)

        # Get model predictions
        with torch.no_grad():
            logits = self.model(source=input_sample, padding_mask=None)['encoder_out']
        
        predicted_ids = torch.argmax(logits, axis=-1)
        predicted_ids = torch.unique_consecutive(predicted_ids.T, dim=1).tolist()

        # Convert token IDs to text
        transcriptions = []
        for ids in predicted_ids:
            transcription = self.token.string(ids)
            transcription = transcription.replace(' ', "").replace('|', " ").strip()
            transcriptions.append(transcription)
        
        total_time = time.perf_counter() - st
        
        return transcriptions, total_time


def main():
    """Main function for pod operation"""
    INPUT_DIR = Path("/app/input")
    OUTPUT_DIR = Path("/app/output")
    MODEL_PATH = "hindi.pt"
    
    # Create directories if they don't exist
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[{datetime.now()}] ASR Pod Service Starting")
    print(f"[{datetime.now()}] Input directory: {INPUT_DIR}")
    print(f"[{datetime.now()}] Output directory: {OUTPUT_DIR}")
    
    # Initialize model
    try:
        transcriber = AudioToText(
            model_path=MODEL_PATH,
            warmup_iterations=1,
            sample_audio_path="./samples/hindi.wav"
        )
    except Exception as e:
        print(f"[{datetime.now()}] ERROR: Failed to load model: {e}")
        sys.exit(1)
    
    print(f"[{datetime.now()}] Listening for audio files in {INPUT_DIR}...")
    processed_files = set()
    
    try:
        while True:
            # Check for new audio files
            audio_files = list(INPUT_DIR.glob("*.wav")) + list(INPUT_DIR.glob("*.mp3")) + list(INPUT_DIR.glob("*.flac"))
            
            for audio_file in audio_files:
                if str(audio_file) not in processed_files:
                    print(f"[{datetime.now()}] Processing {audio_file.name}...")
                    
                    try:
                        # Transcribe
                        transcriptions, inference_time = transcriber.transcribe(str(audio_file))
                        
                        # Prepare output
                        output_data = {
                            "timestamp": datetime.now().isoformat(),
                            "input_file": audio_file.name,
                            "transcription": transcriptions[0] if transcriptions else "",
                            "inference_time_seconds": inference_time,
                            "status": "success"
                        }
                        
                        # Write JSON output
                        output_file = OUTPUT_DIR / f"{audio_file.stem}_transcription.json"
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, ensure_ascii=False, indent=2)
                        
                        print(f"[{datetime.now()}] ✓ Transcription saved to {output_file.name}")
                        print(f"[{datetime.now()}] Text: {transcriptions[0][:100]}...")
                        print(f"[{datetime.now()}] Time: {inference_time:.4f}s")
                        
                        processed_files.add(str(audio_file))
                        
                    except Exception as e:
                        print(f"[{datetime.now()}] ERROR processing {audio_file.name}: {e}")
                        output_data = {
                            "timestamp": datetime.now().isoformat(),
                            "input_file": audio_file.name,
                            "error": str(e),
                            "status": "failed"
                        }
                        output_file = OUTPUT_DIR / f"{audio_file.stem}_error.json"
                        with open(output_file, 'w') as f:
                            json.dump(output_data, f, indent=2)
            
            # Sleep before next check
            time.sleep(5)
            
    except KeyboardInterrupt:
        print(f"\n[{datetime.now()}] Pod service shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"[{datetime.now()}] FATAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
