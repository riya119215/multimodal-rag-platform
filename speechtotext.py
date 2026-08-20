import os
import json
import argparse
import whisper

def transcribe_single_audio(audio_path: str, output_path: str = "output.json", model_name: str = "base", language: str = "hi", task: str = "translate"):
    """Transcribe a single audio file and save its segment chunks."""
    if not os.path.exists(audio_path):
        print(f"[!] Audio file not found: {audio_path}")
        return

    print(f"[*] Loading Whisper model '{model_name}'...")
    model = whisper.load_model(model_name)
    
    print(f"[*] Transcribing '{audio_path}' (language={language}, task={task})...")
    result = model.transcribe(audio=audio_path, language=language, task=task)
    
    chunks = []
    for idx, segment in enumerate(result.get("segments", [])):
        chunks.append({
            "chunk_id": idx,
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "text": segment["text"].strip()
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"file": audio_path, "chunks": chunks, "text": result.get("text", "")}, f, ensure_ascii=False, indent=2)
        
    print(f"[+] Saved {len(chunks)} chunks to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe a single audio file.")
    parser.add_argument("--audio", type=str, default="audios/1_Let's create a graph using Pandas and Matplotlib.mp3", help="Path to audio file")
    parser.add_argument("--output", type=str, default="output.json", help="Path to output JSON")
    args = parser.parse_args()
    
    transcribe_single_audio(args.audio, args.output)
