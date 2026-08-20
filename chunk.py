import os
import json
import argparse
import whisper

def format_timestamp(seconds: float) -> str:
    """Convert seconds to a human-readable MM:SS or HH:MM:SS format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"

def parse_audio_name(filename: str):
    """Safely extract video number and title from audio filename."""
    name_without_ext = os.path.splitext(filename)[0]
    if "_" in name_without_ext:
        parts = name_without_ext.split("_", 1)
        number = parts[0].strip()
        title = parts[1].strip()
    else:
        number = "N/A"
        title = name_without_ext.strip()
    return number, title

def process_audios(audio_dir: str = "audios", json_dir: str = "jsons", model_name: str = "base", language: str = "hi", task: str = "translate"):
    """
    Transcribe and translate audio files into timestamped JSON chunks using OpenAI Whisper.
    """
    os.makedirs(json_dir, exist_ok=True)
    
    if not os.path.exists(audio_dir):
        print(f"[!] Audio directory '{audio_dir}' not found.")
        return

    audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith((".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"))]
    
    if not audio_files:
        print(f"[!] No audio files found in '{audio_dir}'.")
        return

    print(f"[*] Loading Whisper model: '{model_name}'...")
    model = whisper.load_model(model_name)
    print(f"[+] Whisper model loaded successfully. Processing {len(audio_files)} audio files...\n")

    for i, audio in enumerate(audio_files, 1):
        audio_path = os.path.join(audio_dir, audio)
        number, title = parse_audio_name(audio)
        
        print(f"[{i}/{len(audio_files)}] Transcribing: {audio}")
        print(f"    ↳ Video ID: {number} | Title: {title}")

        try:
            result = model.transcribe(
                audio=audio_path,
                language=language,
                task=task,
                word_timestamps=False,
            )

            chunks = []
            for chunk_idx, segment in enumerate(result.get("segments", [])):
                chunks.append({
                    "chunk_id": chunk_idx,
                    "number": number,
                    "title": title,
                    "start": round(segment["start"], 2),
                    "end": round(segment["end"], 2),
                    "start_formatted": format_timestamp(segment["start"]),
                    "end_formatted": format_timestamp(segment["end"]),
                    "text": segment["text"].strip(),
                })

            chunks_with_metadata = {
                "file_name": audio,
                "video_id": number,
                "title": title,
                "total_chunks": len(chunks),
                "full_text": result.get("text", "").strip(),
                "chunks": chunks
            }

            output_filename = os.path.splitext(audio)[0] + ".json"
            output_path = os.path.join(json_dir, output_filename)
            
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=2)

            print(f"    ✓ Saved {len(chunks)} chunks -> {output_path}\n")

        except Exception as e:
            print(f"    [!] Error processing {audio}: {e}\n")

    print("[+] Audio transcription & chunking complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio files into timestamped JSON chunks using Whisper.")
    parser.add_argument("--audio_dir", type=str, default="audios", help="Directory containing audio files")
    parser.add_argument("--json_dir", type=str, default="jsons", help="Directory to save JSON transcripts")
    parser.add_argument("--model", type=str, default="base", help="Whisper model size (tiny, base, small, medium, large)")
    parser.add_argument("--lang", type=str, default="hi", help="Audio source language")
    parser.add_argument("--task", type=str, default="translate", help="Whisper task: 'translate' to English or 'transcribe'")
    
    args = parser.parse_args()
    process_audios(
        audio_dir=args.audio_dir,
        json_dir=args.json_dir,
        model_name=args.model,
        language=args.lang,
        task=args.task
    )