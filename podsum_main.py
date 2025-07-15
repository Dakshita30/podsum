import whisper
import gradio as gr
from transformers import pipeline
from pydub import AudioSegment
from reportlab.pdfgen 
import canvas
from reportlab.lib.pagesizes import A4
import tempfile
import os

model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def make_pdf(text, filename, title):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y, title)
    y -= 30
    c.setFont("Helvetica", 12)
    for line in text.split('\n'):
        for wrap_line in [line[i:i + 100] for i in range(0, len(line), 100)]:
            c.drawString(50, y, wrap_line)
            y -= 20
            if y < 50:
                c.showPage()
                y = height - 50
                c.setFont("Helvetica", 12)
    c.save()

def transcribe_and_summarize(audio_path):
    try:
        print("🎧 Received audio path:", audio_path)

        if isinstance(audio_path, tuple):
            audio_path = audio_path[0]
        if not isinstance(audio_path, str) or not os.path.isfile(audio_path):
            raise ValueError("Invalid audio input: file not found.")

        print("🔄 Starting transcription...")
        audio = AudioSegment.from_file(audio_path)
        chunk_length_ms = 60 * 1000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        full_transcript = ""
        for i, chunk in enumerate(chunks):
            print(f"🔊 Transcribing chunk {i+1}/{len(chunks)}")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                chunk.export(temp_audio.name, format="wav")
                result = model.transcribe(temp_audio.name)
                full_transcript += result["text"].strip() + " "
                os.unlink(temp_audio.name)

        full_transcript = full_transcript.strip()
        print("✅ Finished transcription.")
        print("🧾 Transcript length (chars):", len(full_transcript))

        # ⛔ Cap long transcripts to avoid model overload
        capped_transcript = full_transcript[:4000]
        print("✂️ Starting summarization on capped text (max 4000 chars)")

        # Split into chunks
        max_chunk = 1000
        transcript_chunks = []
        text = capped_transcript
        while len(text) > max_chunk:
            split_idx = text[:max_chunk].rfind(".")
            if split_idx == -1:
                split_idx = max_chunk
            transcript_chunks.append(text[:split_idx + 1].strip())
            text = text[split_idx + 1:]
        if text.strip():
            transcript_chunks.append(text.strip())

        summary = ""
        for i, chunk in enumerate(transcript_chunks):
            print(f"📄 Summarizing chunk {i+1}/{len(transcript_chunks)}")
            summary += summarizer(chunk, max_length=130, min_length=30, do_sample=False)[0]["summary_text"] + " "

        print("✅ Finished summarization. Generating PDFs...")

        make_pdf(full_transcript, "transcript.pdf", "Transcript")
        make_pdf(summary.strip(), "summary.pdf", "Summary")

        print("🚀 All done. Returning results.")
        return full_transcript, summary.strip(), "transcript.pdf", "summary.pdf"

    except Exception as e:
        print("💥 Error during processing:", str(e))
        return f"🔥 Error: {str(e)}", "", None, None

interface = gr.Interface(
    fn=transcribe_and_summarize,
    inputs=gr.Audio(label="🎧 Upload Audio File", type="filepath"),
    outputs=[
        gr.Textbox(label="📝 Transcript"),
        gr.Textbox(label="✂️ Summary"),
        gr.File(label="📄 Download Transcript (.pdf)"),
        gr.File(label="📄 Download Summary (.pdf)")
    ],
    title="🎙️ PodSum: Audio Summarizer",
    description="Upload a podcast/audio file to get transcript + summary — with downloadable PDFs. Powered by Whisper + Transformers ✨"
)

interface.launch()
