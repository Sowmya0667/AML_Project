import io
import logging
import speech_recognition as sr
from groq import Groq
from config import config

log = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self):
        self.groq_api_key = config.GROQ_API_KEY
        # Initialize Groq client only if key is available
        self.client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None
        self.recognizer = sr.Recognizer()

    def transcribe(self, audio_bytes: bytes) -> str:
        """
        Transcribes audio bytes to text.
        Attempts Groq Whisper API first. Falls back to Google's free STT.
        """
        if not audio_bytes:
            return ""

        # Try Groq Whisper API
        if self.client:
            try:
                log.info("Attempting transcription via Groq Whisper API...")
                # Groq requires a tuple: (filename, audio_bytes, content_type)
                file_tuple = ("audio.wav", audio_bytes, "audio/wav")
                completion = self.client.audio.transcriptions.create(
                    model="whisper-large-v3",
                    file=file_tuple,
                    response_format="text"
                )
                text = str(completion).strip()
                log.info(f"Groq Transcription successful.")
                return text
            except Exception as e:
                log.warning(f"Groq Whisper failed: {e}. Falling back to Google STT.")
        else:
            log.info("No Groq API key found. Using Google STT fallback.")

        # Fallback to Google STT (SpeechRecognition)
        try:
            log.info("Attempting transcription via Google Web Speech API...")
            audio_file = io.BytesIO(audio_bytes)
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio_data)
            log.info(f"Google Transcription successful.")
            return text
        except sr.UnknownValueError:
            log.warning("Google STT could not understand the audio.")
            return ""
        except sr.RequestError as e:
            log.error(f"Could not request results from Google STT service; {e}")
            return ""
        except Exception as e:
            log.error(f"Fallback transcription failed: {e}")
            return ""
