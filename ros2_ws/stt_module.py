#!/usr/bin/env python3
"""
Módulo de Speech-to-Text usando Whisper da OpenAI (modo offline)
Reconhece apenas inglês para comandos do robô
"""

import whisper
import sounddevice as sd
import numpy as np
import tempfile
import os
from scipy.io import wavfile

class STTModule:
    def __init__(self, model_name="base", language="en"):
        """
        Inicializa o módulo STT com Whisper
        
        Args:
            model_name: Modelo Whisper (tiny, base, small, medium, large)
                       - tiny: mais rápido, menos preciso
                       - base: bom equilíbrio (recomendado)
                       - small: mais preciso, mais lento
                       - medium: alta precisão, mais lento
                       - large: máxima precisão, mais lento
            language: Língua para reconhecimento (padrão: "en" - inglês)
        """
        try:
            self.model = whisper.load_model(model_name)
            self.language = language
            self.sample_rate = 16000  # Whisper usa 16kHz
            self.enabled = True
            print(f"Whisper STT initialized (model: {model_name})")
        except Exception as e:
            print(f"✗ Erro ao inicializar Whisper: {e}")
            self.enabled = False
    
    def record_audio(self, duration=5):
        """
        Grava áudio do microfone
        
        Args:
            duration: Duração da gravação em segundos
            
        Returns:
            numpy array com o áudio gravado
        """
        print(f"Recording for {duration} seconds... Speak now!")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Aguarda a gravação terminar
        print("Recording completed")
        return audio.flatten()
    
    def transcribe_audio(self, audio_data):
        """
        Transcreve áudio para texto usando Whisper
        
        Args:
            audio_data: numpy array com dados de áudio
            
        Returns:
            String com o texto transcrito
        """
        if not self.enabled:
            return None
        
        try:
            # Salvar áudio temporariamente
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_filename = tmp_file.name
                wavfile.write(tmp_filename, self.sample_rate, audio_data)
            
            # Transcrever usando Whisper
            print("Transcribing audio...")
            result = self.model.transcribe(
                tmp_filename,
                language=self.language,
                fp16=False  # Desabilitar FP16 para compatibilidade CPU
            )
            
            # Remover arquivo temporário
            os.remove(tmp_filename)
            
            transcribed_text = result["text"].strip()
            print(f"Your phrase: {transcribed_text}")
            return transcribed_text
            
        except Exception as e:
            print(f"Error in transcription: {e}")
            return None
    
    def listen_and_transcribe(self, duration=5):
        """
        Grava áudio e transcreve em uma única chamada
        
        Args:
            duration: Duração da gravação em segundos
            
        Returns:
            String com o texto transcrito ou None em caso de erro
        """
        if not self.enabled:
            print("STT não está habilitado!")
            return None
        
        audio = self.record_audio(duration)
        return self.transcribe_audio(audio)


# Teste do módulo
if __name__ == "__main__":
    print("=== Teste do Módulo STT ===\n")
    
    # Inicializar STT
    stt = STTModule(model_name="base", language="en")
    
    if stt.enabled:
        print("\nPressione Enter para começar a gravar (5 segundos)...")
        input()
        
        # Gravar e transcrever
        text = stt.listen_and_transcribe(duration=5)
        
        if text:
            print(f"\n📝 Resultado final: '{text}'")
        else:
            print("\n✗ Falha na transcrição")
    else:
        print("STT não pôde ser inicializado")
