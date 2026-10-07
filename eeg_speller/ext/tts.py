"""TTS contract and no-audio placeholder."""


class StubTTS:
    """TTS extension stub; NEED-F-10, REQ-AUDIO-01; MA-32; Q-28/32."""
    def __init__(self, recorder, clock):
        self.recorder, self.clock = recorder, clock

    def speak(self, text, utterance_id):
        # ASSUMPTION[MA-32]
        # STUB[Q-28]
        received = self.clock.t_ns()
        self.recorder.emit("ext:tts", "tts_requested", {"utterance_id": utterance_id,
                           "text": text, "t_received_ns": received})
        # No tts_audio_started event: the stub cannot claim an audio signal exists.
