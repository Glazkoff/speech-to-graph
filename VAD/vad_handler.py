import torchaudio
from VAD.vad_iterator import VADIterator
from baseHandler import BaseHandler
import numpy as np
import torch
from rich.console import Console

from utils.utils import int2float
from df.enhance import enhance, init_df
import logging

logger = logging.getLogger(__name__)

console = Console()


class VADHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        should_speak,
        thresh=0.3,
        sample_rate=16000,
        min_silence_ms=1000,
        min_speech_ms=200,
        max_speech_ms=float("inf"),
        speech_pad_ms=30,
        audio_enhancement=False,
    ):
        self.should_listen = should_listen
        self.should_speak = should_speak
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.audio_enhancement = audio_enhancement
        if audio_enhancement:
            self.enhanced_model, self.df_state, _ = init_df()

        self.is_speaking = False
        self.buffer = []

    def process(self, audio_chunk):
        audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
        audio_float32 = int2float(audio_int16)
        status, audio = self.iterator(torch.from_numpy(audio_float32))

        if status == "start":
            logger.debug("VAD: start of speech detected")
            self.is_speaking = True
            self.should_speak.clear()
            self.buffer = [audio.numpy()]
        elif status == "continue":
            if self.is_speaking:
                self.buffer.append(audio.numpy())
        elif status == "end":
            if self.is_speaking:
                logger.debug("VAD: end of speech detected")
                self.is_speaking = False
                self.buffer.append(audio.numpy())
                array = np.concatenate(self.buffer)
                duration_ms = len(array) / self.sample_rate * 1000

                if self.min_speech_ms <= duration_ms <= self.max_speech_ms:
                    logger.debug(f"Speech duration: {duration_ms:.2f}ms")
                    self.should_speak.set()

                    if self.audio_enhancement:
                        array = self._enhance_audio(array)

                    yield array
                else:
                    logger.debug(
                        f"Speech duration {duration_ms:.2f}ms out of range, skipping"
                    )

                self.buffer = []

    def _enhance_audio(self, audio):
        if self.sample_rate != self.df_state.sr():
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio),
                orig_freq=self.sample_rate,
                new_freq=self.df_state.sr(),
            )

        enhanced = enhance(self.enhanced_model, self.df_state, audio.unsqueeze(0))

        if self.sample_rate != self.df_state.sr():
            enhanced = torchaudio.functional.resample(
                enhanced,
                orig_freq=self.df_state.sr(),
                new_freq=self.sample_rate,
            )

        return enhanced.numpy().squeeze()

    @property
    def min_time_to_debug(self):
        return 0.00001
