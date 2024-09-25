import torch
import logging
from TTS.api import TTS
from baseHandler import BaseHandler
from rich.console import Console
import numpy as np
import librosa

logger = logging.getLogger(__name__)

console = Console()

WHISPER_LANGUAGE_TO_XTTSv2_LANGUAGE = {
    "en": "EN",
    "ru": "RU",
}

WHISPER_LANGUAGE_TO_XTTSv2_SPEAKER = {
    "en": "EN",
    "ru": "RU",
}


class XTTSv2Handler(BaseHandler):

    def setup(
        self,
        should_listen,
        device="mps",
        language="ru",
        speaker_to_id="ru",
        gen_kwargs={},  # Unused
        blocksize=512,
    ):
        self.should_listen = should_listen
        self.device = device
        self.language = language
        cuda_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "Ftfyhh/xttsv2_banana/model_banana"
        self.model = TTS(self.model_id).to(cuda_device)
        # self.speaker_id = self.model.hps.data.spk2id[
        #     WHISPER_LANGUAGE_TO_XTTSv2_SPEAKER[speaker_to_id]
        # ]
        self.speaker_wav = "./audio_samples/audio5.ogg"
        self.blocksize = blocksize
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.tts_to_file(
            text="привет",
            language=self.language,
            speaker_wav=self.speaker_wav,
        )

    def process(self, llm_sentence):
        language_code = None

        if isinstance(llm_sentence, tuple):
            llm_sentence, language_code = llm_sentence

        console.print(f"[green]ASSISTANT: {llm_sentence}")

        if language_code is not None and self.language != language_code:
            try:
                # self.model = TTS(self.model_id).to(cuda_device)
                self.language = language_code
                _ = WHISPER_LANGUAGE_TO_XTTSv2_LANGUAGE[language_code]
            except KeyError:
                console.print(
                    f"[red]Language {language_code} not supported by XTTSv2. Using {self.language} instead."
                )

        if self.device == "mps":
            import time

            start = time.time()
            torch.mps.synchronize()  # Waits for all kernels in all streams on the MPS device to complete.
            torch.mps.empty_cache()  # Frees all memory allocated by the MPS device.
            _ = (
                time.time() - start
            )  # Removing this line makes it fail more often. I'm looking into it.

        try:
            console.print(
                f"[blue]{llm_sentence} - {self.language} - {self.speaker_wav}"
            )
            audio_chunk = self.model.tts(
                text=llm_sentence,
                language=self.language,
                speaker_wav=self.speaker_wav,
            )
        except (AssertionError, RuntimeError) as e:
            logger.error(f"Error in XTTSv2Handler: {e}")
            audio_chunk = np.array([])
        if len(audio_chunk) == 0:
            self.should_listen.set()
            return
        audio_chunk = np.array(audio_chunk)
        audio_chunk = librosa.resample(audio_chunk, orig_sr=24000, target_sr=16000)
        audio_chunk = (audio_chunk * 32768).astype(np.int16)
        for i in range(0, len(audio_chunk), self.blocksize):
            yield np.pad(
                audio_chunk[i : i + self.blocksize],
                (0, self.blocksize - len(audio_chunk[i : i + self.blocksize])),
            )

        self.should_listen.set()
