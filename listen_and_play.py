import socket
import threading
from queue import Queue
from dataclasses import dataclass, field
import sounddevice as sd
from transformers import HfArgumentParser
import os
import wave
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import io


@dataclass
class ListenAndPlayArguments:
    send_rate: int = field(default=16000, metadata={"help": "In Hz. Default is 16000."})
    recv_rate: int = field(default=16000, metadata={"help": "In Hz. Default is 16000."})
    list_play_chunk_size: int = field(
        default=1024,
        metadata={"help": "The size of data chunks (in bytes). Default is 1024."},
    )
    host: str = field(
        default="localhost",
        metadata={
            "help": "The hostname or IP address for listening and playing. Default is 'localhost'."
        },
    )
    send_port: int = field(
        default=12345,
        metadata={"help": "The network port for sending data. Default is 12345."},
    )
    recv_port: int = field(
        default=12346,
        metadata={"help": "The network port for receiving data. Default is 12346."},
    )
    debug_audio: bool = field(
        default=False,
        metadata={"help": "Enable audio file debug mode. Default is False."},
    )
    debug_audio_listen: bool = field(
        default=False,
        metadata={"help": "Enable audio file debug mode. Default is False."},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Enable verbose output. Default is False."},
    )


def listen_and_play(
    send_rate=16000,
    recv_rate=44100,
    list_play_chunk_size=1024,
    host="localhost",
    send_port=12345,
    recv_port=12346,
    debug_audio=False,
    debug_audio_listen=False,
    verbose=False,
):
    send_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    send_socket.connect((host, send_port))

    recv_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    recv_socket.connect((host, recv_port))

    print("Recording and streaming...")

    stop_event = threading.Event()
    recv_queue = Queue()
    send_queue = Queue()
    file_change_event = threading.Event()

    def callback_send(indata, frames, time, status):
        if status:
            print(status)
        if recv_queue.empty():
            data = bytes(indata)
            send_queue.put(data)

    def callback_recv(outdata, frames, time, status):
        if not recv_queue.empty():
            data = recv_queue.get()
            # Convert byte data to a NumPy array and reshape it accordingly
            audio_data = np.frombuffer(data, dtype=np.int16)
            outdata[: len(audio_data)] = audio_data.reshape(-1, 1)
            outdata[len(audio_data) :] = 0  # Fill the rest with zeros if necessary
        else:
            outdata.fill(0)  # Fill the output with silence if there's no data

    def send(stop_event, send_queue):
        while not stop_event.is_set():
            data = send_queue.get()
            send_socket.sendall(data)

    def recv(stop_event, recv_queue):
        def receive_full_chunk(conn, chunk_size):
            data = b""
            while len(data) < chunk_size:
                packet = conn.recv(chunk_size - len(data))
                if not packet:
                    return None  # Connection has been closed
                data += packet
            return data

        while not stop_event.is_set():
            data = receive_full_chunk(recv_socket, list_play_chunk_size * 2)
            if data:
                recv_queue.put(data)

    def ask_for_audio_file():
        audio_samples_dir = "audio_samples"
        audio_files = [
            f for f in os.listdir(audio_samples_dir) if f.endswith((".mp3", ".wav"))
        ]
        print("Available audio files:")
        for i, file in enumerate(audio_files):
            print(f"{i + 1}. {file}")
        file_number = int(input("Enter the number of the audio file to use: ")) - 1
        audio_file_path = os.path.join(audio_samples_dir, audio_files[file_number])
        return audio_file_path

    def audio_input_stream(file_path, chunk_size, sample_rate):
        _, ext = os.path.splitext(file_path)
        if ext.lower() == ".mp3":
            try:
                with sf.SoundFile(file_path, samplerate=sample_rate) as sf_file:
                    if verbose:
                        print(
                            f"Successfully opened MP3 file with soundfile: {file_path}"
                        )
                    while True:
                        data = sf_file.read(chunk_size, dtype="int16")
                        if len(data) == 0:
                            break
                        if len(data) < chunk_size:
                            data = np.pad(data, (0, chunk_size - len(data)), "constant")
                        yield data
            except Exception as e:
                if verbose:
                    print(f"Error reading MP3 with soundfile: {e}")
                    print("Falling back to pydub for MP3 decoding")
                audio = AudioSegment.from_mp3(file_path)
                audio = audio.set_frame_rate(sample_rate)
                audio = audio.set_channels(1)
                samples = audio.get_array_of_samples()
                for i in range(0, len(samples), chunk_size):
                    chunk = samples[i : i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), "constant")
                    yield np.array(chunk, dtype=np.int16)
        else:
            with sf.SoundFile(file_path, samplerate=sample_rate) as sf_file:
                if verbose:
                    print(f"Successfully opened audio file with soundfile: {file_path}")
                while True:
                    data = sf_file.read(chunk_size, dtype="int16")
                    if len(data) == 0:
                        audio_file_path = ask_for_audio_file()
                        return audio_input_stream(
                            audio_file_path, list_play_chunk_size // 2, send_rate
                        )
                        # break

                    if len(data) < chunk_size:
                        data = np.pad(data, (0, chunk_size - len(data)), "constant")
                    yield data

    try:
        if debug_audio or debug_audio_listen:
            audio_file_path = ask_for_audio_file()

            audio_stream = audio_input_stream(
                audio_file_path, list_play_chunk_size // 2, send_rate
            )

            def debug_callback_send(outdata, frames, time, status):

                if status:
                    print(status)
                try:
                    data = next(audio_stream)
                    outdata[:] = data.reshape(-1, 1)
                    send_queue.put(bytes(data))
                except StopIteration:
                    raise sd.CallbackStop()

            if debug_audio_listen:
                send_stream = sd.OutputStream(
                    samplerate=send_rate,
                    channels=1,
                    callback=debug_callback_send,
                    blocksize=list_play_chunk_size // 2,
                    dtype="int16",
                )
            else:
                send_stream = sd.InputStream(
                    samplerate=send_rate,
                    channels=1,
                    callback=debug_callback_send,
                    blocksize=list_play_chunk_size // 2,
                    dtype="int16",
                )
        else:
            send_stream = sd.InputStream(
                samplerate=send_rate,
                channels=1,
                dtype="int16",
                blocksize=list_play_chunk_size,
                callback=callback_send,
            )

        recv_stream = sd.OutputStream(
            samplerate=recv_rate,
            channels=1,
            dtype="int16",
            blocksize=list_play_chunk_size,
            callback=callback_recv,
        )

        with send_stream, recv_stream:
            print("Streams started. Press Enter to stop...")
            send_thread = threading.Thread(target=send, args=(stop_event, send_queue))
            recv_thread = threading.Thread(target=recv, args=(stop_event, recv_queue))
            send_thread.start()
            recv_thread.start()

            input()

    except KeyboardInterrupt:
        print("Finished streaming.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        stop_event.set()
        if "send_thread" in locals():
            send_thread.join()
        if "recv_thread" in locals():
            recv_thread.join()
        send_socket.close()
        recv_socket.close()
        print("Connection closed.")


if __name__ == "__main__":
    parser = HfArgumentParser((ListenAndPlayArguments,))
    (listen_and_play_kwargs,) = parser.parse_args_into_dataclasses()
    listen_and_play(**vars(listen_and_play_kwargs))
