"""Bidirectional local audio stream.

records mic frames to the handler and plays handler audio frames to the speaker.
"""

import time
import asyncio
import logging
from typing import List

import math
import numpy as np
from scipy.signal import resample_poly

from fastrtc import AdditionalOutputs, audio_to_int16, audio_to_float32

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.providers.openai_realtime import OpenAIRealtimeProvider

logger = logging.getLogger(__name__)


def resample_int16_mono(x_int16: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """High-quality mono PCM16 resampler using polyphase filtering."""
    if sr_in == sr_out or x_int16.size == 0:
        return x_int16
    x = x_int16.astype(np.float32) / 32768.0
    g = math.gcd(sr_in, sr_out)
    up, down = sr_out // g, sr_in // g
    y = resample_poly(x, up, down)
    return np.clip(np.round(y * 32768.0), -32768, 32767).astype(np.int16)
    

class LocalStream:
    """LocalStream using Reachy Mini's recorder/player."""

    def __init__(self, handler: OpenAIRealtimeProvider, robot: ReachyMini):
        """Initialize the stream with an OpenAI realtime handler and pipelines."""
        self.handler = handler
        self._robot = robot
        self._stop_event = asyncio.Event()
        self._tasks: List[asyncio.Task[None]] = []
        # Allow the handler to flush the player queue when appropriate.
        self.handler._clear_queue = self.clear_audio_queue

    def launch(self) -> None:
        """Start the recorder/player and run the async processing loops."""
        self._stop_event.clear()
        self._robot.media.start_recording()
        self._robot.media.start_playing()
        time.sleep(1)  # give some time to the pipelines to start

        async def runner() -> None:
            self._tasks = [
                asyncio.create_task(self.handler.start_up(), name="openai-handler"),
                asyncio.create_task(self.record_loop(), name="stream-record-loop"),
                asyncio.create_task(self.play_loop(), name="stream-play-loop"),
            ]
            try:
                await asyncio.gather(*self._tasks)
            except asyncio.CancelledError:
                logger.info("Tasks cancelled during shutdown")
            finally:
                # Ensure handler connection is closed
                await self.handler.shutdown()

        asyncio.run(runner())

    def close(self) -> None:
        """Stop the stream and underlying media pipelines.

        This method:
        - Sets the stop event to signal async loops to terminate
        - Cancels all pending async tasks (openai-handler, record-loop, play-loop)
        - Stops audio recording and playback
        """
        logger.info("Stopping LocalStream...")
        self._stop_event.set()

        # Cancel all running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        self._robot.media.stop_recording()
        self._robot.media.stop_playing()

    def clear_audio_queue(self) -> None:
        logger.info("User intervention: flushing player queue")
        q = self.handler.output_queue
        try:
            while True:
                q.get_nowait()
                q.task_done()
        except asyncio.QueueEmpty:
            pass


    async def record_loop(self) -> None:
        """Read mic frames from the recorder and forward them to the handler."""
        logger.info("Starting receive loop")
        mic_sr = getattr(self._robot.media, "SAMPLE_RATE", 16000)  # fallback to 16k
        while not self._stop_event.is_set():
            audio_frame = self._robot.media.get_audio_sample()
            if audio_frame is not None:
                frame_mono = audio_frame.T[0]  # device provides mono; safe
                frame = audio_to_int16(frame_mono)
                await self.handler.receive((mic_sr, frame))  # CHANGED: use mic_sr
            await asyncio.sleep(0.01)

    async def play_loop(self) -> None:
        """Fetch outputs from the handler: log text and play audio frames."""
        while not self._stop_event.is_set():
            handler_output = await self.handler.emit()

            if isinstance(handler_output, AdditionalOutputs):
                for msg in handler_output.args:
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        logger.info(
                            "role=%s content=%s",
                            msg.get("role"),
                            content if len(content) < 500 else content[:500] + "…",
                        )

            elif isinstance(handler_output, tuple):
                input_sample_rate, audio_frame = handler_output
                device_sample_rate = self._robot.media.get_audio_samplerate()

                # Normalize to int16 mono 1-D no matter what came in
                mono_int16 = np.asarray(audio_frame, dtype=np.int16).reshape(-1)
                if input_sample_rate != device_sample_rate:
                    mono_int16 = resample_int16_mono(mono_int16, input_sample_rate, device_sample_rate)

                audio_frame_float = audio_to_float32(mono_int16)  # -> float32 [-1, 1]
                
                if not hasattr(self, "_logged_play_stats"):
                    self._logged_play_stats = True
                    logger.debug("play: sr=%s len=%s dtype=%s min=%.3f max=%.3f",
                                 device_sample_rate,
                                 audio_frame_float.size,
                                 audio_frame_float.dtype,
                                 float(audio_frame_float.min()),
                                 float(audio_frame_float.max()))

                # device expects 1-D mono float32
                self._robot.media.push_audio_sample(audio_frame_float)

            else:
                logger.debug("Ignoring output type=%s", type(handler_output).__name__)

            await asyncio.sleep(0)  # yield to event loop
