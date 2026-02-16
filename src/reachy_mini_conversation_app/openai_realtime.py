import json
import base64
import random
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import gradio as gr
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample
from websockets.exceptions import ConnectionClosedError

from affect_engine import AffectEngine, AffectConfig
from reachy_mini_conversation_app.embodiment.eyes import EyesAdapter
from reachy_mini_conversation_app.embodiment.movement import MovementAdapter

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import (
    ToolDependencies,
    get_tool_specs,
    dispatch_tool_call,
)


logger = logging.getLogger(__name__)

OPEN_AI_INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OPEN_AI_OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000


class OpenaiRealtimeHandler(AsyncStreamHandler):
    """OpenAI realtime handler for fastrtc Stream with AffectEngine integration."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        debug: bool = False,
        instance_path: Optional[str] = None,
    ):
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OPEN_AI_OUTPUT_SAMPLE_RATE,
            input_sample_rate=OPEN_AI_INPUT_SAMPLE_RATE,
        )

        self.output_sample_rate = OPEN_AI_OUTPUT_SAMPLE_RATE
        self.input_sample_rate = OPEN_AI_INPUT_SAMPLE_RATE

        self.deps = deps
        self.connection: Any = None
        self.output_queue: asyncio.Queue[
            Tuple[int, NDArray[np.int16]] | AdditionalOutputs
        ] = asyncio.Queue()

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        self._shutdown_requested = False
        self._connected_event = asyncio.Event()
        
        self.debug = debug

        # -------------------------------
        # Affect Engine (internal only)
        # -------------------------------

        affect_config = AffectConfig(
            nrc_lexicon_path=config.AFFECT_ENGINE_DATA_PATH,
            debug=debug,
        )

        self.affect_engine = AffectEngine(affect_config)


    def copy(self) -> "OpenaiRealtimeHandler":
        return OpenaiRealtimeHandler(
        deps=self.deps,
        gradio_mode=self.gradio_mode,
        debug=self.debug,
        instance_path=self.instance_path,
        )


    # --------------------------------------------------
    # STARTUP
    # --------------------------------------------------

    async def start_up(self) -> None:
        openai_api_key = config.OPENAI_API_KEY

        if not openai_api_key or not openai_api_key.strip():
            logger.warning("OPENAI_API_KEY missing. Using placeholder.")
            openai_api_key = "DUMMY"

        self.client = AsyncOpenAI(api_key=openai_api_key)

        # Start AffectEngine
        await self.affect_engine.start()
        
        # Tap into movement 
        self.movement_adapter = MovementAdapter(self.deps.movement_manager)
        self.affect_engine.subscribe(self.movement_adapter.update)

        # Wire up eyes
        self.eyes_adapter = EyesAdapter()
        await self.eyes_adapter.connect()
        self.affect_engine.subscribe(self.eyes_adapter.update)

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_realtime_session()
                return
            except ConnectionClosedError as e:
                logger.warning(
                    "Realtime websocket closed (attempt %d/%d): %s",
                    attempt,
                    max_attempts,
                    e,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(2 ** (attempt - 1))
                    continue
                raise
            finally:
                self.connection = None
                self._connected_event.clear()

    # --------------------------------------------------
    # REALTIME SESSION
    # --------------------------------------------------

    async def _dispatch_tool(self, event) -> None:
        """Execute a tool call without blocking the audio event loop."""
        tool_name = getattr(event, "name", None)
        args_json_str = getattr(event, "arguments", None)
        call_id = getattr(event, "call_id", None)

        if not isinstance(tool_name, str):
            return

        try:
            tool_result = await dispatch_tool_call(
                tool_name,
                args_json_str,
                self.deps,
            )
        except Exception as e:
            tool_result = {"error": str(e)}

        if not isinstance(call_id, str):
            return

        try:
            await self.connection.conversation.item.create(
                item={
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": json.dumps(tool_result),
                },
            )

            await self.connection.response.create(
                response={
                    "instructions": "Use the tool result and answer concisely in speech.",
                },
            )
        except Exception as e:
            logger.exception("Tool follow-up failed for %s: %s", tool_name, e)



    async def _run_realtime_session(self) -> None:
        async with self.client.realtime.connect(model=config.MODEL_NAME) as conn:
            await conn.session.update(
                session={
                    "type": "realtime",
                    "instructions": get_session_instructions(),
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": self.input_sample_rate},
                            "transcription": {
                                "model": "gpt-4o-transcribe",
                                "language": "en",
                            },
                            "turn_detection": {
                                "type": "server_vad",
                                "interrupt_response": True,
                            },
                        },
                        "output": {
                            "format": {"type": "audio/pcm", "rate": self.output_sample_rate},
                            "voice": get_session_voice(),
                        },
                    },
                    "tools": get_tool_specs(),
                    "tool_choice": "auto",
                },
            )

            logger.info("Realtime session ready.")
            self.connection = conn
            self._connected_event.set()

            async for event in self.connection:

                # ---------------------------------------
                # LISTENING STATE
                # ---------------------------------------

                if event.type == "input_audio_buffer.speech_started":
                    if self.deps.head_wobbler:
                        self.deps.head_wobbler.reset()
                        self.deps.movement_manager.set_listening(True)

                if event.type == "input_audio_buffer.speech_stopped":
                    self.deps.movement_manager.set_listening(False)


                # ---------------------------------------
                # USER TRANSCRIPTS (DO NOT FEED ENGINE)
                # ---------------------------------------

                if event.type == "conversation.item.input_audio_transcription.completed":
                    await self.output_queue.put(
                        AdditionalOutputs(
                            {"role": "user", "content": event.transcript}
                        )
                    )

                # ---------------------------------------
                # ASSISTANT TRANSCRIPTS (FEED ENGINE)
                # ---------------------------------------

                if event.type in (
                    "response.audio_transcript.done",
                    "response.output_audio_transcript.done",
                ):
                    transcript = event.transcript

                    logger.debug(
                    "AFFECT TRIGGER event=%s len=%d hash=%d",
                    event.type,
                    len(transcript),
                    hash(transcript),
                    )

                    asyncio.create_task(self._feed_affect(transcript))


                    await self.output_queue.put(
                        AdditionalOutputs(
                            {"role": "assistant", "content": transcript}
                        )
                    )

                    # Feed assistant speech into AffectEngine
                    asyncio.create_task(self._feed_affect(transcript))


                # ---------------------------------------
                # AUDIO DELTA
                # ---------------------------------------

                if event.type in ("response.audio.delta", "response.output_audio.delta"):

                    # Feed audio into head wobble (speech motion)
                    if self.deps.head_wobbler:
                        self.deps.head_wobbler.feed(event.delta)

                    self.last_activity_time = asyncio.get_event_loop().time()

                    await self.output_queue.put(
                        (
                            self.output_sample_rate,
                            np.frombuffer(
                            base64.b64decode(event.delta),
                            dtype=np.int16,
                            ).reshape(1, -1),
                        )
                    )


                # ---------------------------------------
                # TOOL CALLS
                # ---------------------------------------
                if event.type == "response.function_call_arguments.done":
                    asyncio.create_task(self._dispatch_tool(event))

                # ---------------------------------------
                # ERROR
                # ---------------------------------------

                if event.type == "error":
                    logger.error("Realtime error: %s", event)

    async def _feed_affect(self, transcript: str) -> None:
        try:
            t0 = time.perf_counter()

            logger.debug("AFFECT START hash=%d", hash(transcript))

            result = await self.affect_engine.process_text(transcript)

            dt = time.perf_counter() - t0

            logger.debug(
                "AFFECT DONE hash=%d latency=%.3fs vibe=%s",
                hash(transcript),
                dt,
                result.get("vibe"),
            )
        except Exception as e:
        	logger.exception("AffectEngine failed: %s", e)


    # --------------------------------------------------
    # AUDIO INPUT
    # --------------------------------------------------

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        if not self.connection:
            return

        input_sample_rate, audio_frame = frame

        if audio_frame.ndim == 2:
            audio_frame = audio_frame[:, 0]

        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(
                audio_frame,
                int(len(audio_frame) * self.input_sample_rate / input_sample_rate),
            )

        audio_frame = audio_to_int16(audio_frame)

        try:
            audio_message = base64.b64encode(
                audio_frame.tobytes()
            ).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_message)
        except Exception:
            pass

    # --------------------------------------------------
    # EMIT
    # --------------------------------------------------

    async def emit(
        self,
    ) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:

        return await wait_for_item(self.output_queue)

    # --------------------------------------------------
    # SHUTDOWN
    # --------------------------------------------------

    async def shutdown(self) -> None:
        self._shutdown_requested = True

        # Stop AffectEngine
        await self.affect_engine.stop()

        if self.connection:
            try:
                await self.connection.close()
            except Exception:
                pass

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Stop Eyes
        if hasattr(self, "eyes_adapter"):
            await self.eyes_adapter.close()


    # --------------------------------------------------
    # UTILS
    # --------------------------------------------------

    def format_timestamp(self) -> str:
        loop_time = asyncio.get_event_loop().time()
        elapsed = loop_time - self.start_time
        dt = datetime.now()
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed:.1f}s]"
        
    
