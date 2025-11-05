"""OpenAI Realtime API provider for voice conversation."""

import json
import base64
import asyncio
import logging
from typing import Any, Tuple, Literal, cast
from datetime import datetime

import numpy as np
import gradio as gr
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, wait_for_item
from numpy.typing import NDArray

from .base import BaseConversationProvider

from reachy_mini_conversation_app.tools import (
    ALL_TOOL_SPECS,
    ToolDependencies,
    dispatch_tool_call,
)
from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import SESSION_INSTRUCTIONS


logger = logging.getLogger(__name__)


class OpenAIRealtimeProvider(BaseConversationProvider):
    """OpenAI Realtime API provider for voice conversation."""

    def __init__(self, deps: ToolDependencies):
        """Initialize the OpenAI provider."""
        super().__init__(deps)
        
        # OpenAI-specific type annotations
        self.output_sample_rate: Literal[24000]
        self.target_input_rate: Literal[24000] = 24000
        self.resample_ratio = self.target_input_rate / self.input_sample_rate
        
        self.connection: Any = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()
        
        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False
    
    def get_output_sample_rate(self) -> int:
        """OpenAI outputs 24kHz PCM."""
        return 24000
    
    def get_input_sample_rate(self) -> int:
        """ReSpeaker mic is 16kHz."""
        return 16000

    def resample_audio(self, audio: NDArray[np.int16]) -> NDArray[np.int16]:
        """Resample audio using linear interpolation."""
        if self.input_sample_rate == self.target_input_rate:
            return audio

        # Use numpy's interp for simple linear resampling
        input_length = len(audio)
        output_length = int(input_length * self.resample_ratio)

        input_time = np.arange(input_length)
        output_time = np.linspace(0, input_length - 1, output_length)

        resampled = np.interp(output_time, input_time, audio.astype(np.float32))
        return cast(NDArray[np.int16], resampled.astype(np.int16))

    async def start_up(self) -> None:
        """Start the handler."""
        try:
            logger.info(f"Connecting to OpenAI Realtime API with model: {config.MODEL_NAME}")
            self.client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
            
            logger.debug("Establishing realtime connection...")
            async with self.client.realtime.connect(model=config.MODEL_NAME) as conn:
                logger.info("Connected to OpenAI Realtime API")
                
                try:
                    logger.debug("Updating session configuration...")
                    await conn.session.update(
                        session={
                            "type": "realtime",
                            "instructions": SESSION_INSTRUCTIONS,
                            "audio": {
                                "input": {
                                    "format": {
                                        "type": "audio/pcm",
                                        "rate": self.target_input_rate,
                                    },
                                    "transcription": {
                                        "model": "whisper-1",
                                        "language": "en"
                                    },
                                    "turn_detection": {
                                        "type": "server_vad",
                                        "interrupt_response": True,
                                    },
                                },
                                "output": {
                                    "format": {
                                        "type": "audio/pcm",
                                        "rate": self.output_sample_rate,
                                    },
                                    "voice": "cedar",
                                },
                            },
                            "tools": ALL_TOOL_SPECS,  # type: ignore[typeddict-item]
                            "tool_choice": "auto",
                        },
                    )
                    logger.info("Realtime session updated successfully")
                except Exception as e:
                    logger.exception("Realtime session.update failed")
                    raise RuntimeError(f"Failed to configure OpenAI session: {e}") from e

                # Manage events received from the openai server
                self.connection = conn
                logger.info("Starting event loop...")
                async for event in self.connection:
                    logger.debug(f"OpenAI event: {event.type}")
                    
                    # Handle speech detection events
                    if event.type == "input_audio_buffer.speech_started":
                        if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                            self._clear_queue()
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.reset()
                        self.deps.movement_manager.set_listening(True)
                        logger.debug("User speech started")

                    if event.type == "input_audio_buffer.speech_stopped":
                        self.deps.movement_manager.set_listening(False)
                        logger.debug("User speech stopped - server will auto-commit with VAD")

                    # Handle response completion events
                    if event.type in (
                        "response.audio.done",            # GA
                        "response.output_audio.done",     # GA alias
                        "response.audio.completed",       # legacy (for safety)
                        "response.completed",             # text-only completion
                    ):
                        logger.debug("response completed")

                    if event.type == "response.created":
                        logger.debug("Response created")

                    if event.type == "response.done":
                        # Doesn't mean the audio is done playing
                        logger.debug("Response done")

                    # Handle partial transcription (user speaking in real-time)
                    if event.type == "conversation.item.input_audio_transcription.partial":
                        logger.debug(f"User partial transcript: {event.transcript}")
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "user_partial", "content": event.transcript})
                        )

                    # Handle completed transcription (user finished speaking)
                    if event.type == "conversation.item.input_audio_transcription.completed":
                        logger.debug(f"User transcript: {event.transcript}")
                        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": event.transcript}))

                    # Handle transcription failures with detailed error messages
                    if event.type == "conversation.item.input_audio_transcription.failed":
                        error_details = getattr(event, 'error', {})
                        error_code = error_details.get('code', 'unknown') if isinstance(error_details, dict) else 'unknown'
                        error_message = error_details.get('message', str(error_details)) if isinstance(error_details, dict) else str(error_details)
                        
                        logger.error("=" * 60)
                        logger.error("❌ OPENAI TRANSCRIPTION FAILED")
                        logger.error("=" * 60)
                        logger.error(f"Error code: {error_code}")
                        logger.error(f"Error message: {error_message}")
                        
                        # Diagnose common issues
                        error_str = str(error_message).lower()
                        if "insufficient" in error_str or "quota" in error_str or "billing" in error_str:
                            logger.error("")
                            logger.error("💳 BILLING/CREDITS ISSUE DETECTED")
                            logger.error("   → Check: https://platform.openai.com/account/billing")
                            logger.error("   → You may be out of credits")
                            logger.error("   → Add payment method or purchase credits")
                        elif "rate" in error_str and "limit" in error_str:
                            logger.error("")
                            logger.error("⏱️  RATE LIMIT EXCEEDED")
                            logger.error("   → Slow down requests or upgrade tier")
                        else:
                            logger.error("")
                            logger.error("🎤 AUDIO PROCESSING ERROR")
                            logger.error("   → Audio format may be incorrect")
                            logger.error("   → Check microphone is working")
                        
                        logger.error("=" * 60)

                    # Handle assistant transcription
                    if event.type in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
                        logger.debug(f"Assistant transcript: {event.transcript}")
                        await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": event.transcript}))

                    # Handle audio delta (critical for audio playback!)
                    if event.type in ("response.audio.delta", "response.output_audio.delta"):
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.feed(event.delta)
                        self.last_activity_time = asyncio.get_event_loop().time()
                        logger.debug("last activity time updated to %s", self.last_activity_time)
                        await self.output_queue.put(
                            (
                                self.output_sample_rate,
                                np.frombuffer(base64.b64decode(event.delta), dtype=np.int16).reshape(1, -1),
                            ),
                        )

                    # ---- tool-calling plumbing ----
                    if event.type == "response.function_call_arguments.done":
                        tool_name = getattr(event, "name", None)
                        args_json_str = getattr(event, "arguments", None)
                        call_id = getattr(event, "call_id", None)

                        if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
                            logger.error("Invalid tool call: tool_name=%s, args=%s", tool_name, args_json_str)
                            continue

                        try:
                            tool_result = await dispatch_tool_call(tool_name, args_json_str, self.deps)
                            logger.debug("Tool '%s' executed successfully", tool_name)
                            logger.debug("Tool result: %s", tool_result)
                        except Exception as e:
                            logger.error("Tool '%s' failed", tool_name)
                            tool_result = {"error": str(e)}

                        # send the tool result back
                        if isinstance(call_id, str):
                            await self.connection.conversation.item.create(
                                item={
                                    "type": "function_call_output",
                                    "call_id": call_id,
                                    "output": json.dumps(tool_result),
                                },
                            )

                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "role": "assistant",
                                    "content": json.dumps(tool_result),
                                    "metadata": {"title": f"🛠️ Used tool {tool_name}", "status": "done"},
                                },
                            ),
                        )

                        if tool_name == "camera" and "b64_im" in tool_result:
                            # use raw base64, don't json.dumps (which adds quotes)
                            b64_im = tool_result["b64_im"]
                            if not isinstance(b64_im, str):
                                logger.warning("Unexpected type for b64_im: %s", type(b64_im))
                                b64_im = str(b64_im)
                            await self.connection.conversation.item.create(
                                item={
                                    "type": "message",
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "input_image",
                                            "image_url": f"data:image/jpeg;base64,{b64_im}",
                                        },
                                    ],
                                },
                            )
                            logger.info("Added camera image to conversation")

                            if self.deps.camera_worker is not None:
                                np_img = self.deps.camera_worker.get_latest_frame()
                                img = gr.Image(value=np_img)

                                await self.output_queue.put(
                                    AdditionalOutputs(
                                        {
                                            "role": "assistant",
                                            "content": img,
                                        },
                                    ),
                                )

                        # if this tool call was triggered by an idle signal, don't make the robot speak
                        # for other tool calls, let the robot reply out loud
                        if self.is_idle_tool_call:
                            self.is_idle_tool_call = False
                        else:
                            await self.connection.response.create(
                                response={
                                    "instructions": "Use the tool result just returned and answer concisely in speech.",
                                },
                            )

                        # re synchronize the head wobble after a tool call that may have taken some time
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.reset()

                    # server error
                    if event.type == "error":
                        err = getattr(event, "error", None)
                        msg = getattr(err, "message", str(err) if err else "unknown error")
                        code = getattr(err, "code", "")

                        logger.error("Realtime error [%s]: %s (raw=%s)", code, msg, err)

                        # Only show user-facing errors, not internal state errors
                        if code not in ("input_audio_buffer_commit_empty", "conversation_already_has_active_response"):
                            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"}))
                    
        except Exception as e:
            logger.error("=" * 60)
            logger.error("FAILED TO START OPENAI REALTIME PROVIDER")
            logger.error("=" * 60)
            logger.exception(f"Error: {e}")
            
            # Check common issues
            if "401" in str(e) or "authentication" in str(e).lower():
                logger.error("❌ Authentication failed - check your OPENAI_API_KEY in .env")
                logger.error(f"   Current key starts with: {config.OPENAI_API_KEY[:10]}..." if config.OPENAI_API_KEY else "   No API key set!")
            elif "404" in str(e) or "not found" in str(e).lower():
                logger.error(f"❌ Model '{config.MODEL_NAME}' not found")
                logger.error("   Make sure you have access to the realtime API")
            elif "connection" in str(e).lower() or "timeout" in str(e).lower():
                logger.error("❌ Connection failed - check your internet connection")
            elif "429" in str(error_message) or ("rate" in error_str and "limit" in error_str):
                logger.error("")
                logger.error("⏱️  RATE LIMIT EXCEEDED (429)")
                logger.error("   → You're sending too many requests")
                logger.error("   → Wait a moment and try again")
                logger.error("   → Or upgrade to a higher tier")
            
            logger.error("=" * 60)
            raise  # Re-raise so the app knows it failed
            
    # Microphone receive
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and send it to the openai server."""
        if not self.connection:
            return
        _, array = frame
        array = array.squeeze()

        # Resample if needed
        if self.input_sample_rate != self.target_input_rate:
            array = self.resample_audio(array)

        audio_message = base64.b64encode(array.tobytes()).decode("utf-8")
        await self.connection.input_audio_buffer.append(audio=audio_message)

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # sends to the stream the stuff put in the output queue by the openai event handler
        # This is called periodically by the fastrtc Stream

        # Handle idle
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 15.0 and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle signal skipped (connection closed?): %s", e)
                return None

            self.last_activity_time = asyncio.get_event_loop().time()  # avoid repeated resets

        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        if self.connection:
            await self.connection.close()
            self.connection = None

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()  # monotonic
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()  # wall-clock
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle signal to the openai server."""
        logger.debug("Sending idle signal")
        self.is_idle_tool_call = True
        timestamp_msg = f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] You've been idle for a while. Feel free to get creative - dance, show an emotion, look around, do nothing, or just be yourself!"
        if not self.connection:
            logger.debug("No connection, cannot send idle signal")
            return
        await self.connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": timestamp_msg}],
            },
        )
        await self.connection.response.create(
            response={
                "instructions": "You MUST respond with function calls only - no speech or text. Choose appropriate actions for idle behavior.",
                "tool_choice": "required",
            },
        )