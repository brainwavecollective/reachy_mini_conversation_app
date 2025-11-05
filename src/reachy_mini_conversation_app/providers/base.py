"""Base class for conversation providers."""

from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np
from numpy.typing import NDArray
from fastrtc import AsyncStreamHandler


class BaseConversationProvider(AsyncStreamHandler, ABC):
    """Minimal base for conversation providers.
    
    Each provider must implement:
    - get_output_sample_rate(): TTS output rate
    - get_input_sample_rate(): Mic input rate  
    - start_up(): Initialize connections
    - receive(): Handle incoming mic audio
    - emit(): Return audio/text to play
    - shutdown(): Clean up
    """
    
    def __init__(self, deps, **kwargs):
        """Initialize with tool dependencies.
        
        Args:
            deps: ToolDependencies instance with robot, camera, etc.
            **kwargs: Additional provider-specific config
        """
        super().__init__(
            expected_layout="mono",
            output_sample_rate=self.get_output_sample_rate(),
            input_sample_rate=self.get_input_sample_rate(),
        )
        self.deps = deps
    
    @abstractmethod
    def get_output_sample_rate(self) -> int:
        """Return the sample rate for TTS output audio.
        
        Returns:
            int: Sample rate in Hz (e.g., 24000)
        """
        pass
    
    @abstractmethod
    def get_input_sample_rate(self) -> int:
        """Return the sample rate for microphone input.
        
        Returns:
            int: Sample rate in Hz (e.g., 16000)
        """
        pass
    
    @abstractmethod
    async def start_up(self) -> None:
        """Initialize the provider.
        
        Connect to APIs, load models, start background tasks, etc.
        Called once when the stream starts.
        """
        pass
    
    @abstractmethod
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Handle incoming microphone audio frame.
        
        Args:
            frame: Tuple of (sample_rate, audio_array)
                   audio_array is int16 mono audio
        """
        pass
    
    @abstractmethod
    async def emit(self) -> Tuple[int, NDArray[np.int16]] | Any | None:
        """Return next audio frame or UI update to play/display.
        
        Returns:
            - Tuple[int, NDArray[np.int16]]: Audio to play (rate, data)
            - AdditionalOutputs: UI updates (transcripts, images, etc.)
            - None: Nothing to emit right now
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Clean up the provider.
        
        Close connections, stop background tasks, clear queues, etc.
        Called once when the stream stops.
        """
        pass
    
    def copy(self) -> "BaseConversationProvider":
        """Create a copy of the handler for new sessions.
        
        Override this if your provider needs special copy logic.
        Default implementation creates a new instance with same deps.
        """
        return self.__class__(self.deps)
