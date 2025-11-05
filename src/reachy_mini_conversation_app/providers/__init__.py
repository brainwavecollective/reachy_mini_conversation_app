"""Conversation providers for Reachy Mini."""

from .base import BaseConversationProvider

__all__ = ["BaseConversationProvider", "create_provider"]


def create_provider(provider_name: str, deps) -> BaseConversationProvider:
    """Factory to create the appropriate conversation provider.
    
    Args:
        provider_name: Name of provider ("openai", "anthropic", etc.)
        deps: ToolDependencies instance
        
    Returns:
        BaseConversationProvider instance
        
    Raises:
        ValueError: If provider_name is unknown
    """
    if provider_name == "openai":
        from .openai_realtime import OpenAIRealtimeProvider
        return OpenAIRealtimeProvider(deps)
    
    elif provider_name == "anthropic":
        from .anthropic_elevenlabs import AnthropicElevenLabsProvider
        return AnthropicElevenLabsProvider(deps)
    
    elif provider_name == "elevenlabs":
        # Could be a different LLM + ElevenLabs combo
        from .elevenlabs import ElevenLabsProvider
        return ElevenLabsProvider(deps)
    
    elif provider_name == "local":
        from .local import LocalProvider
        return LocalProvider(deps)
    
    else:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available: openai, anthropic, elevenlabs, local"
        )

