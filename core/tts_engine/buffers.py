from __future__ import annotations
from typing import Optional


class AudioBuffer:
    """Helper to manage prebuffering logic for streaming audio."""
    
    def __init__(self, prebuffer_samples: int):
        self.prebuffer_samples = prebuffer_samples
        self.buffer = bytearray()
        self.prebuffering = True
    
    def process(self, chunk: bytes) -> Optional[bytes]:
        """
        Add a chunk to the buffer and return output if ready.
        
        Args:
            chunk: Audio bytes to add to buffer
            
        Returns:
            Buffered audio bytes if prebuffering is complete, None otherwise
        """
        self.buffer.extend(chunk)
        
        if self.prebuffering:
            # Check if we have enough samples (2 bytes per sample, mono)
            if len(self.buffer) >= self.prebuffer_samples * 2:
                self.prebuffering = False
                result = bytes(self.buffer)
                self.buffer.clear()
                return result
            return None
        else:
            # After prebuffering, return chunks directly
            result = bytes(self.buffer)
            self.buffer.clear()
            return result


class SyncFuture:
    """Wrapper to make synchronous results look like futures."""
    
    def __init__(self, value):
        self._value = value
    
    def result(self):
        return self._value
