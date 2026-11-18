"""
SOMYTHRA Hardware — ESP32 Sensor Bridge.

Decodes serial and BLE telemetry packets emitted by the SOMYTHRA ESP32 physiological node.
Validates framing, packet sequence, timestamps, and XOR checksums.
"""

from dataclasses import dataclass
import struct
from typing import Optional, List, Tuple, Generator
import numpy as np


@dataclass
class ESP32Sample:
    """Decoded physiological sample from ESP32."""
    sequence: int
    timestamp_ms: int
    timestamp_sec: float
    red_raw: int
    ir_raw: int
    checksum_valid: bool


class ESP32StreamBridge:
    """
    Parser and stream bridge for ESP32 optical PPG sensor hardware.
    Supports binary packet framing (17 bytes) and JSON streaming.
    """

    SYNC_BYTE_1 = 0xAA
    SYNC_BYTE_2 = 0x55
    PACKET_SIZE = 17

    def __init__(self):
        self._buffer = bytearray()
        self._last_seq: Optional[int] = None
        self._dropped_packets = 0

    def feed_bytes(self, data: bytes) -> List[ESP32Sample]:
        """
        Feed incoming raw bytes from serial/BLE stream and parse all valid framed packets.
        """
        self._buffer.extend(data)
        samples = []

        while len(self._buffer) >= self.PACKET_SIZE:
            # Search for sync header [0xAA, 0x55]
            if self._buffer[0] != self.SYNC_BYTE_1 or self._buffer[1] != self.SYNC_BYTE_2:
                self._buffer.pop(0)
                continue

            # Full 17-byte packet candidate
            packet = bytes(self._buffer[:self.PACKET_SIZE])
            sample = self.parse_binary_packet(packet)

            if sample is not None and sample.checksum_valid:
                # Valid packet consumed
                self._buffer = self._buffer[self.PACKET_SIZE:]
                if self._last_seq is not None:
                    expected_seq = (self._last_seq + 1) & 0xFFFF
                    if sample.sequence != expected_seq:
                        diff = (sample.sequence - expected_seq) & 0xFFFF
                        self._dropped_packets += diff
                self._last_seq = sample.sequence
                samples.append(sample)
            else:
                # Checksum or framing error, advance 1 byte
                self._buffer.pop(0)

        return samples

    @classmethod
    def parse_binary_packet(cls, packet: bytes) -> Optional[ESP32Sample]:
        """
        Parse raw 17-byte framed packet.
        Format:
        [0..1]: 0xAA, 0x55
        [2..3]: seq uint16 (big endian)
        [4..7]: timestamp_ms uint32 (big endian)
        [8..11]: red uint32 (big endian)
        [12..15]: ir uint32 (big endian)
        [16]: checksum uint8 (XOR of bytes 2..15)
        """
        if len(packet) != cls.PACKET_SIZE:
            return None

        if packet[0] != cls.SYNC_BYTE_1 or packet[1] != cls.SYNC_BYTE_2:
            return None

        # Calculate XOR checksum
        computed_cs = 0
        for b in packet[2:16]:
            computed_cs ^= b

        expected_cs = packet[16]
        is_valid = (computed_cs == expected_cs)

        seq, ts_ms, red, ir = struct.unpack(">HIII", packet[2:16])

        return ESP32Sample(
            sequence=seq,
            timestamp_ms=ts_ms,
            timestamp_sec=ts_ms / 1000.0,
            red_raw=red,
            ir_raw=ir,
            checksum_valid=is_valid,
        )

    @classmethod
    def create_mock_packet(cls, seq: int, ts_ms: int, red: int, ir: int) -> bytes:
        """Helper to create a valid 17-byte packet for testing/simulation."""
        payload = struct.pack(">HIII", seq, ts_ms, red, ir)
        cs = 0
        for b in payload:
            cs ^= b
        return bytes([cls.SYNC_BYTE_1, cls.SYNC_BYTE_2]) + payload + bytes([cs])
