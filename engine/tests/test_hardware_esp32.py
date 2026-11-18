"""
Tests for SOMYTHRA ESP32 Hardware Bridge & Packet Parsing.
"""

import pytest
from somythra.hardware.esp32 import ESP32StreamBridge, ESP32Sample


def test_esp32_valid_packet_parsing():
    bridge = ESP32StreamBridge()
    packet = ESP32StreamBridge.create_mock_packet(seq=101, ts_ms=15000, red=52000, ir=64000)

    sample = ESP32StreamBridge.parse_binary_packet(packet)
    assert sample is not None
    assert sample.sequence == 101
    assert sample.timestamp_ms == 15000
    assert sample.timestamp_sec == 15.0
    assert sample.red_raw == 52000
    assert sample.ir_raw == 64000
    assert sample.checksum_valid is True


def test_esp32_corrupted_checksum_rejection():
    packet = bytearray(ESP32StreamBridge.create_mock_packet(seq=1, ts_ms=10, red=1000, ir=2000))
    packet[16] ^= 0xFF  # Corrupt checksum byte

    sample = ESP32StreamBridge.parse_binary_packet(bytes(packet))
    assert sample is not None
    assert sample.checksum_valid is False


def test_esp32_streaming_feed_with_garbage_bytes():
    bridge = ESP32StreamBridge()

    # Prepend and append garbage noise around valid packets
    noise = b"\x00\xFF\x12\x34\xAA\x11"
    pkt1 = ESP32StreamBridge.create_mock_packet(seq=1, ts_ms=100, red=50000, ir=60000)
    pkt2 = ESP32StreamBridge.create_mock_packet(seq=2, ts_ms=110, red=50100, ir=60100)

    stream_bytes = noise + pkt1 + b"\x55\xAA" + pkt2 + b"\x99"

    samples = bridge.feed_bytes(stream_bytes)
    assert len(samples) == 2
    assert samples[0].sequence == 1
    assert samples[1].sequence == 2
    assert samples[0].red_raw == 50000
    assert samples[1].red_raw == 50100
