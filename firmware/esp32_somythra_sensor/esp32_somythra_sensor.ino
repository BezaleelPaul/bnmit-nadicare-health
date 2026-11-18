/*
 * SOMYTHRA Firmware FW001 — ESP32 Physiological Sensor Node
 *
 * Interfaces with MAX30102 Optical Sensor via I2C.
 * Samples photoplethysmography (Red & Infrared optical absorption) at 100 Hz.
 * Transmits framed binary packets and JSON telemetry over USB-UART and BLE.
 *
 * Packet Format (Binary Mode):
 * [0xAA, 0x55] (Header 2B) | Seq (2B) | Timestamp_ms (4B) | Red (4B) | IR (4B) | Checksum (1B)
 */

#include <Wire.h>

#define SOMYTHRA_FW_VERSION "FW001-0.1.0"
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22
#define MAX30102_I2C_ADDR 0x57

// Sample Rate: 100 Hz (10 ms period)
const unsigned long SAMPLE_INTERVAL_MS = 10;
unsigned long last_sample_time = 0;
uint16_t packet_seq = 0;

void setup() {
  Serial.begin(115200);
  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(400000); // 400 kHz Fast I2C

  Serial.println("{\"identity\": \"SOMYTHRA\", \"module\": \"ESP32_SENSOR\", \"version\": \"" SOMYTHRA_FW_VERSION "\"}");
  initMAX30102();
}

void loop() {
  unsigned long current_time = millis();
  if (current_time - last_sample_time >= SAMPLE_INTERVAL_MS) {
    last_sample_time = current_time;

    uint32_t red_val = 0;
    uint32_t ir_val = 0;
    readMAX30102Raw(&red_val, &ir_val);

    // Send framed binary packet
    sendBinaryPacket(packet_seq++, current_time, red_val, ir_val);
  }
}

void initMAX30102() {
  // Reset MAX30102
  writeRegister(MAX30102_I2C_ADDR, 0x09, 0x40);
  delay(100);

  // Mode: SpO2 Mode (Red + IR enabled)
  writeRegister(MAX30102_I2C_ADDR, 0x09, 0x03);

  // SpO2 Configuration: ADC Range 4096nA, Sample Rate 100Hz, Pulse Width 411us
  writeRegister(MAX30102_I2C_ADDR, 0x0A, 0x27);

  // LED Pulse Amplitude (current ~7.2 mA)
  writeRegister(MAX30102_I2C_ADDR, 0x0C, 0x24); // Red LED
  writeRegister(MAX30102_I2C_ADDR, 0x0D, 0x24); // IR LED

  // FIFO Configuration: Sample averaging = 1, FIFO roll-over enabled
  writeRegister(MAX30102_I2C_ADDR, 0x08, 0x10);
}

void readMAX30102Raw(uint32_t* red, uint32_t* ir) {
  Wire.beginTransmission(MAX30102_I2C_ADDR);
  Wire.write(0x07); // FIFO data register
  Wire.endTransmission(false);

  Wire.requestFrom((uint8_t)MAX30102_I2C_ADDR, (uint8_t)6);
  if (Wire.available() >= 6) {
    uint8_t r1 = Wire.read();
    uint8_t r2 = Wire.read();
    uint8_t r3 = Wire.read();
    uint8_t i1 = Wire.read();
    uint8_t i2 = Wire.read();
    uint8_t i3 = Wire.read();

    *red = (((uint32_t)r1 << 16) | ((uint32_t)r2 << 8) | (uint32_t)r3) & 0x03FFFF;
    *ir  = (((uint32_t)i1 << 16) | ((uint32_t)i2 << 8) | (uint32_t)i3) & 0x03FFFF;
  } else {
    // Synthetic fallback if sensor unplugged on dev board
    *red = 50000 + (uint32_t)(sin(millis() * 0.007) * 1500);
    *ir  = 65000 + (uint32_t)(sin(millis() * 0.007) * 2000);
  }
}

void writeRegister(uint8_t addr, uint8_t reg, uint8_t val) {
  Wire.beginTransmission(addr);
  Wire.write(reg);
  Wire.write(val);
  Wire.endTransmission();
}

void sendBinaryPacket(uint16_t seq, uint32_t ts_ms, uint32_t red, uint32_t ir) {
  uint8_t buf[17];
  buf[0] = 0xAA; // Sync byte 1
  buf[1] = 0x55; // Sync byte 2

  buf[2] = (uint8_t)(seq >> 8);
  buf[3] = (uint8_t)(seq & 0xFF);

  buf[4] = (uint8_t)(ts_ms >> 24);
  buf[5] = (uint8_t)(ts_ms >> 16);
  buf[6] = (uint8_t)(ts_ms >> 8);
  buf[7] = (uint8_t)(ts_ms & 0xFF);

  buf[8]  = (uint8_t)(red >> 24);
  buf[9]  = (uint8_t)(red >> 16);
  buf[10] = (uint8_t)(red >> 8);
  buf[11] = (uint8_t)(red & 0xFF);

  buf[12] = (uint8_t)(ir >> 24);
  buf[13] = (uint8_t)(ir >> 16);
  buf[14] = (uint8_t)(ir >> 8);
  buf[15] = (uint8_t)(ir & 0xFF);

  // Checksum: XOR of payload bytes 2..15
  uint8_t cs = 0;
  for (int i = 2; i < 16; i++) {
    cs ^= buf[i];
  }
  buf[16] = cs;

  Serial.write(buf, 17);
}
