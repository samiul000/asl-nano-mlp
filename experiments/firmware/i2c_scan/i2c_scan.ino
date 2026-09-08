#include <Wire.h>

void setup() {
    Serial.begin(115200);
    Wire.begin();
    Wire.setClock(100000);
    Serial.println("{scan}");
}

void loop() {
    byte count = 0;
    for (byte addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            Serial.print("{i2c,0x");
            if (addr < 16) Serial.print('0');
            Serial.print(addr, HEX);
            Serial.println("}");
            count++;
        }
        delay(5);
    }
    if (!count) Serial.println("{none}");
    delay(2000);
}
