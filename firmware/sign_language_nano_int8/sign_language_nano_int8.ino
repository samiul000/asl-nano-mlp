/*
 * Sign Language Recognition on Arduino Nano — INT8 Quantized
 * 
 * This firmware receives 42-dimensional feature vectors over serial,
 * runs a small INT8-quantized MLP model to classify the sign, and
 * displays the result on an SSD1306 OLED display/Serial port.
 *
 * INT8 variant: weights are int8 (-128..127), accumulation is int32.
 * 50% weight storage savings vs INT16 variant, ~1.1% accuracy drop.
 *
 * Built by Al Mahmud Samiul
 * Email: amsamiul.dev@gmail.com; Website: https://amsamiul.vercel.app
*/

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "model_int8.h"

#define SCREEN_W 128
#define SCREEN_H 64
#define OLED_ADDR 0x3C

Adafruit_SSD1306 display(SCREEN_W, SCREEN_H, &Wire, -1);
bool hasOLED = false;

// ---- layer sizes ----
#define L0_IN  42
#define L0_OUT 32
#define L1_OUT 16
#define L2_OUT 5

// ---- serial protocol ----
#define PKT_BUF 240
#define PKT_TIMEOUT_MS 500
#define START_CHAR '<'
#define END_CHAR   '>'
#define RESPOND

// ---- inference buffers (total ~200 B) ----
int16_t  input[L0_IN];       // 84 B
int16_t  hidden1[L0_OUT];    // 64 B
int16_t  hidden2[L1_OUT];    // 32 B
float    output[L2_OUT];     // 20 B

// ---- serial state machine ----
char     pktBuf[PKT_BUF];
uint8_t  pktIdx = 0;
unsigned long pktStart = 0;
bool     pktReady = false;

// ---- stats ----
char     lastSign = '?';
float    lastConf = 0.0;
unsigned long lastInferUs = 0;
unsigned long frameCount = 0;
unsigned long fpsStart = 0;
float    fps = 0;
uint16_t freeRam = 0;

// ---- PROGMEM safe read ----
float pgm_read_float_p(const void* addr) {
    float f;
    memcpy_P(&f, addr, sizeof(float));
    return f;
}

// ---- forward pass (INT8 weights, int32 accumulate) ----
void mlp_infer() {
    unsigned long t0 = micros();

    // layer 0: input(42 int16) × W0(42×32 int8) → hidden1(32 int16)
    for (int j = 0; j < L0_OUT; j++) {
        int32_t acc = 0;
        for (int i = 0; i < L0_IN; i++)
            acc += (int32_t)input[i] * (int8_t)pgm_read_byte(&W0[j * L0_IN + i]);
        float z = (float)acc * pgm_read_float_p(&W0_scale[j]) + pgm_read_float_p(&bias0[j]);
        hidden1[j] = z > 0 ? (int16_t)min(z, 32767.0f) : 0;
    }

    // layer 1: hidden1(32 int16) × W1(32×16 int8) → hidden2(16 int16)
    for (int j = 0; j < L1_OUT; j++) {
        int32_t acc = 0;
        for (int i = 0; i < L0_OUT; i++)
            acc += (int32_t)hidden1[i] * (int8_t)pgm_read_byte(&W1[j * L0_OUT + i]);
        float z = (float)acc * pgm_read_float_p(&W1_scale[j]) + pgm_read_float_p(&bias1[j]);
        hidden2[j] = z > 0 ? (int16_t)min(z, 32767.0f) : 0;
    }

    // layer 2: hidden2(16 int16) × W2(16×5 int8) → output(5 float)
    for (int j = 0; j < L2_OUT; j++) {
        int32_t acc = 0;
        for (int i = 0; i < L1_OUT; i++)
            acc += (int32_t)hidden2[i] * (int8_t)pgm_read_byte(&W2[j * L1_OUT + i]);
        output[j] = (float)acc * pgm_read_float_p(&W2_scale[j]) + pgm_read_float_p(&bias2[j]);
    }

    lastInferUs = micros() - t0;
}

float expf_approx(float x) {
    return exp(x);
}

void softmax() {
    float mx = output[0];
    for (int i = 1; i < L2_OUT; i++) if (output[i] > mx) mx = output[i];
    float sum = 0;
    for (int i = 0; i < L2_OUT; i++) { output[i] = expf_approx(output[i] - mx); sum += output[i]; }
    for (int i = 0; i < L2_OUT; i++) output[i] /= sum;
}

char predict() {
    mlp_infer();
    softmax();
    int best = 0;
    for (int i = 1; i < L2_OUT; i++) if (output[i] > output[best]) best = i;
    lastConf = output[best] * 100.0f;
    return 'A' + best;
}

bool parse_packet() {
    char *start = strchr(pktBuf, START_CHAR);
    char *end = strchr(pktBuf, END_CHAR);
    if (!start || !end || end <= start) return false;
    start++;
    int len = end - start;
    if (len < 3) return false;

    char *csSep = strrchr(start, ',');
    if (!csSep || csSep == start) return false;
    int csVal = atoi(csSep + 1);
    *csSep = '\0';

    int idx = 0;
    char *tok = start;
    while (tok && idx < L0_IN) {
        char *next = strchr(tok, ',');
        if (next) *next = '\0';
        input[idx++] = atoi(tok);
        tok = next ? next + 1 : NULL;
    }
    if (idx != L0_IN) return false;

    int32_t sum = 0;
    for (int i = 0; i < L0_IN; i++) sum += input[i];
    if ((sum % 100) != csVal) return false;
    return true;
}

void serial_recv() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == START_CHAR) {
            pktIdx = 0;
            pktStart = millis();
        }
        if (pktIdx < PKT_BUF - 1) {
            pktBuf[pktIdx++] = c;
            pktBuf[pktIdx] = '\0';
        }
        if (c == END_CHAR || c == '\n') {
            if (strchr(pktBuf, END_CHAR)) pktReady = true;
        }
    }
    if (pktIdx > 0 && millis() - pktStart > PKT_TIMEOUT_MS) {
        pktIdx = 0;
        pktBuf[0] = '\0';
    }
}

void update_display() {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);

    display.setCursor(0, 0);
    display.print("Sign : "); display.println(lastSign);

    display.setCursor(0, 12);
    display.print("Conf : "); display.print((int)lastConf); display.println("%");

    display.setCursor(0, 24);
    display.print("Lat  : "); display.print(lastInferUs / 1000.0f, 1); display.println("ms");

    display.setCursor(0, 36);
    display.print("RAM  : "); display.print(freeRam); display.println("B");

    display.setCursor(0, 48);
    display.print("FPS  : "); display.println((int)fps);

    display.display();
}

void setup() {
    Serial.begin(115200);
    hasOLED = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
    if (hasOLED) {
        display.clearDisplay();
        display.setTextSize(1);
        display.setTextColor(SSD1306_WHITE);
        display.setCursor(0, 0);
        display.println("Sign Lang MLP");
        display.println("INT8 variant");
        display.println("Waiting...");
        display.display();
    } else {
        Serial.println("No OLED - serial only (INT8)");
    }
    fpsStart = millis();
}

void loop() {
    serial_recv();

    if (pktReady) {
        pktReady = false;
        if (parse_packet()) {
            lastSign = predict();
            frameCount++;
#ifdef RESPOND
            Serial.print('{');
            Serial.print(lastSign);
            Serial.print(',');
            Serial.print(lastInferUs);
            Serial.print(',');
            Serial.print(lastConf, 1);
            Serial.print(',');
            Serial.print((int)fps);
            Serial.print(',');
            Serial.print(freeRam);
            Serial.println('}');
#endif
        }
        pktIdx = 0;
        pktBuf[0] = '\0';
    }

    unsigned long now = millis();
    if (now - fpsStart >= 1000) {
        fps = (float)frameCount * 1000.0f / (now - fpsStart);
        frameCount = 0;
        fpsStart = now;
        freeRam = getFreeRam();
        if (hasOLED) update_display();
    }
}

uint16_t getFreeRam() {
    extern uint16_t __heap_start, *__brkval;
    uint16_t v;
    return (uint16_t)&v - (__brkval == 0 ? (uint16_t)&__heap_start : (uint16_t)__brkval);
}
