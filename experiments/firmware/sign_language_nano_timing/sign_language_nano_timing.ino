/*
 * Timing-probe firmware — Dataset D per-layer measurement.
 * Copy of sign_language_nano (INT16) with micros() timers around
 * layer 0, layer 1, output layer, and softmax. Response format:
 *
 *   {sign,infer_us,l0_us,l1_us,l2_us,sm_us,confidence,fps,free_ram}
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include "model.h"

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

// ---- inference buffers ----
int16_t  input[L0_IN];
int16_t  hidden1[L0_OUT];
int16_t  hidden2[L1_OUT];
float    output[L2_OUT];

// ---- per-layer timings ----
unsigned long t_l0 = 0, t_l1 = 0, t_l2 = 0, t_sm = 0;

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

float pgm_read_float_p(const void* addr) {
    float f;
    memcpy_P(&f, addr, sizeof(float));
    return f;
}

void mlp_infer() {
    unsigned long t0 = micros();

    // layer 0
    for (int j = 0; j < L0_OUT; j++) {
        int32_t acc = 0;
        for (int i = 0; i < L0_IN; i++)
            acc += (int32_t)input[i] * (int16_t)pgm_read_word(&W0[j * L0_IN + i]);
        float z = (float)acc * pgm_read_float_p(&W0_scale[j]) + pgm_read_float_p(&bias0[j]);
        hidden1[j] = z > 0 ? (int16_t)min(z, 32767.0f) : 0;
    }
    t_l0 = micros() - t0;

    // layer 1
    unsigned long t1 = micros();
    for (int j = 0; j < L1_OUT; j++) {
        int32_t acc = 0;
        for (int i = 0; i < L0_OUT; i++)
            acc += (int32_t)hidden1[i] * (int16_t)pgm_read_word(&W1[j * L0_OUT + i]);
        float z = (float)acc * pgm_read_float_p(&W1_scale[j]) + pgm_read_float_p(&bias1[j]);
        hidden2[j] = z > 0 ? (int16_t)min(z, 32767.0f) : 0;
    }
    t_l1 = micros() - t1;

    // output layer
    unsigned long t2 = micros();
    for (int j = 0; j < L2_OUT; j++) {
        int32_t acc = 0;
        for (int i = 0; i < L1_OUT; i++)
            acc += (int32_t)hidden2[i] * (int16_t)pgm_read_word(&W2[j * L1_OUT + i]);
        output[j] = (float)acc * pgm_read_float_p(&W2_scale[j]) + pgm_read_float_p(&bias2[j]);
    }
    t_l2 = micros() - t2;

    lastInferUs = micros() - t0;
}

void softmax() {
    float mx = output[0];
    for (int i = 1; i < L2_OUT; i++) if (output[i] > mx) mx = output[i];
    float sum = 0;
    for (int i = 0; i < L2_OUT; i++) { output[i] = exp(output[i] - mx); sum += output[i]; }
    for (int i = 0; i < L2_OUT; i++) output[i] /= sum;
}

char predict() {
    mlp_infer();
    unsigned long ts = micros();
    softmax();
    t_sm = micros() - ts;
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
    return (sum % 100) == csVal;
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

void setup() {
    Serial.begin(115200);
    hasOLED = display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDR);
    fpsStart = millis();
    Serial.println("{boot}");
}

void loop() {
    serial_recv();

    if (pktReady) {
        pktReady = false;
        if (parse_packet()) {
            lastSign = predict();
            frameCount++;
            // {sign,infer_us,l0_us,l1_us,l2_us,sm_us,confidence,fps,free_ram}
            Serial.print('{');
            Serial.print(lastSign);
            Serial.print(',');
            Serial.print(lastInferUs);
            Serial.print(',');
            Serial.print(t_l0);
            Serial.print(',');
            Serial.print(t_l1);
            Serial.print(',');
            Serial.print(t_l2);
            Serial.print(',');
            Serial.print(t_sm);
            Serial.print(',');
            Serial.print(lastConf, 1);
            Serial.print(',');
            Serial.print((int)fps);
            Serial.print(',');
            Serial.print(freeRam);
            Serial.println('}');
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
    }
}

uint16_t getFreeRam() {
    extern uint16_t __heap_start, *__brkval;
    uint16_t v;
    return (uint16_t)&v - (__brkval == 0 ? (uint16_t)&__heap_start : (uint16_t)__brkval);
}
