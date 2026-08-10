# Serial Protocol

## Packet Format (PC → Arduino)

```
<v1,v2,...,v42,CS>
```

- Start delimiter: `<`
- End delimiter: `>`
- 42 comma-separated int16 values (0–1000)
- Checksum: `CS` = sum of all 42 values mod 100 (2 digits)
- Baud rate: 115200

## Example

```
<532,412,510,401,489,388,521,490,478,365,512,400,498,378,467,356,505,390,492,368,475,345,508,382,488,360,470,340,500,375,485,355,465,335,495,370,480,350,460,330,490,365,485>
```

## Response Format (Arduino → PC)

```
{sign,infer_us,confidence,fps,free_ram}\n
```

- `sign` : predicted letter (A–E)
- `infer_us` : inference time in microseconds
- `confidence` : softmax confidence (0.0–1.0)
- `fps` : frames per second
- `free_ram` : free SRAM in bytes

Example: `{B,10700,0.97,22,950}\n`

## Checksum Calculation

```python
values = [532, 412, 510, ...]  # 42 values
cs = sum(values) % 100  # e.g. 85
packet = f"<{','.join(str(v) for v in values)},{cs}>"
```

## Arduino Parsing

1. Detect `<` → reset buffer
2. Accumulate chars until `>` or `\n`
3. Split on last `,` → checksum
4. Parse 42 values
5. Validate: `(sum(values) % 100) == checksum`
6. On failure: discard packet, wait for next `<`

Timeout: if no complete packet within 500ms, buffer resets.

## Buffer Size

Serial receive buffer: 240 bytes (increased from 80 to handle ~170 char packets).
