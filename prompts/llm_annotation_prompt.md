# LLM Annotation Prompt — Gudea Narrative Segmentation & Dynamics

## Definition of a segment
A segment is the smallest contiguous text unit in which the dominant **Function** and **Cognitive_Frame** remain stable.
Start a new segment when there is a detectable shift in **intention**, **interaction mode**, **epistemic position**, or **symbolic risk**.

## Task
Analyze the text step by step from stroke **1** to **814**.
Do **not** skip strokes and do **not** overlap them.
Segments must cover the entire range contiguously.

For each segment, assign labels using the controlled vocabulary and output JSON objects following the schema below.

## Controlled vocabulary

### Function (choose exactly 1)
- preparation | contact | exchange | disruption | negotiation | stabilization | return

### Cognitive_Frame (choose 3–5)
- activation_through_attraction | authority_invocation | trickster | ritual_descent
- boundary_testing | reciprocity | caution

### Transition_From / Transition_To (choose exactly 1 each; from Function list)
- preparation | contact | exchange | disruption | negotiation | stabilization | return

### Anomaly_Type (choose exactly 1)
- sensory | status | temporal | spatial | normative

### Risk_Mode (choose exactly 1)
- seduction | deception | overload | coercion | none

### Outcome_Tag (choose exactly 1)
- channel_opened | channel_blocked | partial_transfer | misread_signal | stabilization

### Markers
Keywords that are explicit, repetitive, or salient (e.g., crown, radiance, sacred body, banquet, beer, me, boat).

## Output requirements (strict)
- Output **only JSON** (no commentary).
- Output a **JSON array** of segment objects.
- `segment_id` must be sequential: `gudea_001`, `gudea_002`, ...
- `text_en` must be short: scene name + stroke range (e.g., `"Boat offering (strokes 120–146)"`).
- `evidence` must contain **short direct quotations** from the segment (1–3 short quotes).
- Do not summarize the story.
- Do not invent events not present in the text.
- Provide your best annotation in **one attempt**.

## JSON schema
[
  {
    "segment_id": "gudea_001",
    "text_en": "",
    "function": "",
    "cognitive_frame": [],
    "transition_from": "",
    "transition_to": "",
    "markers": [],
    "anomaly_type": "",
    "exchange_channel": "_",
    "risk_mode": "",
    "outcome_tag": "",
    "evidence": [],
    "confidence": 0.0
  }
]
