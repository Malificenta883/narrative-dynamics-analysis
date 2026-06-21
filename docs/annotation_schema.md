# Annotation Schema (Field Definitions)

This document defines the meaning of each field used in the annotated
narrative segments.

---

**Segment_ID**  
Sequential episode identifier.

---

**Text_EN**  
Short scene annotation (1–2 lines). Not prose.

---

**Function**  
The role of the scene within the exchange ritual:  
preparation | contact | exchange | disruption | negotiation | stabilization | return

---

**Cognitive_Frame**  
The cognitive regime activated in the segment:  
activation_through_attraction | authority_invocation | trickster | ritual_descent | boundary_testing | reciprocity | caution

---

**Transition_From / Transition_To**  
Functional phase shift (from / to values selected from the *Function* list).

---

**Markers**  
Explicit textual signals (keywords or images), e.g.:  
crown, radiance, sacred body, banquet, beer, me, boat.

---

**Anomaly_Type**  
Source of disturbance for the attention system:  
sensory | status | temporal | spatial | normative

---

**Risk_Mode**  
Mode of potential distortion:  
seduction | deception | overload | coercion | none

---

**Outcome_Tag**  
Result of the interactional step:  
channel_opened | channel_blocked | partial_transfer | misread_signal | stabilization

---

**Evidence**  
Minimal quotation fragments used to anchor the segment to the source corpus.

---

**Exchange_Channel**  
Reserved for future development to track spatial/temporal exchange vectors. Currently set to `"_"` by default.

---

**Confidence**  
Epistemic certainty of the AI segment mapping. Calculated strictly via evidence matching:  
**1.0:** Direct, literal presence of Markers in the Evidence text. Unambiguous mapping to Function.  
**0.75:** Markers are implicitly present (synonyms/metaphors) but the Cognitive_Frame is logically indisputable.  
**0.50:** Forced categorization due to ambiguous text; multiple Function or Risk_Mode options could equally apply.
