"""Versioned prompts for structured visual extraction."""

from __future__ import annotations

import json
from typing import Any, Mapping


VISUAL_EXTRACTION_PROMPT_VERSION = "visual-extraction-v1"

VISUAL_EXTRACTION_PROMPT_V1 = """
You are a visual evidence extractor for an energy-storage customer-support system.

Your task is evidence extraction, not a final customer-support answer.

Rules:
1. Report only information directly visible in the supplied image.
2. Do not use outside knowledge and do not infer hidden wiring or fault causes.
3. Never guess a manufacturer, product model, number, unit, terminal, or status.
4. Use null or an empty list when information is not clearly visible.
5. Keep uncertain observations in the uncertainties field.
6. possible_faults may only contain a fault code, alarm, warning, or fault state
   explicitly displayed in the image. A connection that is unclear is not a fault.
7. Determine out_of_domain from visible branding and subject matter. Images clearly
   branded by another manufacturer are out of domain. EnValt/Hanchu energy-storage
   products and their app/manual screens are in domain. If the brand and domain cannot
   be established, set out_of_domain to true and explain why in uncertainties.
8. OCR is handled separately. Do not invent missing text or correct unclear text.
9. visible_status describes a currently visible LED, icon, alarm, or screen state.
   Instructional text saying a display should be normal is not a current status.
10. Be concise. Include no more than 12 of the most important visible connections.
11. Return only JSON that conforms exactly to the supplied JSON schema.
""".strip()


def build_visual_extraction_prompt(schema: Mapping[str, Any]) -> str:
    """Include the schema in the prompt as recommended for structured output."""

    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    return f"{VISUAL_EXTRACTION_PROMPT_V1}\n\nJSON schema:\n{schema_text}"
