from __future__ import annotations

import random
from pathlib import Path

from loguru import logger

from energy_grid_research_agent.config import get_settings

_IEC_STANDARDS = ["IEC 61850", "IEC 60909", "IEC 61968", "IEC 62351", "IEC 60870"]
_FAULT_TYPES = [
    "three-phase fault",
    "single-line-to-ground fault",
    "line-to-line fault",
    "transformer differential protection trip",
    "busbar protection operation",
]
_COMPONENTS = [
    "transformer",
    "circuit breaker",
    "protection relay",
    "busbar",
    "power cable",
    "capacitor bank",
    "reactor",
]


def _generate_standard(index: int, rng: random.Random) -> str:
    std = rng.choice(_IEC_STANDARDS)
    voltage = rng.choice([11, 33, 66, 110, 220, 400])
    component = rng.choice(_COMPONENTS)
    short_circuit_ka = rng.randint(10, 40)
    page = rng.randint(10, 200)
    return f"""{std} — Grid Standard Document {index}
Section 3: Protection Requirements
Page {page}

3.1 Scope
This section specifies protection requirements for {component} installations
operating at {voltage} kV in accordance with {std}.

3.2 Fault Tolerance Requirements
Under normal operating conditions the protection system shall operate within
80 ms for close-in faults and within 150 ms for remote-end faults.
The impedance protection zone 1 shall cover 80 % of the protected line section.

3.3 Measurement Criteria
Power factor shall be maintained above 0.95 lagging during peak load.
Voltage deviation must remain within ±5 % of nominal at the point of common coupling.
Short-circuit level at {voltage} kV busbar: {short_circuit_ka} kA symmetrical.

3.4 Communication Requirements
Under {std} Part 8-1, GOOSE messages shall have a maximum transfer time of 4 ms
for protection-critical applications.
""".strip()


def _generate_fault_report(index: int, rng: random.Random) -> str:
    fault = rng.choice(_FAULT_TYPES)
    component = rng.choice(_COMPONENTS)
    std = rng.choice(_IEC_STANDARDS)
    line = f"Line-{rng.randint(100, 999)}"
    clearance_ms = rng.randint(60, 180)
    overrun_min = rng.randint(5, 45)
    page = rng.randint(1, 50)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    substation = rng.randint(10, 99)
    return f"""Fault Incident Report — {index:04d}
Date: 2025-{month:02d}-{day:02d}
Location: Substation {substation}, {line}
Page {page}

Executive Summary
A {fault} was recorded on {line} involving the {component}.
Protection operated in {clearance_ms} ms, within {std} tolerances.

Root Cause Analysis
Insulation degradation on the {component} contributed to the fault inception.
Post-fault inspection revealed partial discharge activity above 500 pC
on phase B, indicating moisture ingress through cable termination.

Compliance Assessment
The fault clearance time complied with {std} zone 1 requirements.
However, post-fault restoration exceeded the 30-minute target by
{overrun_min} minutes due to manual switching operations.

Recommendations
1. Replace {component} insulation system at next scheduled outage.
2. Install online partial discharge monitoring per {std} guidelines.
3. Review switching procedures to reduce manual intervention time.
""".strip()


def generate_corpus(output_dir: Path | None = None) -> Path:
    settings = get_settings()
    out = Path(output_dir or settings.corpus.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)

    for i in range(1, settings.corpus.num_standards + 1):
        text = _generate_standard(i, rng)
        (out / f"grid_standard_{i:02d}.txt").write_text(text)
        logger.info(f"Generated standard document {i}")

    for i in range(1, settings.corpus.num_fault_reports + 1):
        text = _generate_fault_report(i, rng)
        (out / f"fault_report_{i:04d}.txt").write_text(text)
        logger.info(f"Generated fault report {i}")

    total = settings.corpus.num_standards + settings.corpus.num_fault_reports
    logger.success(f"Corpus: {total} documents → {out}")
    return out


if __name__ == "__main__":
    generate_corpus()
