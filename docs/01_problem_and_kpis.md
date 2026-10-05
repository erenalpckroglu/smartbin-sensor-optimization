# Problem & KPIs

**Bossard SmartBin Cloud.** A weighing scale under each bin keeps the stock level in ARIMS up to date. A vibration sensor wakes the scale up when someone touches the bin.

- A weighing costs about 3000× the sleep power. The battery must last at least 5 years.
- Every weighing is sent over the radio network and processed by the IoT cloud and ARIMS.
- A missed stock change means the stock level is wrong, which leads to late replenishment, express orders and stock-outs at the customer.

| Situation | Meaning | Who pays |
|---|---|---|
| R1 too sensitive | many empty weighings (TOO_MANY) | Bossard: battery, radio, cloud, service visits |
| R2 correct | only real changes are weighed | nobody |
| R3 too insensitive | changes missed (TOO_FEW) | the customer: availability; then Bossard: reputation |

## KPIs (Bossard's definitions, kept unchanged)

- **Fraction TOO_MANY** = TOO_MANY / automatic weighings. Target ≤ 0.5; alarm > 0.8.
- **TOO_FEW per week**. Target ≤ 2.

## Supporting KPIs (ours)

- **Miss rate** = TOO_FEW / (detected changes + TOO_FEW): the share of real stock changes the sensor slept through.
- **Weighings per day**: the battery and cloud load proxy.
- **Share of scales in the target zone**: the single number that combines both KPIs per scale.

## If the target is not reachable

If the target cannot be reached with sensitivity alone, the pipeline does **not** silently move the goalposts. It reports the achievable frontier ("best median TOO_MANY while TOO_FEW ≤ 2") next to the target. It also names what would be needed to close the gap: crosstalk mitigation and a per-scale adaptive setting.

## KPI 2.0 (our proposal, configured in `config.yaml -> kpi2`)

Real-data finding: at the same sensitivity, the TOO_MANY *fraction* is driven by how often a bin is used.
- Spearman −0.82 for 20 kg and −0.67 for 6 kg.
- The least-used fifth of bins sits at about 1.0, the most-used fifth at about 0.6.

The reason: rarely used bins have few real weighings in the denominator. The fraction therefore punishes slow-moving parts, not bad settings.

There is a second problem. A setting chosen to meet the Bossard KPIs misses 38–52% of real stock changes. TOO_FEW/week stays under 2 anyway, because the 24 h check bundles a whole day of misses into one record.

KPI 2.0 measures load and service directly:

| KPI | Target | Why |
|---|---|---|
| TOO_MANY per week (absolute) | ≤ 7 | Battery, radio and cloud pay per weighing, not per ratio |
| Miss rate = TOO_FEW / (detected changes + TOO_FEW) | ≤ 10% | Share of real stock changes the sensor slept through |
| TOO_FEW per week | ≤ 2 | Bossard's own service floor, kept |
