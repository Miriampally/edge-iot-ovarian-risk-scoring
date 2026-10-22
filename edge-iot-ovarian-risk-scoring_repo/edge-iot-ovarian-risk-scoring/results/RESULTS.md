# Simulation results (modeled)

All values come from a parameterised model, not measurements. Mean ± 95% CI over repeats where shown.

## Parameters

| Parameter | Value |
| --- | --- |
| study_mb | 112.0 |
| lan_mbps | 1000.0 |
| wan_mbps | 21.2 |
| rtt_ms | 49.8 |
| outage_prob | 0.05 |
| outage_s | (60.0, 600.0) |
| edge_infer_s | 6.5 |
| cloud_infer_s | 4.0 |
| overhead_s | 0.05 |
| cv | 0.15 |
| model_mb | 100.0 |
| visits_per_round | 50 |
| result_kb | 2.0 |
| seed | 1 |
| visits per repeat / repeats | 1000 / 20 |

**Break-even uplink:** with these parameters the cloud median time matches the edge median at about 442 Mbps; below that the edge is faster.

## E1: time to result vs uplink bandwidth (s)

| Uplink (Mbps) | Edge p50 | Edge p95 | Cloud p50 | Cloud p95 |
| --- | --- | --- | --- | --- |
| 5 | 7.6 ± 0.0 | 9.2 ± 0.0 | 213.5 ± 0.6 | 279.0 ± 15.7 |
| 10 | 7.6 ± 0.0 | 9.2 ± 0.0 | 109.4 ± 0.4 | 177.4 ± 25.1 |
| 20 | 7.6 ± 0.0 | 9.2 ± 0.0 | 57.3 ± 0.1 | 130.2 ± 28.6 |
| 50 | 7.6 ± 0.0 | 9.2 ± 0.0 | 26.1 ± 0.1 | 88.9 ± 30.0 |
| 100 | 7.6 ± 0.0 | 9.2 ± 0.0 | 15.6 ± 0.0 | 51.3 ± 23.6 |
| 200 | 7.6 ± 0.0 | 9.2 ± 0.0 | 10.4 ± 0.0 | 60.2 ± 24.0 |
| 500 | 7.6 ± 0.0 | 9.2 ± 0.0 | 7.3 ± 0.0 | 50.0 ± 25.2 |
| 1000 | 7.6 ± 0.0 | 9.2 ± 0.0 | 6.3 ± 0.0 | 67.1 ± 29.8 |

## E2: time to result vs outage probability at 50 Mbps (s)

| Outage (%) | Edge p50 | Edge p95 | Cloud p50 | Cloud p95 |
| --- | --- | --- | --- | --- |
| 0 | 7.6 ± 0.0 | 9.2 ± 0.0 | 25.8 ± 0.1 | 29.9 ± 0.0 |
| 2 | 7.6 ± 0.0 | 9.2 ± 0.0 | 25.9 ± 0.1 | 30.2 ± 0.0 |
| 5 | 7.6 ± 0.0 | 9.2 ± 0.0 | 26.0 ± 0.1 | 72.9 ± 25.0 |
| 10 | 7.6 ± 0.0 | 9.2 ± 0.0 | 26.3 ± 0.0 | 356.2 ± 17.3 |
| 15 | 7.6 ± 0.0 | 9.2 ± 0.0 | 26.7 ± 0.1 | 438.1 ± 8.7 |
| 20 | 7.6 ± 0.0 | 9.2 ± 0.0 | 26.9 ± 0.1 | 483.6 ± 9.7 |
| 30 | 7.6 ± 0.0 | 9.2 ± 0.0 | 27.7 ± 0.1 | 530.2 ± 5.6 |

## E3: response time under load (s; one edge GPU; 21.2 Mbps shared uplink)

| Arrivals/h | Edge 6.5 s p50 | Edge 11.2 s p50 | Cloud p50 | Edge 6.5 s p95 | Edge 11.2 s p95 | Cloud p95 |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 7.6 | 12.4 | 57.2 | 9.5 | 16.2 | 105.4 |
| 50 | 7.7 | 12.6 | 84.9 | 10.7 | 20.7 | 234.4 |
| 100 | 7.8 | 13.1 | 19767.7 | 13.0 | 25.9 | 35248.2 |
| 200 | 8.2 | 17.6 | 50751.7 | 16.7 | 48.8 | 87557.2 |
| 300 | 9.1 | 64.2 | 60557.1 | 23.3 | 240.8 | 104903.4 |
| 400 | 12.8 | 3787.3 | 65204.3 | 36.8 | 6334.4 | 113218.0 |
| 500 | 26.7 | 6535.7 | 68305.9 | 94.1 | 11397.9 | 118432.8 |
| 600 | 886.2 | 8725.2 | 70272.1 | 1418.9 | 15102.6 | 121953.6 |

Approximate capacity (visits/h): edge 6.5 s = 554, edge 11.2 s = 321, shared uplink = 74. Above capacity the queue grows without bound, so those response times depend on run length.

## E4: uplink per visit

| Model (MB) | Visits/round | Edge (MB/visit) | Cloud (MB/visit) | Saved (%) |
| --- | --- | --- | --- | --- |
| 25 | 10 | 2.502 | 112 | 97.77 |
| 25 | 25 | 1.002 | 112 | 99.11 |
| 25 | 50 | 0.502 | 112 | 99.55 |
| 25 | 100 | 0.252 | 112 | 99.78 |
| 25 | 250 | 0.102 | 112 | 99.91 |
| 25 | 500 | 0.052 | 112 | 99.95 |
| 100 | 10 | 10.002 | 112 | 91.07 |
| 100 | 25 | 4.002 | 112 | 96.43 |
| 100 | 50 | 2.002 | 112 | 98.21 |
| 100 | 100 | 1.002 | 112 | 99.11 |
| 100 | 250 | 0.402 | 112 | 99.64 |
| 100 | 500 | 0.202 | 112 | 99.82 |
| 400 | 10 | 40.002 | 112 | 64.28 |
| 400 | 25 | 16.002 | 112 | 85.71 |
| 400 | 50 | 8.002 | 112 | 92.86 |
| 400 | 100 | 4.002 | 112 | 96.43 |
| 400 | 250 | 1.602 | 112 | 98.57 |
| 400 | 500 | 0.802 | 112 | 99.28 |

## Model assumptions

- Time to result ends when the result is available to the on-site clinician; remote delivery is queued.
- De-identification runs at the edge in both deployments, so only de-identified data leave the clinic.
- Transfer times are size / bandwidth times a uniform factor in [0.9, 1.4] plus round-trip time.
- Inference times are normal with the stated coefficient of variation.
- Outages add a uniform wait to the cloud path only. E3 excludes outages to isolate the load effect.
- E3: edge is a FIFO queue with one GPU; cloud shares one FIFO uplink and has unlimited cloud GPUs.
- E4 counts uplink only and ignores the downlink used to roll out global models.
