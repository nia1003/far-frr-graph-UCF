| Dataset | Method | FAR | FRR | FAR + FRR | Latency, parallel run (ms) | Latency, sequential run (ms) |
|---|---|---|---|---|---|---|
| Celeb-DF-v2 | MILP (optimised on this dataset) | 0.2059 | 0.1461 | 0.3519 | 2676.2 | 3001.8 |
| Celeb-DF-v2 | MILP (FF++ parameters, cross-dataset) | 0.6294 | 0.0281 | 0.6575 | 2676.2 | 3001.8 |
| Celeb-DF-v2 | Average Score | 0.2118 | 0.1910 | 0.4028 | 2676.2 | 3001.8 |
| Celeb-DF-v2 | Relative Majority | 0.2235 | 0.2079 | 0.4314 | 2676.2 | 3001.8 |
| Celeb-DF-v2 | Cascade (5% grid, per-video search) | 0.1500 | 0.1854 | 0.3354 | 1234.4 | 1234.4 |
| DFDC | MILP (optimised on this dataset) | 0.3951 | 0.1970 | 0.5921 | 365.5 | 642.5 |
| DFDC | MILP (FF++ parameters, cross-dataset) | 0.6325 | 0.0916 | 0.7241 | 365.5 | 642.5 |
| DFDC | Average Score | 0.4948 | 0.1499 | 0.6447 | 365.5 | 642.5 |
| DFDC | Relative Majority | 0.4357 | 0.2078 | 0.6435 | 365.5 | 642.5 |
| DFDC | Cascade (5% grid, per-video search) | 0.3478 | 0.2346 | 0.5824 | 432.0 | 432.0 |
| FaceForensics++ | MILP (optimised on this dataset) | 0.0339 | 0.0143 | 0.0482 | 1035.1 | 1374.6 |
| FaceForensics++ | Average Score | 0.0286 | 0.0643 | 0.0929 | 1035.1 | 1374.6 |
| FaceForensics++ | Relative Majority | 0.0196 | 0.0786 | 0.0982 | 1035.1 | 1374.6 |
| FaceForensics++ | Cascade (5% grid, per-video search) | 0.0500 | 0.0000 | 0.0500 | 152.9 | 152.9 |
