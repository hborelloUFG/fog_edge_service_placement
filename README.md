## Title
Service Placement in Fog Computing Environments Based on Heuristics


### Abstract
Service placement and server allocation in fog and edge computing environments, particularly for latency-sensitive applications, are challenging due to limited server resources and diverse service requirements. This work addresses these latency reduction challenges through three specialized heuristics that seek to optimize the use of available resources by nodes and the network. The proposed heuristics minimize the number of allocated servers, maximize the efficient use of residual resources, and reduce path lengths between allocated servers, achieving these optimizations while maintaining acceptable application processing times. The experimental results highlight the flexibility of these heuristics, allowing tradeoffs between solution quality and processing time, which increases their suitability for resource-constrained environments. By tailoring placement strategies to specific latency and execution time needs, the proposed approach strengthens the adaptability and responsiveness of service deployment in dynamic edge and fog computing contexts.

#### Files

Implementation of the three heuristics for service placement:
- h1_min_nodes_alloc.py
- h2_min_residual.py
- h3_min_hops.py

Implementation of the three ILP Models for service placement:
- m1_min_nodes_alloc.py
- m2_min_residual.py
- m3_min_hops.py