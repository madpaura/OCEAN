# OCEAN Use Cases Deep Dive Analysis

This document provides a comprehensive analysis of the use cases implemented in the OCEAN CXL Emulation Framework.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Use Case Architecture](#2-use-case-architecture)
3. [Production Profiling](#3-production-profiling)
4. [Procurement Decision Support](#4-procurement-decision-support)
5. [Memory Tiering Engine](#5-memory-tiering-engine)
6. [Dynamic Migration Engine](#6-dynamic-migration-engine)
7. [Predictive Placement Optimizer](#7-predictive-placement-optimizer)
8. [Topology-Guided Procurement](#8-topology-guided-procurement)
9. [Integration Patterns](#9-integration-patterns)
10. [Use Case Comparison Matrix](#10-use-case-comparison-matrix)

---

## 1. Overview

The OCEAN use cases demonstrate practical applications of CXL memory simulation for enterprise decision-making. They are organized into **6 main categories**:

| Use Case | Purpose | Key Technology |
|----------|---------|----------------|
| **Production Profiling** | Profile real workloads with CXL configurations | Parallel execution, metrics parsing |
| **Procurement Decision** | Hardware cost/performance analysis | TCO modeling, multi-criteria scoring |
| **Memory Tiering** | Dynamic memory placement policies | ML (RandomForest), adaptive learning |
| **Dynamic Migration** | Real-time data migration policies | Anomaly detection, trigger-based |
| **Predictive Placement** | ML-based optimal data placement | Deep learning (PyTorch), clustering |
| **Topology-Guided Procurement** | Hardware selection based on hotness | Hotness prediction, topology matching |

---

## 2. Use Case Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OCEAN Use Cases Framework                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Input Layer                                   │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ YAML Configs │  │ Workload     │  │ Hardware     │               │   │
│  │  │              │  │ Binaries     │  │ Catalogs     │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      Processing Engines                              │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ Production   │  │ Procurement  │  │ Memory       │               │   │
│  │  │ Profiler     │  │ Analyzer     │  │ Tiering      │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  │                                                                      │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ Dynamic      │  │ Predictive   │  │ Topology     │               │   │
│  │  │ Migration    │  │ Placement    │  │ Procurement  │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        CXLMemSim Core                                │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  subprocess.run() → CXLMemSim binary → Parse stdout metrics  │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        Output Layer                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │ JSON Reports │  │ PNG Charts   │  │ CSV Data     │               │   │
│  │  │              │  │ (matplotlib) │  │              │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Common Dependencies

```python
# Core dependencies across all use cases
pandas          # Data manipulation and analysis
numpy           # Numerical computing
matplotlib      # Visualization
seaborn         # Statistical visualization
pyyaml          # Configuration parsing
scikit-learn    # Machine learning (RandomForest, KMeans, IsolationForest)
torch           # Deep learning (Predictive Placement)
```

---

## 3. Production Profiling

**Location:** `use_cases/production_profiling/`

### 3.1 Purpose

Profile production workloads with various CXL configurations to understand performance characteristics before deployment.

### 3.2 Class Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      ProductionProfiler                          │
├─────────────────────────────────────────────────────────────────┤
│ - cxlmemsim_path: Path                                          │
│ - output_dir: Path                                              │
│ - results: List[Dict]                                           │
├─────────────────────────────────────────────────────────────────┤
│ + profile_workload(workload_config: Dict) → Dict                │
│ + run_production_suite(suite_config_file: str) → None           │
│ - _parse_output(output: str) → Dict                             │
│ - _save_results() → None                                        │
│ - _generate_report() → None                                     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.3 Key Features

| Feature | Description |
|---------|-------------|
| **Parallel Execution** | Uses `multiprocessing.Pool` for concurrent workload profiling |
| **Metrics Parsing** | Extracts local/remote accesses, latency, bandwidth from stdout |
| **Visualization** | Generates 4-panel comparison charts (time, ratio, latency, bandwidth) |
| **CI/CD Integration** | `ci_profiling.sh` script for automated pipeline integration |

### 3.4 Configuration Example

```yaml
# example_suite.yaml
parallel_jobs: 4
workloads:
  - name: "database_workload"
    binary: "/path/to/workload"
    interval: 10
    timeout: 3600
    cpuset: "0,2"
    dram_latency: 85
    
cxl_configurations:
  - name: "baseline"
    dram_latency: 85
    capacity: [100]
    
  - name: "cxl_2tier"
    dram_latency: 85
    latency: [150, 180]
    bandwidth: [30000, 25000]
    capacity: [50, 50]
    topology: "(1,(2))"
```

### 3.5 Workflow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Load Config  │───►│ Create Task  │───►│ Parallel     │───►│ Parse        │
│ (YAML)       │    │ Combinations │    │ Execution    │    │ Results      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Summary      │◄───│ Generate     │◄───│ Create       │◄───│ Aggregate    │
│ Report       │    │ Charts       │    │ DataFrame    │    │ Metrics      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

---

## 4. Procurement Decision Support

**Location:** `use_cases/procurement_decision/`

### 4.1 Purpose

Help evaluate cost/performance tradeoffs for different CXL hardware configurations to make data-driven purchasing decisions.

### 4.2 Class Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      ProcurementAnalyzer                         │
├─────────────────────────────────────────────────────────────────┤
│ - cxlmemsim_path: Path                                          │
│ - calibration_data: Dict (optional gem5 calibration)            │
├─────────────────────────────────────────────────────────────────┤
│ + evaluate_hardware_config(hw_config, workload) → Dict          │
│ + run_procurement_analysis(config_file, output_dir) → Dict      │
│ - _parse_metrics(output: str) → Dict                            │
│ - _calculate_performance_score(metrics, workload) → float       │
│ - _calculate_hardware_cost(hw_config) → float                   │
│ - _estimate_power_consumption(hw_config, metrics) → float       │
│ - _generate_procurement_report(results, output_dir) → None      │
│ - _generate_tco_analysis(results, tco_params, output_dir)       │
│ - _generate_recommendation(results, requirements) → Dict        │
│ - _describe_tradeoff(alt_config, best_config) → str             │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Cost Model

```
Total Hardware Cost = Base System Cost
                    + (Local Memory GB × DRAM Cost/GB)
                    + (CXL Memory GB × CXL Cost/GB)
                    + CXL Device Cost (if CXL memory > 0)
                    + CXL Switch Cost (if switched topology)
```

### 4.4 TCO Calculation

```
3-Year TCO = Initial Hardware Cost
           + (Annual Electricity Cost × 3)
           + (Annual Maintenance Cost × 3)

Where:
  Annual Electricity = Avg Power (W) × 24 × 365 × $/kWh / 1000
  Annual Maintenance = Initial Cost × 2%
```

### 4.5 Multi-Criteria Scoring

```python
# Recommendation scoring algorithm
def calculate_score(result, filtered_results, weights):
    perf_norm = result["performance_score"] / max(r["performance_score"])
    cost_norm = 1 - (result["total_cost"] / max(r["total_cost"]))
    power_norm = 1 - (result["power_estimate"] / max(r["power_estimate"]))
    
    total_score = (
        weights["performance"] * perf_norm +  # default: 0.4
        weights["cost"] * cost_norm +          # default: 0.4
        weights["power"] * power_norm          # default: 0.2
    )
    return total_score
```

### 4.6 Output Visualizations

| Chart | Description |
|-------|-------------|
| **Cost vs Performance Scatter** | Hardware options plotted by cost and performance |
| **Cost Efficiency Bar** | Cost per performance unit comparison |
| **Power Consumption Bar** | Average power consumption by configuration |
| **Performance Heatmap** | Performance matrix (hardware × workload) |
| **TCO Breakdown** | Stacked bar chart of TCO components |

---

## 5. Memory Tiering Engine

**Location:** `use_cases/memory_tiering/`

### 5.1 Purpose

Implement and evaluate intelligent memory placement and migration policies for CXL memory tiering.

### 5.2 Class Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      MemoryTieringEngine                         │
├─────────────────────────────────────────────────────────────────┤
│ - cxlmemsim_path: Path                                          │
│ - policies: Dict[str, Callable]                                 │
│ - performance_history: List                                     │
│ - ml_model: RandomForestRegressor                               │
│ - access_patterns: Dict                                         │
│ - endpoint_hotness_history: Dict                                │
│ - endpoint_performance_metrics: Dict                            │
├─────────────────────────────────────────────────────────────────┤
│ + register_policy(name: str, policy_func: Callable) → None      │
│ + create_static_policy(tier_allocation: List[float]) → Callable │
│ + create_hotness_based_policy(hot_threshold: float) → Callable  │
│ + create_ml_policy(training_data: List[Dict]) → Callable        │
│ + create_adaptive_policy(adaptation_rate: float) → Callable     │
│ + create_endpoint_aware_hotness_policy(...) → Callable          │
│ + evaluate_policy(policy_name, workload_config, duration) → Dict│
│ + run_policy_comparison(config, output_dir) → List[Dict]        │
│ - _simulate_access_pattern(workload_config) → Dict              │
│ - _run_simulation(workload_config, allocation) → Dict           │
│ - _generate_comparison_report(results, output_path) → None      │
│ - _generate_endpoint_hotness_report(output_path) → None         │
└─────────────────────────────────────────────────────────────────┘
```

### 5.3 Policy Types

| Policy | Algorithm | Use Case |
|--------|-----------|----------|
| **Static Balanced** | Fixed 50/50 allocation | Baseline comparison |
| **Static Local Heavy** | Fixed 80/20 local/CXL | Latency-sensitive workloads |
| **Static CXL Heavy** | Fixed 30/70 local/CXL | Capacity-constrained systems |
| **Hotness-Based** | Threshold-based allocation | Dynamic workloads |
| **Endpoint-Aware** | Per-endpoint hotness weighting | Multi-tier CXL |
| **ML-Based** | RandomForest prediction | Complex access patterns |
| **Adaptive** | Gradient-based learning | Self-optimizing systems |

### 5.4 Hotness-Based Policy Logic

```python
def hotness_policy(workload_info, access_pattern):
    endpoint_hotness = access_pattern.get("endpoint_hotness", {})
    
    if endpoint_hotness:
        total_hotness = sum(endpoint_hotness.values())
        
        if total_hotness > hot_threshold:
            # High hotness: prioritize local memory (60%)
            allocations[0] = 0.6
            # Distribute 40% to CXL based on inverse hotness
            for i, (endpoint, hotness) in enumerate(endpoint_hotness.items()):
                allocations[i+1] = 0.4 * (1.0 - hotness)
        else:
            # Low hotness: more even distribution
            allocations[0] = 0.3
            # Distribute 70% evenly across CXL endpoints
            
    return allocations
```

### 5.5 Access Pattern Simulation

The engine simulates realistic access patterns based on workload type:

| Workload Type | Pattern Characteristics |
|---------------|------------------------|
| **Database** | Periodic hot/cold cycles, phase-shifted per endpoint |
| **Analytics** | Sequential with bursts, skewed endpoint usage |
| **Web** | Random with locality, uniform distribution |
| **General** | Sinusoidal variation, decreasing with distance |

### 5.6 ML Model Training

```python
# Feature extraction for ML policy
features = [
    data.get("memory_intensity", 0),
    data.get("access_locality", 0),
    data.get("read_write_ratio", 0),
    data.get("working_set_size", 0),
    data.get("cache_miss_rate", 0)
]

# Target: optimal memory allocation [local_ratio, cxl_ratio]
targets = data.get("optimal_allocation", [0.5, 0.5])

# Model: RandomForestRegressor with 100 estimators
ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
ml_model.fit(X, y)
```

---

## 6. Dynamic Migration Engine

**Location:** `use_cases/dynamic_migration/`

### 6.1 Purpose

Implement intelligent data migration policies with real-time hotness monitoring and adaptive learning.

### 6.2 Key Data Structures

```python
class MigrationTrigger(Enum):
    HOTNESS_THRESHOLD = "hotness_threshold"
    LOAD_IMBALANCE = "load_imbalance"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    PERIODIC = "periodic"
    CONGESTION = "congestion"
    ANOMALY_DETECTED = "anomaly_detected"

class MigrationPolicy(Enum):
    CONSERVATIVE = "conservative"   # Only critical triggers
    BALANCED = "balanced"           # Most triggers, moderated
    AGGRESSIVE = "aggressive"       # All triggers proactively
    ADAPTIVE = "adaptive"           # Learns from outcomes
    PREDICTIVE = "predictive"       # ML-based prediction

@dataclass
class MigrationCandidate:
    page_id: int
    current_location: str
    target_location: str
    benefit_score: float
    cost_score: float
    net_benefit: float
    trigger: MigrationTrigger
    timestamp: float

@dataclass
class EndpointState:
    endpoint_id: str
    capacity_gb: int
    used_gb: float
    hotness_score: float
    access_rate: float
    bandwidth_utilization: float
    latency_percentiles: Dict[int, float]  # 50th, 95th, 99th
    migration_in_progress: int
```

### 6.3 Class Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DynamicMigrationEngine                        │
├─────────────────────────────────────────────────────────────────┤
│ - cxlmemsim_path: Path                                          │
│ - topology: Dict                                                │
│ - endpoint_states: Dict[str, EndpointState]                     │
│ - policy_states: Dict[MigrationPolicy, PolicyState]             │
│ - migration_queue: PriorityQueue                                │
│ - migration_history: deque (maxlen=1000)                        │
│ - anomaly_detector: IsolationForest                             │
│ - executor: ThreadPoolExecutor                                  │
│ - monitoring_active: bool                                       │
│ - current_policy: MigrationPolicy                               │
├─────────────────────────────────────────────────────────────────┤
│ + set_migration_policy(policy: MigrationPolicy) → None          │
│ + start_monitoring(interval: float) → None                      │
│ + stop_monitoring() → None                                      │
│ - _monitoring_loop(interval: float) → None                      │
│ - _update_endpoint_states() → None                              │
│ - _check_migration_triggers() → List[Tuple]                     │
│ - _detect_anomalies() → List[Dict]                              │
│ - _process_migration_triggers(triggers) → None                  │
│ - _should_act_on_trigger(trigger_type, data) → bool             │
│ - _predict_future_benefit(trigger_type, data) → float           │
│ - _generate_migration_candidates(trigger_type, data) → None     │
│ - _process_migration_queue() → None                             │
│ - _execute_migration(candidate) → MigrationOutcome              │
└─────────────────────────────────────────────────────────────────┘
```

### 6.4 Trigger Detection Logic

```python
def _check_migration_triggers(self):
    triggers = []
    
    # 1. Hotness Threshold (>0.8 on non-local endpoints)
    for endpoint_id, state in self.endpoint_states.items():
        if state.hotness_score > 0.8 and endpoint_id != "endpoint_1":
            triggers.append((HOTNESS_THRESHOLD, {"endpoint": endpoint_id}))
    
    # 2. Load Imbalance (std deviation > 30%)
    utilizations = [s.used_gb / s.capacity_gb for s in states]
    if np.std(utilizations) > 0.3:
        triggers.append((LOAD_IMBALANCE, {"imbalance": imbalance}))
    
    # 3. Performance Degradation (p99 > 2x p50)
    for state in states:
        if state.latency_percentiles[99] > state.latency_percentiles[50] * 2:
            triggers.append((PERFORMANCE_DEGRADATION, {...}))
    
    # 4. Congestion (bandwidth utilization > 90%)
    for state in states:
        if state.bandwidth_utilization > 0.9:
            triggers.append((CONGESTION, {...}))
    
    # 5. Anomaly Detection (IsolationForest)
    if self.anomaly_detector:
        anomalies = self._detect_anomalies()
        if anomalies:
            triggers.append((ANOMALY_DETECTED, {"anomalies": anomalies}))
    
    return triggers
```

### 6.5 Policy Decision Matrix

| Policy | HOTNESS | LOAD_IMBALANCE | PERF_DEGRADE | CONGESTION | PERIODIC | ANOMALY |
|--------|---------|----------------|--------------|------------|----------|---------|
| Conservative | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Balanced | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ |
| Aggressive | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Adaptive | Learned | Learned | Learned | Learned | Learned | Learned |
| Predictive | Predicted | Predicted | Predicted | Predicted | Predicted | Predicted |

### 6.6 Anomaly Detection

Uses **IsolationForest** from scikit-learn:

```python
# Features for anomaly detection
features = [
    outcome.candidate.benefit_score,
    outcome.candidate.cost_score,
    outcome.actual_improvement,
    outcome.migration_duration_ms
]

# Train with 10% contamination threshold
anomaly_detector = IsolationForest(contamination=0.1)
anomaly_detector.fit(features_scaled)

# Detect anomalies (returns -1 for anomalies)
predictions = anomaly_detector.predict(recent_features)
```

---

## 7. Predictive Placement Optimizer

**Location:** `use_cases/predictive_placement/`

### 7.1 Purpose

Use deep learning to predict optimal data placement across CXL topology based on access patterns.

### 7.2 Deep Learning Architecture

```python
class DeepPlacementNet(nn.Module):
    """4-layer neural network for placement prediction"""
    
    def __init__(self, input_features: int, num_endpoints: int):
        super().__init__()
        self.fc1 = nn.Linear(input_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_endpoints)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x
```

### 7.3 Key Data Structures

```python
class AccessPattern(Enum):
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    STRIDED = "strided"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"

@dataclass
class MemoryPage:
    page_id: int
    size_kb: int
    access_count: int
    last_access_time: float
    access_pattern: AccessPattern
    current_location: str
    heat_score: float = 0.0
    predicted_future_accesses: int = 0

@dataclass
class PlacementDecision:
    page_id: int
    current_location: str
    recommended_location: str
    expected_improvement: float  # percentage
    confidence: float  # 0-1
    reasoning: str
```

### 7.4 Placement Score Calculation

```python
def _calculate_placement_score(self, page, endpoint, page_clusters):
    # Base scores
    latency_score = 100 / (endpoint.latency_ns + 1)
    bandwidth_score = endpoint.bandwidth_gbps / 100
    
    # Access frequency weight
    access_weight = min(page.predicted_future_accesses / 1000, 1.0)
    
    # Distance penalty
    distance_penalty = 1.0 / (1.0 + endpoint.distance_from_cpu * 0.2)
    
    # Capacity availability
    available_ratio = 1.0 - (endpoint.used_gb / endpoint.capacity_gb)
    capacity_score = min(available_ratio * 2, 1.0)
    
    # Congestion penalty
    congestion_penalty = 1.0 - endpoint.congestion_level
    
    # Pattern-specific scoring
    if page.access_pattern == AccessPattern.SEQUENTIAL:
        pattern_score = bandwidth_score * 1.5  # Needs high bandwidth
    elif page.access_pattern == AccessPattern.RANDOM:
        pattern_score = latency_score * 1.5    # Needs low latency
    elif page.access_pattern == AccessPattern.TEMPORAL:
        pattern_score = distance_penalty * 2.0  # Needs proximity
    
    # Hot data bonus for local memory
    if endpoint.endpoint_id == "endpoint_1" and page.heat_score > 0.7:
        pattern_score *= 1.5
    
    # Weighted combination
    total_score = (
        latency_score * 0.3 +
        bandwidth_score * 0.2 +
        capacity_score * 0.2 +
        pattern_score * 0.2 +
        congestion_penalty * 0.1
    ) * access_weight * distance_penalty
    
    return total_score
```

### 7.5 Page Clustering

Uses **KMeans** clustering to group pages by access pattern similarity:

```python
# Features for clustering
features = [
    page.access_count,
    page.heat_score,
    page.predicted_future_accesses,
    1.0 if page.access_pattern == AccessPattern.SEQUENTIAL else 0.0,
    1.0 if page.access_pattern == AccessPattern.RANDOM else 0.0,
    1.0 if page.access_pattern == AccessPattern.TEMPORAL else 0.0
]

# Cluster pages
n_clusters = min(5, len(pages) // 10)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features_normalized)
```

### 7.6 Workflow

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Load Pages   │───►│ Predict      │───►│ Cluster by   │───►│ Calculate    │
│ & Patterns   │    │ Future       │    │ Access       │    │ Placement    │
│              │    │ Accesses     │    │ Pattern      │    │ Scores       │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                    │
                                                                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Migration    │◄───│ Optimize     │◄───│ Generate     │◄───│ Select Best  │
│ Plan         │    │ Load         │    │ Reasoning    │    │ Placement    │
│              │    │ Balance      │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

---

## 8. Topology-Guided Procurement

**Location:** `use_cases/topology_guided_procurement/`

### 8.1 Purpose

Make data-driven hardware purchasing decisions based on workload hotness patterns and topology optimization.

### 8.2 Hardware Catalog

Pre-defined hardware options representing different topology types:

| Model | Topology | Endpoints | Memory/EP | Latency Profile | Cost/EP |
|-------|----------|-----------|-----------|-----------------|---------|
| CXL-Flat-256 | Flat | 4 | 256GB | [150, 170, 190]ns | $2,000 |
| CXL-Hierarchy-512 | Hierarchical | 8 | 512GB | [180, 220, 280]ns | $3,500 |
| CXL-Star-384 | Star | 6 | 384GB | [160, 200, 240]ns | $2,800 |
| CXL-Mesh-1024 | Mesh | 16 | 1024GB | [200, 280, 360, 440]ns | $6,000 |
| CXL-Economy-128 | Flat | 2 | 128GB | [140, 160]ns | $1,200 |

### 8.3 Hotness Distribution Prediction

```python
def _predict_hotness_distribution(self, workload_config):
    workload_type = workload_config.get("type", "general")
    num_endpoints = workload_config.get("target_endpoints", 4)
    
    if workload_type == "database":
        # Skewed: exponential decay
        for i in range(num_endpoints):
            hotness = 0.9 * np.exp(-i * 0.5)
            
    elif workload_type == "analytics":
        # Uniform with sinusoidal variation
        for i in range(num_endpoints):
            hotness = 0.4 + 0.2 * np.sin(i * np.pi / num_endpoints)
            
    elif workload_type == "ml_training":
        # Bimodal: training vs validation
        for i in range(num_endpoints):
            hotness = 0.7 if i < num_endpoints // 2 else 0.3
            
    return hotness_dist
```

### 8.4 Topology Fit Evaluation

```python
def _evaluate_topology_fit(self, topology_type, hotness_dist, num_endpoints):
    hotness_skew = np.std(hotness_values) / np.mean(hotness_values)
    
    if topology_type == "flat":
        # Good for uniform distribution
        return 1.0 - min(hotness_skew, 1.0)
        
    elif topology_type == "hierarchical":
        # Good for moderate skew
        return 1.0 - abs(hotness_skew - 0.5) * 2
        
    elif topology_type == "star":
        # Good for centralized hot data
        return min(hotness_skew * 1.5, 1.0)
        
    elif topology_type == "mesh":
        # Good for distributed access
        return 1.0 - abs(hotness_skew - 0.3) * 2
```

### 8.5 Recommendation Output

```python
@dataclass
class ProcurementRecommendation:
    hardware_option: HardwareOption
    topology_config: str
    total_cost: float
    performance_score: float
    scalability_score: float
    tco_3_year: float
    reasoning: List[str]
    risk_factors: List[str]
```

---

## 9. Integration Patterns

### 9.1 CI/CD Integration

```bash
# ci_profiling.sh - Example CI/CD integration
#!/bin/bash

# Run profiling suite
python3 production_profiler.py \
    --cxlmemsim ../../build/CXLMemSim \
    --config ci_suite.yaml \
    --output ./ci_results

# Check performance regression
python3 check_regression.py \
    --baseline ./baseline_results.json \
    --current ./ci_results/profiling_results_*.json \
    --threshold 0.1  # 10% regression threshold
```

### 9.2 Kubernetes Integration

```yaml
# ConfigMap for CXL topology optimizer
apiVersion: v1
kind: ConfigMap
metadata:
  name: cxl-topology-optimizer
data:
  policy: "adaptive"
  hotness_threshold: "0.8"
  migration_cooldown: "60"
```

### 9.3 Prometheus Metrics Export

```python
from prometheus_client import Counter, Histogram, Gauge

# Migration metrics
migration_counter = Counter('cxl_migrations_total', 
                          'Total CXL memory migrations',
                          ['policy', 'trigger'])
                          
migration_duration = Histogram('cxl_migration_duration_seconds',
                             'Migration duration distribution',
                             buckets=[0.001, 0.01, 0.1, 1.0, 10.0])
                             
endpoint_hotness = Gauge('cxl_endpoint_hotness',
                        'Current endpoint hotness score',
                        ['endpoint_id'])
```

---

## 10. Use Case Comparison Matrix

### 10.1 Feature Comparison

| Feature | Profiling | Procurement | Tiering | Migration | Placement | Topo-Proc |
|---------|-----------|-------------|---------|-----------|-----------|-----------|
| **ML/AI** | ❌ | ❌ | ✅ RF | ✅ IF | ✅ DL | ❌ |
| **Real-time** | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| **Hotness Tracking** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Cost Analysis** | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| **TCO Modeling** | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| **Parallel Exec** | ✅ | ✅ | ❌ | ✅ | ✅ | ✅ |
| **Visualization** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

### 10.2 Input/Output Summary

| Use Case | Input | Output |
|----------|-------|--------|
| **Profiling** | Workload binaries, CXL configs | Performance metrics, comparison charts |
| **Procurement** | Hardware configs, workloads, budget | Recommendations, TCO analysis |
| **Tiering** | Workloads, policies, training data | Policy comparison, best policy |
| **Migration** | Topology, thresholds, policy | Migration decisions, outcomes |
| **Placement** | Pages, access patterns, topology | Placement decisions, migration plan |
| **Topo-Proc** | Workload requirements, constraints | Hardware recommendations, risk analysis |

### 10.3 When to Use Each

| Scenario | Recommended Use Case |
|----------|---------------------|
| "What CXL config is best for my workload?" | Production Profiling |
| "Which hardware should I buy?" | Procurement Decision + Topology-Guided |
| "How should I distribute data across tiers?" | Memory Tiering Engine |
| "When should I migrate data?" | Dynamic Migration Engine |
| "Where should I place this data?" | Predictive Placement Optimizer |
| "Does this topology match my access patterns?" | Topology-Guided Procurement |

---

## Appendix: Running the Use Cases

### Quick Start

```bash
# Run all demos
cd use_cases
./run_all_demos.sh

# Run specific use case
cd production_profiling
python3 production_profiler.py \
    --cxlmemsim ../../build/CXLMemSim \
    --config example_suite.yaml \
    --output ./results
```

### Test Suite

```bash
# Run integration tests
cd use_cases
python3 test_use_cases.py
```

### Dependencies Installation

```bash
pip install pandas matplotlib numpy pyyaml scikit-learn torch seaborn
```

---

*Deep Dive Analysis for OCEAN CXL Emulation Framework Use Cases*  
*UC Santa Cruz Sluglab - 2025*
