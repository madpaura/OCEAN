# OCEAN Architecture Document

## Open-source CXL Emulation at Hyperscale Architecture and Networking

**Version:** 1.0  
**Authors:** UC Santa Cruz Sluglab  
**License:** LGPL-2.1 OR BSD-2-Clause

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Core Components](#3-core-components)
4. [Component Architecture](#4-component-architecture)
5. [Communication Layers](#5-communication-layers)
6. [Memory Management](#6-memory-management)
7. [Policy Framework](#7-policy-framework)
8. [QEMU Integration](#8-qemu-integration)
9. [Sequence Flow Diagrams](#9-sequence-flow-diagrams)
10. [Data Structures](#10-data-structures)
11. [Build and Deployment](#11-build-and-deployment)

---

## 1. Executive Summary

OCEAN is a comprehensive CXL 3.0 emulation framework that enables full CXL functionality including:
- Multi-host memory sharing and pooling
- Fabric management
- Dynamic memory allocation
- Coherent memory sharing across multiple hosts

The framework achieves performance within ~3x of projected native CXL 3.0 speeds with complete compatibility with existing CXL software stacks.

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OCEAN Framework                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   QEMU VM   │    │   QEMU VM   │    │   QEMU VM   │    │   QEMU VM   │  │
│  │   (Host 1)  │    │   (Host 2)  │    │   (Host 3)  │    │   (Host N)  │  │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│         │                  │                  │                  │          │
│         └──────────────────┴──────────────────┴──────────────────┘          │
│                                    │                                        │
│                    ┌───────────────┴───────────────┐                        │
│                    │     Communication Layer       │                        │
│                    │   (TCP / SHM / RDMA)          │                        │
│                    └───────────────┬───────────────┘                        │
│                                    │                                        │
│                    ┌───────────────┴───────────────┐                        │
│                    │     CXLMemSim Server          │                        │
│                    │   (Thread-per-Connection)     │                        │
│                    └───────────────┬───────────────┘                        │
│                                    │                                        │
│         ┌──────────────────────────┼──────────────────────────┐             │
│         │                          │                          │             │
│  ┌──────┴──────┐           ┌───────┴───────┐          ┌───────┴───────┐    │
│  │   CXL       │           │    Shared     │          │    Policy     │    │
│  │ Controller  │◄─────────►│    Memory     │◄────────►│   Framework   │    │
│  │             │           │    Manager    │          │               │    │
│  └──────┬──────┘           └───────────────┘          └───────────────┘    │
│         │                                                                   │
│  ┌──────┴──────────────────────────────────────────────────────────────┐   │
│  │                        CXL Topology Tree                             │   │
│  │  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐       │   │
│  │  │ CXLSwitch│───►│ CXLSwitch│───►│CXLExpander│   │CXLExpander│      │   │
│  │  └──────────┘    └──────────┘    └──────────┘    └──────────┘       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Design Principles

1. **Modular Architecture**: Separation of concerns between controller, endpoints, policies, and communication
2. **Thread-Safe Operations**: Extensive use of mutexes, shared_mutex, and atomic operations
3. **Configurable Topology**: Newick tree format for flexible CXL fabric configuration
4. **Multiple Communication Modes**: TCP, Shared Memory (SHM), and RDMA support
5. **Coherency Protocol**: MESI-like protocol (Invalid, Shared, Exclusive, Modified)

---

## 3. Core Components

### 3.1 Component Hierarchy

```
┌─────────────────────────────────────────────────────────────────┐
│                        CXLController                            │
│  (Inherits from CXLSwitch, manages entire CXL fabric)          │
├─────────────────────────────────────────────────────────────────┤
│  - AllocationPolicy*                                            │
│  - MigrationPolicy*                                             │
│  - PagingPolicy*                                                │
│  - CachingPolicy*                                               │
│  - LRUCache                                                     │
│  - CXLCounter                                                   │
│  - device_map<int, CXLMemExpander*>                            │
│  - thread_map<tid, thread_info>                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 ▼
┌─────────────────────┐           ┌─────────────────────┐
│     CXLSwitch       │           │   CXLMemExpander    │
│  (Fabric Switch)    │           │  (Memory Device)    │
├─────────────────────┤           ├─────────────────────┤
│  - expanders[]      │           │  - bandwidth        │
│  - switches[]       │           │  - latency          │
│  - CXLSwitchEvent   │           │  - capacity         │
│  - congestion_lat   │           │  - occupation[]     │
│  - timeseries_map   │           │  - request_queue    │
└─────────────────────┘           │  - credits (R/W)    │
                                  │  - CXLMemExpanderEvent│
                                  └─────────────────────┘
```

### 3.2 Source File Organization

| Directory | Purpose |
|-----------|---------|
| `src/` | Core C++ implementation (controller, server, helpers) |
| `include/` | Header files with class definitions |
| `microbench/` | C/C++ workloads and benchmark utilities |
| `qemu_integration/` | QEMU integration layer and VM launch scripts |
| `script/` | Setup and utility scripts |
| `use_cases/` | Example use cases and demos |
| `workloads/` | Application workloads (GROMACS, TIGON) |

---

## 4. Component Architecture

### 4.1 CXLController (`cxlcontroller.h`, `cxlcontroller.cpp`)

The central orchestrator that manages the entire CXL fabric.

```cpp
class CXLController : public CXLSwitch {
    // Policy Management
    AllocationPolicy *allocation_policy;
    MigrationPolicy *migration_policy;
    PagingPolicy *paging_policy;
    CachingPolicy *caching_policy;
    
    // Memory Tracking
    std::map<uint64_t, occupation_info> occupation;
    std::unordered_map<int, CXLMemExpander*> device_map;
    
    // Thread Management
    std::unordered_map<uint64_t, thread_info> thread_map;
    std::queue<lbr> ring_buffer;  // LBR ring buffer
    
    // Caching
    LRUCache lru_cache;  // Thread-safe LRU cache
    
    // Counters
    CXLCounter counter;  // local, remote, hitm, backinv
};
```

**Key Methods:**
- `construct_topo()`: Parses Newick tree format to build topology
- `insert()`: Handles memory access requests
- `calculate_latency()`: Computes access latency through topology
- `perform_migration()`: Executes data migration based on policy
- `perform_back_invalidation()`: Handles cache coherency invalidations

### 4.2 CXLMemExpander (`cxlendpoint.h`, `cxlendpoint.cpp`)

Represents a CXL Type-3 memory device.

```cpp
class CXLMemExpander : public CXLEndPoint {
    // Device Properties
    EmuCXLBandwidth bandwidth;  // read/write bandwidth
    EmuCXLLatency latency;      // read/write latency
    uint64_t capacity;
    
    // Queue Management (CXL Protocol)
    std::deque<CXLRequest> request_queue_;
    std::map<uint64_t, CXLRequest> in_flight_requests_;
    
    // Credit-based Flow Control
    std::atomic<size_t> read_credits_;
    std::atomic<size_t> write_credits_;
    
    // Pipeline Latencies
    double frontend_latency_;   // 10ns
    double forward_latency_;    // 15ns
    double response_latency_;   // 20ns
};
```

**CXL Protocol Constants:**
- `MAX_QUEUE_SIZE`: 64 entries
- `FLIT_SIZE`: 66 bytes (528/8)
- `INITIAL_CREDITS`: 2 (ResCrd[2])

### 4.3 CXLSwitch (`cxlendpoint.h`)

Represents a CXL fabric switch for hierarchical topologies.

```cpp
class CXLSwitch : public CXLEndPoint {
    std::vector<CXLMemExpander*> expanders;
    std::vector<CXLSwitch*> switches;
    CXLSwitchEvent counter;  // load, store, conflict
    double congestion_latency;  // 200ns default
};
```

### 4.4 LRUCache (`cxlcontroller.h`)

Thread-safe LRU cache with read-write lock optimization.

```cpp
class LRUCache {
    int capacity;
    std::unordered_map<uint64_t, LRUCacheEntry> cache;
    std::list<uint64_t> lru_list;
    mutable std::shared_mutex rwmutex_;
    
    // Operations
    std::optional<uint64_t> get(uint64_t key, uint64_t timestamp);
    void put(uint64_t key, uint64_t value, uint64_t timestamp);
    bool remove(uint64_t key);
    bool contains(uint64_t key) const;
};
```

---

## 5. Communication Layers

### 5.1 Communication Mode Selection

```
┌─────────────────────────────────────────────────────────────┐
│                   Communication Modes                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    TCP      │  │    SHM      │  │    RDMA     │         │
│  │  (Default)  │  │ (/dev/shm)  │  │ (InfiniBand)│         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┴────────────────┘                 │
│                          │                                  │
│              ┌───────────┴───────────┐                      │
│              │  ThreadPerConnection  │                      │
│              │       Server          │                      │
│              └───────────────────────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 TCP Communication

Standard socket-based communication for cross-machine deployment.

```cpp
struct ServerRequest {
    uint8_t op_type;      // 0=READ, 1=WRITE, 2=GET_SHM_INFO
    uint64_t addr;
    uint64_t size;
    uint64_t timestamp;
    uint8_t data[64];     // Cacheline data
};

struct ServerResponse {
    uint8_t status;
    uint64_t latency_ns;
    uint8_t data[64];
};
```

### 5.3 Shared Memory Communication (`shm_communication.h`)

High-performance IPC via `/dev/shm` for same-machine deployment.

```cpp
struct ShmRingBuffer {
    static constexpr size_t RING_SIZE = 1024;
    std::atomic<uint32_t> head;
    std::atomic<uint32_t> tail;
    std::atomic<uint32_t> pending_count;
    
    struct Entry {
        std::atomic<bool> request_ready;
        std::atomic<bool> response_ready;
        ShmRequest request;
        ShmResponse response;
    } entries[RING_SIZE];
};
```

### 5.4 RDMA Communication (`rdma_communication.h`)

InfiniBand RDMA for ultra-low latency communication.

```cpp
struct RDMARequest {
    uint8_t op_type;
    uint64_t addr;
    uint64_t size;
    uint64_t timestamp;
    uint8_t host_id;
    uint64_t virtual_addr;
    uint8_t data[64];
};

class RDMAConnection {
    struct ibv_context* context;
    struct ibv_pd* pd;
    struct ibv_mr* mr;
    struct ibv_qp* qp;
    // Credit-based flow control
    std::atomic<bool> connected;
};
```

---

## 6. Memory Management

### 6.1 Shared Memory Manager (`shared_memory_manager.h`)

Manages real shared memory allocation for CXL memory simulation.

```
┌─────────────────────────────────────────────────────────────┐
│                  Shared Memory Layout                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                    Header                             │  │
│  │  - magic (0x434D454D53484D43)                        │  │
│  │  - version                                            │  │
│  │  - total_size                                         │  │
│  │  - data_offset                                        │  │
│  │  - metadata_offset                                    │  │
│  │  - num_cachelines                                     │  │
│  │  - base_addr                                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Cacheline Data Area                      │  │
│  │  [64B][64B][64B][64B]...[64B]                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Metadata Area (Local Cache)              │  │
│  │  - CachelineMetadata per cacheline                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 Coherency States

```cpp
enum CoherencyState {
    INVALID,    // Cacheline not valid
    SHARED,     // Multiple readers, no writers
    EXCLUSIVE,  // Single owner, clean
    MODIFIED    // Single owner, dirty
};

struct CachelineMetadata {
    CoherencyState state;
    std::set<int> sharers;      // Thread IDs in SHARED state
    int owner;                   // Owner thread ID
    uint64_t last_access_time;
    bool has_dirty_update;       // Back invalidation flag
    uint64_t version;           // Consistency version
    std::mutex lock;            // Per-cacheline lock
};
```

### 6.3 Coherency Protocol Flow

```
┌─────────────────────────────────────────────────────────────┐
│                  Coherency State Transitions                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                      ┌─────────┐                            │
│           ┌─────────►│ INVALID │◄─────────┐                 │
│           │          └────┬────┘          │                 │
│           │               │               │                 │
│      Invalidate      Read │          Invalidate             │
│           │               ▼               │                 │
│           │          ┌─────────┐          │                 │
│           ├──────────│ SHARED  │──────────┤                 │
│           │          └────┬────┘          │                 │
│           │               │               │                 │
│           │          Write│               │                 │
│           │               ▼               │                 │
│           │         ┌──────────┐          │                 │
│           └─────────│EXCLUSIVE │──────────┘                 │
│                     └────┬─────┘                            │
│                          │                                  │
│                     Write│                                  │
│                          ▼                                  │
│                     ┌──────────┐                            │
│                     │ MODIFIED │                            │
│                     └──────────┘                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Policy Framework

### 7.1 Policy Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                      Policy Framework                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    Policy (Base)                     │   │
│  │  virtual int compute_once(CXLController*) = 0       │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│    ┌─────────────────────┼─────────────────────┐           │
│    │                     │                     │           │
│    ▼                     ▼                     ▼           │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│ │ Allocation   │  │  Migration   │  │   Paging     │       │
│ │   Policy     │  │   Policy     │  │   Policy     │       │
│ └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│        │                 │                 │               │
│        ▼                 ▼                 ▼               │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│ │InterleavePolicy│ │HeatAwareMigration│ │HugePagePolicy│   │
│ │NUMAPolicy    │  │FrequencyBased│  │PageTableAware│       │
│ └──────────────┘  │LoadBalancing │  └──────────────┘       │
│                   │LocalityBased │                         │
│                   │LifetimeBased │                         │
│                   │HybridMigration│                        │
│                   └──────────────┘                         │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                  CachingPolicy                       │   │
│  │  - FIFOPolicy                                        │   │
│  │  - FrequencyBasedInvalidationPolicy                 │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Policy Implementations

| Policy Type | Implementation | Description |
|-------------|----------------|-------------|
| **Allocation** | `InterleavePolicy` | Saturate local 90%, interleave remote by latency |
| **Allocation** | `NUMAPolicy` | NUMA-aware allocation based on latency scores |
| **Migration** | `HeatAwareMigrationPolicy` | Migrate based on access frequency threshold |
| **Migration** | `FrequencyBasedMigrationPolicy` | Hot/cold data classification |
| **Migration** | `LoadBalancingMigrationPolicy` | Balance load across devices |
| **Migration** | `LocalityBasedMigrationPolicy` | Migrate based on access patterns |
| **Migration** | `LifetimeBasedMigrationPolicy` | Migrate based on data age |
| **Migration** | `HybridMigrationPolicy` | Combine multiple strategies |
| **Paging** | `HugePagePolicy` | TLB simulation for 4K/2M/1G pages |
| **Paging** | `PageTableAwarePolicy` | Page table walk latency modeling |
| **Caching** | `FIFOPolicy` | First-in-first-out invalidation |
| **Caching** | `FrequencyBasedInvalidationPolicy` | Frequency-based back invalidation |

---

## 8. QEMU Integration

### 8.1 Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    QEMU Integration                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    QEMU Guest                        │   │
│  │  ┌─────────────────────────────────────────────┐    │   │
│  │  │              Guest Application               │    │   │
│  │  └────────────────────┬────────────────────────┘    │   │
│  │                       │                              │   │
│  │  ┌────────────────────▼────────────────────────┐    │   │
│  │  │           CXL DAX Device (/dev/dax0.0)      │    │   │
│  │  └────────────────────┬────────────────────────┘    │   │
│  └───────────────────────┼─────────────────────────────┘   │
│                          │                                  │
│  ┌───────────────────────▼─────────────────────────────┐   │
│  │              qemu_cxl_memsim.c                       │   │
│  │  - cxl_type3_read()                                  │   │
│  │  - cxl_type3_write()                                 │   │
│  │  - Hotness tracking                                  │   │
│  │  - Back invalidation support                         │   │
│  └───────────────────────┬─────────────────────────────┘   │
│                          │                                  │
│  ┌───────────────────────▼─────────────────────────────┐   │
│  │              CXLMemSim Server                        │   │
│  │  - ThreadPerConnectionServer                         │   │
│  │  - SharedMemoryManager                               │   │
│  │  - Coherency Protocol                                │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 QEMU Client API

```c
// Initialize connection to CXLMemSim server
int cxlmemsim_init(const char *host, int port);

// CXL Type-3 Read operation
MemTxResult cxl_type3_read(void* d, uint64_t addr, uint64_t *data,
                           unsigned size, MemTxAttrs attrs);

// CXL Type-3 Write operation
MemTxResult cxl_type3_write(void *d, uint64_t addr, uint64_t data,
                            unsigned size, MemTxAttrs attrs);

// Hotness tracking
uint64_t cxlmemsim_get_hotness(uint64_t addr);

// Back invalidation support
int cxlmemsim_check_invalidation(uint64_t phys_addr, size_t size, void *data);
void cxlmemsim_register_invalidation(uint64_t phys_addr, void *data, size_t size);

// Cleanup
void cxlmemsim_cleanup(void);
```

---

## 9. Sequence Flow Diagrams

### 9.1 CXL Read Operation

```
┌────────┐     ┌────────────┐     ┌────────────┐     ┌──────────────┐     ┌────────────┐
│  QEMU  │     │   Client   │     │   Server   │     │  Controller  │     │  Expander  │
│  Guest │     │   (TCP)    │     │  (Thread)  │     │              │     │            │
└───┬────┘     └─────┬──────┘     └─────┬──────┘     └──────┬───────┘     └─────┬──────┘
    │                │                  │                   │                   │
    │ cxl_type3_read │                  │                   │                   │
    │───────────────►│                  │                   │                   │
    │                │                  │                   │                   │
    │                │  ServerRequest   │                   │                   │
    │                │  (op=READ)       │                   │                   │
    │                │─────────────────►│                   │                   │
    │                │                  │                   │                   │
    │                │                  │ memory_barrier    │                   │
    │                │                  │──────────────────►│                   │
    │                │                  │                   │                   │
    │                │                  │ get_cacheline_    │                   │
    │                │                  │ metadata()        │                   │
    │                │                  │──────────────────►│                   │
    │                │                  │                   │                   │
    │                │                  │ check_back_       │                   │
    │                │                  │ invalidations()   │                   │
    │                │                  │──────────────────►│                   │
    │                │                  │                   │                   │
    │                │                  │ handle_read_      │                   │
    │                │                  │ coherency()       │                   │
    │                │                  │──────────────────►│                   │
    │                │                  │                   │                   │
    │                │                  │                   │ read_cacheline()  │
    │                │                  │                   │──────────────────►│
    │                │                  │                   │                   │
    │                │                  │                   │◄──────────────────│
    │                │                  │                   │      data         │
    │                │                  │                   │                   │
    │                │                  │ calculate_        │                   │
    │                │                  │ latency()         │                   │
    │                │                  │──────────────────►│                   │
    │                │                  │                   │                   │
    │                │  ServerResponse  │                   │                   │
    │                │  (data, latency) │                   │                   │
    │                │◄─────────────────│                   │                   │
    │                │                  │                   │                   │
    │◄───────────────│                  │                   │                   │
    │     data       │                  │                   │                   │
    │                │                  │                   │                   │
```

### 9.2 CXL Write Operation with Coherency

```
┌────────┐     ┌────────────┐     ┌────────────┐     ┌──────────────┐     ┌────────────┐
│  QEMU  │     │   Client   │     │   Server   │     │  Controller  │     │  Expander  │
│  Guest │     │   (TCP)    │     │  (Thread)  │     │              │     │            │
└───┬────┘     └─────┬──────┘     └─────┬──────┘     └──────┬───────┘     └─────┬──────┘
    │                │                  │                   │                   │
    │cxl_type3_write │                  │                   │                   │
    │───────────────►│                  │                   │                   │
    │                │                  │                   │                   │
    │                │  ServerRequest   │                   │                   │
    │                │  (op=WRITE,data) │                   │                   │
    │                │─────────────────►│                   │                   │
    │                │                  │                   │                   │
    │                │                  │ get_cacheline_    │                   │
    │                │                  │ metadata()        │                   │
    │                │                  │──────────────────►│                   │
    │                │                  │                   │                   │
    │                │                  │ Check coherency   │                   │
    │                │                  │ state             │                   │
    │                │                  │──────────────────►│                   │
    │                │                  │                   │                   │
    │                │                  │ ┌─────────────────┴───────────────┐   │
    │                │                  │ │ If SHARED: invalidate_sharers() │   │
    │                │                  │ │ If EXCLUSIVE/MODIFIED:          │   │
    │                │                  │ │   invalidate owner              │   │
    │                │                  │ └─────────────────┬───────────────┘   │
    │                │                  │                   │                   │
    │                │                  │ handle_write_     │                   │
    │                │                  │ coherency()       │                   │
    │                │                  │──────────────────►│                   │
    │                │                  │                   │                   │
    │                │                  │                   │ write_cacheline() │
    │                │                  │                   │──────────────────►│
    │                │                  │                   │                   │
    │                │                  │                   │◄──────────────────│
    │                │                  │                   │       ack         │
    │                │                  │                   │                   │
    │                │                  │ msync() to        │                   │
    │                │                  │ physical memory   │                   │
    │                │                  │──────────────────►│                   │
    │                │                  │                   │                   │
    │                │                  │ register_back_    │                   │
    │                │                  │ invalidation()    │                   │
    │                │                  │──────────────────►│                   │
    │                │                  │                   │                   │
    │                │  ServerResponse  │                   │                   │
    │                │  (status,latency)│                   │                   │
    │                │◄─────────────────│                   │                   │
    │                │                  │                   │                   │
    │◄───────────────│                  │                   │                   │
    │      ack       │                  │                   │                   │
    │                │                  │                   │                   │
```

### 9.3 Multi-Host Memory Sharing

```
┌──────────┐  ┌──────────┐  ┌──────────────────┐  ┌────────────────┐
│  Host 1  │  │  Host 2  │  │  CXLMemSim       │  │ Shared Memory  │
│  (QEMU)  │  │  (QEMU)  │  │  Server          │  │ (/dev/shm)     │
└────┬─────┘  └────┬─────┘  └────────┬─────────┘  └───────┬────────┘
     │             │                 │                    │
     │ WRITE(A,X)  │                 │                    │
     │────────────────────────────►  │                    │
     │             │                 │                    │
     │             │                 │ write_cacheline(A) │
     │             │                 │───────────────────►│
     │             │                 │                    │
     │             │                 │ msync(A)           │
     │             │                 │───────────────────►│
     │             │                 │                    │
     │             │                 │ set_state(A,       │
     │             │                 │   MODIFIED,        │
     │             │                 │   owner=Host1)     │
     │             │                 │───────────────────►│
     │             │                 │                    │
     │◄────────────────────────────  │                    │
     │    ack      │                 │                    │
     │             │                 │                    │
     │             │ READ(A)         │                    │
     │             │────────────────►│                    │
     │             │                 │                    │
     │             │                 │ get_metadata(A)    │
     │             │                 │───────────────────►│
     │             │                 │                    │
     │             │                 │ state=MODIFIED,    │
     │             │                 │ owner=Host1        │
     │             │                 │◄───────────────────│
     │             │                 │                    │
     │             │                 │ downgrade_owner()  │
     │             │                 │ (notify Host1)     │
     │ coherency   │                 │                    │
     │ downgrade   │                 │                    │
     │◄────────────────────────────  │                    │
     │             │                 │                    │
     │             │                 │ set_state(A,       │
     │             │                 │   SHARED,          │
     │             │                 │   sharers=[1,2])   │
     │             │                 │───────────────────►│
     │             │                 │                    │
     │             │                 │ read_cacheline(A)  │
     │             │                 │───────────────────►│
     │             │                 │                    │
     │             │◄────────────────│                    │
     │             │   data(X)       │                    │
     │             │                 │                    │
```

### 9.4 Data Migration Flow

```
┌────────────┐     ┌────────────────┐     ┌────────────┐     ┌────────────┐
│ Controller │     │MigrationPolicy │     │ Expander 1 │     │ Expander 2 │
│            │     │                │     │  (Source)  │     │  (Target)  │
└─────┬──────┘     └───────┬────────┘     └─────┬──────┘     └─────┬──────┘
      │                    │                    │                   │
      │ compute_once()     │                    │                   │
      │───────────────────►│                    │                   │
      │                    │                    │                   │
      │                    │ analyze access     │                   │
      │                    │ patterns           │                   │
      │                    │───────────────────►│                   │
      │                    │                    │                   │
      │                    │◄───────────────────│                   │
      │                    │   access_count     │                   │
      │                    │                    │                   │
      │ get_migration_     │                    │                   │
      │ list()             │                    │                   │
      │───────────────────►│                    │                   │
      │                    │                    │                   │
      │◄───────────────────│                    │                   │
      │ [(addr, size),...]│                    │                   │
      │                    │                    │                   │
      │ perform_migration()│                    │                   │
      │                    │                    │                   │
      │ find source        │                    │                   │
      │ expander           │                    │                   │
      │───────────────────────────────────────►│                   │
      │                    │                    │                   │
      │                    │                    │ copy data         │
      │                    │                    │──────────────────►│
      │                    │                    │                   │
      │                    │                    │ inc_migrate_out() │
      │                    │                    │                   │
      │                    │                    │                   │
      │                    │                    │   inc_migrate_in()│
      │                    │                    │◄──────────────────│
      │                    │                    │                   │
      │ update occupation  │                    │                   │
      │◄───────────────────────────────────────│                   │
      │                    │                    │                   │
```

### 9.5 Server Initialization Flow

```
┌──────────┐     ┌────────────────┐     ┌─────────────────┐     ┌────────────┐
│   Main   │     │ThreadPerConn   │     │SharedMemory     │     │    CXL     │
│          │     │Server          │     │Manager          │     │ Controller │
└────┬─────┘     └───────┬────────┘     └────────┬────────┘     └─────┬──────┘
     │                   │                       │                    │
     │ parse args        │                       │                    │
     │                   │                       │                    │
     │ create policies   │                       │                    │
     │──────────────────────────────────────────────────────────────►│
     │                   │                       │                    │
     │ new CXLController │                       │                    │
     │──────────────────────────────────────────────────────────────►│
     │                   │                       │                    │
     │ load topology     │                       │                    │
     │──────────────────────────────────────────────────────────────►│
     │                   │                       │                    │
     │                   │                       │                    │
     │ new Server(port,  │                       │                    │
     │   controller,     │                       │                    │
     │   capacity)       │                       │                    │
     │──────────────────►│                       │                    │
     │                   │                       │                    │
     │                   │ new SharedMemory      │                    │
     │                   │ Manager(capacity)     │                    │
     │                   │──────────────────────►│                    │
     │                   │                       │                    │
     │ server.start()    │                       │                    │
     │──────────────────►│                       │                    │
     │                   │                       │                    │
     │                   │ shm_manager->         │                    │
     │                   │ initialize()          │                    │
     │                   │──────────────────────►│                    │
     │                   │                       │                    │
     │                   │                       │ create_shared_     │
     │                   │                       │ memory()           │
     │                   │                       │                    │
     │                   │                       │ map_shared_        │
     │                   │                       │ memory()           │
     │                   │                       │                    │
     │                   │                       │ initialize_        │
     │                   │                       │ header()           │
     │                   │                       │                    │
     │                   │◄──────────────────────│                    │
     │                   │       success         │                    │
     │                   │                       │                    │
     │                   │ socket()/bind()/      │                    │
     │                   │ listen()              │                    │
     │                   │                       │                    │
     │ server.run()      │                       │                    │
     │──────────────────►│                       │                    │
     │                   │                       │                    │
     │                   │ accept loop           │                    │
     │                   │ (spawn threads)       │                    │
     │                   │                       │                    │
```

---

## 10. Data Structures

### 10.1 Counter Classes

```cpp
// Atomic counter with compile-time name
template <const char *Name> 
class AtomicCounter {
    std::atomic<uint64_t> value = 0;
    void increment() noexcept;
    uint64_t get() const noexcept;
};

// Switch event counters
class CXLSwitchEvent {
    AtomicCounter<"load"> load;
    AtomicCounter<"store"> store;
    AtomicCounter<"conflict"> conflict;
};

// Memory expander event counters
class CXLMemExpanderEvent {
    AtomicCounter<"load"> load;
    AtomicCounter<"store"> store;
    AtomicCounter<"migrate_in"> migrate_in;
    AtomicCounter<"migrate_out"> migrate_out;
    AtomicCounter<"hit_old"> hit_old;
};

// Global counters
class CXLCounter {
    AtomicCounter<"local"> local;
    AtomicCounter<"remote"> remote;
    AtomicCounter<"hitm"> hitm;
    AtomicCounter<"backinv"> backinv;
};
```

### 10.2 Request/Response Structures

```cpp
// CXL Request (in queue)
struct CXLRequest {
    uint64_t timestamp;
    uint64_t address;
    uint64_t tid;
    bool is_read;
    bool is_write;
    uint64_t issue_time;
    uint64_t complete_time;
};

// Occupation info (memory tracking)
struct occupation_info {
    uint64_t timestamp;
    uint64_t address;
    uint64_t access_count;
};

// Thread info (ROB tracking)
struct thread_info {
    rob_info rob;
    std::queue<int> llcm_type;
    std::queue<int> llcm_type_rob;
};

// ROB info
struct rob_info {
    std::map<int, int64_t> m_bandwidth, m_count;
    int64_t llcm_base, llcm_count, ins_count;
};
```

### 10.3 Page Types

```cpp
enum page_type { 
    CACHELINE,      // 64 bytes
    PAGE,           // 4KB
    HUGEPAGE_2M,    // 2MB
    HUGEPAGE_1G     // 1GB
};
```

---

## 11. Build and Deployment

### 11.1 Build Configuration

```bash
# Standard build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Server mode (without bpftime)
cmake -S . -B build -DSERVER_MODE=ON
cmake --build build -j

# With RDMA support
cmake -S . -B build -DENABLE_RDMA=ON
cmake --build build -j
```

### 11.2 Dependencies

| Dependency | Purpose |
|------------|---------|
| CMake ≥ 3.25 | Build system |
| GCC/Clang (C++20) | Compiler |
| libspdlog-dev | Logging |
| libcxxopts-dev | Command-line parsing |
| librdmacm / libibverbs | RDMA support (optional) |
| Linux headers | BPF support |

### 11.3 Running the Server

```bash
# Basic server
./build/cxlmemsim_server --port 9999 --capacity 256

# With topology
./build/cxlmemsim_server --port 9999 --capacity 256 \
    --topology topology_simple.txt

# With shared memory communication
./build/cxlmemsim_server --port 9999 --capacity 256 \
    --comm-mode shm

# With file backing (for VM sharing)
./build/cxlmemsim_server --port 9999 --capacity 256 \
    --backing-file /path/to/backing.dat

# Debug logging
SPDLOG_LEVEL=debug ./build/cxlmemsim_server --port 9999
```

### 11.4 Topology File Format (Newick)

```
# Simple topology with 2 expanders
(1,2)

# Hierarchical topology with switches
((1,2),(3,4))

# Complex topology
(((1,2),3),(4,(5,6)))
```

### 11.5 Environment Variables

| Variable | Description |
|----------|-------------|
| `SPDLOG_LEVEL` | Log level (trace, debug, info, warn, error) |
| `CXL_BASE_ADDR` | Base address for CXL memory (default: 0) |
| `CXL_TRANSPORT_MODE` | Transport mode (tcp, shm, rdma) |
| `CXL_DAX_PATH` | DAX device path for guest |
| `CXL_DAX_RESET` | Reset allocation counter |
| `CXL_SHIM_VERBOSE` | Enable verbose shim logging |

---

## Appendix A: File Reference

### Core Implementation Files

| File | Description |
|------|-------------|
| `src/main_server.cc` | Thread-per-connection server implementation |
| `src/cxlcontroller.cpp` | CXL controller logic |
| `src/cxlendpoint.cpp` | Expander and switch implementations |
| `src/shared_memory_manager.cc` | Shared memory management |
| `src/shm_communication.cpp` | Shared memory IPC |
| `src/rdma_communication.cpp` | RDMA communication |
| `src/policy.cpp` | Policy implementations |

### Header Files

| File | Description |
|------|-------------|
| `include/cxlcontroller.h` | Controller, policies, LRU cache |
| `include/cxlendpoint.h` | Expander, switch, CXL protocol |
| `include/cxlcounter.h` | Atomic counters |
| `include/policy.h` | All policy classes |
| `include/shm_communication.h` | SHM structures |
| `include/rdma_communication.h` | RDMA structures |
| `include/monitor.h` | Performance monitoring |

### QEMU Integration

| File | Description |
|------|-------------|
| `qemu_integration/src/qemu_cxl_memsim.c` | QEMU client library |
| `qemu_integration/launch_qemu_cxl.sh` | VM launch scripts |
| `qemu_integration/setup_cxl_numa.sh` | NUMA setup |

---

*Document generated for OCEAN CXL Emulation Framework*  
*UC Santa Cruz Sluglab - 2025*
