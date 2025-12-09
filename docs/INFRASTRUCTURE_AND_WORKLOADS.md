# OCEAN Infrastructure Setup & Workload Execution Guide

## Open-source CXL Emulation at Hyperscale Architecture and Networking

**Version:** 1.0  
**Generated:** December 2024  
**Repository:** OCEAN - CXL 3.0 Emulation Framework

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Repository Structure Overview](#2-repository-structure-overview)
3. [Infrastructure Setup](#3-infrastructure-setup)
4. [Core Components Architecture](#4-core-components-architecture)
5. [Script Directory Analysis](#5-script-directory-analysis)
6. [Workload Execution Framework](#6-workload-execution-framework)
7. [Microbenchmarks](#7-microbenchmarks)
8. [QEMU Integration](#8-qemu-integration)
9. [Use Cases and Demos](#9-use-cases-and-demos)
10. [Calibration and Analysis Tools](#10-calibration-and-analysis-tools)
11. [Build and Deployment](#11-build-and-deployment)
12. [Quick Reference Commands](#12-quick-reference-commands)

---

## 1. Executive Summary

OCEAN is a comprehensive CXL 3.0 emulation framework that enables:
- **Multi-host memory sharing and pooling**
- **Fabric management**
- **Dynamic memory allocation**
- **Coherent memory sharing across multiple hosts**

The framework achieves performance within ~3x of projected native CXL 3.0 speeds with complete compatibility with existing CXL software stacks.

### Key Capabilities
| Feature | Description |
|---------|-------------|
| CXL 3.0 Emulation | Full protocol support including Type-1, Type-2, Type-3 devices |
| Multi-Host Support | QEMU-based VM orchestration for multi-node simulation |
| Policy Framework | Allocation, Migration, Paging, and Caching policies |
| Workload Support | GROMACS, TIGON, GAPBS, LLaMA, and custom microbenchmarks |
| Communication | TCP, Shared Memory (SHM), and RDMA transport modes |

---

## 2. Repository Structure Overview

```
OCEAN/
├── src/                    # Core C++ implementation
│   ├── main_server.cc      # CXLMemSim server entry point
│   ├── cxlcontroller.cpp   # Central CXL fabric controller
│   ├── cxlendpoint.cpp     # CXL Type-3 memory device
│   ├── policy.cpp          # Policy implementations
│   ├── shared_memory_manager.cc  # SHM management
│   └── ...
├── include/                # Header files
│   ├── cxlcontroller.h     # Controller class definitions
│   ├── policy.h            # Policy interfaces
│   └── ...
├── script/                 # Setup and utility scripts (24 files)
│   ├── setup_host.sh       # Host environment setup
│   ├── setup_network.sh    # Network bridge configuration
│   ├── setup_cxl_numa.sh   # CXL NUMA configuration
│   ├── calibrate_memory_latency.py  # Latency calibration
│   ├── get_all_results.py  # Workload runner
│   └── ...
├── microbench/             # Microbenchmarks (36 files)
│   ├── ld*.cpp             # Load benchmarks
│   ├── st*.cpp             # Store benchmarks
│   ├── test_dax_*.c        # DAX litmus tests
│   └── ...
├── qemu_integration/       # QEMU integration layer
│   ├── launch_qemu_cxl*.sh # VM launch scripts
│   ├── start_server.sh     # Server startup
│   └── ...
├── workloads/              # Application workloads
│   ├── gromacs/            # Molecular dynamics simulation
│   └── tigon/              # Distributed database
├── use_cases/              # Advanced use case demos
│   ├── topology_guided_procurement/
│   ├── predictive_placement/
│   ├── dynamic_migration/
│   └── ...
├── artifact/               # Results and logs storage
└── docs/                   # Documentation
```

---

## 2.5 Workload Execution Modes - Where Do Workloads Run?

OCEAN supports **two distinct execution modes** for workloads:

### Mode 1: Host-Based Execution (Process Tracing)

```
┌─────────────────────────────────────────────────────────────────┐
│                         HOST SYSTEM                              │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    CXLMemSim Binary                       │   │
│  │  ./CXLMemSim -t ./microbench/ld1 -p 1000                 │   │
│  │                         │                                 │   │
│  │         ┌───────────────┴───────────────┐                │   │
│  │         ▼                               ▼                │   │
│  │  ┌─────────────┐              ┌─────────────────┐        │   │
│  │  │   Target    │   ptrace     │   CXL Memory    │        │   │
│  │  │  Process    │◄────────────►│   Controller    │        │   │
│  │  │ (workload)  │   intercept  │   + Policies    │        │   │
│  │  └─────────────┘              └─────────────────┘        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Workload runs: DIRECTLY ON HOST (traced by CXLMemSim)          │
│  CXL Memory:    EMULATED via latency injection                  │
└─────────────────────────────────────────────────────────────────┘
```

**How it works:**
1. CXLMemSim **spawns the target workload** as a child process
2. Uses **ptrace** and **PEBS** (Processor Event-Based Sampling) to intercept memory accesses
3. **Injects latency** to simulate CXL memory behavior
4. Collects PMU counters for analysis

**Use cases:**
- Microbenchmarks (`ld1`, `st64`, etc.)
- Single-node workloads (GAPBS, LLaMA, GROMACS single-node)
- Performance characterization and calibration

**Example:**
```bash
# Workload runs on HOST, CXL behavior is emulated
./CXLMemSim -t ./microbench/ld1 -p 1000 \
    -l "200,250,200,250,200,250" \
    -b "50,50,50,50,50,50"
```

---

### Mode 2: Guest VM Execution (Multi-Host Emulation)

```
┌─────────────────────────────────────────────────────────────────┐
│                         HOST SYSTEM                              │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐     │
│  │              CXLMemSim Server (Port 9999)               │     │
│  │  ┌─────────────────────────────────────────────────┐   │     │
│  │  │  Shared Memory: /dev/shm/cxlmemsim_shared (2GB) │   │     │
│  │  └─────────────────────────────────────────────────┘   │     │
│  └──────────────────────────┬─────────────────────────────┘     │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐               │
│         │                   │                   │               │
│  ┌──────┴──────┐    ┌───────┴───────┐   ┌───────┴───────┐      │
│  │  QEMU VM 0  │    │   QEMU VM 1   │   │   QEMU VM N   │      │
│  │   (node0)   │    │    (node1)    │   │    (nodeN)    │      │
│  │             │    │               │   │               │      │
│  │ ┌─────────┐ │    │ ┌───────────┐ │   │ ┌───────────┐ │      │
│  │ │WORKLOAD │ │    │ │ WORKLOAD  │ │   │ │ WORKLOAD  │ │      │
│  │ │(GROMACS)│ │    │ │ (GROMACS) │ │   │ │ (TIGON)   │ │      │
│  │ └────┬────┘ │    │ └─────┬─────┘ │   │ └─────┬─────┘ │      │
│  │      │      │    │       │       │   │       │       │      │
│  │ ┌────┴────┐ │    │ ┌─────┴─────┐ │   │ ┌─────┴─────┐ │      │
│  │ │CXL Type3│ │    │ │CXL Type-3 │ │   │ │CXL Type-3 │ │      │
│  │ │/dev/dax │ │    │ │ /dev/dax  │ │   │ │ /dev/dax  │ │      │
│  │ └─────────┘ │    │ └───────────┘ │   │ └───────────┘ │      │
│  └─────────────┘    └───────────────┘   └───────────────┘      │
│                                                                  │
│  Workloads run: INSIDE GUEST VMs                                │
│  CXL Memory:    REAL shared memory via DAX device               │
│  Multi-host:    VMs communicate via bridge network              │
└─────────────────────────────────────────────────────────────────┘
```

**How it works:**
1. **CXLMemSim Server** runs on host, manages shared memory pool
2. **QEMU VMs** boot with CXL Type-3 device backed by shared memory
3. **Inside each VM**: CXL memory appears as `/dev/dax0.0` (DAX device)
4. **Workloads run inside VMs** and access CXL memory via DAX
5. **MPI shim layer** redirects MPI allocations to CXL memory

**Use cases:**
- Multi-host distributed workloads (GROMACS MPI, TIGON)
- CXL memory pooling scenarios
- Coherent memory sharing experiments
- Real CXL software stack testing

**Example:**
```bash
# On HOST: Start server
./start_server.sh 9999 topology_simple.txt

# On HOST: Launch VMs
sudo ./launch_qemu_cxl.sh   # VM0
sudo ./launch_qemu_cxl1.sh  # VM1

# INSIDE VM (Guest OS): Run workload
export CXL_DAX_PATH="/dev/dax0.0"
export LD_PRELOAD=/root/libmpi_cxl_shim.so
mpirun -np 2 -hostfile hostfile ./gmx_mpi mdrun -s input.tpr
```

---

### Comparison of Execution Modes

| Aspect | Mode 1: Host-Based | Mode 2: Guest VM |
|--------|-------------------|------------------|
| **Where workload runs** | Directly on host | Inside QEMU guest OS |
| **CXL memory** | Emulated (latency injection) | Real shared memory (DAX) |
| **Multi-host support** | No (single process) | Yes (multiple VMs) |
| **Accuracy** | Approximate (PMU-based) | High (actual memory access) |
| **Setup complexity** | Low (single binary) | High (QEMU + VMs) |
| **Use case** | Benchmarking, calibration | Distributed apps, pooling |
| **Scripts** | `get_all_results.py` | `launch_qemu_cxl*.sh` |

---

### Guest OS Environment (Inside VMs)

When workloads run inside QEMU VMs, the guest OS sees:

```
Guest OS View:
├── /dev/dax0.0          # CXL memory as DAX device
├── NUMA Node 1          # CXL memory as separate NUMA node
├── 192.168.100.X/24     # Network to other VMs
└── /mnt/hostshm/        # 9p mount to host /dev/shm (optional)
```

**Guest OS Setup** (automatic via `setup_cxl_numa.sh`):
1. Load CXL kernel modules
2. Create CXL region and DAX namespace
3. Configure network interface
4. (Optional) Online CXL memory as system RAM

---

## 3. Infrastructure Setup

### 3.1 Prerequisites

#### System Requirements
- **OS**: Linux with kernel 5.15+ (CXL support)
- **CPU**: x86_64 with AVX, AVX2, AVX512 support
- **RAM**: Minimum 16GB (32GB recommended for multi-VM setups)
- **Storage**: 50GB+ for QEMU images and artifacts

#### Required Packages
```bash
# Core dependencies
sudo apt update && sudo apt install \
    llvm-dev clang libbpf-dev libclang-dev \
    python3-pip libcxxopts-dev libboost-dev \
    nvidia-cuda-dev libfmt-dev libspdlog-dev \
    librdmacm-dev

# QEMU build dependencies
sudo apt-get install \
    libglib2.0-dev libgcrypt20-dev zlib1g-dev \
    autoconf automake libtool bison flex \
    libpixman-1-dev bc make ninja-build \
    libncurses-dev libelf-dev libssl-dev \
    debootstrap libcap-ng-dev libattr1-dev \
    libslirp-dev libslirp0 libpmem-dev

# Python dependencies
pip install tomli pandas matplotlib numpy pyyaml scikit-learn torch seaborn
```

### 3.2 Host Setup Script (`script/setup_host.sh`)

The primary setup script performs:
1. **Package installation** - All required system dependencies
2. **Git submodule initialization** - Fetches bpftime and other dependencies
3. **QEMU build** - Compiles custom QEMU with CXL and libpmem support

```bash
#!/bin/bash
# Execute from repository root
bash ./script/setup_host.sh
```

**QEMU Configuration:**
```bash
../configure --prefix=/usr/local \
    --target-list=x86_64-softmmu \
    --enable-debug \
    --enable-libpmem \
    --enable-slirp
```

### 3.3 Network Setup (`script/setup_network.sh`)

Creates a bridge network for multi-VM communication:

```bash
# Usage: setup_network.sh <num_vms>
bash ./script/setup_network.sh 2

# Creates:
# - br0: Bridge interface (192.168.100.1/24)
# - tap0, tap1, ...: TAP interfaces for each VM
```

**Network Topology:**
```
┌─────────────────────────────────────────────────┐
│                   Host System                    │
│  ┌─────────────────────────────────────────┐    │
│  │              br0 (192.168.100.1/24)      │    │
│  └─────┬─────────────┬─────────────┬───────┘    │
│        │             │             │            │
│     tap0          tap1          tap2           │
│        │             │             │            │
│  ┌─────┴─────┐ ┌─────┴─────┐ ┌─────┴─────┐     │
│  │  QEMU VM  │ │  QEMU VM  │ │  QEMU VM  │     │
│  │  node0    │ │  node1    │ │  node2    │     │
│  │ .100.10   │ │ .100.11   │ │ .100.12   │     │
│  └───────────┘ └───────────┘ └───────────┘     │
└─────────────────────────────────────────────────┘
```

### 3.4 CXL NUMA Configuration (`script/setup_cxl_numa.sh`)

Configures CXL memory as a NUMA node inside VMs:

**Key Functions:**
1. **Kernel Module Loading:**
   ```bash
   modprobe cxl_core cxl_pci cxl_acpi cxl_port cxl_mem
   modprobe dax device_dax kmem
   ```

2. **CXL Region Creation:**
   ```bash
   cxl create-region -m -d decoder0.0 -w 1 mem0 -s 256M
   ```

3. **DAX Namespace Setup:**
   ```bash
   ndctl create-namespace -m dax -r region0
   ```

4. **NUMA Node Configuration:**
   ```bash
   daxctl reconfigure-device --mode=system-ram <dax_device>
   ```

5. **Network Configuration:**
   ```bash
   ip link set enp0s2 up
   ip addr add 192.168.100.10/24 dev enp0s2
   ip route add default via 192.168.100.1
   ```

---

## 4. Core Components Architecture

### 4.1 CXLMemSim Server (`src/main_server.cc`)

The central server that manages CXL memory simulation:

```
┌─────────────────────────────────────────────────────────────┐
│                     CXLMemSim Server                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │ TCP Listener    │  │ SHM Manager     │                   │
│  │ (Port 9999)     │  │ (/dev/shm/...)  │                   │
│  └────────┬────────┘  └────────┬────────┘                   │
│           │                    │                            │
│           └──────────┬─────────┘                            │
│                      ▼                                      │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              CXL Controller                          │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │  │ Allocation  │ │ Migration   │ │ Caching     │    │   │
│  │  │ Policy      │ │ Policy      │ │ Policy      │    │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  └─────────────────────────────────────────────────────┘   │
│                      │                                      │
│  ┌───────────────────┴───────────────────┐                 │
│  │           CXL Topology Tree            │                 │
│  │  ┌──────────┐    ┌──────────────────┐ │                 │
│  │  │ CXLSwitch│───►│ CXLMemExpander   │ │                 │
│  │  └──────────┘    │ (Type-3 Device)  │ │                 │
│  │                  └──────────────────┘ │                 │
│  └────────────────────────────────────────┘                 │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Policy Framework

| Policy Type | Options | Description |
|-------------|---------|-------------|
| **Allocation** | `none`, `interleave`, `numa` | Memory allocation strategy |
| **Migration** | `none`, `heataware`, `frequency`, `loadbalance`, `locality`, `lifetime`, `hybrid` | Data migration triggers |
| **Paging** | `none`, `hugepage`, `pagetableaware` | Page management |
| **Caching** | `none`, `fifo`, `frequency` | Cache replacement |

### 4.3 Communication Modes

| Mode | Environment Variable | Use Case |
|------|---------------------|----------|
| **TCP** | `CXL_MEMSIM_HOST`, `CXL_MEMSIM_PORT` | Remote/distributed setup |
| **SHM** | `CXL_TRANSPORT_MODE=shm` | Local high-performance |
| **RDMA** | RDMA libraries required | Low-latency networking |

---

## 5. Script Directory Analysis

### 5.1 Setup Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `setup_host.sh` | Install dependencies, build QEMU | `bash ./script/setup_host.sh` |
| `setup_network.sh` | Create bridge network for VMs | `bash ./script/setup_network.sh <num_vms>` |
| `setup_cxl_numa.sh` | Configure CXL as NUMA node (in VM) | Auto-run at VM boot |

### 5.2 Workload Execution Scripts

| Script | Purpose | Key Features |
|--------|---------|--------------|
| `get_all_results.py` | Automated workload runner | Multi-workload, policy combinations, artifact logging |
| `get_number.py` | Microbenchmark runner | Load/store benchmark execution |
| `run_gromacs.sh` | GROMACS setup and run | Molecular dynamics simulation |

### 5.3 Calibration and Analysis Scripts

| Script | Purpose | Input/Output |
|--------|---------|--------------|
| `calibrate_memory_latency.py` | Calibrate RoBSim parameters | gem5 trace → calibration JSON |
| `apply_calibration.py` | Apply calibration to rob.cpp | JSON config → source modification |
| `calibrate_example.sh` | Example calibration workflow | Demo script |

### 5.4 Result Processing Scripts

| Script | Purpose | Output Format |
|--------|---------|---------------|
| `get_pebs.py` | PEBS sampling analysis | PDF plots |
| `get_latency.py` | Latency extraction | Statistics |
| `get_slowdown.py` | Slowdown calculation | Metrics |
| `get_gem5_slowdown.py` | gem5 comparison | Ratios |
| `get_policy.py` | Policy effectiveness | Analysis |

### 5.5 Result Visualization Scripts

| Script | Purpose |
|--------|---------|
| `ld_result.py` | Load benchmark results |
| `st_result.py` | Store benchmark results |
| `ld_st_result.py` | Combined load/store results |
| `wb_result.py` | Write-back results |
| `ipc_rob_result.py` | IPC and ROB analysis |

---

## 6. Workload Execution Framework

### 6.1 Automated Workload Runner (`get_all_results.py`)

**Supported Workloads:**

| Workload | Path | Description |
|----------|------|-------------|
| `gapbs` | `../workloads/gapbs` | Graph algorithms (bc, bfs, cc, pr, sssp, tc) |
| `gromacs` | `../workloads/gromacs/build/bin` | Molecular dynamics |
| `llama` | `../workloads/llama.cpp/build/bin` | LLM inference |
| `vsag` | `/usr/bin/python3` | Vector similarity search |
| `mlc` | `../workloads/MLC` | Memory latency checker |
| `mcf` | `../workloads/mcf` | SPEC CPU benchmark |
| `memcached` | `./` | Key-value store |

**Usage Examples:**

```bash
# Run all workloads with default settings
python3 script/get_all_results.py --run-original --run-cxlmemsim

# Run specific workload with policy combinations
python3 script/get_all_results.py \
    --workloads gapbs \
    --run-cxlmemsim \
    --run-policy-combinations \
    --latency "200,250,200,250,200,250" \
    --bandwidth "50,50,50,50,50,50"

# Run with specific policies
python3 script/get_all_results.py \
    --workloads gromacs \
    --allocation-policies numa interleave \
    --migration-policies heataware frequency
```

**Command Line Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--workloads` | Workloads to run | All |
| `--programs` | Specific programs | All |
| `--run-original` | Run without CXLMemSim | False |
| `--run-cxlmemsim` | Run with CXLMemSim | False |
| `--pebs-period` | PEBS sampling period | 1000 |
| `--latency` | Latency settings (6 values) | "200,250,200,250,200,250" |
| `--bandwidth` | Bandwidth settings (6 values) | "50,50,50,50,50,50" |
| `--run-policy-combinations` | Test all policy combos | False |
| `--collect-system-info` | Gather dmesg, dmidecode, lspci | False |

### 6.2 GROMACS Workload

**MPI CXL Shim Layer** (`workloads/gromacs/`):

Intercepts MPI calls to redirect memory operations to CXL memory:

```bash
# Build the shim library
cd workloads/gromacs
make

# Run GROMACS with CXL memory
export LD_PRELOAD=/path/to/libmpi_cxl_shim.so
export CXL_DAX_PATH=/dev/dax0.0
export CXL_DAX_RESET=1
export CXL_SHIM_VERBOSE=1

mpirun --allow-run-as-root -np 2 -hostfile hostfile \
    -x CXL_DAX_PATH -x CXL_DAX_RESET -x CXL_SHIM_VERBOSE -x LD_PRELOAD \
    ./gmx_mpi mdrun -s benchMEM.tpr -nsteps 10000 -resethway -ntomp 1
```

**Environment Variables:**

| Variable | Description |
|----------|-------------|
| `CXL_DAX_PATH` | DAX device path (e.g., `/dev/dax0.0`) |
| `CXL_MEM_SIZE` | CXL memory pool size (default: 4GB) |
| `CXL_SHIM_VERBOSE` | Enable verbose logging |
| `CXL_SHIM_ALLOC` | Redirect MPI_Alloc_mem to CXL |
| `CXL_SHIM_WIN` | Redirect MPI window allocations |

### 6.3 TIGON Workload

Distributed database workload:

```bash
cd workloads/tigon
./scripts/setup.sh HOST
./emulation/image/make_vm_img.sh
sudo ./emulation/start_vms.sh --using-old-img --cxl 0 5 2 0 1
./scripts/setup.sh VMS 2
./scripts/run.sh COMPILE_SYNC 2
./scripts/run_tpcc_dax.sh TwoPLPasha 2 3 mixed 10 15 1 0 1 Clock OnDemand 200000000 1 WriteThrough None 15 5 GROUP_WAL 20000 0 0
```

---

## 7. Microbenchmarks

### 7.1 Benchmark Categories

Located in `microbench/`:

| Category | Files | Description |
|----------|-------|-------------|
| **Load Benchmarks** | `ld.cpp`, `ld_serial.cpp`, `ld_nt.cpp`, `ld_base.cpp` | Memory load patterns |
| **Store Benchmarks** | `st.cpp`, `st_serial.cpp` | Memory store patterns |
| **Memory Allocation** | `malloc.c`, `calloc.c`, `sbrk.c`, `mmap_*.c` | Allocation methods |
| **Cache Tests** | `cache-miss.c`, `cache-thrash.c` | Cache behavior |
| **DAX Litmus Tests** | `test_dax_litmus_*.c` | Memory ordering tests |
| **Bandwidth** | `bw.cpp` | Bandwidth measurement |
| **Pointer Chasing** | `ptr-chasing.cpp` | Latency measurement |

### 7.2 Fence Count Variants

Benchmarks are compiled with different fence counts (1, 2, 4, 8, 16, 32, 64, 128, 256):

```cmake
add_executable(ld1 ld.cpp)
target_compile_definitions(ld1 PRIVATE -DFENCE_COUNT=1)

add_executable(ld256 ld.cpp)
target_compile_definitions(ld256 PRIVATE -DFENCE_COUNT=256)
```

**Available Variants:**
- `ld1` through `ld256` - Regular loads
- `ld_serial1` through `ld_serial256` - Serial loads
- `ld_nt1` through `ld_nt256` - Non-temporal loads
- `st1` through `st256` - Stores
- `st_serial1` through `st_serial256` - Serial stores

### 7.3 DAX Litmus Tests

Memory ordering verification:

| Test | Purpose |
|------|---------|
| `test_dax_litmus_mp` | Message passing pattern |
| `test_dax_litmus_sb` | Store buffering pattern |
| `test_dax_litmus_atomic` | Atomic operations |
| `test_dax_litmus_tearing` | Store tearing detection |

### 7.4 Building Microbenchmarks

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Binaries are in build/microbench/
ls microbench/
```

---

## 8. QEMU Integration

### 8.1 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Host System                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                  CXLMemSim Server                    │    │
│  │                    (Port 9999)                       │    │
│  └──────────────────────────┬──────────────────────────┘    │
│                             │                               │
│           ┌─────────────────┼─────────────────┐             │
│           │                 │                 │             │
│  ┌────────┴────────┐ ┌──────┴──────┐ ┌───────┴───────┐     │
│  │    QEMU VM 0    │ │  QEMU VM 1  │ │   QEMU VM N   │     │
│  │  (node0)        │ │  (node1)    │ │   (nodeN)     │     │
│  │                 │ │             │ │               │     │
│  │ ┌─────────────┐ │ │ ┌─────────┐ │ │ ┌───────────┐ │     │
│  │ │ CXL Type-3  │ │ │ │CXL Type3│ │ │ │CXL Type-3 │ │     │
│  │ │ Memory Dev  │ │ │ │Memory   │ │ │ │Memory Dev │ │     │
│  │ └─────────────┘ │ │ └─────────┘ │ │ └───────────┘ │     │
│  └─────────────────┘ └─────────────┘ └───────────────┘     │
│                                                             │
│  Shared Memory: /dev/shm/cxlmemsim_shared                  │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Starting the Server

```bash
cd qemu_integration/build

# Start server with default topology
./start_server.sh 9999 topology_simple.txt

# Or directly:
./cxlmemsim_server 9999 topology_simple.txt
```

### 8.3 Launching VMs

**VM Launch Script** (`launch_qemu_cxl.sh`):

```bash
#!/bin/bash
QEMU_BINARY=/usr/local/bin/qemu-system-x86_64
export CXL_TRANSPORT_MODE=shm
export CXL_MEMSIM_HOST=127.0.0.1
export CXL_MEMSIM_PORT=9999

exec $QEMU_BINARY \
    --enable-kvm \
    -cpu qemu64,+xsave,+rdtscp,+avx,+avx2,+sse4.1,+sse4.2,+avx512f,... \
    -m 16G,maxmem=32G,slots=8 \
    -smp 4 \
    -M q35,cxl=on \
    -kernel ./bzImage \
    -append "root=/dev/sda rw console=ttyS0,115200 nokaslr" \
    -drive file=./qemu.img,index=0,media=disk,format=raw \
    -netdev tap,id=net0,ifname=tap0,script=no,downscript=no \
    -device virtio-net-pci,netdev=net0,mac=52:54:00:00:00:01 \
    -fsdev local,security_model=none,id=fsdev0,path=/dev/shm \
    -device virtio-9p-pci,id=fs0,fsdev=fsdev0,mount_tag=hostshm \
    -device pxb-cxl,bus_nr=12,bus=pcie.0,id=cxl.1 \
    -device cxl-rp,port=0,bus=cxl.1,id=root_port13,chassis=0,slot=0 \
    -device cxl-type3,bus=root_port13,persistent-memdev=cxl-mem1,... \
    -object memory-backend-file,id=cxl-mem1,share=on,mem-path=/dev/shm/cxlmemsim_shared,size=2G \
    -M cxl-fmw.0.targets.0=cxl.1,cxl-fmw.0.size=4G \
    -nographic
```

**Key QEMU Options:**

| Option | Purpose |
|--------|---------|
| `-M q35,cxl=on` | Enable CXL support |
| `-device pxb-cxl` | CXL host bridge |
| `-device cxl-rp` | CXL root port |
| `-device cxl-type3` | CXL Type-3 memory device |
| `-object memory-backend-file` | Shared memory backend |
| `-M cxl-fmw.0.*` | CXL fixed memory window |

### 8.4 Multi-VM Setup

```bash
# Terminal 1: Start server
./start_server.sh 9999 topology_simple.txt

# Terminal 2: Launch first VM
sudo ./launch_qemu_cxl.sh

# Terminal 3: Launch second VM
sudo ./launch_qemu_cxl1.sh

# Inside VMs: Configure network
# VM0: 192.168.100.10
# VM1: 192.168.100.11
```

---

## 9. Use Cases and Demos

### 9.1 Available Use Cases

Located in `use_cases/`:

| Use Case | Directory | Purpose |
|----------|-----------|---------|
| **Topology-Guided Procurement** | `topology_guided_procurement/` | Hardware purchasing decisions |
| **Predictive Placement** | `predictive_placement/` | ML-based data placement |
| **Dynamic Migration** | `dynamic_migration/` | Adaptive migration policies |
| **Memory Tiering** | `memory_tiering/` | Multi-tier memory management |
| **Production Profiling** | `production_profiling/` | Workload analysis |
| **Procurement Decision** | `procurement_decision/` | TCO analysis |

### 9.2 Running All Demos

```bash
cd use_cases
./run_all_demos.sh
```

**Demo Output:**
```
Results saved to: demo_results_YYYYMMDD_HHMMSS/
├── DEMO_SUMMARY.md
├── procurement_results/
│   ├── hardware_comparison.png
│   └── procurement_summary.md
├── placement_results/
│   ├── placement_distribution.png
│   └── migration_plan.md
└── migration_results/
    ├── policy_comparison.png
    └── policy_recommendations.md
```

### 9.3 Individual Use Case Execution

**Topology-Guided Procurement:**
```bash
cd use_cases/topology_guided_procurement
python3 topology_procurement_advisor.py \
    --cxlmemsim ../../build/CXLMemSim \
    --workloads workload_requirements.yaml \
    --constraints procurement_constraints.yaml \
    --output ./results
```

**Predictive Placement:**
```bash
cd use_cases/predictive_placement
python3 topology_placement_predictor.py \
    --cxlmemsim ../../build/CXLMemSim \
    --topology topology_config.yaml \
    --workload workload_trace.yaml \
    --output ./results
```

**Dynamic Migration:**
```bash
cd use_cases/dynamic_migration
python3 migration_policy_engine.py \
    --cxlmemsim ../../build/CXLMemSim \
    --topology migration_topology.yaml \
    --evaluate \
    --duration 300 \
    --output ./results
```

---

## 10. Calibration and Analysis Tools

### 10.1 Memory Latency Calibration

**Purpose:** Calibrate RoBSim parameters to match gem5 simulation data.

**Workflow:**
```
gem5 Trace ─────┐
                ├──► calibrate_memory_latency.py ──► calibrated_params.json
CXLMemSim Trace ┘                                          │
                                                           ▼
                                              apply_calibration.py
                                                           │
                                                           ▼
                                                    rob.cpp (modified)
```

**Usage:**
```bash
# Step 1: Generate calibration parameters
python3 script/calibrate_memory_latency.py \
    --gem5-trace gem5_trace.out \
    --cxlmemsim-trace cxlmemsim_trace.out \
    --output-config calibrated_params.json \
    --target-ratio 1.0 \
    --output-plot comparison.png

# Step 2: Apply calibration
python3 script/apply_calibration.py \
    --config calibrated_params.json \
    --rob-file src/rob.cpp

# Step 3: Rebuild
cd build && make -j$(nproc)
```

**Calibration Parameters:**

| Parameter | Description |
|-----------|-------------|
| `base_latency_multiplier` | Scaling factor for base latency |
| `congestion_factor` | Congestion impact factor |
| `min_latency_threshold` | Minimum latency floor |
| `stall_multiplier` | Stall cycle scaling |
| `instruction_latency_adjustment` | Per-instruction type adjustments |

### 10.2 PEBS Analysis (`get_pebs.py`)

Analyzes PEBS (Processor Event-Based Sampling) data:

```bash
# Run tests with different PEBS periods
python3 script/get_pebs.py \
    --targets ./microbench/ld1 \
    --pebs-periods 1 10 100 1000 10000 \
    --output-dir ./results \
    --runs 3

# Plot from existing logs
python3 script/get_pebs.py \
    --only-plot \
    --output-dir ./results \
    --output-file combined_pebs_analysis.pdf
```

### 10.3 Result Extraction

**Latency Extraction:**
```bash
python3 script/get_latency.py <log_file>
```

**Slowdown Calculation:**
```bash
python3 script/get_slowdown.py <baseline_log> <cxl_log>
```

**Policy Analysis:**
```bash
python3 script/get_policy.py --artifact-dir ../artifact
```

---

## 11. Build and Deployment

### 11.1 CMake Configuration

```bash
# Standard build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Server-only build (no bpftime dependencies)
cmake -S . -B build -DSERVER_MODE=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# With RDMA support
cmake -S . -B build -DENABLE_RDMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

### 11.2 Build Targets

| Target | Description |
|--------|-------------|
| `cxlmemsim` | Static library of CXL core functionality |
| `cxlmemsim_server` | Main server executable |
| `cxlmemsim_server_rdma` | RDMA-enabled server |
| `cxlmemsim_latency` | Latency calculator (server mode only) |
| Microbenchmarks | `ld*`, `st*`, `test_dax_*`, etc. |

### 11.3 QEMU Integration Build

```bash
cd qemu_integration
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
```

**Outputs:**
- `libCXLMemSim.so` - QEMU interception library
- `cxlmemsim_server` - Server executable

---

## 12. Quick Reference Commands

### 12.1 Complete Setup Sequence

```bash
# 1. Clone and setup
git clone https://github.com/cxl-emu/OCEAN.git
cd OCEAN
bash ./script/setup_host.sh

# 2. Network setup (for 2 VMs)
bash ./script/setup_network.sh 2

# 3. Build QEMU integration
cd qemu_integration
mkdir build && cd build
cmake .. && make -j$(nproc)
sudo make install

# 4. Download VM images
wget https://asplos.dev/about/qemu.img
wget https://asplos.dev/about/bzImage

# 5. Start server
./start_server.sh 9999 topology_simple.txt

# 6. Launch VMs (separate terminals)
sudo ../launch_qemu_cxl.sh
sudo ../launch_qemu_cxl1.sh
```

### 12.2 Running Workloads

```bash
# Microbenchmarks
cd build
./microbench/ld1
./microbench/st64

# With CXLMemSim
./CXLMemSim -t ./microbench/ld1 -p 1000 -l "100,100,100,100,100,100"

# Full workload suite
python3 script/get_all_results.py \
    --run-cxlmemsim \
    --run-policy-combinations \
    --collect-system-info
```

### 12.3 Analysis Commands

```bash
# Calibration
python3 script/calibrate_memory_latency.py \
    --gem5-trace trace.out \
    --cxlmemsim-trace cxl_trace.out \
    --output-config params.json

# PEBS analysis
python3 script/get_pebs.py --only-plot --output-dir ./results

# Use case demos
cd use_cases && ./run_all_demos.sh
```

### 12.4 Environment Variables Reference

| Variable | Purpose | Example |
|----------|---------|---------|
| `CXL_MEMSIM_HOST` | Server IP | `127.0.0.1` |
| `CXL_MEMSIM_PORT` | Server port | `9999` |
| `CXL_TRANSPORT_MODE` | Transport type | `shm`, `tcp`, `rdma` |
| `CXL_DAX_PATH` | DAX device path | `/dev/dax0.0` |
| `CXL_DAX_RESET` | Reset allocation counter | `1` |
| `CXL_SHIM_VERBOSE` | Enable verbose logging | `1` |
| `SPDLOG_LEVEL` | Log level | `debug`, `info` |
| `CXL_BASE_ADDR` | Base address offset | `0` |

---

## Appendix A: File Reference

### Script Directory (`script/`)

| File | Lines | Purpose |
|------|-------|---------|
| `setup_host.sh` | 18 | Host environment setup |
| `setup_network.sh` | 21 | Network bridge creation |
| `setup_cxl_numa.sh` | 161 | CXL NUMA configuration |
| `calibrate_memory_latency.py` | 426 | Latency calibration |
| `apply_calibration.py` | 151 | Apply calibration to source |
| `get_all_results.py` | 406 | Workload automation |
| `get_pebs.py` | 295 | PEBS analysis |
| `get_number.py` | 342 | Microbenchmark runner |
| `get_latency.py` | ~100 | Latency extraction |
| `get_slowdown.py` | ~100 | Slowdown calculation |
| `get_policy.py` | ~300 | Policy analysis |
| `run_gromacs.sh` | 8 | GROMACS setup |
| `*_result.py` | Various | Result visualization |

### Microbenchmark Directory (`microbench/`)

| Category | Count | Files |
|----------|-------|-------|
| Load benchmarks | 45+ | `ld*.cpp` variants |
| Store benchmarks | 18+ | `st*.cpp` variants |
| Memory allocation | 5 | `malloc.c`, `calloc.c`, `sbrk.c`, `mmap_*.c` |
| DAX tests | 4 | `test_dax_litmus_*.c` |
| Utility | 5 | `bw.cpp`, `ptr-chasing.cpp`, etc. |

---

## Appendix B: Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| CXL device not found | Load kernel modules: `modprobe cxl_core cxl_pci` |
| Connection refused | Ensure server is running on correct port |
| QEMU CXL not working | Verify QEMU built with `--enable-libpmem` |
| Permission denied | Run with `sudo` for TAP/bridge operations |
| Memory size mismatch | Align QEMU and server memory configurations |

### Debug Commands

```bash
# Check CXL devices
cxl list -M

# Check NUMA topology
numactl --hardware

# Check DAX devices
ls /sys/bus/dax/devices/

# Check kernel modules
lsmod | grep cxl

# Server logs
SPDLOG_LEVEL=debug ./cxlmemsim_server 9999 topology.txt
```

---

*Document generated for OCEAN CXL 3.0 Emulation Framework*
