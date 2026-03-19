# MuraltZ Ground Control Station — System Diagrams

These diagrams use [Mermaid](https://mermaid.js.org/) syntax. They render in GitHub, GitLab, VS Code (with Mermaid extension), and many markdown viewers.

---

## 1. High-Level System Architecture

```mermaid
flowchart TB
    subgraph CubeSat["🛰️ CubeSat (Raspberry Pi)"]
        FS[Flight Software]
        CAM[Camera]
        IMU[IMU]
        FS --> CAM
        FS --> IMU
    end

    subgraph GCS["💻 Ground Control Station"]
        subgraph Receiver["Receiver (port 5000)"]
            L[Listener]
            PH[Packet Handler]
            QC[Quality Check]
            L --> PH
            PH --> QC
        end

        subgraph Processing["Processing Pipeline"]
            MS[Mosaic Stitcher]
            MG[Mosaic Grid]
            SD[Shadow Detector]
            HC[Hazard Classifier]
            YD[YOLO Detector]
            CD[Change Detector]
            RP[Route Planner]
            MS --> MG
            MG --> SD
            SD --> HC
            HC --> YD
            YD --> CD
            CD --> RP
        end

        subgraph State["State"]
            MS2[MissionState]
        end

        subgraph Uplink["Uplink (port 5001)"]
            CMD[Commander]
        end

        subgraph Dashboard["Dashboard (port 3000)"]
            FLASK[Flask API]
            UI[Web UI]
            FLASK --> UI
        end

        QC -->|callback| Processing
        Processing --> MS2
        FLASK --> MS2
        FLASK --> CMD
    end

    CubeSat -->|"TCP 5000\nimages + telemetry"| Receiver
    Uplink -->|"TCP 5001\ncommands"| CubeSat
    FLASK -->|"poll"| MS2
```

---

## 2. Image Reception Flow (Sequence)

```mermaid
sequenceDiagram
    participant CS as CubeSat
    participant L as Listener
    participant PH as Packet Handler
    participant QC as Quality Check
    participant P as Pipeline
    participant FS as File System

    CS->>L: Connect TCP :5000
    CS->>L: Send JSON header\n{type, filename, file_size, md5, metadata}
    L->>L: Read header

    alt type = "image"
        CS->>L: Send image bytes (throttled 1200 B/s)
        L->>L: Accumulate bytes
        L->>PH: validate_transfer(data, size, md5)
        PH-->>L: ValidationResult (ok/fail)
        alt MD5 OK
            L->>FS: Save to data/received_images/
            L->>QC: run_ground_quality_check(path)
            QC-->>L: {passed, notes}
            L->>P: process(path, metadata, quality)
            L->>CS: Send ACK
        else MD5 FAIL
            L->>CS: Send NACK
        end
    else type = "telemetry"
        CS->>L: Send telemetry JSON
        L->>L: Parse, save, update cache
        L->>CS: Send ACK
    end
```

---

## 3. CV Pipeline Flow (Per Image)

```mermaid
flowchart TB
    subgraph Input["Input"]
        IMG[Image + Metadata]
    end

    subgraph Stage1["1. Mosaic Stitching"]
        SP[SuperPoint\nkeypoints]
        LG[LightGlue\nmatching]
        H[homography]
        BL[Multi-band blend]
        SP --> LG
        LG --> H
        H --> BL
        BL --> BBOX[mosaic_bbox]
    end

    subgraph Stage2["2. Grid Update"]
        MG[MosaicGrid.update_from_mosaic]
        GC[grid_cell = mosaic_px_to_grid]
        MG --> GC
    end

    subgraph Stage3["3. Shadow Detection"]
        OT[Otsu threshold]
        CR[Contour regions]
        OT --> CR
        CR --> SM[shadow_mask, shadow_pct]
    end

    subgraph Stage4["4. Hazard Classification"]
        LBP[LBP variance]
        ED[Edge density]
        CC[Contour coverage]
        LBP --> CL[Classify → SAFE/MODERATE/SHADOW/HAZARD/IMPASSABLE]
        ED --> CL
        CC --> CL
        CL --> HC[Apply to grid]
    end

    subgraph Stage4b["4b. YOLO + Fusion"]
        YOLO[YOLO detect\ncraters/boulders]
        FUSE[Fuse with classical]
        YOLO --> FUSE
    end

    subgraph Stage5["5. Change Detection"]
        TM[Template match\nalignment]
        SSIM[SSIM diff]
        CE[Change events]
        TM --> SSIM
        SSIM --> CE
    end

    subgraph Stage6["6. Route Planning"]
        ASTAR[A* on cost grid]
        R3[Fastest, Safest, Balanced]
        ASTAR --> R3
    end

    subgraph Output["Output"]
        CG[cost_grid.json]
        MS2[mission_state.json]
        HM[hazard_map.png]
        CM[change_map.png]
        RM[route_map.png]
    end

    IMG --> Stage1
    BBOX --> Stage2
    Stage2 --> Stage3
    SM --> Stage4
    Stage4 --> Stage4b
    Stage4b --> Stage5
    Stage5 --> Stage6
    Stage6 --> Output
```

---

## 4. Pipeline Stage Dependencies

```mermaid
flowchart LR
    subgraph Pipeline["Pipeline.process()"]
        direction TB
        P1["1. MosaicStitcher\nregister_image()"]
        P2["2. MosaicGrid\nupdate_from_mosaic()"]
        P3["3. ShadowDetector\nrun()"]
        P3b["3b. SlopeEstimator\nestimate()"]
        P4["4. HazardClassifier\nclassify()"]
        P4b["4b. YOLODetector + fuse"]
        P5["5. ChangeDetector\ndetect()"]
        P6["6. RoutePlanner\nplan_*()"]
        P7["7. save_cost_grid"]
    end

    P1 -->|mosaic_bbox| P2
    P2 -->|grid_cell| P4
    P3 -->|shadow_mask| P3b
    P3 -->|shadow_mask| P4
    P3b -->|slope_map| P2
    P4 --> P4b
    P4b --> P5
    P5 --> P6
    P6 --> P7
```

---

## 5. Dashboard Data Flow

```mermaid
flowchart TB
    subgraph Browser["Browser"]
        UI[Single-Page UI]
        POLL[Polling Intervals]
    end

    subgraph Flask["Flask (port 3000)"]
        API["/api/* routes"]
    end

    subgraph DataSources["Data Sources"]
        MS[MissionState]
        TP[Telemetry Parser]
        FS[File System\nimages, JSON]
        P[Pipeline]
    end

    UI -->|"fetch every 2-5s"| POLL
    POLL -->|"/api/status"| API
    POLL -->|"/api/routes"| API
    POLL -->|"/api/cost_grid"| API
    POLL -->|"/api/changes"| API
    POLL -->|"/api/quality_log"| API
    POLL -->|"/api/log"| API
    POLL -->|"/api/latest_image"| API

    API --> MS
    API --> TP
    API --> FS
    API --> P

    subgraph UserActions["User Actions → POST"]
        START["/api/start_pass"]
        END["/api/end_pass"]
        CELL["/api/set_cell"]
        PLAN["/api/plan_routes"]
        CMD["/api/command"]
    end

    UI --> START
    UI --> END
    UI --> CELL
    UI --> PLAN
    UI --> CMD
    START --> API
    END --> API
    CELL --> API
    PLAN --> API
    CMD --> API
```

---

## 6. Transfer Protocol (CubeSat → GCS)

```mermaid
sequenceDiagram
    participant CS as CubeSat
    participant GCS as GCS Listener

    Note over CS,GCS: Data Port 5000

    CS->>GCS: Connect TCP
    CS->>GCS: JSON header + newline

    rect rgb(240, 248, 255)
        Note over CS,GCS: For type="image"
        loop Throttled (1200 B/s)
            CS->>GCS: Chunk of bytes
        end
    end

    GCS->>GCS: Verify MD5
    GCS->>GCS: Save file
    GCS->>GCS: Run pipeline

    alt MD5 OK
        GCS->>CS: ACK (0x06)
    else MD5 FAIL
        GCS->>CS: NACK (0x15)
    end

    CS->>CS: Mark sent or retry
```

---

## 7. Command Protocol (GCS → CubeSat)

```mermaid
sequenceDiagram
    participant GCS as GCS Commander
    participant CS as CubeSat Command Listener

    Note over GCS,CS: Command Port 5001

    GCS->>CS: Connect TCP
    GCS->>CS: JSON + newline\n{"cmd": "start_pass"}
    CS->>CS: Parse command
    CS->>CS: Execute (state transition)
    CS->>GCS: ACK (0x06) or NACK (0x15)
    GCS->>GCS: Close connection
```

---

## 8. Mosaic → Grid → Route Flow

```mermaid
flowchart TB
    subgraph Images["Incoming Images"]
        I1[Image 1]
        I2[Image 2]
        I3[Image N]
    end

    subgraph Mosaic["Mosaic Stitcher"]
        direction TB
        M1[SuperPoint + LightGlue]
        M2[Homography]
        M3[Blend onto canvas]
        M1 --> M2
        M2 --> M3
    end

    subgraph Grid["Mosaic Grid"]
        direction TB
        G1["Canvas size → rows, cols"]
        G2["Each cell = 80px"]
        G3["Cost grid, hazard grid"]
        G1 --> G2
        G2 --> G3
    end

    subgraph Route["Route Planning"]
        direction TB
        R1["Start/End from dashboard"]
        R2["A* on cost grid"]
        R3["Fastest, Safest, Balanced"]
        R1 --> R2
        R2 --> R3
    end

    Images --> Mosaic
    Mosaic -->|"mosaic_bbox per image"| Grid
    Grid -->|"grid_cell = px_to_grid(bbox center)"| G3
    G3 --> Route
```

---

## 9. Component Relationships

```mermaid
flowchart TB
    subgraph Core["Core (server.py)"]
        MS[MissionState]
        PL[Pipeline]
        CMD[Commander]
    end

    subgraph Receiver["receiver/"]
        L[listener]
        PH[packet_handler]
        QC[quality_check]
        TP[telemetry_parser]
        DS[downlink_state]
    end

    subgraph Processing["processing/"]
        MST[mosaic_stitcher]
        MGR[mosaic_grid]
        SHD[shadow_detector]
        HZC[hazard_classifier]
        YOLO[yolo_detector]
        CHG[change_detector]
        RTE[route_planner]
    end

    subgraph Dashboard["dashboard/"]
        APP[app.py]
    end

    L -->|callback| PL
    PL --> MS
    PL --> MST
    MST --> MGR
    MGR --> SHD
    SHD --> HZC
    HZC --> MGR
    HZC --> YOLO
    YOLO --> CHG
    CHG --> RTE
    RTE --> MGR
    APP --> MS
    APP --> PL
    APP --> CMD
```

---

## 10. State Machine (CubeSat)

```mermaid
stateDiagram-v2
    [*] --> BOOT
    BOOT --> WAITING: success
    BOOT --> SAFE_MODE: timeout

    WAITING --> IMAGING: start_pass
    SAFE_MODE --> WAITING: resume_normal

    IMAGING --> PROCESSING: end_pass
    IMAGING --> SAFE_MODE: error

    PROCESSING --> DOWNLINK: idle complete
    DOWNLINK --> WAITING: window complete
    DOWNLINK --> WAITING: all sent

    note right of IMAGING
        Captures images
        set_cell overrides target
    end note

    note right of DOWNLINK
        Throttled 1200 B/s
        Pushes to GCS :5000
    end note
```

---

## 11. File Naming Convention

```mermaid
flowchart LR
    subgraph Image["Image File"]
        A["pass{N}_img{MM}_{YYYYMMDD_HHMMSS}.jpg"]
    end

    subgraph Metadata["Metadata Sidecar"]
        B["pass{N}_img{MM}_{YYYYMMDD_HHMMSS}_meta.json"]
    end

    subgraph Legend["Legend"]
        L1["N = pass number"]
        L2["MM = image index (00-19)"]
        L3["Timestamp = capture UTC"]
    end
```

---

## 12. Data Storage Layout

```mermaid
flowchart TB
    subgraph Data["data/"]
        RI[received_images/\n*.jpg + *_meta.json]
        TE[telemetry/\n*.json]
        MSF[mission_state.json]
        
        subgraph Processed["processed/"]
            SM[shadow_masks/]
            HM[hazard_maps/]
            CM[change_maps/]
            MO[mosaics/]
            RT[routes/]
            MD[mosaic_database/]
            YD[yolo_detections/]
        end
    end

    subgraph Logs["data/logs/"]
        GCS[gcs.log]
    end

    RI --> Pipeline
    Pipeline --> HM
    Pipeline --> CM
    Pipeline --> MO
    Pipeline --> RT
    Pipeline --> MSF
```

---

## Viewing These Diagrams

- **GitHub/GitLab:** Render automatically in markdown preview
- **VS Code:** Install "Markdown Preview Mermaid Support" extension
- **Online:** Paste into [mermaid.live](https://mermaid.live) to edit/export as PNG/SVG
- **CLI:** `npx @mermaid-js/mermaid-cli mmdc -i DIAGRAMS.md -o docs/` (requires mermaid-cli)
