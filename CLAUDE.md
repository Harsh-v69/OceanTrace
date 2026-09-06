### **1\. Project Objective & Constraints**

* **Goal:** Build a single, reliable FastAPI application merging SAMUDRA NETRA (backend, SAR detection, drift physics, RBAC, offline-first UI) and POSEatSea (AIS Autoencoder, LSTM trajectory, anomaly scaling) for SIH26143.  
* **Strict Rules:** One unified API (no separate Streamlit app). Retain the best overlapping components via benchmarking. Expose real ML metrics (IoU, top-1 attribution); explicitly label detections as "Oil-like anomaly."  
* **Performance:** Optimize for offline-first local CPU execution using lazy model loading, caching, and deterministic simulated met-ocean data.

### **2\. Architecture & Data Model**

* **Backend:** FastAPI structure (`/api`, `/core`, `/models`, `/services`, `/ml`).  
* **Database:** SQLite designed for seamless PostGIS migration.  
* **Security:** JWT/Session auth with strict RBAC (Pilot, Regional, National roles). Map roles to GeoJSON maritime boundaries using point-in-polygon lookups.  
* **Alerting:** Real Twilio SMS integration (`SmsProvider` interface with a local Mock fallback) triggered by anomaly confidence thresholds.

### **3\. Unified ML Pipeline**

* **SAR Ingestion & Detection:** Sentinel-1/2 processing, Refined-Lee filtering, adaptive dark-spot segmentation, and wind/context look-alike filtering.  
* **Physics-Based Tracking:** Lagrangian RK4 drift model for backward hindcasting (origin reconstruction) and 48h forward forecasting.  
* **AIS Intelligence:** Normalize vessel pings and process through POSEatSea’s Autoencoder and LSTM. **Must** use their pre-trained `StandardScaler`.  
* **Attribution Fusion:** Rank vessels using configurable, normalized weights (spatiotemporal alignment, AIS blackouts, speed/course anomalies, route deviation). Feedback AIS correlation to refine the initial release-time estimate.