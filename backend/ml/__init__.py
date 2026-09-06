"""
Machine-learning subsystems (populated from Phase 3 onward).

Layout mirrors the merge decision in ``docs/MERGE_ARCHITECTURE.md``:

* ``sar/``         - SAMUDRA NETRA classical detector + RF/GB classifier
* ``drift/``       - SAMUDRA NETRA Lagrangian RK4 drift / hindcast / forecast
* ``attribution/`` - SAMUDRA NETRA baseline + Phase-6 unified fusion scoring
* ``ais/``         - AIS track processing + POSEatSea anomaly autoencoder
* ``trajectory/``  - POSEatSea LSTM route-deviation model (Mauritius-AOI gated)

Nothing is imported at package load; ``backend.ml.registry`` loads each model
lazily, once per process, on first use.
"""
