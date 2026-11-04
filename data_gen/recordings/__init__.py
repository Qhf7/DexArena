"""Recorded-demonstration pipeline: the pending-episode store, dataset layout, integrity
contracts and certificates, review/export into LeRobot datasets, and exact state/action
replay of a recording.

Modules:

- ``recording_integrity``: the recording contract (schema version, payload validation, hashes)
- ``episode_store``: pending episodes on disk (``episode.npz`` + ``meta.json``) and seed restore
- ``dataset_layout``: where pending sessions and exported datasets live, per end effector
- ``lerobot_integrity``: LeRobot round-trip checks, replay sidecars and dataset certificates
- ``review_demos``: review pending episodes and export them into certified datasets
- ``replay_recording``: zero-error state replay and action replay of a recording
- ``replay_compare_video``: side-by-side video of a recording and its replay
"""
