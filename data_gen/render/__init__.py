"""Offline multi-view re-rendering of recorded (state-only) demonstrations.

Nothing under ``tasks/`` imports this package: the cameras defined here are in no
environment's code-fingerprint closure, so changing them re-renders videos and
never re-versions a dataset.
"""
