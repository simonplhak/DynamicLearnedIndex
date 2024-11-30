"""Includes detection of the execution environment and its configurations for experiments."""

from execution_environment.detect import detect_environment
from execution_environment.environment import Environment

__all__ = ['Environment', 'detect_environment']
