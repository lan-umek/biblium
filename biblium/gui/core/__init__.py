# -*- coding: utf-8 -*-
"""
GUI Core Module
===============
Base classes, events, state management, and threading utilities.
"""

from biblium.gui.core.events import EventBus, event_bus
from biblium.gui.core.state import StateManager, ReportQueue, report_queue, ReportItem
from biblium.gui.core.threading import BackgroundTask, TaskManager
from biblium.gui.core.base_widgets import BaseFrame, BaseCanvas
from biblium.gui.core.undo_manager import UndoManager, get_undo_manager, reset_undo_manager

__all__ = [
    "EventBus",
    "event_bus",
    "StateManager",
    "ReportQueue",
    "report_queue",
    "ReportItem",
    "BackgroundTask",
    "TaskManager",
    "BaseFrame",
    "BaseCanvas",
    "UndoManager",
    "get_undo_manager",
    "reset_undo_manager",
]
