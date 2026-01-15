# -*- coding: utf-8 -*-
"""
Event Bus
=========
Central event system for decoupled component communication.
"""

from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
import weakref
from functools import wraps


@dataclass
class Event:
    """Event data container"""
    name: str
    data: Any = None
    source: Optional[str] = None


class EventBus:
    """
    Central event bus for application-wide communication.
    
    Components can subscribe to events and emit events without
    direct dependencies on each other.
    
    Usage:
        # Subscribe to an event
        event_bus.subscribe("dataset_loaded", self.on_dataset_loaded)
        
        # Emit an event
        event_bus.emit("dataset_loaded", {"path": "data.csv", "n_docs": 1234})
        
        # Unsubscribe
        event_bus.unsubscribe("dataset_loaded", self.on_dataset_loaded)
    """
    
    # Event names as constants
    DATASET_LOADED = "dataset_loaded"
    DATASET_FILTERED = "dataset_filtered"
    DATASET_RESET = "dataset_reset"
    DATASET_UPDATED = "dataset_updated"
    
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_PROGRESS = "analysis_progress"
    ANALYSIS_COMPLETED = "analysis_completed"
    ANALYSIS_FAILED = "analysis_failed"
    ANALYSIS_CANCELLED = "analysis_cancelled"
    
    TAB_OPENED = "tab_opened"
    TAB_CLOSED = "tab_closed"
    TAB_CHANGED = "tab_changed"
    
    PANEL_CHANGED = "panel_changed"
    RESET_PANELS = "reset_panels"
    
    SETTINGS_CHANGED = "settings_changed"
    THEME_CHANGED = "theme_changed"
    
    STATUS_MESSAGE = "status_message"
    STATUS_PROGRESS = "status_progress"
    STATUS_CLEAR = "status_clear"
    
    RESULTS_UPDATED = "results_updated"
    PLOT_GENERATED = "plot_generated"
    
    ERROR_OCCURRED = "error_occurred"
    WARNING_OCCURRED = "warning_occurred"
    
    # Report queue events
    REPORT_ITEM_ADDED = "report_item_added"
    REPORT_ITEM_REMOVED = "report_item_removed"
    REPORT_QUEUE_CLEARED = "report_queue_cleared"
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._once_subscribers: Dict[str, List[Callable]] = {}
        self._history: List[Event] = []
        self._history_limit = 100
        self._paused = False
        self._queue: List[Event] = []
    
    def subscribe(self, event_name: str, callback: Callable, weak: bool = False):
        """
        Subscribe to an event.
        
        Parameters
        ----------
        event_name : str
            Name of the event to subscribe to.
        callback : Callable
            Function to call when event is emitted.
            Should accept (event_data) as argument.
        weak : bool
            If True, use weak reference (auto-unsubscribe when object is deleted).
        """
        if event_name not in self._subscribers:
            self._subscribers[event_name] = []
        
        if weak:
            # Use weak reference for methods
            if hasattr(callback, '__self__'):
                ref = weakref.WeakMethod(callback)
            else:
                ref = weakref.ref(callback)
            self._subscribers[event_name].append(ref)
        else:
            self._subscribers[event_name].append(callback)
    
    def subscribe_once(self, event_name: str, callback: Callable):
        """Subscribe to an event for one-time execution."""
        if event_name not in self._once_subscribers:
            self._once_subscribers[event_name] = []
        self._once_subscribers[event_name].append(callback)
    
    def unsubscribe(self, event_name: str, callback: Callable):
        """Unsubscribe from an event."""
        if event_name in self._subscribers:
            # Handle both regular and weak references
            self._subscribers[event_name] = [
                cb for cb in self._subscribers[event_name]
                if not self._callback_matches(cb, callback)
            ]
    
    def _callback_matches(self, stored: Any, target: Callable) -> bool:
        """Check if stored callback matches target."""
        if isinstance(stored, (weakref.ref, weakref.WeakMethod)):
            ref_callback = stored()
            return ref_callback is None or ref_callback == target
        return stored == target
    
    def emit(self, event_name: str, data: Any = None, source: str = None):
        """
        Emit an event to all subscribers.
        
        Parameters
        ----------
        event_name : str
            Name of the event.
        data : Any
            Data to pass to subscribers.
        source : str, optional
            Identifier of the event source.
        """
        event = Event(name=event_name, data=data, source=source)
        
        # Record in history
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history.pop(0)
        
        # Queue if paused
        if self._paused:
            self._queue.append(event)
            return
        
        self._dispatch(event)
    
    def _dispatch(self, event: Event):
        """Dispatch event to subscribers."""
        event_name = event.name
        
        # Regular subscribers
        if event_name in self._subscribers:
            dead_refs = []
            for i, callback in enumerate(self._subscribers[event_name]):
                try:
                    # Handle weak references
                    if isinstance(callback, (weakref.ref, weakref.WeakMethod)):
                        actual_callback = callback()
                        if actual_callback is None:
                            dead_refs.append(i)
                            continue
                        actual_callback(event.data)
                    else:
                        callback(event.data)
                except Exception as e:
                    # Silently ignore TclError for destroyed widgets
                    error_str = str(e)
                    if "application has been destroyed" in error_str or "invalid command name" in error_str:
                        dead_refs.append(i)
                    else:
                        print(f"Error in event handler for '{event_name}': {e}")
            
            # Clean up dead weak references and handlers for destroyed widgets
            for i in reversed(dead_refs):
                try:
                    self._subscribers[event_name].pop(i)
                except:
                    pass
        
        # One-time subscribers
        if event_name in self._once_subscribers:
            callbacks = self._once_subscribers.pop(event_name)
            for callback in callbacks:
                try:
                    callback(event.data)
                except Exception as e:
                    error_str = str(e)
                    if "application has been destroyed" not in error_str and "invalid command name" not in error_str:
                        print(f"Error in one-time event handler for '{event_name}': {e}")
    
    def pause(self):
        """Pause event emission (events are queued)."""
        self._paused = True
    
    def resume(self):
        """Resume event emission and dispatch queued events."""
        self._paused = False
        queued = self._queue.copy()
        self._queue.clear()
        for event in queued:
            self._dispatch(event)
    
    def clear(self, event_name: str = None):
        """Clear subscribers for an event or all events."""
        if event_name:
            self._subscribers.pop(event_name, None)
            self._once_subscribers.pop(event_name, None)
        else:
            self._subscribers.clear()
            self._once_subscribers.clear()
    
    def get_history(self, event_name: str = None, limit: int = None) -> List[Event]:
        """Get event history, optionally filtered by event name."""
        history = self._history
        if event_name:
            history = [e for e in history if e.name == event_name]
        if limit:
            history = history[-limit:]
        return history
    
    def has_subscribers(self, event_name: str) -> bool:
        """Check if an event has any subscribers."""
        return bool(
            self._subscribers.get(event_name) or 
            self._once_subscribers.get(event_name)
        )


def on_event(event_name: str):
    """
    Decorator to mark a method as an event handler.
    
    Usage:
        @on_event("dataset_loaded")
        def handle_dataset(self, data):
            ...
    """
    def decorator(func):
        func._event_handler = event_name
        return func
    return decorator


def auto_subscribe(obj, event_bus: EventBus):
    """
    Automatically subscribe all @on_event decorated methods.
    
    Usage:
        class MyPanel:
            def __init__(self):
                auto_subscribe(self, event_bus)
            
            @on_event("dataset_loaded")
            def on_dataset(self, data):
                ...
    """
    for name in dir(obj):
        method = getattr(obj, name, None)
        if callable(method) and hasattr(method, '_event_handler'):
            event_name = method._event_handler
            event_bus.subscribe(event_name, method, weak=True)


# Global event bus instance
event_bus = EventBus()
