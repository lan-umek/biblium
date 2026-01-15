# -*- coding: utf-8 -*-
"""
Undo Manager
============
Manages undo/redo history for data operations.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass
import copy

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class UndoState:
    """Represents a state that can be undone/redone."""
    description: str
    data: Any  # Usually a DataFrame
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UndoManager:
    """
    Manages undo/redo history.
    
    Usage:
        manager = UndoManager(max_history=20)
        
        # Save state before making changes
        manager.save_state(df.copy(), "Filter by year > 2020")
        
        # Perform operation...
        df = df[df['Year'] > 2020]
        
        # Undo
        if manager.can_undo():
            old_df = manager.undo()
            
        # Redo  
        if manager.can_redo():
            new_df = manager.redo()
    """
    
    def __init__(self, max_history: int = 20):
        """
        Initialize UndoManager.
        
        Parameters
        ----------
        max_history : int
            Maximum number of states to keep in history.
        """
        self.max_history = max_history
        self._undo_stack: List[UndoState] = []
        self._redo_stack: List[UndoState] = []
        self._current_state: Optional[UndoState] = None
        self._callbacks: List[Callable] = []
    
    def save_state(self, data: Any, description: str = "", metadata: Dict = None) -> None:
        """
        Save current state before making changes.
        
        Parameters
        ----------
        data : Any
            The data to save (usually a DataFrame copy).
        description : str
            Description of the operation being performed.
        metadata : dict, optional
            Additional metadata to store with the state.
        """
        # If we have a current state, push it to undo stack
        if self._current_state is not None:
            self._undo_stack.append(self._current_state)
            
            # Trim history if needed
            while len(self._undo_stack) > self.max_history:
                self._undo_stack.pop(0)
        
        # Set new current state
        self._current_state = UndoState(
            description=description,
            data=data,
            metadata=metadata or {}
        )
        
        # Clear redo stack when new action is performed
        self._redo_stack.clear()
        
        # Notify listeners
        self._notify()
    
    def undo(self) -> Optional[Any]:
        """
        Undo the last operation.
        
        Returns
        -------
        Any or None
            The previous state's data, or None if can't undo.
        """
        if not self.can_undo():
            return None
        
        # Push current state to redo stack
        if self._current_state is not None:
            self._redo_stack.append(self._current_state)
        
        # Pop from undo stack
        self._current_state = self._undo_stack.pop()
        
        # Notify listeners
        self._notify()
        
        return self._current_state.data
    
    def redo(self) -> Optional[Any]:
        """
        Redo the last undone operation.
        
        Returns
        -------
        Any or None
            The next state's data, or None if can't redo.
        """
        if not self.can_redo():
            return None
        
        # Push current state to undo stack
        if self._current_state is not None:
            self._undo_stack.append(self._current_state)
        
        # Pop from redo stack
        self._current_state = self._redo_stack.pop()
        
        # Notify listeners
        self._notify()
        
        return self._current_state.data
    
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self._undo_stack) > 0
    
    def can_redo(self) -> bool:
        """Check if redo is available."""
        return len(self._redo_stack) > 0
    
    def get_undo_description(self) -> str:
        """Get description of what will be undone."""
        if self._undo_stack:
            return self._undo_stack[-1].description
        return ""
    
    def get_redo_description(self) -> str:
        """Get description of what will be redone."""
        if self._redo_stack:
            return self._redo_stack[-1].description
        return ""
    
    def get_current_description(self) -> str:
        """Get description of current state."""
        if self._current_state:
            return self._current_state.description
        return ""
    
    def clear(self) -> None:
        """Clear all history."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._current_state = None
        self._notify()
    
    def get_history(self) -> List[str]:
        """Get list of operation descriptions in history."""
        history = [s.description for s in self._undo_stack]
        if self._current_state:
            history.append(f"→ {self._current_state.description}")
        for s in reversed(self._redo_stack):
            history.append(f"  {s.description}")
        return history
    
    def add_callback(self, callback: Callable) -> None:
        """Add a callback to be notified when state changes."""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable) -> None:
        """Remove a callback."""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def _notify(self) -> None:
        """Notify all callbacks of state change."""
        for callback in self._callbacks:
            try:
                callback(self.can_undo(), self.can_redo())
            except Exception:
                pass
    
    @property
    def undo_count(self) -> int:
        """Number of undo steps available."""
        return len(self._undo_stack)
    
    @property
    def redo_count(self) -> int:
        """Number of redo steps available."""
        return len(self._redo_stack)


# Global undo manager instance
_undo_manager: Optional[UndoManager] = None


def get_undo_manager() -> UndoManager:
    """Get the global UndoManager instance."""
    global _undo_manager
    if _undo_manager is None:
        _undo_manager = UndoManager()
    return _undo_manager


def reset_undo_manager() -> None:
    """Reset the global UndoManager."""
    global _undo_manager
    if _undo_manager is not None:
        _undo_manager.clear()
    _undo_manager = UndoManager()
