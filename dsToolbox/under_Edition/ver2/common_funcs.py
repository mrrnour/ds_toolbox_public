"""
Backward Compatibility Module - Data Science Toolbox
====================================================

DEPRECATED: This module has been renamed to `utilities.py` for clarity.

Please update your imports:
    OLD: from dsToolbox.common_funcs import TextProcessor
    NEW: from dsToolbox.utilities import TextProcessor

This compatibility module will be removed in a future version.

Author: Data Science Toolbox Contributors
License: MIT License
"""

import warnings

# Show deprecation warning
warnings.warn(
    "common_funcs.py is deprecated and has been renamed to utilities.py. "
    "Please update your imports: 'from dsToolbox.utilities import ...' "
    "This compatibility module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Import everything from the new module for backward compatibility
from .utilities import *
    """
    Perform comprehensive comparison between two lists including fuzzy matching.
    
    **DEPRECATED**: This function has been moved to the ComparativeVisualization class
    with enhanced functionality. Please use:
    `ComparativeVisualization.create_comprehensive_list_comparison()` instead.
    
    Parameters:
    -----------
    list_a : List[Any]
        First list of items to compare
    list_b : List[Any]
        Second list of items to compare  
    similarity_threshold : float, default=60.0
        Minimum similarity percentage for considering items as matches
    include_venn_diagram : bool, default=False
        Whether to generate a Venn diagram visualization
        
    Returns:
    --------
    dict
        Comprehensive comparison results including matches, unique items, and statistics
        
    Examples:
    --------
    **Old usage (deprecated):**
    >>> results = compare_lists(list1, list2, similarity_threshold=70, include_venn_diagram=True)
    
    **New usage (recommended):**
    >>> visualizer = ComparativeVisualization()
    >>> results = visualizer.create_comprehensive_list_comparison(list1, list2, similarity_threshold=70)
    """
    warnings.warn(
        "compare_lists() is deprecated and will be removed in a future version. "
        "Please use ComparativeVisualization.create_comprehensive_list_comparison() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    visualizer = ComparativeVisualization()
    return visualizer.create_comprehensive_list_comparison(
        list_a=list_a,
        list_b=list_b,
        similarity_threshold=similarity_threshold,
        include_venn_diagram=include_venn_diagram
    )