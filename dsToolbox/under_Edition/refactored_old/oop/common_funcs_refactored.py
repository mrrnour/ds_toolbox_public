"""
Data Science Toolbox - Comprehensive Object-Oriented Refactored Module

This module provides a complete suite of data science utilities organized into 
well-structured classes following SOLID principles and best practices.

Author: Reza Nourzadeh - reza.nourzadeh@gmail.com
Refactored: 2025 - Object-oriented architecture with comprehensive error handling
"""

import numpy as np
import pandas as pd
import logging
import time
import re
import os
import sys
import datetime as dt
import math
import importlib.util
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pylab as pl
import seaborn as sns

# Statistical analysis imports
import scipy.stats as stats
import scipy
from difflib import SequenceMatcher

# ML imports
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression, 
    chi2, SelectPercentile, SelectFpr, SelectFdr, SelectFwe, f_classif
)

# Web scraping imports (conditional)
try:
    import requests
    import urllib3
    from requests_ntlm import HttpNtlmAuth
    from bs4 import BeautifulSoup
    WEB_SCRAPING_AVAILABLE = True
except ImportError:
    WEB_SCRAPING_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False


class DataScienceToolboxError(Exception):
    """Base exception class for DataScienceToolbox errors."""
    pass


class ValidationError(DataScienceToolboxError):
    """Raised when input validation fails."""
    pass


class ConfigurationError(DataScienceToolboxError):
    """Raised when configuration is invalid."""
    pass


class DataProcessingError(DataScienceToolboxError):
    """Raised when data processing operations fail."""
    pass


@dataclass
class PlotConfiguration:
    """
    Configuration class for plot customization.
    
    Attributes:
        figure_size (Tuple[int, int]): Width and height of the figure
        color_palette (str): Color palette name
        theme (str): Plot theme
        font_size (int): Default font size
        title_size (int): Title font size
        show_legend (bool): Whether to show legend
        save_format (str): Format for saving plots
        dpi (int): Dots per inch for saved plots
        
    Example:
        >>> config = PlotConfiguration(
        ...     figure_size=(12, 8),
        ...     color_palette='viridis',
        ...     theme='whitegrid'
        ... )
    """
    figure_size: Tuple[int, int] = (10, 6)
    color_palette: str = 'Set2'
    theme: str = 'whitegrid'
    font_size: int = 10
    title_size: int = 14
    show_legend: bool = True
    save_format: str = 'png'
    dpi: int = 300


class InputValidator:
    """
    Centralized input validation utilities.
    
    This class provides comprehensive validation methods for various data types
    and structures commonly used in data science workflows.
    """
    
    @staticmethod
    def validate_dataframe(
        data: Any,
        required_columns: Optional[List[str]] = None,
        min_rows: int = 1,
        allow_empty: bool = False
    ) -> pd.DataFrame:
        """
        Validate and ensure input is a proper DataFrame.
        
        Parameters:
            data: Input data to validate
            required_columns: List of required column names
            min_rows: Minimum number of rows required
            allow_empty: Whether to allow empty DataFrames
            
        Returns:
            pd.DataFrame: Validated DataFrame
            
        Raises:
            ValidationError: If validation fails
            
        Example:
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> validated = InputValidator.validate_dataframe(df, ['A', 'B'])
        """
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as e:
                raise ValidationError(f"Cannot convert input to DataFrame: {str(e)}")
        
        if not allow_empty and data.empty:
            raise ValidationError("DataFrame cannot be empty")
            
        if len(data) < min_rows:
            raise ValidationError(f"DataFrame must have at least {min_rows} rows, got {len(data)}")
            
        if required_columns:
            missing_cols = set(required_columns) - set(data.columns)
            if missing_cols:
                raise ValidationError(f"Missing required columns: {missing_cols}")
                
        return data
    
    @staticmethod
    def validate_numeric_series(
        data: Any,
        allow_nan: bool = False,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None
    ) -> pd.Series:
        """
        Validate numeric Series data.
        
        Parameters:
            data: Input data to validate
            allow_nan: Whether to allow NaN values
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            
        Returns:
            pd.Series: Validated numeric Series
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(data, pd.Series):
            try:
                data = pd.Series(data)
            except Exception as e:
                raise ValidationError(f"Cannot convert input to Series: {str(e)}")
        
        if not allow_nan and data.isna().any():
            raise ValidationError("Series contains NaN values")
            
        if not pd.api.types.is_numeric_dtype(data):
            try:
                data = pd.to_numeric(data, errors='coerce')
                if data.isna().any() and not allow_nan:
                    raise ValidationError("Series contains non-numeric values")
            except Exception as e:
                raise ValidationError(f"Cannot convert Series to numeric: {str(e)}")
        
        if min_value is not None and (data < min_value).any():
            raise ValidationError(f"Series contains values below minimum: {min_value}")
            
        if max_value is not None and (data > max_value).any():
            raise ValidationError(f"Series contains values above maximum: {max_value}")
            
        return data
    
    @staticmethod
    def validate_string_list(
        data: Any,
        min_length: int = 1,
        allow_empty_strings: bool = False
    ) -> List[str]:
        """
        Validate list of strings.
        
        Parameters:
            data: Input data to validate
            min_length: Minimum length of the list
            allow_empty_strings: Whether to allow empty strings
            
        Returns:
            List[str]: Validated list of strings
            
        Raises:
            ValidationError: If validation fails
        """
        if not isinstance(data, list):
            if isinstance(data, (pd.Series, np.ndarray)):
                data = data.tolist()
            else:
                raise ValidationError("Input must be a list or array-like")
        
        if len(data) < min_length:
            raise ValidationError(f"List must have at least {min_length} elements")
        
        string_data = []
        for item in data:
            if not isinstance(item, str):
                string_data.append(str(item))
            else:
                string_data.append(item)
        
        if not allow_empty_strings and any(not s.strip() for s in string_data):
            raise ValidationError("List contains empty strings")
            
        return string_data


class TextProcessor:
    """
    Advanced text processing and normalization utilities.
    
    This class provides comprehensive text processing capabilities including
    normalization, regex operations, fuzzy matching, and filename sanitization.
    """
    
    def __init__(self, default_encoding: str = 'utf-8'):
        """
        Initialize TextProcessor.
        
        Parameters:
            default_encoding: Default text encoding to use
        """
        self.default_encoding = default_encoding
        self.logger = logging.getLogger(__name__)
    
    def search_with_regex_patterns(
        self,
        regex_patterns: Union[str, List[str]],
        target_list: List[str]
    ) -> Tuple[List[str], List[bool]]:
        """
        Search for regular expression patterns in a list of strings.
        
        Parameters:
            regex_patterns: Single regex pattern or list of patterns to search for
            target_list: List of strings to search within
            
        Returns:
            Tuple[List[str], List[bool]]: 
                - List of matching strings
                - Boolean mask indicating matches
                
        Raises:
            ValidationError: If inputs are invalid
            
        Example:
            >>> processor = TextProcessor()
            >>> patterns = [r'\.vol_flag$', r'_date']
            >>> targets = ['bi_alt_account_id', 'snapshot_date', 'tv_vol_flag']
            >>> matches, mask = processor.search_with_regex_patterns(patterns, targets)
            >>> print(matches)  # ['snapshot_date', 'tv_vol_flag']
        """
        target_list = InputValidator.validate_string_list(target_list)
        
        if isinstance(regex_patterns, str):
            regex_patterns = [regex_patterns]
        
        regex_patterns = InputValidator.validate_string_list(regex_patterns)
        
        matched_strings = []
        try:
            for pattern in regex_patterns:
                compiled_pattern = re.compile(pattern)
                pattern_matches = list(filter(compiled_pattern.search, target_list))
                matched_strings.extend(pattern_matches)
        except re.error as e:
            raise ValidationError(f"Invalid regex pattern: {str(e)}")
        
        # Remove duplicates while preserving order
        matched_strings = list(dict.fromkeys(matched_strings))
        boolean_mask = np.isin(target_list, matched_strings).tolist()
        
        return matched_strings, boolean_mask
    
    def normalize_text(
        self,
        text: str,
        remove_spaces: bool = True,
        lowercase: bool = True,
        special_chars_pattern: str = r'[^a-zA-Z0-9\s]',
        replacement_char: str = '',
        max_length: Optional[int] = None,
        fallback_text: str = 'unnamed'
    ) -> str:
        """
        Flexible text normalization with comprehensive options.
        
        Parameters:
            text: Text to normalize
            remove_spaces: Whether to remove spaces
            lowercase: Whether to convert to lowercase
            special_chars_pattern: Regex pattern for characters to replace
            replacement_char: Character to replace matched patterns with
            max_length: Maximum length of output (truncate if longer)
            fallback_text: Text to return if result is empty
            
        Returns:
            str: Normalized text
            
        Example:
            >>> processor = TextProcessor()
            >>> result = processor.normalize_text("Hello, World! 123", lowercase=True)
            >>> print(result)  # "helloworld123"
        """
        if not isinstance(text, str):
            text = str(text)
        
        try:
            if lowercase:
                text = text.lower()
            
            if special_chars_pattern:
                text = re.sub(special_chars_pattern, replacement_char, text)
            
            if remove_spaces:
                text = re.sub(r'\s+', replacement_char, text)
            
            text = text.strip()
            
            if max_length and len(text) > max_length:
                text = text[:max_length]
            
            if not text or text.isspace():
                text = fallback_text
                
        except Exception as e:
            self.logger.warning(f"Text normalization failed: {str(e)}, using fallback")
            text = fallback_text
            
        return text
    
    def sanitize_filename(
        self,
        filename: str,
        max_length: int = 100,
        replacement_char: str = '_'
    ) -> str:
        """
        Sanitize filename for safe file system storage.
        
        Parameters:
            filename: Original filename
            max_length: Maximum filename length
            replacement_char: Character to replace invalid characters
            
        Returns:
            str: Sanitized filename
            
        Example:
            >>> processor = TextProcessor()
            >>> safe_name = processor.sanitize_filename("my/file<name>.txt")
            >>> print(safe_name)  # "my_file_name_.txt"
        """
        if not isinstance(filename, str):
            filename = str(filename)
        
        # Remove or replace invalid filename characters
        invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
        sanitized = re.sub(invalid_chars, replacement_char, filename)
        
        # Remove leading/trailing whitespace and dots
        sanitized = sanitized.strip(' .')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = 'unnamed_file'
        
        # Truncate if too long, preserving extension if possible
        if len(sanitized) > max_length:
            name, ext = os.path.splitext(sanitized)
            if ext and len(ext) <= 10:  # Reasonable extension length
                name = name[:max_length - len(ext)]
                sanitized = name + ext
            else:
                sanitized = sanitized[:max_length]
        
        return sanitized
    
    def clean_column_names(
        self,
        column_names: List[str],
        replacement_rules: Optional[Dict[str, str]] = None,
        lowercase: bool = False
    ) -> List[str]:
        """
        Clean column names for use in data analysis.
        
        Parameters:
            column_names: List of column names to clean
            replacement_rules: Dictionary of specific replacement rules
            lowercase: Whether to convert to lowercase
            
        Returns:
            List[str]: Cleaned column names
            
        Example:
            >>> processor = TextProcessor()
            >>> columns = ['First Name', 'Last-Name', 'Email@Address']
            >>> clean_cols = processor.clean_column_names(columns, lowercase=True)
            >>> print(clean_cols)  # ['first_name', 'last_name', 'emailaddress']
        """
        column_names = InputValidator.validate_string_list(column_names)
        
        if replacement_rules is None:
            replacement_rules = {}
        
        cleaned_columns = []
        for col_name in column_names:
            # Apply specific replacement rules first
            for old, new in replacement_rules.items():
                col_name = col_name.replace(old, new)
            
            # Standard cleaning
            cleaned = self.normalize_text(
                col_name,
                remove_spaces=True,
                lowercase=lowercase,
                special_chars_pattern=r'[^a-zA-Z0-9_]',
                replacement_char='_'
            )
            
            # Ensure it doesn't start with a number
            if cleaned and cleaned[0].isdigit():
                cleaned = 'col_' + cleaned
            
            cleaned_columns.append(cleaned)
        
        return cleaned_columns
    
    def find_fuzzy_matches(
        self,
        source_list: List[str],
        target_list: List[str],
        similarity_threshold: float = 60.0
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        Find fuzzy string matches between two lists.
        
        Parameters:
            source_list: List of source strings
            target_list: List of target strings to match against
            similarity_threshold: Minimum similarity score (0-100)
            
        Returns:
            Dict[str, List[Tuple[str, float]]]: Mapping of source strings to 
                list of (target_string, similarity_score) tuples
                
        Example:
            >>> processor = TextProcessor()
            >>> source = ['apple', 'banana']
            >>> target = ['aple', 'bananna', 'orange']
            >>> matches = processor.find_fuzzy_matches(source, target, 70.0)
        """
        source_list = InputValidator.validate_string_list(source_list)
        target_list = InputValidator.validate_string_list(target_list)
        
        if not 0 <= similarity_threshold <= 100:
            raise ValidationError("Similarity threshold must be between 0 and 100")
        
        fuzzy_matches = {}
        
        for source_item in source_list:
            matches = []
            for target_item in target_list:
                similarity_ratio = SequenceMatcher(None, source_item, target_item).ratio() * 100
                if similarity_ratio >= similarity_threshold:
                    matches.append((target_item, round(similarity_ratio, 2)))
            
            # Sort by similarity score (descending)
            matches.sort(key=lambda x: x[1], reverse=True)
            fuzzy_matches[source_item] = matches
        
        return fuzzy_matches


class DateTimeProcessor:
    """
    Comprehensive date and time processing utilities.
    
    This class provides robust date/time validation, conversion, and manipulation
    capabilities with proper error handling and flexible format support.
    """
    
    def __init__(self, default_format: str = '%Y-%m-%d'):
        """
        Initialize DateTimeProcessor.
        
        Parameters:
            default_format: Default datetime format string
        """
        self.default_format = default_format
        self.logger = logging.getLogger(__name__)
    
    def validate_timestamp_format(
        self,
        start_date: str,
        end_date: str,
        required_format: str = '%Y-%m-%d'
    ) -> Tuple[dt.datetime, dt.datetime]:
        """
        Validate timestamp format and convert to datetime objects.
        
        Parameters:
            start_date: Start date string
            end_date: End date string
            required_format: Expected date format
            
        Returns:
            Tuple[dt.datetime, dt.datetime]: Parsed start and end datetime objects
            
        Raises:
            ValidationError: If date parsing fails
            
        Example:
            >>> processor = DateTimeProcessor()
            >>> start, end = processor.validate_timestamp_format('2023-01-01', '2023-12-31')
        """
        try:
            start_dt = dt.datetime.strptime(start_date, required_format)
            end_dt = dt.datetime.strptime(end_date, required_format)
            
            if start_dt > end_dt:
                raise ValidationError("Start date must be before or equal to end date")
                
            return start_dt, end_dt
            
        except ValueError as e:
            raise ValidationError(f"Invalid date format. Expected {required_format}: {str(e)}")
    
    def calculate_business_days_elapsed(
        self,
        start_date: str,
        end_date: str,
        quarters_breakdown: bool = True
    ) -> Union[int, Dict[str, int]]:
        """
        Calculate elapsed business days between dates with optional quarterly breakdown.
        
        Parameters:
            start_date: Start date string (YYYY-MM-DD format)
            end_date: End date string (YYYY-MM-DD format)
            quarters_breakdown: Whether to provide quarterly breakdown
            
        Returns:
            Union[int, Dict[str, int]]: Total days or quarterly breakdown
            
        Example:
            >>> processor = DateTimeProcessor()
            >>> result = processor.calculate_business_days_elapsed('2023-01-01', '2023-12-31')
        """
        start_dt, end_dt = self.validate_timestamp_format(start_date, end_date)
        
        total_days = (end_dt - start_dt).days + 1
        
        if not quarters_breakdown:
            return total_days
        
        # Calculate quarterly breakdown
        quarterly_days = {}
        current_date = start_dt
        
        while current_date <= end_dt:
            quarter = f"Q{(current_date.month - 1) // 3 + 1}_{current_date.year}"
            
            # Find end of current quarter or end_date, whichever is earlier
            quarter_end_month = ((current_date.month - 1) // 3 + 1) * 3
            quarter_end = dt.datetime(current_date.year, quarter_end_month, 1)
            
            # Get last day of quarter
            if quarter_end_month == 12:
                quarter_end = quarter_end.replace(day=31)
            else:
                quarter_end = (quarter_end.replace(month=quarter_end_month + 1) - dt.timedelta(days=1))
            
            quarter_end = min(quarter_end, end_dt)
            
            days_in_quarter = (quarter_end - current_date).days + 1
            quarterly_days[quarter] = quarterly_days.get(quarter, 0) + days_in_quarter
            
            current_date = quarter_end + dt.timedelta(days=1)
        
        quarterly_days['total'] = total_days
        return quarterly_days
    
    def generate_date_range(
        self,
        year_range: Tuple[int, int] = (2018, 2099),
        month_range: Tuple[int, int] = (1, 12),
        include_day: bool = False,
        output_format: str = '%Y-%m'
    ) -> List[str]:
        """
        Generate list of formatted dates within specified ranges.
        
        Parameters:
            year_range: Tuple of (start_year, end_year)
            month_range: Tuple of (start_month, end_month)
            include_day: Whether to include day component
            output_format: Format string for output dates
            
        Returns:
            List[str]: List of formatted date strings
            
        Example:
            >>> processor = DateTimeProcessor()
            >>> dates = processor.generate_date_range((2023, 2024), (1, 3))
            >>> print(dates)  # ['2023-01', '2023-02', '2023-03', '2024-01', ...]
        """
        if year_range[0] > year_range[1]:
            raise ValidationError("Start year must be <= end year")
        
        if not (1 <= month_range[0] <= 12) or not (1 <= month_range[1] <= 12):
            raise ValidationError("Months must be between 1 and 12")
        
        date_list = []
        
        for year in range(year_range[0], year_range[1] + 1):
            for month in range(month_range[0], month_range[1] + 1):
                if include_day:
                    # Generate all days in the month
                    import calendar
                    days_in_month = calendar.monthrange(year, month)[1]
                    for day in range(1, days_in_month + 1):
                        date_obj = dt.datetime(year, month, day)
                        date_list.append(date_obj.strftime(output_format))
                else:
                    date_obj = dt.datetime(year, month, 1)
                    date_list.append(date_obj.strftime(output_format))
        
        return date_list
    
    def convert_to_readable_time(self, seconds: float) -> str:
        """
        Convert seconds to human-readable time format.
        
        Parameters:
            seconds: Time in seconds
            
        Returns:
            str: Human-readable time string
            
        Example:
            >>> processor = DateTimeProcessor()
            >>> readable = processor.convert_to_readable_time(3661)
            >>> print(readable)  # "1 hour, 1 minute, 1 second"
        """
        if not isinstance(seconds, (int, float)) or seconds < 0:
            raise ValidationError("Seconds must be a non-negative number")
        
        seconds = int(seconds)
        
        if seconds == 0:
            return "0 seconds"
        
        time_units = [
            (86400, 'day'),
            (3600, 'hour'),
            (60, 'minute'),
            (1, 'second')
        ]
        
        parts = []
        for unit_seconds, unit_name in time_units:
            if seconds >= unit_seconds:
                count = seconds // unit_seconds
                seconds %= unit_seconds
                unit_display = unit_name if count == 1 else f"{unit_name}s"
                parts.append(f"{count} {unit_display}")
        
        if len(parts) == 1:
            return parts[0]
        elif len(parts) == 2:
            return f"{parts[0]} and {parts[1]}"
        else:
            return ", ".join(parts[:-1]) + f", and {parts[-1]}"
    
    def convert_dates_to_numeric(
        self,
        dataframe: pd.DataFrame,
        date_columns: List[str],
        reference_date: Optional[str] = None,
        unit: str = 'days'
    ) -> pd.DataFrame:
        """
        Convert date columns to numeric values relative to a reference date.
        
        Parameters:
            dataframe: Input DataFrame
            date_columns: List of column names containing dates
            reference_date: Reference date (defaults to minimum date found)
            unit: Time unit for conversion ('days', 'hours', 'seconds')
            
        Returns:
            pd.DataFrame: DataFrame with converted numeric date columns
            
        Example:
            >>> df = pd.DataFrame({'date': ['2023-01-01', '2023-01-02']})
            >>> processor = DateTimeProcessor()
            >>> result = processor.convert_dates_to_numeric(df, ['date'])
        """
        dataframe = InputValidator.validate_dataframe(dataframe, date_columns)
        df_copy = dataframe.copy()
        
        # Convert date columns to datetime
        for col in date_columns:
            try:
                df_copy[col] = pd.to_datetime(df_copy[col])
            except Exception as e:
                raise DataProcessingError(f"Failed to convert column {col} to datetime: {str(e)}")
        
        # Determine reference date
        if reference_date is None:
            all_dates = []
            for col in date_columns:
                all_dates.extend(df_copy[col].dropna().tolist())
            if not all_dates:
                raise ValidationError("No valid dates found in specified columns")
            reference_dt = min(all_dates)
        else:
            try:
                reference_dt = pd.to_datetime(reference_date)
            except Exception as e:
                raise ValidationError(f"Invalid reference date: {str(e)}")
        
        # Convert to numeric
        unit_multipliers = {'days': 1, 'hours': 24, 'seconds': 86400}
        if unit not in unit_multipliers:
            raise ValidationError(f"Unit must be one of {list(unit_multipliers.keys())}")
        
        multiplier = unit_multipliers[unit]
        
        for col in date_columns:
            df_copy[col] = (df_copy[col] - reference_dt).dt.days * multiplier
        
        return df_copy


class DataFrameProcessor:
    """
    Comprehensive DataFrame processing and manipulation utilities.
    
    This class provides advanced DataFrame operations including column manipulation,
    merging, memory optimization, and data type conversions with robust error handling.
    """
    
    def __init__(self):
        """Initialize DataFrameProcessor."""
        self.logger = logging.getLogger(__name__)
    
    def reorder_columns(
        self,
        dataframe: pd.DataFrame,
        columns_to_move: List[str],
        reference_column: str,
        position: Literal['before', 'after'] = 'after'
    ) -> pd.DataFrame:
        """
        Reorder DataFrame columns by moving specified columns relative to reference column.
        
        Parameters:
            dataframe: Input DataFrame
            columns_to_move: List of columns to move
            reference_column: Reference column for positioning
            position: Whether to place columns 'before' or 'after' reference
            
        Returns:
            pd.DataFrame: DataFrame with reordered columns
            
        Raises:
            ValidationError: If columns don't exist
            
        Example:
            >>> df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3], 'D': [4]})
            >>> processor = DataFrameProcessor()
            >>> result = processor.reorder_columns(df, ['A', 'D'], 'B', 'after')
        """
        dataframe = InputValidator.validate_dataframe(dataframe)
        
        # Validate columns exist
        all_required_cols = columns_to_move + [reference_column]
        missing_cols = set(all_required_cols) - set(dataframe.columns)
        if missing_cols:
            raise ValidationError(f"Columns not found: {missing_cols}")
        
        # Get current column order
        current_columns = dataframe.columns.tolist()
        
        # Remove columns to move from their current positions
        remaining_columns = [col for col in current_columns if col not in columns_to_move]
        
        # Find position of reference column in remaining columns
        try:
            ref_index = remaining_columns.index(reference_column)
        except ValueError:
            raise ValidationError(f"Reference column '{reference_column}' not found")
        
        # Insert columns at appropriate position
        if position == 'after':
            insert_index = ref_index + 1
        else:  # before
            insert_index = ref_index
        
        # Build new column order
        new_columns = (
            remaining_columns[:insert_index] +
            columns_to_move +
            remaining_columns[insert_index:]
        )
        
        return dataframe[new_columns]
    
    def merge_with_date_intervals(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        group_column: str,
        left_date_col: str = 'date',
        right_start_col: str = 'start_date',
        right_end_col: str = 'end_date',
        interval_closed: str = 'both'
    ) -> pd.DataFrame:
        """
        Merge DataFrames based on date intervals.
        
        Parameters:
            left_df: Left DataFrame with dates
            right_df: Right DataFrame with date intervals
            group_column: Column to group by before merging
            left_date_col: Date column in left DataFrame
            right_start_col: Start date column in right DataFrame
            right_end_col: End date column in right DataFrame
            interval_closed: Whether interval is 'left', 'right', 'both', or 'neither'
            
        Returns:
            pd.DataFrame: Merged DataFrame
            
        Example:
            >>> left = pd.DataFrame({'group': ['A'], 'date': ['2023-01-15']})
            >>> right = pd.DataFrame({'group': ['A'], 'start_date': ['2023-01-01'], 'end_date': ['2023-01-31']})
            >>> processor = DataFrameProcessor()
            >>> result = processor.merge_with_date_intervals(left, right, 'group')
        """
        # Validate inputs
        left_df = InputValidator.validate_dataframe(left_df, [group_column, left_date_col])
        right_df = InputValidator.validate_dataframe(right_df, [group_column, right_start_col, right_end_col])
        
        if interval_closed not in ['left', 'right', 'both', 'neither']:
            raise ValidationError("interval_closed must be 'left', 'right', 'both', or 'neither'")
        
        # Convert date columns to datetime
        left_copy = left_df.copy()
        right_copy = right_df.copy()
        
        try:
            left_copy[left_date_col] = pd.to_datetime(left_copy[left_date_col])
            right_copy[right_start_col] = pd.to_datetime(right_copy[right_start_col])
            right_copy[right_end_col] = pd.to_datetime(right_copy[right_end_col])
        except Exception as e:
            raise DataProcessingError(f"Date conversion failed: {str(e)}")
        
        merged_results = []
        
        for group_value in left_copy[group_column].unique():
            left_group = left_copy[left_copy[group_column] == group_value].copy()
            right_group = right_copy[right_copy[group_column] == group_value].copy()
            
            if right_group.empty:
                continue
            
            # Perform interval matching
            matched_rows = []
            for _, left_row in left_group.iterrows():
                date_val = left_row[left_date_col]
                
                for _, right_row in right_group.iterrows():
                    start_date = right_row[right_start_col]
                    end_date = right_row[right_end_col]
                    
                    # Check if date falls within interval
                    if interval_closed == 'both':
                        in_interval = start_date <= date_val <= end_date
                    elif interval_closed == 'left':
                        in_interval = start_date <= date_val < end_date
                    elif interval_closed == 'right':
                        in_interval = start_date < date_val <= end_date
                    else:  # neither
                        in_interval = start_date < date_val < end_date
                    
                    if in_interval:
                        merged_row = pd.concat([left_row, right_row]).drop_duplicates()
                        matched_rows.append(merged_row)
            
            if matched_rows:
                group_result = pd.DataFrame(matched_rows)
                merged_results.append(group_result)
        
        if merged_results:
            return pd.concat(merged_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def calculate_cell_weights(
        self,
        dataframe: pd.DataFrame,
        axis: int = 0,
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Calculate cell weights (proportions) along specified axis.
        
        Parameters:
            dataframe: Input DataFrame with numeric data
            axis: Axis along which to calculate weights (0=rows, 1=columns)
            normalize: Whether to normalize to sum to 1
            
        Returns:
            pd.DataFrame: DataFrame with calculated weights
            
        Example:
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> processor = DataFrameProcessor()
            >>> weights = processor.calculate_cell_weights(df, axis=0)
        """
        dataframe = InputValidator.validate_dataframe(dataframe)
        
        # Ensure all columns are numeric
        numeric_df = dataframe.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValidationError("DataFrame must contain numeric columns")
        
        if axis not in [0, 1]:
            raise ValidationError("Axis must be 0 (rows) or 1 (columns)")
        
        # Calculate weights
        if axis == 0:  # Along rows
            row_sums = numeric_df.sum(axis=1)
            weights = numeric_df.div(row_sums, axis=0)
        else:  # Along columns
            col_sums = numeric_df.sum(axis=0)
            weights = numeric_df.div(col_sums, axis=1)
        
        # Handle division by zero
        weights = weights.fillna(0)
        
        if normalize:
            total_sum = weights.sum().sum()
            if total_sum > 0:
                weights = weights / total_sum
        
        return weights
    
    def optimize_memory_usage(
        self,
        dataframe: pd.DataFrame,
        object_to_string_columns: Union[str, List[str]] = 'all_columns',
        string_to_category_columns: Union[str, List[str]] = 'all_columns',
        downcast_integers: bool = True,
        downcast_floats: bool = True,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Optimize DataFrame memory usage through intelligent type conversion.
        
        Parameters:
            dataframe: Input DataFrame
            object_to_string_columns: Columns to convert from object to string
            string_to_category_columns: Columns to convert from string to category
            downcast_integers: Whether to downcast integer types
            downcast_floats: Whether to downcast float types
            verbose: Whether to print memory usage information
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Optimized DataFrame and optimization report
            
        Example:
            >>> df = pd.DataFrame({'A': ['x', 'y', 'x'] * 1000, 'B': [1.0, 2.0, 3.0] * 1000})
            >>> processor = DataFrameProcessor()
            >>> optimized_df, report = processor.optimize_memory_usage(df)
        """
        dataframe = InputValidator.validate_dataframe(dataframe)
        df_optimized = dataframe.copy()
        
        # Calculate initial memory usage
        initial_memory = df_optimized.memory_usage(deep=True).sum()
        
        optimization_report = {
            'initial_memory_mb': initial_memory / (1024 ** 2),
            'optimizations_applied': [],
            'column_changes': {}
        }
        
        # Object to string conversion
        if object_to_string_columns == 'all_columns':
            object_cols = df_optimized.select_dtypes(include=['object']).columns.tolist()
        elif isinstance(object_to_string_columns, list):
            object_cols = object_to_string_columns
        else:
            object_cols = []
        
        for col in object_cols:
            if col in df_optimized.columns:
                original_dtype = str(df_optimized[col].dtype)
                df_optimized[col] = df_optimized[col].astype('string')
                optimization_report['column_changes'][col] = {
                    'from': original_dtype, 'to': 'string'
                }
        
        if object_cols:
            optimization_report['optimizations_applied'].append('object_to_string')
        
        # String to category conversion
        if string_to_category_columns == 'all_columns':
            string_cols = df_optimized.select_dtypes(include=['object', 'string']).columns.tolist()
        elif isinstance(string_to_category_columns, list):
            string_cols = string_to_category_columns
        else:
            string_cols = []
        
        for col in string_cols:
            if col in df_optimized.columns:
                # Convert to category if cardinality is reasonable
                unique_ratio = df_optimized[col].nunique() / len(df_optimized)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    original_dtype = str(df_optimized[col].dtype)
                    df_optimized[col] = df_optimized[col].astype('category')
                    optimization_report['column_changes'][col] = {
                        'from': original_dtype, 'to': 'category'
                    }
        
        if string_cols:
            optimization_report['optimizations_applied'].append('string_to_category')
        
        # Downcast integers
        if downcast_integers:
            int_cols = df_optimized.select_dtypes(include=['int']).columns.tolist()
            for col in int_cols:
                original_dtype = str(df_optimized[col].dtype)
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
                new_dtype = str(df_optimized[col].dtype)
                if original_dtype != new_dtype:
                    optimization_report['column_changes'][col] = {
                        'from': original_dtype, 'to': new_dtype
                    }
            
            if int_cols:
                optimization_report['optimizations_applied'].append('downcast_integers')
        
        # Downcast floats
        if downcast_floats:
            float_cols = df_optimized.select_dtypes(include=['float']).columns.tolist()
            for col in float_cols:
                original_dtype = str(df_optimized[col].dtype)
                df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
                new_dtype = str(df_optimized[col].dtype)
                if original_dtype != new_dtype:
                    optimization_report['column_changes'][col] = {
                        'from': original_dtype, 'to': new_dtype
                    }
            
            if float_cols:
                optimization_report['optimizations_applied'].append('downcast_floats')
        
        # Calculate final memory usage
        final_memory = df_optimized.memory_usage(deep=True).sum()
        optimization_report['final_memory_mb'] = final_memory / (1024 ** 2)
        optimization_report['memory_reduction_mb'] = optimization_report['initial_memory_mb'] - optimization_report['final_memory_mb']
        optimization_report['memory_reduction_percent'] = (optimization_report['memory_reduction_mb'] / optimization_report['initial_memory_mb']) * 100
        
        if verbose:
            print(f"Memory usage reduced from {optimization_report['initial_memory_mb']:.2f} MB to {optimization_report['final_memory_mb']:.2f} MB")
            print(f"Reduction: {optimization_report['memory_reduction_mb']:.2f} MB ({optimization_report['memory_reduction_percent']:.1f}%)")
        
        return df_optimized, optimization_report
    
    def calculate_null_percentages(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate null value percentages for each column.
        
        Parameters:
            dataframe: Input DataFrame
            
        Returns:
            pd.Series: Null percentages by column
            
        Example:
            >>> df = pd.DataFrame({'A': [1, None, 3], 'B': [None, None, 6]})
            >>> processor = DataFrameProcessor()
            >>> null_pcts = processor.calculate_null_percentages(df)
        """
        dataframe = InputValidator.validate_dataframe(dataframe)
        
        null_counts = dataframe.isnull().sum()
        total_rows = len(dataframe)
        
        if total_rows == 0:
            return pd.Series(dtype=float)
        
        null_percentages = (null_counts / total_rows * 100).round(2)
        return null_percentages.sort_values(ascending=False)


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis and hypothesis testing utilities.
    
    This class provides advanced statistical methods including hypothesis testing,
    correlation analysis, and categorical data analysis with proper statistical
    interpretation and reporting.
    """
    
    def __init__(self, default_alpha: float = 0.05):
        """
        Initialize StatisticalAnalyzer.
        
        Parameters:
            default_alpha: Default significance level for hypothesis tests
        """
        if not 0 < default_alpha < 1:
            raise ValidationError("Alpha must be between 0 and 1")
        
        self.default_alpha = default_alpha
        self.logger = logging.getLogger(__name__)
    
    def perform_two_sample_t_test(
        self,
        dataframe: pd.DataFrame,
        parameter_column: str,
        group_column: str,
        group_names: Optional[List[str]] = None,
        alpha: Optional[float] = None,
        equal_variances: bool = False
    ) -> Dict[str, Any]:
        """
        Perform two-sample t-test between groups.
        
        Parameters:
            dataframe: Input DataFrame
            parameter_column: Column containing numeric values to test
            group_column: Column containing group labels
            group_names: Specific group names to compare (uses first 2 if None)
            alpha: Significance level (uses default if None)
            equal_variances: Whether to assume equal variances
            
        Returns:
            Dict[str, Any]: Test results including statistics and interpretation
            
        Example:
            >>> df = pd.DataFrame({
            ...     'score': [1, 2, 3, 4, 5, 6],
            ...     'group': ['A', 'A', 'A', 'B', 'B', 'B']
            ... })
            >>> analyzer = StatisticalAnalyzer()
            >>> result = analyzer.perform_two_sample_t_test(df, 'score', 'group')
        """
        dataframe = InputValidator.validate_dataframe(dataframe, [parameter_column, group_column])
        
        if alpha is None:
            alpha = self.default_alpha
        
        # Get unique groups
        unique_groups = dataframe[group_column].unique()
        if len(unique_groups) < 2:
            raise ValidationError("Need at least 2 groups for t-test")
        
        if group_names is None:
            group_names = unique_groups[:2].tolist()
        elif len(group_names) != 2:
            raise ValidationError("Exactly 2 group names required for t-test")
        
        # Extract data for each group
        group1_data = dataframe[dataframe[group_column] == group_names[0]][parameter_column]
        group2_data = dataframe[dataframe[group_column] == group_names[1]][parameter_column]
        
        # Validate numeric data
        group1_data = InputValidator.validate_numeric_series(group1_data, allow_nan=True)
        group2_data = InputValidator.validate_numeric_series(group2_data, allow_nan=True)
        
        # Remove NaN values
        group1_clean = group1_data.dropna()
        group2_clean = group2_data.dropna()
        
        if len(group1_clean) < 2 or len(group2_clean) < 2:
            raise ValidationError("Each group must have at least 2 non-null values")
        
        # Perform t-test
        try:
            t_statistic, p_value = stats.ttest_ind(
                group1_clean, 
                group2_clean, 
                equal_var=equal_variances
            )
        except Exception as e:
            raise DataProcessingError(f"T-test calculation failed: {str(e)}")
        
        # Calculate descriptive statistics
        group1_stats = {
            'mean': group1_clean.mean(),
            'std': group1_clean.std(),
            'count': len(group1_clean)
        }
        
        group2_stats = {
            'mean': group2_clean.mean(),
            'std': group2_clean.std(),
            'count': len(group2_clean)
        }
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((group1_stats['count'] - 1) * group1_stats['std']**2 + 
                             (group2_stats['count'] - 1) * group2_stats['std']**2) / 
                            (group1_stats['count'] + group2_stats['count'] - 2))
        
        cohens_d = (group1_stats['mean'] - group2_stats['mean']) / pooled_std if pooled_std > 0 else 0
        
        # Interpret results
        is_significant = p_value < alpha
        
        effect_size_interpretation = 'negligible'
        if abs(cohens_d) >= 0.8:
            effect_size_interpretation = 'large'
        elif abs(cohens_d) >= 0.5:
            effect_size_interpretation = 'medium'
        elif abs(cohens_d) >= 0.2:
            effect_size_interpretation = 'small'
        
        result = {
            'test_type': 'Two-sample t-test',
            'parameter': parameter_column,
            'groups': group_names,
            't_statistic': round(t_statistic, 4),
            'p_value': round(p_value, 6),
            'alpha': alpha,
            'is_significant': is_significant,
            'cohens_d': round(cohens_d, 4),
            'effect_size': effect_size_interpretation,
            'group_statistics': {
                group_names[0]: group1_stats,
                group_names[1]: group2_stats
            },
            'interpretation': self._interpret_t_test_results(
                is_significant, p_value, alpha, group_names, 
                group1_stats['mean'], group2_stats['mean']
            )
        }
        
        return result
    
    def _interpret_t_test_results(
        self, 
        is_significant: bool, 
        p_value: float, 
        alpha: float,
        group_names: List[str],
        mean1: float,
        mean2: float
    ) -> str:
        """Generate interpretation text for t-test results."""
        if is_significant:
            direction = "higher" if mean1 > mean2 else "lower"
            return (f"The difference between {group_names[0]} (mean={mean1:.3f}) and "
                   f"{group_names[1]} (mean={mean2:.3f}) is statistically significant "
                   f"(p={p_value:.4f} < {alpha}). {group_names[0]} has significantly "
                   f"{direction} values than {group_names[1]}.")
        else:
            return (f"No statistically significant difference found between "
                   f"{group_names[0]} and {group_names[1]} (p={p_value:.4f} >= {alpha}).")
    
    def perform_kruskal_wallis_test(
        self,
        group_data: pd.Series,
        value_data: pd.Series,
        alpha: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform Kruskal-Wallis H-test for comparing multiple groups.
        
        Parameters:
            group_data: Series containing group labels
            value_data: Series containing values to compare
            alpha: Significance level
            
        Returns:
            Dict[str, Any]: Test results and interpretation
            
        Example:
            >>> groups = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'])
            >>> values = pd.Series([1, 2, 3, 4, 5, 6])
            >>> analyzer = StatisticalAnalyzer()
            >>> result = analyzer.perform_kruskal_wallis_test(groups, values)
        """
        group_data = InputValidator.validate_string_list(group_data.tolist())
        value_data = InputValidator.validate_numeric_series(value_data, allow_nan=True)
        
        if alpha is None:
            alpha = self.default_alpha
        
        # Prepare data by groups
        groups_dict = {}
        for group, value in zip(group_data, value_data):
            if pd.notna(value):
                if group not in groups_dict:
                    groups_dict[group] = []
                groups_dict[group].append(value)
        
        if len(groups_dict) < 2:
            raise ValidationError("Need at least 2 groups for Kruskal-Wallis test")
        
        # Check minimum sample sizes
        for group, values in groups_dict.items():
            if len(values) < 3:
                raise ValidationError(f"Group '{group}' has fewer than 3 observations")
        
        # Perform test
        try:
            h_statistic, p_value = stats.kruskal(*groups_dict.values())
        except Exception as e:
            raise DataProcessingError(f"Kruskal-Wallis test failed: {str(e)}")
        
        # Calculate descriptive statistics for each group
        group_stats = {}
        for group, values in groups_dict.items():
            group_stats[group] = {
                'median': np.median(values),
                'mean': np.mean(values),
                'count': len(values),
                'std': np.std(values, ddof=1)
            }
        
        is_significant = p_value < alpha
        
        result = {
            'test_type': 'Kruskal-Wallis H-test',
            'h_statistic': round(h_statistic, 4),
            'p_value': round(p_value, 6),
            'alpha': alpha,
            'degrees_of_freedom': len(groups_dict) - 1,
            'is_significant': is_significant,
            'group_statistics': group_stats,
            'interpretation': self._interpret_kruskal_wallis_results(
                is_significant, p_value, alpha, list(groups_dict.keys())
            )
        }
        
        return result
    
    def _interpret_kruskal_wallis_results(
        self, 
        is_significant: bool, 
        p_value: float, 
        alpha: float,
        group_names: List[str]
    ) -> str:
        """Generate interpretation text for Kruskal-Wallis test results."""
        if is_significant:
            return (f"The Kruskal-Wallis test indicates significant differences "
                   f"between the groups {group_names} (p={p_value:.4f} < {alpha}). "
                   f"Post-hoc tests are recommended to identify which specific "
                   f"groups differ from each other.")
        else:
            return (f"No statistically significant differences found between "
                   f"the groups {group_names} (p={p_value:.4f} >= {alpha}).")
    
    def perform_chi_square_independence_test(
        self,
        categorical_var1: pd.Series,
        categorical_var2: pd.Series,
        alpha: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Perform Chi-square test of independence between two categorical variables.
        
        Parameters:
            categorical_var1: First categorical variable
            categorical_var2: Second categorical variable
            alpha: Significance level
            
        Returns:
            Dict[str, Any]: Test results and interpretation
            
        Example:
            >>> var1 = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'])
            >>> var2 = pd.Series(['X', 'Y', 'X', 'Y', 'X', 'Y'])
            >>> analyzer = StatisticalAnalyzer()
            >>> result = analyzer.perform_chi_square_independence_test(var1, var2)
        """
        if alpha is None:
            alpha = self.default_alpha
        
        # Create contingency table
        try:
            contingency_table = pd.crosstab(categorical_var1, categorical_var2)
        except Exception as e:
            raise DataProcessingError(f"Failed to create contingency table: {str(e)}")
        
        if contingency_table.empty:
            raise ValidationError("Contingency table is empty")
        
        # Check minimum expected frequencies
        chi2_stat, p_value, dof, expected_freq = stats.chi2_contingency(contingency_table)
        
        min_expected = np.min(expected_freq)
        cells_below_5 = np.sum(expected_freq < 5)
        total_cells = expected_freq.size
        
        # Warnings for assumption violations
        warnings = []
        if min_expected < 1:
            warnings.append("Some expected frequencies are below 1")
        if cells_below_5 / total_cells > 0.2:
            warnings.append("More than 20% of cells have expected frequencies below 5")
        
        is_significant = p_value < alpha
        
        # Calculate effect size (Cramér's V)
        n = contingency_table.sum().sum()
        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
        
        result = {
            'test_type': 'Chi-square test of independence',
            'chi2_statistic': round(chi2_stat, 4),
            'p_value': round(p_value, 6),
            'degrees_of_freedom': dof,
            'alpha': alpha,
            'is_significant': is_significant,
            'cramers_v': round(cramers_v, 4),
            'contingency_table': contingency_table,
            'expected_frequencies': expected_freq,
            'warnings': warnings,
            'interpretation': self._interpret_chi_square_results(
                is_significant, p_value, alpha, cramers_v, warnings
            )
        }
        
        return result
    
    def _interpret_chi_square_results(
        self,
        is_significant: bool,
        p_value: float,
        alpha: float,
        cramers_v: float,
        warnings: List[str]
    ) -> str:
        """Generate interpretation text for Chi-square test results."""
        effect_size_desc = 'negligible'
        if cramers_v >= 0.5:
            effect_size_desc = 'large'
        elif cramers_v >= 0.3:
            effect_size_desc = 'medium'
        elif cramers_v >= 0.1:
            effect_size_desc = 'small'
        
        interpretation = ""
        if is_significant:
            interpretation = (f"The chi-square test indicates a statistically significant "
                            f"association between the variables (p={p_value:.4f} < {alpha}). "
                            f"The effect size is {effect_size_desc} (Cramér's V = {cramers_v:.3f}).")
        else:
            interpretation = (f"No statistically significant association found between "
                            f"the variables (p={p_value:.4f} >= {alpha}).")
        
        if warnings:
            interpretation += f" Warning: {'; '.join(warnings)}. Results should be interpreted with caution."
        
        return interpretation


# Continue with more classes...
class FeatureEngineer:
    """
    Comprehensive feature engineering utilities for machine learning.
    
    This class provides advanced feature transformation, encoding, and engineering
    methods optimized for machine learning workflows with proper validation.
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        self.logger = logging.getLogger(__name__)
        self.fitted_encoders = {}
    
    def convert_categorical_to_numeric(
        self,
        dataframe: pd.DataFrame,
        categorical_columns: Optional[List[str]] = None,
        encoding_method: str = 'OneHotEncoder',
        drop_first: bool = True,
        handle_unknown: str = 'ignore'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Convert categorical columns to numeric using various encoding methods.
        
        Parameters:
            dataframe: Input DataFrame
            categorical_columns: List of categorical columns (auto-detect if None)
            encoding_method: 'OneHotEncoder', 'LabelEncoder', or 'OrdinalEncoder'
            drop_first: Whether to drop first category in one-hot encoding
            handle_unknown: How to handle unknown categories ('ignore' or 'error')
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Encoded DataFrame and encoding info
            
        Example:
            >>> df = pd.DataFrame({'cat': ['A', 'B', 'A'], 'num': [1, 2, 3]})
            >>> engineer = FeatureEngineer()
            >>> encoded_df, info = engineer.convert_categorical_to_numeric(df)
        """
        dataframe = InputValidator.validate_dataframe(dataframe)
        
        if categorical_columns is None:
            categorical_columns = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_columns:
            return dataframe.copy(), {'message': 'No categorical columns found'}
        
        # Validate categorical columns exist
        missing_cols = set(categorical_columns) - set(dataframe.columns)
        if missing_cols:
            raise ValidationError(f"Categorical columns not found: {missing_cols}")
        
        df_encoded = dataframe.copy()
        encoding_info = {'method': encoding_method, 'columns_processed': []}
        
        if encoding_method == 'OneHotEncoder':
            # One-hot encoding
            for col in categorical_columns:
                try:
                    dummies = pd.get_dummies(
                        df_encoded[col], 
                        prefix=col, 
                        drop_first=drop_first,
                        dtype=int
                    )
                    
                    # Store encoding info
                    encoding_info['columns_processed'].append({
                        'original_column': col,
                        'new_columns': dummies.columns.tolist(),
                        'categories': df_encoded[col].unique().tolist()
                    })
                    
                    # Add dummy columns and remove original
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded = df_encoded.drop(columns=[col])
                    
                except Exception as e:
                    self.logger.warning(f"Failed to encode column {col}: {str(e)}")
                    
        elif encoding_method == 'LabelEncoder':
            # Label encoding
            for col in categorical_columns:
                try:
                    unique_values = df_encoded[col].unique()
                    label_map = {val: idx for idx, val in enumerate(unique_values)}
                    
                    df_encoded[col] = df_encoded[col].map(label_map)
                    
                    encoding_info['columns_processed'].append({
                        'original_column': col,
                        'label_mapping': label_map,
                        'categories': unique_values.tolist()
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to encode column {col}: {str(e)}")
                    
        elif encoding_method == 'OrdinalEncoder':
            # Ordinal encoding (similar to label encoding but with explicit ordering)
            for col in categorical_columns:
                try:
                    unique_values = sorted(df_encoded[col].unique())
                    ordinal_map = {val: idx for idx, val in enumerate(unique_values)}
                    
                    df_encoded[col] = df_encoded[col].map(ordinal_map)
                    
                    encoding_info['columns_processed'].append({
                        'original_column': col,
                        'ordinal_mapping': ordinal_map,
                        'categories': unique_values
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Failed to encode column {col}: {str(e)}")
        else:
            raise ValidationError(f"Unknown encoding method: {encoding_method}")
        
        return df_encoded, encoding_info
    
    def create_sparse_categorical_features(
        self,
        dataframe: pd.DataFrame,
        product_column: str,
        price_column: str,
        separator: str = "; "
    ) -> pd.DataFrame:
        """
        Create sparse categorical features from delimited strings.
        
        Parameters:
            dataframe: Input DataFrame
            product_column: Column containing delimited product names
            price_column: Column containing corresponding prices
            separator: Delimiter used in product column
            
        Returns:
            pd.DataFrame: DataFrame with sparse categorical features
            
        Example:
            >>> df = pd.DataFrame({
            ...     'products': ['A; B', 'B; C', 'A'],
            ...     'prices': [10, 20, 5]
            ... })
            >>> engineer = FeatureEngineer()
            >>> result = engineer.create_sparse_categorical_features(df, 'products', 'prices')
        """
        dataframe = InputValidator.validate_dataframe(dataframe, [product_column, price_column])
        
        # Get all unique products
        all_products = set()
        for products_str in dataframe[product_column].fillna(''):
            if products_str:
                products = [p.strip() for p in str(products_str).split(separator) if p.strip()]
                all_products.update(products)
        
        all_products = sorted(list(all_products))
        
        if not all_products:
            raise ValidationError(f"No products found in column {product_column}")
        
        # Create sparse features
        sparse_features = pd.DataFrame(index=dataframe.index)
        
        for product in all_products:
            # Initialize column with zeros
            sparse_features[f"{product}_flag"] = 0
            sparse_features[f"{product}_price"] = 0.0
        
        # Fill in the sparse features
        for idx, row in dataframe.iterrows():
            products_str = str(row[product_column]) if pd.notna(row[product_column]) else ''
            price_val = row[price_column] if pd.notna(row[price_column]) else 0
            
            if products_str:
                products = [p.strip() for p in products_str.split(separator) if p.strip()]
                
                for product in products:
                    if product in all_products:
                        sparse_features.loc[idx, f"{product}_flag"] = 1
                        sparse_features.loc[idx, f"{product}_price"] = float(price_val)
        
        # Combine with original dataframe (excluding the original columns)
        result_df = dataframe.drop(columns=[product_column, price_column])
        result_df = pd.concat([result_df, sparse_features], axis=1)
        
        return result_df
    
    def discretize_continuous_variables(
        self,
        series: pd.Series,
        target_series: Optional[pd.Series] = None,
        bins: Union[int, List[float]] = 4,
        labels: Optional[List[str]] = None,
        method: str = 'cut'
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """
        Discretize continuous variables into categorical bins.
        
        Parameters:
            series: Continuous variable to discretize
            target_series: Optional target variable for informed binning
            bins: Number of bins or bin edges
            labels: Labels for bins
            method: 'cut' (equal-width) or 'qcut' (equal-frequency)
            
        Returns:
            Tuple[pd.Series, Dict[str, Any]]: Discretized series and binning info
            
        Example:
            >>> import pandas as pd
            >>> series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            >>> engineer = FeatureEngineer()
            >>> discretized, info = engineer.discretize_continuous_variables(series, bins=3)
        """
        series = InputValidator.validate_numeric_series(series, allow_nan=True)
        
        if method not in ['cut', 'qcut']:
            raise ValidationError("Method must be 'cut' or 'qcut'")
        
        # Remove NaN values for binning calculation
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            raise ValidationError("Series contains no non-null values")
        
        # Generate default labels if not provided
        if labels is None:
            if isinstance(bins, int):
                labels = [f"Q{i+1}" for i in range(bins)]
            else:
                labels = [f"Bin_{i+1}" for i in range(len(bins)-1)]
        
        try:
            if method == 'cut':
                discretized_clean, bin_edges = pd.cut(
                    series_clean, 
                    bins=bins, 
                    labels=labels, 
                    retbins=True,
                    duplicates='drop'
                )
            else:  # qcut
                discretized_clean, bin_edges = pd.qcut(
                    series_clean,
                    q=bins,
                    labels=labels,
                    retbins=True,
                    duplicates='drop'
                )
                
        except Exception as e:
            raise DataProcessingError(f"Discretization failed: {str(e)}")
        
        # Map back to original series (preserving NaN values)
        discretized_series = pd.Series(index=series.index, dtype='category')
        discretized_series.loc[series_clean.index] = discretized_clean
        
        binning_info = {
            'method': method,
            'bin_edges': bin_edges.tolist(),
            'labels': labels,
            'n_bins': len(labels),
            'null_count': series.isnull().sum()
        }
        
        return discretized_series, binning_info


class DataVisualizer:
    """
    Comprehensive data visualization utilities.
    
    This class provides advanced plotting capabilities for statistical analysis,
    EDA, and data exploration with consistent styling and proper error handling.
    """
    
    def __init__(self, config: Optional[PlotConfiguration] = None):
        """
        Initialize DataVisualizer.
        
        Parameters:
            config: Plot configuration settings
        """
        self.config = config or PlotConfiguration()
        self.logger = logging.getLogger(__name__)
        
        # Set default style
        plt.style.use('default')
        sns.set_palette(self.config.color_palette)
        sns.set_style(self.config.theme)
    
    def create_correlation_heatmap(
        self,
        dataframe: pd.DataFrame,
        correlation_method: str = 'kendall',
        show_diagonal: bool = True,
        annot: bool = True,
        figure_size: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Create correlation heatmap with customizable options.
        
        Parameters:
            dataframe: Input DataFrame with numeric columns
            correlation_method: Correlation method ('pearson', 'kendall', 'spearman')
            show_diagonal: Whether to show diagonal values
            annot: Whether to annotate cells with correlation values
            figure_size: Figure size (width, height)
            save_path: Path to save the plot
            **kwargs: Additional arguments for seaborn heatmap
            
        Returns:
            plt.Figure: Matplotlib figure object
            
        Example:
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
            >>> visualizer = DataVisualizer()
            >>> fig = visualizer.create_correlation_heatmap(df)
        """
        # Validate input
        dataframe = InputValidator.validate_dataframe(dataframe)
        numeric_df = dataframe.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            raise ValidationError("DataFrame must contain numeric columns")
        
        if correlation_method not in ['pearson', 'kendall', 'spearman']:
            raise ValidationError("correlation_method must be 'pearson', 'kendall', or 'spearman'")
        
        # Calculate correlation matrix
        try:
            corr_matrix = numeric_df.corr(method=correlation_method)
        except Exception as e:
            raise DataProcessingError(f"Correlation calculation failed: {str(e)}")
        
        # Set figure size
        if figure_size is None:
            figure_size = self.config.figure_size
        
        # Create plot
        fig, ax = plt.subplots(figsize=figure_size)
        
        # Mask diagonal if requested
        mask = None
        if not show_diagonal:
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Default heatmap parameters
        heatmap_params = {
            'annot': annot,
            'cmap': 'RdYlBu_r',
            'center': 0,
            'square': True,
            'linewidths': 0.5,
            'cbar_kws': {"shrink": .8},
            'fmt': '.2f' if annot else None
        }
        
        # Update with user parameters
        heatmap_params.update(kwargs)
        
        # Create heatmap
        try:
            sns.heatmap(corr_matrix, mask=mask, ax=ax, **heatmap_params)
        except Exception as e:
            raise DataProcessingError(f"Heatmap creation failed: {str(e)}")
        
        ax.set_title(f'Correlation Matrix ({correlation_method.title()})', 
                    fontsize=self.config.title_size, pad=20)
        
        plt.tight_layout()
        
        if save_path:
            try:
                fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                self.logger.info(f"Plot saved to {save_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save plot: {str(e)}")
        
        return fig
    
    def create_distribution_plots(
        self,
        dataframe: pd.DataFrame,
        dependent_variable: str,
        group_variable: str,
        plot_types: List[str] = ['histogram', 'boxplot', 'violin'],
        figure_size: Optional[Tuple[int, int]] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, plt.Figure]:
        """
        Create multiple distribution plots for grouped data.
        
        Parameters:
            dataframe: Input DataFrame
            dependent_variable: Numeric variable to plot
            group_variable: Grouping variable
            plot_types: List of plot types to create
            figure_size: Figure size for each plot
            save_path: Base path for saving plots (will append plot type)
            
        Returns:
            Dict[str, plt.Figure]: Dictionary mapping plot types to figures
            
        Example:
            >>> df = pd.DataFrame({
            ...     'value': np.random.normal(0, 1, 100),
            ...     'group': np.random.choice(['A', 'B'], 100)
            ... })
            >>> visualizer = DataVisualizer()
            >>> figures = visualizer.create_distribution_plots(df, 'value', 'group')
        """
        # Validate input
        dataframe = InputValidator.validate_dataframe(dataframe, [dependent_variable, group_variable])
        
        # Validate dependent variable is numeric
        dep_series = InputValidator.validate_numeric_series(
            dataframe[dependent_variable], 
            allow_nan=True
        )
        
        valid_plot_types = ['histogram', 'boxplot', 'violin', 'kde', 'strip']
        invalid_types = set(plot_types) - set(valid_plot_types)
        if invalid_types:
            raise ValidationError(f"Invalid plot types: {invalid_types}. Valid types: {valid_plot_types}")
        
        if figure_size is None:
            figure_size = self.config.figure_size
        
        figures = {}
        
        # Remove rows with NaN values in dependent variable
        clean_df = dataframe.dropna(subset=[dependent_variable])
        
        if clean_df.empty:
            raise ValidationError(f"No valid data found for variable {dependent_variable}")
        
        for plot_type in plot_types:
            try:
                fig, ax = plt.subplots(figsize=figure_size)
                
                if plot_type == 'histogram':
                    for group in clean_df[group_variable].unique():
                        group_data = clean_df[clean_df[group_variable] == group][dependent_variable]
                        ax.hist(group_data, alpha=0.7, label=str(group), bins=20)
                    ax.set_xlabel(dependent_variable)
                    ax.set_ylabel('Frequency')
                    ax.set_title(f'Distribution of {dependent_variable} by {group_variable}')
                    ax.legend()
                    
                elif plot_type == 'boxplot':
                    sns.boxplot(data=clean_df, x=group_variable, y=dependent_variable, ax=ax)
                    ax.set_title(f'Boxplot of {dependent_variable} by {group_variable}')
                    
                elif plot_type == 'violin':
                    sns.violinplot(data=clean_df, x=group_variable, y=dependent_variable, ax=ax)
                    ax.set_title(f'Violin Plot of {dependent_variable} by {group_variable}')
                    
                elif plot_type == 'kde':
                    for group in clean_df[group_variable].unique():
                        group_data = clean_df[clean_df[group_variable] == group][dependent_variable]
                        sns.kdeplot(group_data, label=str(group), ax=ax)
                    ax.set_xlabel(dependent_variable)
                    ax.set_ylabel('Density')
                    ax.set_title(f'Density Plot of {dependent_variable} by {group_variable}')
                    ax.legend()
                    
                elif plot_type == 'strip':
                    sns.stripplot(data=clean_df, x=group_variable, y=dependent_variable, ax=ax)
                    ax.set_title(f'Strip Plot of {dependent_variable} by {group_variable}')
                
                plt.tight_layout()
                figures[plot_type] = fig
                
                # Save plot if path provided
                if save_path:
                    save_file = f"{save_path}_{plot_type}.{self.config.save_format}"
                    try:
                        fig.savefig(save_file, dpi=self.config.dpi, bbox_inches='tight')
                        self.logger.info(f"Plot saved to {save_file}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save {plot_type} plot: {str(e)}")
                        
            except Exception as e:
                self.logger.error(f"Failed to create {plot_type} plot: {str(e)}")
                continue
        
        return figures
    
    def create_scatter_plot_3d(
        self,
        dataframe: pd.DataFrame,
        x_column: str,
        y_column: str,
        z_column: str,
        color_column: Optional[str] = None,
        size_column: Optional[str] = None,
        title: str = "3D Scatter Plot",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create 3D scatter plot with optional color and size encoding.
        
        Parameters:
            dataframe: Input DataFrame
            x_column: X-axis column name
            y_column: Y-axis column name  
            z_column: Z-axis column name
            color_column: Optional column for color encoding
            size_column: Optional column for size encoding
            title: Plot title
            save_path: Path to save the plot
            
        Returns:
            plt.Figure: 3D scatter plot figure
            
        Example:
            >>> df = pd.DataFrame({
            ...     'x': [1, 2, 3], 'y': [4, 5, 6], 'z': [7, 8, 9],
            ...     'category': ['A', 'B', 'C']
            ... })
            >>> visualizer = DataVisualizer()
            >>> fig = visualizer.create_scatter_plot_3d(df, 'x', 'y', 'z', 'category')
        """
        # Validate required columns
        required_cols = [x_column, y_column, z_column]
        if color_column:
            required_cols.append(color_column)
        if size_column:
            required_cols.append(size_column)
            
        dataframe = InputValidator.validate_dataframe(dataframe, required_cols)
        
        # Validate numeric columns
        for col in [x_column, y_column, z_column]:
            InputValidator.validate_numeric_series(dataframe[col], allow_nan=True)
        
        if size_column:
            InputValidator.validate_numeric_series(dataframe[size_column], allow_nan=True)
        
        # Remove rows with NaN in required columns
        clean_df = dataframe.dropna(subset=[x_column, y_column, z_column])
        
        if clean_df.empty:
            raise ValidationError("No valid data points for 3D plotting")
        
        # Create 3D plot
        fig = plt.figure(figsize=self.config.figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Prepare color and size arrays
        colors = None
        sizes = None
        
        if color_column:
            if clean_df[color_column].dtype in ['object', 'category']:
                # Categorical colors
                unique_cats = clean_df[color_column].unique()
                color_map = {cat: i for i, cat in enumerate(unique_cats)}
                colors = [color_map[cat] for cat in clean_df[color_column]]
            else:
                # Continuous colors
                colors = clean_df[color_column]
        
        if size_column:
            # Normalize sizes to reasonable range
            size_data = clean_df[size_column]
            min_size, max_size = 20, 200
            sizes = min_size + (size_data - size_data.min()) / (size_data.max() - size_data.min()) * (max_size - min_size)
        else:
            sizes = 50
        
        # Create scatter plot
        scatter = ax.scatter(
            clean_df[x_column],
            clean_df[y_column], 
            clean_df[z_column],
            c=colors,
            s=sizes,
            alpha=0.7,
            cmap='viridis'
        )
        
        # Set labels and title
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.set_zlabel(z_column)
        ax.set_title(title, fontsize=self.config.title_size)
        
        # Add colorbar if colors are used
        if color_column and clean_df[color_column].dtype not in ['object', 'category']:
            plt.colorbar(scatter, ax=ax, label=color_column)
        
        plt.tight_layout()
        
        if save_path:
            try:
                fig.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
                self.logger.info(f"3D plot saved to {save_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save 3D plot: {str(e)}")
        
        return fig


class LoggingManager:
    """
    Comprehensive logging configuration and management utilities.
    
    This class provides centralized logging setup, custom loggers, and
    output redirection capabilities for data science workflows.
    """
    
    def __init__(self):
        """Initialize LoggingManager."""
        self.active_loggers = {}
        self.log_files = {}
    
    def create_custom_logger(
        self,
        logger_name: str,
        log_file_path: Optional[str] = None,
        log_level: int = logging.INFO,
        console_output: bool = True,
        file_format: Optional[str] = None,
        console_format: Optional[str] = None
    ) -> logging.Logger:
        """
        Create and configure a custom logger.
        
        Parameters:
            logger_name: Unique name for the logger
            log_file_path: Optional path to log file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            console_output: Whether to output to console
            file_format: Custom format string for file output
            console_format: Custom format string for console output
            
        Returns:
            logging.Logger: Configured logger instance
            
        Example:
            >>> manager = LoggingManager()
            >>> logger = manager.create_custom_logger(
            ...     'my_analysis',
            ...     log_file_path='analysis.log',
            ...     log_level=logging.DEBUG
            ... )
            >>> logger.info("Analysis started")
        """
        # Default formats
        if file_format is None:
            file_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        if console_format is None:
            console_format = '%(levelname)s - %(message)s'
        
        # Create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        
        # Clear any existing handlers to avoid duplication
        logger.handlers.clear()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(console_format)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_file_path:
            try:
                # Create directory if it doesn't exist
                log_dir = os.path.dirname(log_file_path)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)
                
                file_handler = logging.FileHandler(log_file_path)
                file_handler.setLevel(log_level)
                file_formatter = logging.Formatter(file_format)
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
                
                self.log_files[logger_name] = log_file_path
                
            except Exception as e:
                logger.warning(f"Failed to create file handler for {log_file_path}: {str(e)}")
        
        # Store logger reference
        self.active_loggers[logger_name] = logger
        
        return logger
    
    def setup_analysis_logger(
        self,
        analysis_name: str,
        output_directory: str = "./logs",
        include_timestamp: bool = True
    ) -> logging.Logger:
        """
        Set up a logger specifically for data analysis workflows.
        
        Parameters:
            analysis_name: Name of the analysis
            output_directory: Directory for log files
            include_timestamp: Whether to include timestamp in filename
            
        Returns:
            logging.Logger: Configured analysis logger
            
        Example:
            >>> manager = LoggingManager()
            >>> logger = manager.setup_analysis_logger("customer_segmentation")
            >>> logger.info("Starting customer segmentation analysis")
        """
        # Create output directory
        Path(output_directory).mkdir(parents=True, exist_ok=True)
        
        # Generate log filename
        if include_timestamp:
            timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_filename = f"{analysis_name}_{timestamp}.log"
        else:
            log_filename = f"{analysis_name}.log"
        
        log_file_path = os.path.join(output_directory, log_filename)
        
        # Create logger with analysis-specific format
        file_format = '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s'
        console_format = '%(levelname)-8s | %(message)s'
        
        logger = self.create_custom_logger(
            logger_name=f"analysis_{analysis_name}",
            log_file_path=log_file_path,
            log_level=logging.INFO,
            console_output=True,
            file_format=file_format,
            console_format=console_format
        )
        
        logger.info(f"Analysis logger initialized for '{analysis_name}'")
        logger.info(f"Log file: {log_file_path}")
        
        return logger
    
    def load_configuration_from_file(
        self,
        config_file_path: str,
        logger: Optional[logging.Logger] = None
    ) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file.
        
        Parameters:
            config_file_path: Path to configuration file
            logger: Optional logger for status messages
            
        Returns:
            Dict[str, Any]: Configuration dictionary
            
        Raises:
            ConfigurationError: If file loading fails
            
        Example:
            >>> manager = LoggingManager()
            >>> config = manager.load_configuration_from_file("config.yaml")
        """
        if not os.path.exists(config_file_path):
            raise ConfigurationError(f"Configuration file not found: {config_file_path}")
        
        if logger is None:
            logger = logging.getLogger(__name__)
        
        file_extension = os.path.splitext(config_file_path)[1].lower()
        
        try:
            with open(config_file_path, 'r', encoding='utf-8') as file:
                if file_extension in ['.yaml', '.yml']:
                    try:
                        import yaml
                        config = yaml.safe_load(file)
                    except ImportError:
                        raise ConfigurationError("PyYAML required for YAML files")
                        
                elif file_extension == '.json':
                    import json
                    config = json.load(file)
                    
                else:
                    raise ConfigurationError(f"Unsupported file format: {file_extension}")
                    
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
        
        if logger:
            logger.info(f"Configuration loaded from {config_file_path}")
        
        return config or {}
    
    def custom_print_with_logging(
        self,
        message: str,
        logger: Optional[logging.Logger] = None,
        log_level: int = logging.INFO,
        print_to_console: bool = True
    ) -> None:
        """
        Print message with optional logging.
        
        Parameters:
            message: Message to print/log
            logger: Optional logger to use
            log_level: Logging level for the message
            print_to_console: Whether to also print to console
            
        Example:
            >>> manager = LoggingManager()
            >>> logger = manager.create_custom_logger("test")
            >>> manager.custom_print_with_logging("Hello World", logger)
        """
        if print_to_console:
            print(message)
        
        if logger:
            logger.log(log_level, message)


class FileSystemOperations:
    """
    File system operations and utilities for data science workflows.
    
    This class provides robust file handling, path validation, and 
    directory management with proper error handling.
    """
    
    def __init__(self):
        """Initialize FileSystemOperations."""
        self.logger = logging.getLogger(__name__)
    
    def validate_and_create_path(
        self,
        path: str,
        create_if_missing: bool = True,
        path_type: str = 'directory'
    ) -> str:
        """
        Validate path and optionally create directory structure.
        
        Parameters:
            path: Path to validate
            create_if_missing: Whether to create missing directories
            path_type: Either 'directory' or 'file'
            
        Returns:
            str: Validated and normalized path
            
        Raises:
            ValidationError: If path validation fails
            
        Example:
            >>> fs_ops = FileSystemOperations()
            >>> valid_path = fs_ops.validate_and_create_path("/home/user/analysis")
        """
        if not isinstance(path, str) or not path.strip():
            raise ValidationError("Path must be a non-empty string")
        
        # Normalize path
        normalized_path = os.path.normpath(os.path.expanduser(path))
        
        if path_type == 'file':
            # For files, check the directory part
            directory = os.path.dirname(normalized_path)
            if directory and not os.path.exists(directory):
                if create_if_missing:
                    try:
                        os.makedirs(directory, exist_ok=True)
                        self.logger.info(f"Created directory: {directory}")
                    except Exception as e:
                        raise ValidationError(f"Cannot create directory {directory}: {str(e)}")
                else:
                    raise ValidationError(f"Directory does not exist: {directory}")
                    
        elif path_type == 'directory':
            if not os.path.exists(normalized_path):
                if create_if_missing:
                    try:
                        os.makedirs(normalized_path, exist_ok=True)
                        self.logger.info(f"Created directory: {normalized_path}")
                    except Exception as e:
                        raise ValidationError(f"Cannot create directory {normalized_path}: {str(e)}")
                else:
                    raise ValidationError(f"Directory does not exist: {normalized_path}")
        
        else:
            raise ValidationError("path_type must be 'directory' or 'file'")
        
        return normalized_path
    
    def setup_output_directory(
        self,
        output_directory: str,
        files_to_copy: Optional[List[str]] = None,
        overwrite_existing: bool = False
    ) -> str:
        """
        Set up output directory with optional file copying.
        
        Parameters:
            output_directory: Path to output directory
            files_to_copy: Optional list of files to copy to output directory
            overwrite_existing: Whether to overwrite existing files
            
        Returns:
            str: Path to created output directory
            
        Example:
            >>> fs_ops = FileSystemOperations()
            >>> output_dir = fs_ops.setup_output_directory(
            ...     "/path/to/output",
            ...     ["config.yaml", "data.csv"]
            ... )
        """
        # Create output directory
        output_path = self.validate_and_create_path(output_directory, create_if_missing=True)
        
        # Copy files if specified
        if files_to_copy:
            for file_path in files_to_copy:
                if not os.path.exists(file_path):
                    self.logger.warning(f"File not found, skipping: {file_path}")
                    continue
                
                filename = os.path.basename(file_path)
                destination = os.path.join(output_path, filename)
                
                if os.path.exists(destination) and not overwrite_existing:
                    self.logger.info(f"File already exists, skipping: {filename}")
                    continue
                
                try:
                    import shutil
                    shutil.copy2(file_path, destination)
                    self.logger.info(f"Copied {filename} to {output_path}")
                except Exception as e:
                    self.logger.error(f"Failed to copy {filename}: {str(e)}")
        
        return output_path
    
    def safe_file_write(
        self,
        file_path: str,
        content: str,
        encoding: str = 'utf-8',
        backup_existing: bool = True
    ) -> bool:
        """
        Safely write content to file with optional backup.
        
        Parameters:
            file_path: Path to write file
            content: Content to write
            encoding: File encoding
            backup_existing: Whether to backup existing file
            
        Returns:
            bool: True if write successful
            
        Example:
            >>> fs_ops = FileSystemOperations()
            >>> success = fs_ops.safe_file_write("/path/to/file.txt", "Hello World")
        """
        try:
            # Validate path
            file_path = self.validate_and_create_path(file_path, create_if_missing=True, path_type='file')
            
            # Backup existing file if requested
            if backup_existing and os.path.exists(file_path):
                backup_path = f"{file_path}.backup"
                try:
                    import shutil
                    shutil.copy2(file_path, backup_path)
                    self.logger.info(f"Created backup: {backup_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to create backup: {str(e)}")
            
            # Write content
            with open(file_path, 'w', encoding=encoding) as f:
                f.write(content)
            
            self.logger.info(f"Successfully wrote file: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write file {file_path}: {str(e)}")
            return False


class WebDataRetriever:
    """
    Web scraping and data retrieval utilities.
    
    This class provides secure web scraping capabilities with authentication
    support and proper error handling for data collection workflows.
    
    Note: Requires optional web scraping dependencies (requests, requests-ntlm, beautifulsoup4)
    """
    
    def __init__(self):
        """Initialize WebDataRetriever."""
        self.logger = logging.getLogger(__name__)
        
        if not WEB_SCRAPING_AVAILABLE:
            self.logger.warning("Web scraping dependencies not available. Install requests, requests-ntlm, and beautifulsoup4 for full functionality.")
    
    def download_files_with_authentication(
        self,
        base_url: str,
        save_location: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        file_extensions: Optional[List[str]] = None,
        chunk_size: int = 8192
    ) -> List[str]:
        """
        Download files from authenticated web location.
        
        Parameters:
            base_url: Base URL to download from
            save_location: Local directory to save files
            username: Authentication username
            password: Authentication password
            file_extensions: List of file extensions to download (e.g., ['.pdf', '.xlsx'])
            chunk_size: Chunk size for downloading large files
            
        Returns:
            List[str]: List of successfully downloaded file paths
            
        Raises:
            ConfigurationError: If web scraping dependencies unavailable
            
        Example:
            >>> retriever = WebDataRetriever()
            >>> files = retriever.download_files_with_authentication(
            ...     "https://example.com/files/",
            ...     "/local/download/path",
            ...     username="user",
            ...     password="pass",
            ...     file_extensions=['.csv', '.xlsx']
            ... )
        """
        if not WEB_SCRAPING_AVAILABLE:
            raise ConfigurationError("Web scraping dependencies not available")
        
        # Create save directory
        save_path = Path(save_location)
        save_path.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = []
        
        try:
            # Create session with authentication
            session = requests.Session()
            session.verify = False  # Disable SSL verification (use with caution)
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            if username and password:
                session.auth = HttpNtlmAuth(username, password)
            
            self.logger.info(f"Accessing {base_url}")
            
            # Get initial page
            response = session.get(base_url)
            response.raise_for_status()
            
            # Parse HTML to find file links
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all links
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link['href']
                
                # Filter by file extensions if specified
                if file_extensions:
                    if not any(href.lower().endswith(ext.lower()) for ext in file_extensions):
                        continue
                
                # Construct full URL
                if href.startswith('http'):
                    file_url = href
                else:
                    file_url = f"{base_url.rstrip('/')}/{href.lstrip('/')}"
                
                # Extract filename
                filename = os.path.basename(href)
                if not filename:
                    continue
                
                local_path = save_path / filename
                
                try:
                    self.logger.info(f"Downloading {filename}")
                    
                    file_response = session.get(file_url)
                    file_response.raise_for_status()
                    
                    # Write file in chunks
                    with open(local_path, 'wb') as f:
                        for chunk in file_response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                    
                    downloaded_files.append(str(local_path))
                    self.logger.info(f"Successfully downloaded: {filename}")
                    
                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Failed to download {filename}: {str(e)}")
                    continue
        
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Failed to access {base_url}: {str(e)}")
        
        self.logger.info(f"Downloaded {len(downloaded_files)} files to {save_location}")
        return downloaded_files
    
    def download_webpage_content(
        self,
        url: str,
        save_location: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        output_filename: str = 'webpage_content.html'
    ) -> Optional[str]:
        """
        Download webpage content and save to file.
        
        Parameters:
            url: URL to download
            save_location: Directory to save content
            username: Authentication username
            password: Authentication password
            output_filename: Name for saved file
            
        Returns:
            Optional[str]: Path to saved file if successful
            
        Example:
            >>> retriever = WebDataRetriever()
            >>> saved_path = retriever.download_webpage_content(
            ...     "https://example.com",
            ...     "/local/path"
            ... )
        """
        if not WEB_SCRAPING_AVAILABLE:
            raise ConfigurationError("Web scraping dependencies not available")
        
        # Create save directory
        save_path = Path(save_location)
        save_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create session
            session = requests.Session()
            session.verify = False
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            if username and password:
                session.auth = HttpNtlmAuth(username, password)
            else:
                session.auth = HttpNtlmAuth('', '')
            
            self.logger.info(f"Accessing {url}")
            response = session.get(url)
            response.raise_for_status()
            
            # Save content
            output_path = save_path / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            self.logger.info(f"Successfully saved webpage content to: {output_path}")
            return str(output_path)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error occurred: {str(e)}")
            return None


# =============================================================================
# BACKWARD COMPATIBILITY LAYER
# =============================================================================

"""
This section provides backward compatibility by creating function wrappers
that maintain the original function signatures while using the new class-based
implementations internally.
"""

# Initialize global instances for backward compatibility
_text_processor = TextProcessor()
_datetime_processor = DateTimeProcessor()
_dataframe_processor = DataFrameProcessor()
_statistical_analyzer = StatisticalAnalyzer()
_feature_engineer = FeatureEngineer()
_visualizer = DataVisualizer()
_logging_manager = LoggingManager()
_fs_operations = FileSystemOperations()
if WEB_SCRAPING_AVAILABLE:
    _web_retriever = WebDataRetriever()


# Text processing functions
def inWithReg(regLst, LstAll):
    """Backward compatibility wrapper for search_with_regex_patterns."""
    return _text_processor.search_with_regex_patterns(regLst, LstAll)


def normalize_text(text, remove_spaces=True, lowercase=True, special_chars=r'[^a-zA-Z0-9\s]',
                  replace_with='', max_length=None, fallback_text='unnamed'):
    """Backward compatibility wrapper for normalize_text."""
    return _text_processor.normalize_text(
        text, remove_spaces, lowercase, special_chars, replace_with, max_length, fallback_text
    )


def sanitize_filename(filename, max_length=100):
    """Backward compatibility wrapper for sanitize_filename."""
    return _text_processor.sanitize_filename(filename, max_length)


def clean_column_names(column_list, replacements={}, lowercase=False):
    """Backward compatibility wrapper for clean_column_names."""
    return _text_processor.clean_column_names(column_list, replacements, lowercase)


def find_fuzzy_matches(listA, listB, threshold=60):
    """Backward compatibility wrapper for find_fuzzy_matches."""
    return _text_processor.find_fuzzy_matches(listA, listB, threshold)


# DateTime functions
def check_timestamps(start, end, format_required='%Y-%m-%d'):
    """Backward compatibility wrapper for validate_timestamp_format."""
    return _datetime_processor.validate_timestamp_format(start, end, format_required)


def pass_days(start_date, end_date):
    """Backward compatibility wrapper for calculate_business_days_elapsed."""
    return _datetime_processor.calculate_business_days_elapsed(start_date, end_date, quarters_breakdown=True)


def readableTime(time):
    """Backward compatibility wrapper for convert_to_readable_time."""
    return _datetime_processor.convert_to_readable_time(time)


def datesList(range_date__year=[2018, 2099], range_date__month=[1, 12]):
    """Backward compatibility wrapper for generate_date_range."""
    return _datetime_processor.generate_date_range(
        tuple(range_date__year), tuple(range_date__month)
    )


def date2Num(df, dateCols):
    """Backward compatibility wrapper for convert_dates_to_numeric."""
    return _datetime_processor.convert_dates_to_numeric(df, dateCols)


# DataFrame functions
def movecol(df, cols_to_move=[], ref_col='', place='After'):
    """Backward compatibility wrapper for reorder_columns."""
    position = 'after' if place.lower() == 'after' else 'before'
    return _dataframe_processor.reorder_columns(df, cols_to_move, ref_col, position)


def merge_between(df1, df2, groupCol, closed="both"):
    """Backward compatibility wrapper for merge_with_date_intervals."""
    return _dataframe_processor.merge_with_date_intervals(
        df1, df2, groupCol, interval_closed=closed
    )


def cellWeight(df, axis=0):
    """Backward compatibility wrapper for calculate_cell_weights."""
    return _dataframe_processor.calculate_cell_weights(df, axis)


def reduce_mem_usage(df, obj2str_cols='all_columns', str2cat_cols='all_columns', 
                    downcast_int=True, downcast_float=True, verbose=True):
    """Backward compatibility wrapper for optimize_memory_usage."""
    optimized_df, report = _dataframe_processor.optimize_memory_usage(
        df, obj2str_cols, str2cat_cols, downcast_int, downcast_float, verbose
    )
    return optimized_df


def null_per_column(df):
    """Backward compatibility wrapper for calculate_null_percentages."""
    return _dataframe_processor.calculate_null_percentages(df)


# Statistical functions
def hypothesis_test(df, par, group, group_names, alpha=0.05):
    """Backward compatibility wrapper for perform_two_sample_t_test."""
    result = _statistical_analyzer.perform_two_sample_t_test(df, par, group, group_names, alpha)
    # Return simplified format for backward compatibility
    return result['p_value'], result['is_significant']


def kruskalwallis2(x, y):
    """Backward compatibility wrapper for perform_kruskal_wallis_test."""
    result = _statistical_analyzer.perform_kruskal_wallis_test(x, y)
    return result['h_statistic'], result['p_value']


def chi2_contingency(x, y):
    """Backward compatibility wrapper for perform_chi_square_independence_test."""
    result = _statistical_analyzer.perform_chi_square_independence_test(x, y)
    return result['chi2_statistic'], result['p_value'], result['degrees_of_freedom']


# Feature engineering functions
def cat2num(df, cat_decoder='OneHotEncoder'):
    """Backward compatibility wrapper for convert_categorical_to_numeric."""
    encoded_df, info = _feature_engineer.convert_categorical_to_numeric(df, encoding_method=cat_decoder)
    return encoded_df


def discretizer(x, y=None, labels=["Q1", "Q2", "Q3", "Q4"], method='cut'):
    """Backward compatibility wrapper for discretize_continuous_variables."""
    discretized, info = _feature_engineer.discretize_continuous_variables(
        x, y, bins=len(labels), labels=labels, method=method
    )
    return discretized


# Visualization functions  
def corrmap(df0, method='kendall', diagonal_plot=True, **kwargs):
    """Backward compatibility wrapper for create_correlation_heatmap."""
    return _visualizer.create_correlation_heatmap(
        df0, method, diagonal_plot, **kwargs
    )


# Logging functions
def logmaker(uFile, name, logLevel=logging.INFO):
    """Backward compatibility wrapper for create_custom_logger."""
    return _logging_manager.create_custom_logger(name, uFile, logLevel)


def custom_print(message, logger=None):
    """Backward compatibility wrapper for custom_print_with_logging."""
    _logging_manager.custom_print_with_logging(message, logger)


def setup_logger(log_file):
    """Backward compatibility wrapper for setup_analysis_logger.""" 
    analysis_name = os.path.splitext(os.path.basename(log_file))[0]
    log_dir = os.path.dirname(log_file) or "./logs"
    return _logging_manager.setup_analysis_logger(analysis_name, log_dir, include_timestamp=False)


def load_config(config_file, logger):
    """Backward compatibility wrapper for load_configuration_from_file."""
    return _logging_manager.load_configuration_from_file(config_file, logger)


# File system functions
def check_path(path):
    """Backward compatibility wrapper for validate_and_create_path."""
    try:
        _fs_operations.validate_and_create_path(path, create_if_missing=False)
        return True
    except ValidationError:
        return False


def setOutputFolder(outputFolder, uFiles, overWrite):
    """Backward compatibility wrapper for setup_output_directory."""
    return _fs_operations.setup_output_directory(outputFolder, uFiles, overWrite)


# Web scraping functions (if available)
if WEB_SCRAPING_AVAILABLE:
    def download_intranet_files(url, save_location, username, password, file_extensions, chunk_size, logger):
        """Backward compatibility wrapper for download_files_with_authentication."""
        return _web_retriever.download_files_with_authentication(
            url, save_location, username, password, file_extensions, chunk_size
        )

    def download_webpage_content(url, save_location, username, password, logger):
        """Backward compatibility wrapper for download_webpage_content."""
        return _web_retriever.download_webpage_content(url, save_location, username, password)


# Utility functions that don't fit into main classes
def retrieve_name(var):
    """
    Retrieve variable name using introspection.
    
    Parameters:
        var: Variable to get name for
        
    Returns:
        str: Variable name
        
    Note: This function has limitations and should be used carefully
    """
    import inspect
    frame = inspect.currentframe()
    try:
        frame = frame.f_back
        for name, value in frame.f_locals.items():
            if value is var:
                return name
        for name, value in frame.f_globals.items():
            if value is var:
                return name
    finally:
        del frame
    return "unknown"


def flattenList(ulist):
    """
    Flatten nested list structure recursively.
    
    Parameters:
        ulist: Nested list to flatten
        
    Returns:
        list: Flattened list
        
    Example:
        >>> nested = [[1, 2], [3, [4, 5]], 6]
        >>> result = flattenList(nested)
        >>> print(result)  # [1, 2, 3, 4, 5, 6]
    """
    def _flatten_recursive(lst):
        result = []
        for item in lst:
            if isinstance(item, list):
                result.extend(_flatten_recursive(item))
            else:
                result.append(item)
        return result
    
    if not isinstance(ulist, list):
        raise ValidationError("Input must be a list")
    
    return _flatten_recursive(ulist)


def unique_list(seq):
    """
    Get unique values from sequence preserving order.
    
    Parameters:
        seq: Input sequence
        
    Returns:
        list: List with unique values in original order
        
    Example:
        >>> data = [1, 2, 2, 3, 1, 4]
        >>> unique = unique_list(data)
        >>> print(unique)  # [1, 2, 3, 4]
    """
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]


def remove_extra_none(nested_lst):
    """
    Remove extra None values from nested list structure.
    
    Parameters:
        nested_lst: Nested list that may contain None values
        
    Returns:
        list: Cleaned nested list
        
    Example:
        >>> data = [1, None, [2, None, 3], None]
        >>> cleaned = remove_extra_none(data)
        >>> print(cleaned)  # [1, [2, 3]]
    """
    def clean_recursive(item):
        if isinstance(item, list):
            cleaned = [clean_recursive(sub_item) for sub_item in item if sub_item is not None]
            return [sub_item for sub_item in cleaned if sub_item != []]
        return item
    
    if not isinstance(nested_lst, list):
        return nested_lst
    
    return clean_recursive(nested_lst)


# =============================================================================
# MODULE INITIALIZATION AND EXPORTS
# =============================================================================

# Set up module-level logger
logger = logging.getLogger(__name__)

# Export classes for direct usage
__all__ = [
    # Exception classes
    'DataScienceToolboxError', 'ValidationError', 'ConfigurationError', 'DataProcessingError',
    
    # Configuration classes
    'PlotConfiguration',
    
    # Core classes
    'InputValidator', 'TextProcessor', 'DateTimeProcessor', 'DataFrameProcessor',
    'StatisticalAnalyzer', 'FeatureEngineer', 'DataVisualizer', 'LoggingManager',
    'FileSystemOperations', 'WebDataRetriever',
    
    # Backward compatibility functions
    'inWithReg', 'normalize_text', 'sanitize_filename', 'clean_column_names',
    'find_fuzzy_matches', 'check_timestamps', 'pass_days', 'readableTime',
    'datesList', 'date2Num', 'movecol', 'merge_between', 'cellWeight',
    'reduce_mem_usage', 'null_per_column', 'hypothesis_test', 'kruskalwallis2',
    'chi2_contingency', 'cat2num', 'discretizer', 'corrmap', 'logmaker',
    'custom_print', 'setup_logger', 'load_config', 'check_path',
    'setOutputFolder', 'retrieve_name', 'flattenList', 'unique_list',
    'remove_extra_none'
]

# Add web scraping functions to exports if available
if WEB_SCRAPING_AVAILABLE:
    __all__.extend(['download_intranet_files', 'download_webpage_content'])

logger.info("Data Science Toolbox initialized successfully")
logger.info(f"Available classes: {len([item for item in __all__ if not item.islower()])}")
logger.info(f"Backward compatibility functions: {len([item for item in __all__ if item.islower()])}")
if not WEB_SCRAPING_AVAILABLE:
    logger.warning("Web scraping functionality unavailable - install requests, requests-ntlm, beautifulsoup4")
if not WORDCLOUD_AVAILABLE:
    logger.warning("Word cloud functionality unavailable - install wordcloud")