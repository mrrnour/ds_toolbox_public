"""
Data Utilities - Data Science Toolbox (Enhanced Version)
=========================================================

A comprehensive collection of data science utility functions organized into logical
class groupings for better maintainability and modularity. This enhanced version
includes SQL processing utilities, advanced data encoding functions, and specialized
data manipulation tools.

Classes:
--------
- TextProcessor: Text normalization, cleaning, fuzzy matching
- ListUtilities: List operations, flattening, regex search  
- DataFrameUtilities: DataFrame manipulation, optimization, analysis
- DateTimeUtilities: Date validation, time conversion, range generation
- FileSystemUtilities: Path validation, directory setup
- DataVisualization: Sankey diagrams, 3D scatter plots, word clouds
- SQLProcessor: SQL query parsing, statement splitting, file processing
- EncodingUtilities: Advanced dummy encoding, sparse label processing
- ProductUtilities: Product analysis, column condensing, business logic

Author: Data Science Toolbox Contributors
License: MIT License
"""

# Standard library imports
import os, sys
import re
import time
import logging
import datetime as dt
import zipfile
import warnings
import argparse
from typing import List, Tuple, Dict, Any, Optional, Union

# Third-party imports (with graceful handling)
try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    logging.warning(f"Core dependency not found: {e}")
    raise

try:
    from difflib import SequenceMatcher
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches  
    from matplotlib.patches import Circle
except ImportError as e:
    logging.warning(f"Optional visualization dependency not found: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# TEXT PROCESSING AND NORMALIZATION
# =============================================================================

class TextProcessor:
    """
    Comprehensive text processing utilities for data science workflows.
    
    This class provides methods for text normalization, cleaning, fuzzy matching,
    and filename sanitization commonly needed in data analysis projects.
    """
    
    @staticmethod
    def normalize_text(text: str, 
                      remove_spaces: bool = True, 
                      lowercase: bool = True, 
                      special_chars: str = r'[^a-zA-Z0-9\s]',
                      replace_with: str = '',
                      max_length: Optional[int] = None,
                      fallback_text: str = 'unnamed') -> str:
        """
        Flexible text normalization function for data preprocessing.
        
        This method provides comprehensive text cleaning capabilities including
        case normalization, special character removal, space handling, and
        length limiting with fallback options.
        
        Parameters
        ----------
        text : str
            Input text to normalize
        remove_spaces : bool, default=True
            Whether to remove all spaces from the text
        lowercase : bool, default=True
            Whether to convert text to lowercase
        special_chars : str, default=r'[^a-zA-Z0-9\\s]'
            Regex pattern for characters to match and replace
        replace_with : str, default=''
            Replacement string for matched characters
        max_length : int, optional
            Maximum length to truncate text to
        fallback_text : str, default='unnamed'
            Text to return if result is empty or only whitespace
            
        Returns
        -------
        str
            Normalized text string
            
        Raises
        ------
        ValueError
            If fallback_text is empty or None when needed
            
        Examples
        --------
        >>> processor = TextProcessor()
        >>> processor.normalize_text("Hello World! 123")
        'helloworld123'
        
        >>> processor.normalize_text("Special@#$Characters", replace_with='_')
        'special___characters'
        """
        if not isinstance(text, str):
            text = str(text)
        
        if lowercase:
            text = text.lower()
        
        if special_chars:
            text = re.sub(special_chars, replace_with, text)
        
        if remove_spaces:
            text = re.sub(r'\s+', replace_with, text)
        
        text = text.strip()
        
        if max_length is not None:
            text = text[:max_length]
        
        if not text or text.isspace():
            if not fallback_text:
                raise ValueError("fallback_text cannot be empty when text normalization results in empty string")
            text = fallback_text
        
        return text
    
    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 100) -> str:
        """
        Sanitize filename for safe filesystem operations.
        
        Removes or replaces invalid filesystem characters while preserving
        readability. Uses conservative approach for cross-platform compatibility.
        
        Parameters
        ----------
        filename : str
            Original filename to sanitize
        max_length : int, default=100
            Maximum length for the resulting filename
            
        Returns
        -------
        str
            Sanitized filename safe for filesystem use
            
        Examples
        --------
        >>> TextProcessor.sanitize_filename("My File<>Name.txt")
        'my file__name.txt'
        """
        return TextProcessor.normalize_text(
            filename,
            remove_spaces=False,
            lowercase=True,
            special_chars=r'[\\/*?:"<>|\r\n\t]',
            replace_with='_',
            max_length=max_length,
            fallback_text='unnamed'
        )
    
    @staticmethod
    def clean_column_names(column_list: List[str], 
                          replacements: Dict[str, str] = None, 
                          lowercase: bool = False) -> List[str]:
        """
        Clean column names for DataFrame compatibility.
        
        Standardizes column names to be suitable for pandas DataFrame operations
        by removing special characters, handling spaces, and ensuring valid
        Python identifier format.
        
        Parameters
        ----------
        column_list : list of str
            List of raw column names to clean
        replacements : dict, optional
            Custom string replacements to apply before normalization
        lowercase : bool, default=False
            Whether to convert to lowercase
            
        Returns
        -------
        list of str
            Cleaned column names suitable for DataFrame operations
            
        Raises
        ------
        TypeError
            If column_list is not iterable
            
        Examples
        --------
        >>> TextProcessor.clean_column_names(['First Name', 'Email@Address', '2nd_Value'])
        ['first_name', 'email_address', 'col_2nd_value']
        """
        if not hasattr(column_list, '__iter__'):
            raise TypeError("column_list must be iterable")
        
        if replacements is None:
            replacements = {}
            
        cleaned_columns = []
        
        for col in column_list:
            col = str(col)
            
            # Apply custom replacements first
            for old, new in replacements.items():
                col = col.replace(old, new)
            
            # Use normalize_text for most processing
            col = TextProcessor.normalize_text(
                col,
                remove_spaces=True,
                lowercase=lowercase,
                special_chars=r'[^a-zA-Z0-9_]',
                replace_with='_',
                fallback_text='unnamed_column'
            )
            
            # Collapse multiple underscores and clean edges
            col = re.sub(r'_+', '_', col).strip('_')
            
            # Ensure valid Python identifier
            if col and col[0].isdigit():
                col = 'col_' + col
            
            if not col:
                col = 'unnamed_column'
            
            cleaned_columns.append(col)
        
        return cleaned_columns
    
    @staticmethod
    def find_fuzzy_matches(list_a: List[Any], 
                          list_b: List[Any], 
                          similarity_threshold: float = 60.0) -> Dict[Any, Dict[str, Any]]:
        """
        Find fuzzy string matches between two lists.
        
        Uses normalized text comparison with configurable similarity threshold
        to identify potential matches between list elements. Prevents duplicate
        matching by tracking used indices.
        
        Parameters
        ----------
        list_a : list
            First list for comparison
        list_b : list
            Second list for comparison  
        similarity_threshold : float, default=60.0
            Minimum similarity percentage (0-100) for matches
            
        Returns
        -------
        dict
            Dictionary mapping items from list_a to match information
            
        Raises
        ------
        ValueError
            If similarity_threshold is not between 0 and 100
            
        Examples
        --------
        >>> list_a = ['Apple Inc.', 'Microsoft Corp']
        >>> list_b = ['apple incorporated', 'Google LLC']
        >>> matches = TextProcessor.find_fuzzy_matches(list_a, list_b, threshold=70)
        """
        if not 0 <= similarity_threshold <= 100:
            raise ValueError("similarity_threshold must be between 0 and 100")
        
        matches = {}
        used_b_indices = set()
        
        for i, item_a in enumerate(list_a):
            normalized_a = TextProcessor.normalize_text(str(item_a))
            best_match = None
            best_similarity = 0
            best_index = -1
            
            for j, item_b in enumerate(list_b):
                if j in used_b_indices:
                    continue
                    
                normalized_b = TextProcessor.normalize_text(str(item_b))
                similarity = SequenceMatcher(None, normalized_a, normalized_b).ratio() * 100
                
                if similarity >= similarity_threshold and similarity > best_similarity:
                    best_match = item_b
                    best_similarity = similarity
                    best_index = j
            
            if best_match is not None:
                matches[item_a] = {
                    'match': best_match,
                    'similarity': best_similarity,
                    'normalized_a': normalized_a,
                    'normalized_b': TextProcessor.normalize_text(str(best_match))
                }
                used_b_indices.add(best_index)
        
        return matches
    
    @staticmethod
    def clean_sql_query(query: str, 
                       start_time: Optional[str] = None, 
                       end_time: Optional[str] = None) -> str:
        """
        Clean and format SQL query string with optional time filtering.
        
        This method provides comprehensive SQL query cleaning including whitespace
        normalization, query wrapper removal, and optional time filtering for
        data analysis workflows.
        
        Parameters
        ----------
        query : str
            SQL query string to clean and format
        start_time : str, optional
            Start time for filtering (ISO format recommended)
        end_time : str, optional
            End time for filtering (ISO format recommended)
            
        Returns
        -------
        str
            Cleaned and formatted SQL query string
            
        Raises
        ------
        ValueError
            If query is empty or None
            
        Examples
        --------
        >>> TextProcessor.clean_sql_query("  SELECT * FROM table  ")
        'SELECT * FROM table'
        
        >>> TextProcessor.clean_sql_query(
        ...     "SELECT * FROM logs", 
        ...     start_time="2023-01-01", 
        ...     end_time="2023-12-31"
        ... )
        "SELECT * FROM logs WHERE timestamp BETWEEN '2023-01-01' AND '2023-12-31'"
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Basic cleaning - normalize whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Remove query wrapper if present (format: (query) query)
        if (cleaned.lower().endswith(' query') and 
            cleaned.startswith('(') and 
            cleaned.endswith(')')):
            cleaned = cleaned[1:-6].strip()
        
        # Apply time filtering if provided
        if start_time and end_time:
            # Check if WHERE clause already exists
            if 'WHERE' in cleaned.upper():
                cleaned += f" AND timestamp BETWEEN '{start_time}' AND '{end_time}'"
            else:
                cleaned += f" WHERE timestamp BETWEEN '{start_time}' AND '{end_time}'"
        
        return cleaned


# =============================================================================
# LIST AND COLLECTION UTILITIES
# =============================================================================

class ListUtilities:
    """
    Utility functions for list and collection operations in data science workflows.
    
    Provides methods for list manipulation, flattening, deduplication, and
    regular expression-based filtering commonly needed in data preprocessing.
    """
    
    @staticmethod
    def search_with_regex(regex_patterns: Union[str, List[str]], 
                         target_list: List[str]) -> Tuple[List[str], List[bool]]:
        """
        Search for regex patterns in a list of strings.
        
        Filters a target list using one or more regular expression patterns,
        returning both matching items and boolean indicators.
        
        Parameters
        ----------
        regex_patterns : str or list of str
            Regular expression pattern(s) to search for
        target_list : list of str
            List of strings to search through
            
        Returns
        -------
        tuple of (list, list of bool)
            - List of matching strings from target_list
            - Boolean array indicating matches for each target_list item
            
        Raises
        ------
        TypeError
            If target_list is not iterable or contains non-string elements
        re.error
            If regex_patterns contain invalid regular expressions
            
        Examples
        --------
        >>> patterns = [r'vol_flag$', r'_date$']
        >>> targets = ['tv_vol_flag', 'snapshot_date', 'user_id']
        >>> matches, indicators = ListUtilities.search_with_regex(patterns, targets)
        >>> matches
        ['tv_vol_flag', 'snapshot_date']
        """
        if isinstance(regex_patterns, str):
            regex_patterns = [regex_patterns]
        
        if not hasattr(target_list, '__iter__'):
            raise TypeError("target_list must be iterable")
        
        matches = []
        
        for pattern in regex_patterns:
            try:
                compiled_pattern = re.compile(pattern)
                pattern_matches = list(filter(compiled_pattern.search, target_list))
                matches.extend(pattern_matches)
            except re.error as e:
                raise re.error(f"Invalid regex pattern '{pattern}': {e}")
        
        # Remove duplicates while preserving order
        unique_matches = []
        for item in matches:
            if item not in unique_matches:
                unique_matches.append(item)
        
        # Create boolean indicator array
        indicators = [item in unique_matches for item in target_list]
        
        return unique_matches, indicators
    
    @staticmethod
    def flatten_nested_list(nested_list: List[Any]) -> List[Any]:
        """
        Flatten arbitrarily nested list structure.
        
        Recursively flattens nested lists into a single flat list while
        preserving the original order of elements.
        
        Parameters
        ----------
        nested_list : list
            Nested list structure to flatten
            
        Returns
        -------
        list
            Flattened list containing all elements
            
        Examples
        --------
        >>> nested = [1, [2, 3], [4, [5, 6]], 7]
        >>> ListUtilities.flatten_nested_list(nested)
        [1, 2, 3, 4, 5, 6, 7]
        """
        flattened = []
        
        for item in nested_list:
            if isinstance(item, list):
                flattened.extend(ListUtilities.flatten_nested_list(item))
            else:
                flattened.append(item)
        
        return flattened
    
    @staticmethod
    def get_unique_ordered(sequence: List[Any]) -> List[Any]:
        """
        Get unique elements from sequence while preserving order.
        
        Removes duplicates from a sequence while maintaining the original
        order of first occurrences.
        
        Parameters
        ----------
        sequence : list
            Input sequence with potential duplicates
            
        Returns
        -------
        list
            List with unique elements in original order
            
        Examples
        --------
        >>> ListUtilities.get_unique_ordered([1, 2, 2, 3, 1, 4])
        [1, 2, 3, 4]
        """
        seen = set()
        unique_items = []
        
        for item in sequence:
            if item not in seen:
                seen.add(item)
                unique_items.append(item)
        
        return unique_items
    
    @staticmethod
    def remove_nested_none_values(nested_structure: List[Any]) -> List[Any]:
        """
        Remove None values from nested list structure.
        
        Recursively removes None values from nested lists while preserving
        structure for non-None elements.
        
        Parameters
        ----------
        nested_structure : list
            Nested list potentially containing None values
            
        Returns
        -------
        list
            Cleaned nested structure without None values
            
        Examples
        --------
        >>> nested = [1, None, [2, None, 3], None, 4]
        >>> ListUtilities.remove_nested_none_values(nested)
        [1, [2, 3], 4]
        """
        cleaned = []
        
        for item in nested_structure:
            if item is None:
                continue
            elif isinstance(item, list):
                cleaned_sublist = ListUtilities.remove_nested_none_values(item)
                if cleaned_sublist:  # Only add non-empty sublists
                    cleaned.append(cleaned_sublist)
            else:
                cleaned.append(item)
        
        return cleaned


# =============================================================================
# SQL PROCESSING UTILITIES
# =============================================================================

class SQLProcessor:
    """
    Comprehensive SQL processing utilities for data science workflows.
    
    This class provides methods for parsing SQL files, splitting query statements,
    handling comments, and processing complex SQL scripts commonly used in
    data analysis pipelines.
    """
    
    @staticmethod
    def validate_file_path(file_path: str) -> str:
        """
        Validate that a file path exists and return expanded path.
        
        Parameters
        ----------
        file_path : str
            Path to validate
            
        Returns
        -------
        str
            Expanded and validated file path
            
        Raises
        ------
        argparse.ArgumentTypeError
            If the path does not exist
            
        Examples
        --------
        >>> path = SQLProcessor.validate_file_path('~/queries/script.sql')
        """
        if '~' in file_path:
            file_path = os.path.expanduser(file_path)
        
        if not os.path.exists(file_path):
            msg = f"File ({file_path}) not found!"
            raise argparse.ArgumentTypeError(msg)
        
        return file_path
    
    @staticmethod
    def split_sql_statements(sql_text: str) -> List[str]:
        """
        Split SQL text into individual statements based on semicolon delimiter.
        
        This method provides robust SQL statement parsing that handles:
        - Single and double quoted strings
        - Single-line comments (--)
        - Multi-line comments (/* */)
        - Proper semicolon detection within and outside of quoted strings
        
        Parameters
        ----------
        sql_text : str
            SQL text containing one or more statements
            
        Returns
        -------
        list of str
            List of individual SQL statements
            
        Raises
        ------
        ValueError
            If SQL text contains invalid syntax or illegal state
            
        Examples
        --------
        >>> sql = '''
        ... SELECT * FROM users; -- Get all users
        ... INSERT INTO logs VALUES ('test');
        ... '''
        >>> statements = SQLProcessor.split_sql_statements(sql)
        >>> len(statements)
        2
        """
        if not sql_text or not sql_text.strip():
            return []
        
        results = []
        current = ''
        state = None
        
        for c in sql_text:
            if state is None:  # default state, outside of special entity
                current += c
                if c in '"\'':
                    # quoted string
                    state = c
                elif c == '-':
                    # probably "--" comment
                    state = '-'
                elif c == '/':
                    # probably '/*' comment
                    state = '/'
                elif c == ';':
                    # remove it from the statement
                    current = current[:-1].strip()
                    # and save current stmt unless empty
                    if current:
                        results.append(current)
                    current = ''
            elif state == '-':
                if c != '-':
                    # not a comment
                    state = None
                    current += c
                    continue
                # remove first minus
                current = current[:-1]
                # comment until end of line
                state = '--'
            elif state == '--':
                if c == '\n':
                    # end of comment
                    # and we do include this newline
                    current += c
                    state = None
                # else just ignore
            elif state == '/':
                if c != '*':
                    state = None
                    current += c
                    continue
                # remove starting slash
                current = current[:-1]
                # multiline comment
                state = '/*'
            elif state == '/*':
                if c == '*':
                    # probably end of comment
                    state = '/**'
            elif state == '/**':
                if c == '/':
                    state = None
                else:
                    # not an end
                    state = '/*'
            elif state[0] in '"\'':
                current += c
                if state.endswith('\\'):
                    # prev was backslash, don't check for ender
                    # just revert to regular state
                    state = state[0]
                    continue
                elif c == '\\':
                    # don't check next char
                    state += '\\'
                    continue
                elif c == state[0]:
                    # end of quoted string
                    state = None
            else:
                raise ValueError(f'Illegal parser state: {state}')

        # Handle remaining content
        if current:
            current = current.rstrip(';').strip()
            if current:
                results.append(current)
        
        return results
    
    @staticmethod
    def parse_sql_file(file_path: str) -> List[str]:
        """
        Read and parse SQL statements from a file.
        
        This method combines file validation and SQL statement parsing to
        extract individual queries from SQL script files. Useful for processing
        database migration scripts, stored procedures, or query collections.
        
        Parameters
        ----------
        file_path : str
            Path to the SQL file to parse
            
        Returns
        -------
        list of str
            List of SQL statements found in the file
            
        Raises
        ------
        argparse.ArgumentTypeError
            If file path is invalid
        FileNotFoundError
            If file cannot be opened
        UnicodeDecodeError
            If file contains invalid character encoding
            
        Examples
        --------
        >>> statements = SQLProcessor.parse_sql_file('/path/to/queries.sql')
        >>> print(f"Found {len(statements)} SQL statements")
        """
        validated_path = SQLProcessor.validate_file_path(file_path)
        
        try:
            with open(validated_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            
            sql_statements = SQLProcessor.split_sql_statements(file_content)
            
            logger.info(f"Successfully parsed {len(sql_statements)} SQL statements from {file_path}")
            return sql_statements
            
        except Exception as e:
            logger.error(f"Error parsing SQL file {file_path}: {str(e)}")
            raise


# =============================================================================
# ADVANCED DATA ENCODING UTILITIES
# =============================================================================

class EncodingUtilities:
    """
    Advanced data encoding and transformation utilities for data science workflows.
    
    This class provides optimized methods for dummy variable encoding, sparse label
    processing, and categorical data transformations that are more memory-efficient
    than standard pandas operations for large datasets.
    """
    
    @staticmethod
    def create_optimized_dummy_encoding(series_data: pd.Series, 
                                       delimiter: str = '; ') -> pd.DataFrame:
        """
        Create dummy variables with optimized memory usage for large datasets.
        
        This method provides an alternative to pandas.get_dummies() that is more
        memory-efficient and faster for large datasets with string-separated
        categorical values.
        
        Parameters
        ----------
        series_data : pd.Series
            Series containing delimiter-separated categorical values
        delimiter : str, default='; '
            String delimiter used to separate multiple categories
            
        Returns
        -------
        pd.DataFrame
            DataFrame with dummy-encoded columns for each unique category
            
        Raises
        ------
        TypeError
            If series_data is not a pandas Series
        ValueError
            If series_data is empty or contains no valid categories
            
        Examples
        --------
        >>> data = pd.Series(['cat1; cat2', 'cat2; cat3', 'cat1'])
        >>> encoded = EncodingUtilities.create_optimized_dummy_encoding(data)
        >>> encoded.columns.tolist()
        ['cat1', 'cat2', 'cat3']
        """
        if not isinstance(series_data, pd.Series):
            raise TypeError("series_data must be a pandas Series")
        
        if series_data.empty:
            raise ValueError("series_data cannot be empty")
        
        try:
            # Split the series into expanded DataFrame
            expanded_data = series_data.str.split(delimiter, expand=True)
            
            # Get all unique labels, excluding NaN values
            all_values = expanded_data.values.flatten()
            valid_values = all_values[~pd.isnull(all_values)]
            
            if len(valid_values) == 0:
                raise ValueError("No valid categories found in series_data")
            
            unique_labels = np.sort(np.unique(valid_values))
            
            logger.info(f"Creating dummy encoding for {len(unique_labels)} unique categories")
            
            # Apply encoding logic using vectorized operations
            def encode_row(row_values, all_labels):
                """Create binary encoding for a single row."""
                indicators = np.isin(all_labels, row_values).astype(int)
                return indicators
            
            # Apply encoding to each row
            encoded_matrix = np.apply_along_axis(
                encode_row, 1, expanded_data.values, all_labels=unique_labels
            )
            
            # Create result DataFrame
            result_df = pd.DataFrame(
                encoded_matrix, 
                columns=unique_labels, 
                index=series_data.index
            )
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error in dummy encoding: {str(e)}")
            raise
    
    @staticmethod
    def create_sparse_label_encoding(dataframe: pd.DataFrame,
                                    category_column: str,
                                    value_column: str,
                                    delimiter: str = "; ") -> pd.DataFrame:
        """
        Create sparse label encoding where dummy variables are filled with values.
        
        This method creates dummy-encoded columns for categories but fills them with
        corresponding values instead of binary indicators. Useful for product-price
        relationships or category-value associations.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame containing category and value columns
        category_column : str
            Name of column containing delimiter-separated categories
        value_column : str
            Name of column containing corresponding values for categories
        delimiter : str, default="; "
            String delimiter used to separate categories and values
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns for each category filled with corresponding values
            
        Raises
        ------
        KeyError
            If specified columns are not found in the DataFrame
        ValueError
            If DataFrame is empty or columns contain invalid data
            
        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'products': ['prod1; prod2', 'prod2; prod3'],
        ...     'prices': ['10; 20', '20; 30']
        ... })
        >>> encoded = EncodingUtilities.create_sparse_label_encoding(
        ...     df, 'products', 'prices'
        ... )
        """
        if dataframe.empty:
            raise ValueError("DataFrame cannot be empty")
        
        if category_column not in dataframe.columns:
            raise KeyError(f"Category column '{category_column}' not found in DataFrame")
        
        if value_column not in dataframe.columns:
            raise KeyError(f"Value column '{value_column}' not found in DataFrame")
        
        try:
            logger.info(f"Creating sparse label encoding for {len(dataframe)} rows")
            
            # First create dummy encoding for categories
            dummy_encoded = EncodingUtilities.create_optimized_dummy_encoding(
                dataframe[category_column], delimiter
            )
            
            logger.info("Dummy encoding completed, applying value mapping")
            
            # Combine original data with dummy encoding
            combined_data = pd.concat([
                dataframe[value_column], 
                dataframe[category_column], 
                dummy_encoded
            ], axis=1)
            
            def apply_sparse_encoding(row, delimiter=delimiter):
                """Apply sparse encoding to a single row."""
                try:
                    # Parse categories and values
                    categories = str(row[1]).split(delimiter) if pd.notna(row[1]) else []
                    values_str = str(row[0]) if pd.notna(row[0]) else "0"
                    
                    # Handle different value formats
                    try:
                        values = [float(x.strip()) for x in values_str.split(delimiter)]
                    except ValueError:
                        # If conversion fails, use zeros
                        values = [0] * len(categories)
                    
                    # Ensure values and categories have same length
                    if len(values) < len(categories):
                        values.extend([0] * (len(categories) - len(values)))
                    
                    # Apply values to corresponding dummy columns
                    row_result = row[2:].copy()  # Copy dummy encoded values
                    
                    for i, category in enumerate(categories):
                        if category.strip() and i < len(values):
                            # Find matching column and set value
                            category_cleaned = category.strip().upper()
                            matching_cols = [col for col in dummy_encoded.columns 
                                           if col.upper() == category_cleaned]
                            
                            for col in matching_cols:
                                col_index = dummy_encoded.columns.get_loc(col)
                                if col_index < len(row_result):
                                    row_result.iloc[col_index] = values[i]
                    
                    return row_result
                    
                except Exception as e:
                    logger.warning(f"Error processing row: {e}")
                    return row[2:]  # Return original dummy encoding
            
            # Apply sparse encoding to all rows
            encoded_matrix = combined_data.apply(apply_sparse_encoding, axis=1, result_type='expand')
            
            # Create result DataFrame with proper column names
            result_df = pd.DataFrame(
                encoded_matrix.values,
                columns=[col.upper() for col in dummy_encoded.columns],
                index=dataframe.index,
                dtype=np.float32  # Use float32 to save memory
            )
            
            logger.info(f"Sparse label encoding completed with {len(result_df.columns)} feature columns")
            return result_df
            
        except Exception as e:
            logger.error(f"Error in sparse label encoding: {str(e)}")
            raise
    
    @staticmethod
    def fill_dataframe_with_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Replace non-zero values in DataFrame with their corresponding column names.
        
        This utility method is useful for creating readable representations of
        sparse or binary matrices where you want to see which features are
        active rather than just binary indicators.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame with numeric values
            
        Returns
        -------
        pd.DataFrame
            DataFrame where non-zero values are replaced with column names
            
        Examples
        --------
        >>> df = pd.DataFrame({'A': [1, 0], 'B': [0, 2], 'C': [3, 0]})
        >>> named = EncodingUtilities.fill_dataframe_with_column_names(df)
        >>> # Non-zero values replaced with column names: 'A', 'B', 'C'
        """
        if dataframe.empty:
            return dataframe.copy()
        
        try:
            # Create a matrix of column names tiled to match DataFrame shape
            column_names_matrix = np.tile(
                dataframe.columns, 
                [len(dataframe.index), 1]
            )
            
            # Create boolean mask for non-zero values
            non_zero_mask = dataframe.astype(bool)
            
            # Apply column names where values are non-zero, empty string otherwise
            result_matrix = np.where(
                non_zero_mask,
                column_names_matrix,
                ''
            )
            
            result_df = pd.DataFrame(
                result_matrix,
                columns=dataframe.columns,
                index=dataframe.index
            )
            
            return result_df
            
        except Exception as e:
            logger.error(f"Error filling DataFrame with column names: {str(e)}")
            raise


# =============================================================================
# PRODUCT AND BUSINESS UTILITIES
# =============================================================================

class ProductUtilities:
    """
    Specialized utilities for product analysis and business data processing.
    
    This class provides methods for product data cleaning, column condensing,
    business logic processing, and telecommunications-specific data transformations.
    """
    
    @staticmethod
    def join_non_zero_values(values: List[Any], separator: str = ', ') -> str:
        """
        Join non-zero/non-empty values with specified separator.
        
        Parameters
        ----------
        values : list
            List of values to filter and join
        separator : str, default=', '
            String separator for joining values
            
        Returns
        -------
        str
            Joined string of non-zero values
            
        Examples
        --------
        >>> ProductUtilities.join_non_zero_values([1, 0, 'test', '', 5])
        '1, test, 5'
        """
        try:
            filtered_values = [
                str(val) for val in values 
                if val != 0 and val != '' and val is not None and not pd.isna(val)
            ]
            return separator.join(filtered_values)
        except Exception as e:
            logger.warning(f"Error joining values: {e}")
            return ""
    
    @staticmethod
    def clean_product_descriptions(product_descriptions: pd.DataFrame,
                                  reference_dataframe: pd.DataFrame,
                                  product_id_column: str = 'PRODUCT_ID',
                                  lob_column: str = 'LOB') -> pd.DataFrame:
        """
        Clean and filter product descriptions based on reference DataFrame columns.
        
        This method removes duplicate product descriptions and keeps only those
        products that match columns in the reference DataFrame. Commonly used
        for aligning product metadata with transactional data.
        
        Parameters
        ----------
        product_descriptions : pd.DataFrame
            DataFrame containing product metadata with PRODUCT_ID and LOB columns
        reference_dataframe : pd.DataFrame
            Reference DataFrame whose columns represent valid product IDs
        product_id_column : str, default='PRODUCT_ID'
            Name of the product ID column in product_descriptions
        lob_column : str, default='LOB'
            Name of the line-of-business column in product_descriptions
            
        Returns
        -------
        pd.DataFrame
            Cleaned product descriptions aligned with reference DataFrame
            
        Raises
        ------
        KeyError
            If required columns are missing from product_descriptions
        ValueError
            If shapes don't match after processing
            
        Examples
        --------
        >>> products = pd.DataFrame({
        ...     'PRODUCT_ID': ['prod1', 'prod2', 'prod1'],
        ...     'LOB': ['internet', 'tv', 'internet']
        ... })
        >>> ref_df = pd.DataFrame(columns=['PROD1', 'PROD2'])
        >>> cleaned = ProductUtilities.clean_product_descriptions(products, ref_df)
        """
        if product_descriptions.empty:
            raise ValueError("product_descriptions cannot be empty")
        
        if product_id_column not in product_descriptions.columns:
            raise KeyError(f"Column '{product_id_column}' not found in product_descriptions")
        
        if lob_column not in product_descriptions.columns:
            raise KeyError(f"Column '{lob_column}' not found in product_descriptions")
        
        try:
            logger.info(f"Cleaning {len(product_descriptions)} product descriptions")
            
            # Remove duplicates and clean data
            cleaned_products = product_descriptions[[product_id_column, lob_column]].drop_duplicates()
            cleaned_products[product_id_column] = cleaned_products[product_id_column].str.upper()
            cleaned_products[lob_column] = cleaned_products[lob_column].str.upper()
            
            # Create reference mapping from DataFrame columns
            reference_products = pd.DataFrame(
                reference_dataframe.columns.str.upper(), 
                columns=[product_id_column]
            )
            
            # Merge to keep only matching products
            result = reference_products.merge(
                cleaned_products, 
                on=product_id_column, 
                how='left'
            )
            
            # Validate result dimensions
            expected_rows = len(reference_dataframe.columns)
            actual_rows = len(result)
            
            if actual_rows != expected_rows:
                logger.warning(
                    f"Shape mismatch after cleaning: expected {expected_rows} rows, "
                    f"got {actual_rows} rows"
                )
                logger.debug(f"Reference columns: {list(reference_dataframe.columns)}")
                logger.debug(f"Product IDs: {list(cleaned_products[product_id_column])}")
            
            logger.info(f"Product description cleaning completed: {len(result)} products")
            return result
            
        except Exception as e:
            logger.error(f"Error cleaning product descriptions: {str(e)}")
            raise
    
    @staticmethod
    def condense_dataframe_columns(dataframe: pd.DataFrame,
                                 remove_column_prefix: bool = True,
                                 column_mapping: Dict[str, str] = None) -> pd.Series:
        """
        Condense DataFrame columns into a single string representation per row.
        
        This method combines multiple columns into a single text representation,
        useful for creating human-readable summaries of sparse or indicator matrices.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame to condense
        remove_column_prefix : bool, default=True
            Whether to remove column prefixes (format: prefix_remainder)
        column_mapping : dict, optional
            Dictionary to rename columns before condensing
            
        Returns
        -------
        pd.Series
            Series with condensed string representation for each row
            
        Examples
        --------
        >>> df = pd.DataFrame({
        ...     'cat_product1': [1, 0],
        ...     'cat_product2': [0, 1]
        ... })
        >>> condensed = ProductUtilities.condense_dataframe_columns(df)
        >>> # Returns: ['product1', 'product2']
        """
        if dataframe.empty:
            return pd.Series(dtype=str)
        
        try:
            working_df = dataframe.copy()
            
            # Remove column prefixes if requested
            if remove_column_prefix:
                new_columns = []
                for col in working_df.columns:
                    if '_' in col:
                        # Keep everything after the first underscore
                        new_col = '_'.join(col.split('_')[1:])
                        new_columns.append(new_col if new_col else col)
                    else:
                        new_columns.append(col)
                working_df.columns = new_columns
            
            # Apply column mapping if provided
            if column_mapping:
                working_df.rename(columns=column_mapping, inplace=True)
            
            # Fill with column names for non-zero values
            named_df = EncodingUtilities.fill_dataframe_with_column_names(working_df)
            
            # Join non-zero values for each row
            condensed_series = named_df.apply(
                ProductUtilities.join_non_zero_values, 
                separator=', ',
                axis=1
            )
            
            logger.info(f"Condensed {len(dataframe.columns)} columns into text representation")
            return condensed_series
            
        except Exception as e:
            logger.error(f"Error condensing DataFrame columns: {str(e)}")
            raise
    
    @staticmethod
    def extract_current_products(customer_data: pd.DataFrame, 
                               line_of_business: str) -> pd.DataFrame:
        """
        Extract current products for a specific line of business.
        
        This method filters and processes customer data to extract product
        information for specific business lines (e.g., Internet, TV services).
        Includes business-specific logic for telecommunications data.
        
        Parameters
        ----------
        customer_data : pd.DataFrame
            Customer data containing product indicators
        line_of_business : str
            Line of business to extract ('INTERNET' or 'VIDEO')
            
        Returns
        -------
        pd.DataFrame
            DataFrame with current product information for the specified LOB
            
        Raises
        ------
        ValueError
            If line_of_business is not supported
            
        Examples
        --------
        >>> data = pd.DataFrame({
        ...     'B4_INT10': [1, 0],
        ...     'B4_INT20': [0, 1],
        ...     'B4_BASICTV': [1, 1]
        ... })
        >>> internet_products = ProductUtilities.extract_current_products(data, 'INTERNET')
        """
        if line_of_business not in ['INTERNET', 'VIDEO']:
            raise ValueError("line_of_business must be 'INTERNET' or 'VIDEO'")
        
        if customer_data.empty:
            return pd.DataFrame()
        
        try:
            logger.info(f"Extracting {line_of_business} products from {len(customer_data)} customers")
            
            # Define search patterns based on line of business
            if line_of_business == 'INTERNET':
                search_patterns = [r'B4_INT[0-9]']
            else:  # VIDEO
                search_patterns = [r'B4_.*TV$']
            
            # Find matching columns using regex
            product_columns, _ = ListUtilities.search_with_regex(
                search_patterns, 
                customer_data.columns.tolist()
            )
            
            if not product_columns:
                logger.warning(f"No {line_of_business} product columns found")
                return pd.DataFrame()
            
            # Extract and process product data
            product_data = customer_data[product_columns].copy()
            product_data.fillna(0, inplace=True)
            
            # Remove prefixes from column names
            cleaned_columns = [
                '_'.join(col.split('_')[1:]) if '_' in col else col 
                for col in product_data.columns
            ]
            product_data.columns = cleaned_columns
            
            # Create condensed product representation
            condensed_products = ProductUtilities.condense_dataframe_columns(
                product_data, remove_column_prefix=False
            )
            
            # Handle line-of-business specific logic
            if line_of_business == 'VIDEO':
                # Add additional TV-related columns if available
                additional_columns = []
                if 'B4_BLUECURVE' in customer_data.columns:
                    additional_columns.append(
                        np.sign(customer_data[['B4_BLUECURVE']]).astype(int)
                    )
                if 'B4_TVADDONS' in customer_data.columns:
                    additional_columns.append(
                        np.sign(customer_data[['B4_TVADDONS']]).astype(int)
                    )
                
                if additional_columns:
                    result_df = pd.concat(
                        [condensed_products] + additional_columns, 
                        axis=1
                    )
                    result_df.columns = [
                        'TV_CURRENT_PRODUCT', 'BLUECURVETV', 'TVADDONS'
                    ][:len(result_df.columns)]
                else:
                    result_df = pd.DataFrame({'TV_CURRENT_PRODUCT': condensed_products})
            else:
                result_df = pd.DataFrame({'INTERNET_CURRENT_PRODUCT': condensed_products})
            
            logger.info(f"Extracted {len(result_df.columns)} {line_of_business} product features")
            return result_df
            
        except Exception as e:
            logger.error(f"Error extracting {line_of_business} products: {str(e)}")
            raise
    
    @staticmethod
    def sort_prediction_probabilities(probabilities: pd.Series, 
                                    class_labels: pd.Series) -> List[Any]:
        """
        Sort prediction probabilities and corresponding class labels in descending order.
        
        This method sorts classifier prediction probabilities along with their
        corresponding class labels, useful for ranking predictions and creating
        top-N recommendations.
        
        Parameters
        ----------
        probabilities : pd.Series
            Prediction probabilities from classifier
        class_labels : pd.Series
            Corresponding class labels for probabilities
            
        Returns
        -------
        list
            Combined list of sorted labels and probabilities
            
        Examples
        --------
        >>> probs = pd.Series([0.1, 0.7, 0.2])
        >>> labels = pd.Series(['class_a', 'class_b', 'class_c'])
        >>> sorted_results = ProductUtilities.sort_prediction_probabilities(probs, labels)
        """
        try:
            prob_values = probabilities.values
            label_values = class_labels.values
            
            # Sort by probabilities in descending order
            sort_indices = np.argsort(-prob_values)
            sorted_probabilities = prob_values[sort_indices]
            sorted_labels = label_values[sort_indices]
            
            # Set zero probability labels to empty string
            zero_prob_mask = sorted_probabilities == 0
            sorted_labels[zero_prob_mask] = ''
            
            # Return combined list of labels and probabilities
            return sorted_labels.tolist() + sorted_probabilities.tolist()
            
        except Exception as e:
            logger.error(f"Error sorting prediction probabilities: {str(e)}")
            return []


# =============================================================================
# DATAFRAME MANIPULATION UTILITIES
# =============================================================================

class DataFrameUtilities:
    """
    Comprehensive DataFrame manipulation utilities for data science workflows.
    
    This class provides methods for DataFrame column operations, merging,
    memory optimization, and structural transformations commonly needed
    in data analysis projects.
    """
    
    @staticmethod
    def reorder_columns(dataframe: pd.DataFrame, 
                       columns_to_move: List[str], 
                       reference_column: str, 
                       placement: str = 'After') -> pd.DataFrame:
        """
        Reorder DataFrame columns by moving specified columns relative to a reference.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame to reorder
        columns_to_move : list of str
            Column names to relocate
        reference_column : str
            Reference column for positioning
        placement : {'After', 'Before'}, default='After'
            Where to place columns_to_move relative to reference_column
            
        Returns
        -------
        pd.DataFrame
            DataFrame with reordered columns
            
        Raises
        ------
        KeyError
            If reference_column or any column in columns_to_move not found
        ValueError
            If placement is not 'After' or 'Before'
            
        Examples
        --------
        >>> df = pd.DataFrame({'A': [1], 'B': [2], 'C': [3], 'D': [4]})
        >>> result = DataFrameUtilities.reorder_columns(df, ['D'], 'A', 'After')
        >>> list(result.columns)
        ['A', 'D', 'B', 'C']
        """
        if placement not in ['After', 'Before']:
            raise ValueError("placement must be 'After' or 'Before'")
        
        # Validate columns exist
        missing_cols = [col for col in columns_to_move if col not in dataframe.columns]
        if missing_cols:
            raise KeyError(f"Columns not found: {missing_cols}")
        
        if reference_column not in dataframe.columns:
            raise KeyError(f"Reference column not found: {reference_column}")
        
        column_list = dataframe.columns.tolist()
        ref_index = column_list.index(reference_column)
        
        if placement == 'After':
            segment_1 = column_list[:ref_index + 1]
            segment_2 = columns_to_move
        else:  # Before
            segment_1 = column_list[:ref_index]
            segment_2 = columns_to_move + [reference_column]
        
        # Remove moved columns from segment_1 to avoid duplication
        segment_1 = [col for col in segment_1 if col not in segment_2]
        
        # Remaining columns
        segment_3 = [col for col in column_list if col not in segment_1 + segment_2]
        
        new_column_order = segment_1 + segment_2 + segment_3
        return dataframe[new_column_order]
    
    @staticmethod
    def calculate_cell_proportions(dataframe: pd.DataFrame, axis: int = 0) -> pd.DataFrame:
        """
        Calculate cell values as proportions of row or column totals.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame with numeric values
        axis : {0, 1}, default=0
            - 0: Calculate proportions of column totals
            - 1: Calculate proportions of row totals
            
        Returns
        -------
        pd.DataFrame
            DataFrame with values converted to proportions (0-1 scale)
            
        Examples
        --------
        >>> df = pd.DataFrame({'A': [10, 20], 'B': [30, 40]})
        >>> proportions = DataFrameUtilities.calculate_cell_proportions(df, axis=0)
        """
        if axis not in [0, 1]:
            raise ValueError("axis must be 0 (columns) or 1 (rows)")
        
        if axis == 0:
            return dataframe.div(dataframe.sum(axis=0), axis=1)
        else:
            return dataframe.div(dataframe.sum(axis=1), axis=0)
    
    @staticmethod
    def reduce_memory_usage(dataframe: pd.DataFrame,
                           object_to_string_columns: str = 'all_columns',
                           string_to_category_columns: str = 'all_columns',
                           verbose: bool = True) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage through intelligent type conversion.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame to optimize
        object_to_string_columns : {'all_columns', 'none'} or list, default='all_columns'
            Which object columns to convert to string type
        string_to_category_columns : {'all_columns', 'none'} or list, default='all_columns'
            Which string columns to convert to categorical type
        verbose : bool, default=True
            Whether to print optimization progress and results
            
        Returns
        -------
        pd.DataFrame
            Memory-optimized DataFrame
        """
        start_mem = dataframe.memory_usage(deep=True).sum() / 1024**2
        
        if verbose:
            print(f"Memory usage before optimization: {start_mem:.2f} MB")
        
        df_optimized = dataframe.copy()
        
        # Optimize numeric columns
        for column in df_optimized.columns:
            col_type = df_optimized[column].dtype
            
            if pd.api.types.is_numeric_dtype(col_type):
                column_min = df_optimized[column].min()
                column_max = df_optimized[column].max()
                
                if pd.api.types.is_integer_dtype(col_type):
                    # Integer optimization
                    if column_min > np.iinfo(np.int8).min and column_max < np.iinfo(np.int8).max:
                        df_optimized[column] = df_optimized[column].astype(np.int8)
                    elif column_min > np.iinfo(np.int16).min and column_max < np.iinfo(np.int16).max:
                        df_optimized[column] = df_optimized[column].astype(np.int16)
                    elif column_min > np.iinfo(np.int32).min and column_max < np.iinfo(np.int32).max:
                        df_optimized[column] = df_optimized[column].astype(np.int32)
                
                elif pd.api.types.is_float_dtype(col_type):
                    # Float optimization
                    if column_min > np.finfo(np.float32).min and column_max < np.finfo(np.float32).max:
                        df_optimized[column] = df_optimized[column].astype(np.float32)
        
        end_mem = df_optimized.memory_usage(deep=True).sum() / 1024**2
        
        if verbose:
            print(f"Memory usage after optimization: {end_mem:.2f} MB")
            print(f"Memory reduction: {(start_mem - end_mem) / start_mem:.1%}")
        
        return df_optimized
    
    @staticmethod
    def analyze_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze missing values across DataFrame columns.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame to analyze
            
        Returns
        -------
        pd.DataFrame
            Summary DataFrame with missing value statistics
        """
        total_rows = len(dataframe)
        
        missing_data = []
        for column in dataframe.columns:
            missing_count = dataframe[column].isnull().sum()
            missing_percentage = (missing_count / total_rows) * 100
            
            missing_data.append({
                'Column': column,
                'Data_Type': str(dataframe[column].dtype),
                'Missing_Count': missing_count,
                'Missing_Percentage': missing_percentage,
                'Non_Missing_Count': total_rows - missing_count
            })
        
        result_df = pd.DataFrame(missing_data)
        return result_df.sort_values('Missing_Percentage', ascending=False)
    
    @staticmethod
    def unify_columns(df1: pd.DataFrame, df2: pd.DataFrame, 
                     df1_name: str, df2_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Unify columns between two DataFrames for comparison and analysis.
        
        This method finds common columns between two DataFrames and returns
        new DataFrames containing only the shared columns. Useful for preparing
        DataFrames for concatenation, comparison, or merging operations.
        
        Parameters
        ----------
        df1 : pd.DataFrame
            First DataFrame to unify
        df2 : pd.DataFrame
            Second DataFrame to unify  
        df1_name : str
            Descriptive name for first DataFrame (used in warnings)
        df2_name : str
            Descriptive name for second DataFrame (used in warnings)
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            Tuple containing (df1_unified, df2_unified) with only common columns
            
        Warnings
        --------
        Warns if no common columns are found between the DataFrames
        
        Examples
        --------
        >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        >>> df2 = pd.DataFrame({'A': [7, 8], 'B': [9, 10], 'D': [11, 12]})
        >>> unified1, unified2 = DataFrameUtilities.unify_columns(df1, df2, 'data1', 'data2')
        >>> list(unified1.columns)
        ['A', 'B']
        """
        # Get common columns between DataFrames
        common_columns = list(set(df1.columns).intersection(set(df2.columns)))
        
        if not common_columns:
            warnings.warn(
                f"No common columns found between {df1_name} and {df2_name}. "
                f"{df1_name} columns: {list(df1.columns)}, "
                f"{df2_name} columns: {list(df2.columns)}",
                UserWarning,
                stacklevel=2
            )
            return df1.copy(), df2.copy()
        
        # Sort columns for consistent ordering
        common_columns.sort()
        
        # Return DataFrames with only common columns
        df1_unified = df1[common_columns].copy()
        df2_unified = df2[common_columns].copy()
        
        return df1_unified, df2_unified


# =============================================================================
# DATE AND TIME UTILITIES
# =============================================================================

class DateTimeUtilities:
    """
    Date and time manipulation utilities for data science workflows.
    
    This class provides methods for date validation, time conversion,
    date range generation, and time-based calculations commonly needed
    in temporal data analysis.
    """
    
    @staticmethod
    def validate_timestamp_format(start_date: str, 
                                end_date: str, 
                                date_format: str = '%Y-%m-%d') -> bool:
        """
        Validate timestamp string format.
        
        Parameters
        ----------
        start_date : str
            Start date string to validate
        end_date : str
            End date string to validate
        date_format : str, default='%Y-%m-%d'
            Expected date format string
            
        Returns
        -------
        bool
            True if both dates match format, False otherwise
            
        Examples
        --------
        >>> DateTimeUtilities.validate_timestamp_format('2023-01-01', '2023-12-31')
        True
        """
        try:
            start_parsed = time.strptime(start_date, date_format)
            end_parsed = time.strptime(end_date, date_format)
            return (start_parsed.__class__.__name__ == 'struct_time' and 
                   end_parsed.__class__.__name__ == 'struct_time')
        except ValueError as e:
            logging.warning(f"Date format validation failed: {e}")
            return False
    
    @staticmethod
    def convert_seconds_to_readable(total_seconds: Union[int, float]) -> Tuple[int, int, int, float]:
        """
        Convert seconds to readable time components.
        
        Parameters
        ----------
        total_seconds : int or float
            Total seconds to convert
            
        Returns
        -------
        tuple of (int, int, int, float)
            (days, hours, minutes, seconds) components
            
        Examples
        --------
        >>> DateTimeUtilities.convert_seconds_to_readable(3661)
        (0, 1, 1, 1.0)
        """
        remaining_time = float(total_seconds)
        
        days = int(remaining_time // (24 * 3600))
        remaining_time = remaining_time % (24 * 3600)
        
        hours = int(remaining_time // 3600)
        remaining_time = remaining_time % 3600
        
        minutes = int(remaining_time // 60)
        seconds = remaining_time % 60
        
        return days, hours, minutes, seconds
    
    @staticmethod
    def generate_date_range(start_year: int = 2018,
                          end_year: int = None,
                          first_date: Optional[Union[str, dt.date]] = None,
                          last_date: Optional[Union[str, dt.date]] = None,
                          month_step: int = 1) -> List[dt.date]:
        """
        Generate list of dates within specified range.
        
        Parameters
        ----------
        start_year : int, default=2018
            Starting year for date range
        end_year : int, optional
            Ending year for date range (exclusive)
        first_date : str or datetime.date, optional
            Custom first date
        last_date : str or datetime.date, optional  
            Custom last date
        month_step : int, default=1
            Step size for months
            
        Returns
        -------
        list of datetime.date
            List of generated dates
        """
        if month_step <= 0:
            raise ValueError("month_step must be positive")
        
        if end_year is None:
            end_year = dt.datetime.now().year + 1
        
        if last_date is None:
            last_date = dt.datetime.now().date()
        elif isinstance(last_date, str):
            last_date = dt.datetime.strptime(last_date, '%Y-%m-%d').date()
        
        # Generate year-month combinations
        years = list(range(start_year, end_year))
        months = list(range(1, 13, month_step))
        
        year_month_combinations = [(year, month) for year in years for month in months]
        
        # Convert to dates and filter by date range
        date_list = []
        for year, month in year_month_combinations:
            try:
                current_date = dt.date(year, month, 1)
                
                # Apply first_date filter
                if first_date is not None:
                    if isinstance(first_date, str):
                        first_date_obj = dt.datetime.strptime(first_date, '%Y-%m-%d').date()
                    else:
                        first_date_obj = first_date
                    
                    if current_date < first_date_obj:
                        continue
                
                # Apply last_date filter
                if current_date > last_date:
                    continue
                
                date_list.append(current_date)
                
            except ValueError:
                continue
        
        return date_list


# =============================================================================
# FILE SYSTEM UTILITIES  
# =============================================================================

class FileSystemUtilities:
    """
    File system operations and path utilities for data science workflows.
    
    This class provides methods for path validation, directory creation,
    file operations, and output folder management commonly needed in
    data analysis projects.
    """
    
    @staticmethod
    def validate_path_exists(file_path: str) -> str:
        """
        Validate that a file path exists and return expanded path.
        
        Parameters
        ----------
        file_path : str
            Path to validate
            
        Returns
        -------
        str
            Expanded and validated file path
            
        Raises
        ------
        FileNotFoundError
            If the path does not exist
            
        Examples
        --------
        >>> path = FileSystemUtilities.validate_path_exists('~/data/file.csv')
        """
        if '~' in file_path:
            file_path = os.path.expanduser(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File or directory not found: {file_path}")
        
        return file_path
    
    @staticmethod
    def setup_output_directory(output_directory: str,
                             template_files: List[str] = None,
                             allow_overwrite: bool = False) -> str:
        """
        Setup output directory with optional template file copying.
        
        Parameters
        ----------
        output_directory : str
            Target output directory path
        template_files : list of str, optional
            List of template file paths to copy to output directory
        allow_overwrite : bool, default=False
            Whether to allow overwriting existing directory contents
            
        Returns
        -------
        str
            Absolute path to created output directory
            
        Raises
        ------
        FileExistsError
            If directory exists and allow_overwrite is False
        """
        # Normalize path
        if len(output_directory.split('/')) == 1:
            output_directory = os.path.abspath(os.path.join(os.getcwd(), output_directory))
        else:
            output_directory = os.path.abspath(output_directory)
        
        # Check directory existence and overwrite policy
        if os.path.exists(output_directory) and not allow_overwrite:
            raise FileExistsError(
                f"Output directory exists and overwrite not allowed: {output_directory}"
            )
        
        # Create directory if it doesn't exist
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)
        
        # Copy template files if provided and overwrite is allowed
        if template_files and allow_overwrite:
            import shutil
            
            for template_file in template_files:
                if not os.path.exists(template_file):
                    raise FileNotFoundError(f"Template file not found: {template_file}")
                
                destination_path = os.path.join(
                    output_directory,
                    os.path.basename(template_file)
                )
                shutil.copyfile(template_file, destination_path)
                logging.info(f"Copied template file: {template_file} -> {destination_path}")
        
        return output_directory
    
    @staticmethod
    def extract_zip_archive(download_directory: str, zip_filename: str,
                           extract_folders: Optional[Tuple[str, ...]] = None,
                           exclude_folders: Optional[Tuple[str, ...]] = None) -> bool:
        """
        Extract zip archive with filtering options.
        
        Provides comprehensive zip extraction capabilities with optional filtering
        for specific folders/files and exclusion patterns. Includes progress
        tracking and error handling for robust file operations.
        
        Parameters
        ----------
        download_directory : str
            Directory containing the zip file and target for extraction
        zip_filename : str
            Name of the zip file to extract
        extract_folders : tuple of str, optional
            Specific folders/files to extract. If None, extracts all files.
        exclude_folders : tuple of str, optional
            Folders/files to exclude from extraction
            
        Returns
        -------
        bool
            True if extraction successful, False otherwise
            
        Raises
        ------
        FileNotFoundError
            If the zip file does not exist
            
        Examples
        --------
        >>> FileSystemUtilities.extract_zip_archive(
        ...     '/data', 'dataset.zip', 
        ...     extract_folders=('train/', 'test/'),
        ...     exclude_folders=('sample/',)
        ... )
        True
        """
        try:
            zip_path = os.path.join(download_directory, zip_filename)
            
            if not os.path.exists(zip_path):
                raise FileNotFoundError(f"Zip file not found: {zip_path}")
            
            print("Extracting files...")
            
            with zipfile.ZipFile(zip_path, 'r') as archive:
                all_files = archive.namelist()
                
                if not all_files:
                    warnings.warn("Zip file appears to be empty")
                    return False
                
                files_to_extract = FileSystemUtilities.filter_files_for_extraction(
                    all_files, extract_folders, exclude_folders
                )
                
                if not files_to_extract:
                    warnings.warn("No files selected for extraction after filtering")
                    return False
                
                print(f"Extracting {len(files_to_extract)} files...")
                
                try:
                    from tqdm import tqdm
                    file_iterator = tqdm(files_to_extract, desc="Extracting files")
                except ImportError:
                    file_iterator = files_to_extract
                
                for file_name in file_iterator:
                    try:
                        archive.extract(file_name, download_directory)
                    except Exception as e:
                        warnings.warn(f"Failed to extract {file_name}: {str(e)}")
                        continue
            
            print(f"Successfully extracted files to: {download_directory}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to extract zip archive: {str(e)}")
            return False
    
    @staticmethod
    def filter_files_for_extraction(all_files: List[str],
                                   extract_folders: Optional[Tuple[str, ...]] = None,
                                   exclude_folders: Optional[Tuple[str, ...]] = None) -> List[str]:
        """
        Filter files based on extraction and exclusion criteria.
        
        This method applies inclusion and exclusion filters to a list of files,
        commonly used for selective zip archive extraction or file processing.
        
        Parameters
        ----------
        all_files : list of str
            List of all files to filter
        extract_folders : tuple of str, optional
            Specific folders/files to include. Files must start with these prefixes.
        exclude_folders : tuple of str, optional
            Folders/files to exclude. Files starting with these prefixes are excluded.
            
        Returns
        -------
        list of str
            Filtered list of files meeting the criteria
            
        Examples
        --------
        >>> files = ['train/data.csv', 'test/data.csv', 'sample/example.csv']
        >>> FileSystemUtilities.filter_files_for_extraction(
        ...     files, 
        ...     extract_folders=('train/', 'test/'),
        ...     exclude_folders=('sample/',)
        ... )
        ['train/data.csv', 'test/data.csv']
        """
        files_to_extract = all_files.copy()
        
        # Apply inclusion filter
        if extract_folders:
            files_to_extract = [
                file for file in files_to_extract
                if any(file.startswith(folder) for folder in extract_folders)
            ]
            print(f"Filtered to include only: {extract_folders}")
        
        # Apply exclusion filter  
        if exclude_folders:
            files_to_extract = [
                file for file in files_to_extract
                if not any(file.startswith(folder) for folder in exclude_folders)
            ]
            print(f"Excluded: {exclude_folders}")
        
        return files_to_extract


# =============================================================================
# DATA VISUALIZATION UTILITIES
# =============================================================================

class DataVisualization:
    """
    Comprehensive data visualization utilities for data science workflows.
    
    This class provides methods for creating advanced visualizations including
    Sankey diagrams, 3D scatter plots, and word clouds commonly used in
    exploratory data analysis and presentation of analytical results.
    """
    
    @staticmethod
    def create_sankey_flow_diagram(source_series: pd.Series,
                                 target_series: pd.Series,
                                 flow_values: pd.Series,
                                 minimum_flow_threshold: Union[int, float] = 0,
                                 chart_title: str = "Flow Diagram",
                                 output_filepath: str = "sankey_diagram.html",
                                 node_padding: int = 15,
                                 node_thickness: int = 20,
                                 font_size: int = 10) -> pd.DataFrame:
        """
        Create an interactive Sankey diagram to visualize data flows between categories.
        
        This method generates a Sankey diagram showing flows from source to target
        categories, with flow magnitudes represented by link thickness. Useful for
        visualizing customer journeys, resource flows, or categorical transitions.
        
        Parameters
        ----------
        source_series : pd.Series
            Source category labels for each flow transaction
        target_series : pd.Series  
            Target category labels for each flow transaction
        flow_values : pd.Series
            Numeric values representing flow magnitudes between categories
        minimum_flow_threshold : int or float, default=0
            Minimum flow value to include in visualization (filters small flows)
        chart_title : str, default="Flow Diagram"
            Title to display on the Sankey diagram
        output_filepath : str, default="sankey_diagram.html"
            File path where the interactive HTML plot will be saved
        node_padding : int, default=15
            Spacing between nodes in the diagram
        node_thickness : int, default=20
            Thickness of node rectangles in pixels
        font_size : int, default=10
            Font size for node labels and title
            
        Returns
        -------
        pd.DataFrame
            Aggregated flow data with columns: source, target, flow_value
            
        Raises
        ------
        ImportError
            If required Plotly library is not installed
        ValueError
            If input series have different lengths or contain invalid data
            
        Examples
        --------
        >>> # Customer journey analysis
        >>> sources = pd.Series(['Website', 'Email', 'Social', 'Website'])
        >>> targets = pd.Series(['Purchase', 'Newsletter', 'Purchase', 'Cart'])
        >>> values = pd.Series([100, 50, 75, 25])
        >>> flow_data = DataVisualization.create_sankey_flow_diagram(
        ...     sources, targets, values,
        ...     minimum_flow_threshold=30,
        ...     chart_title="Customer Journey Flow"
        ... )
        """
        # Input validation
        if len(source_series) != len(target_series) != len(flow_values):
            raise ValueError("All input series must have the same length")
        
        if not pd.api.types.is_numeric_dtype(flow_values):
            raise ValueError("flow_values must be numeric")
        
        try:
            import plotly.offline as plotly_offline
        except ImportError:
            raise ImportError(
                "Plotly library required for Sankey diagrams. "
                "Install with: pip install plotly"
            )
        
        # Combine and aggregate flow data
        flow_transactions_raw = pd.concat([
            source_series.rename('source'),
            target_series.rename('target'), 
            flow_values.rename('flow_value')
        ], axis=1)
        
        # Aggregate flows by source-target pairs
        aggregated_flows = flow_transactions_raw.groupby(
            ['source', 'target'], as_index=False
        ).agg({
            'flow_value': 'sum'
        })
        
        # Note: Flow counts could be used for additional analysis if needed
        
        # Filter flows above threshold
        filtered_flows = aggregated_flows[
            aggregated_flows['flow_value'] > minimum_flow_threshold
        ].copy()
        
        if filtered_flows.empty:
            warnings.warn("No flows above threshold found. Returning empty DataFrame.")
            return pd.DataFrame(columns=['source', 'target', 'flow_value'])
        
        # Sort by flow value for better visualization
        filtered_flows = filtered_flows.sort_values('flow_value', ascending=False)
        
        # Extract unique source and target labels
        unique_sources = sorted(filtered_flows['source'].unique())
        unique_targets = sorted(filtered_flows['target'].unique())
        all_labels = unique_sources + unique_targets
        
        # Create source and target indices for Plotly
        source_indices = []
        target_indices = []
        
        for _, row in filtered_flows.iterrows():
            source_idx = unique_sources.index(row['source'])
            target_idx = unique_targets.index(row['target']) + len(unique_sources)
            
            source_indices.append(source_idx)
            target_indices.append(target_idx)
        
        # Configure Sankey diagram
        sankey_data = {
            'type': 'sankey',
            'node': {
                'pad': node_padding,
                'thickness': node_thickness,
                'line': {
                    'color': "black",
                    'width': 0.5
                },
                'label': all_labels
            },
            'link': {
                'source': source_indices,
                'target': target_indices,
                'value': filtered_flows['flow_value'].tolist()
            }
        }
        
        layout = {
            'title': chart_title,
            'font': {'size': font_size}
        }
        
        # Generate and save interactive plot
        figure = {'data': [sankey_data], 'layout': layout}
        plotly_offline.plot(figure, filename=output_filepath)
        
        print(f"Sankey diagram saved to: {output_filepath}")
        print(f"Visualization includes {len(filtered_flows)} flow connections")
        
        return filtered_flows
    
    @staticmethod
    def create_3d_scatter_plot(data_array: np.ndarray,
                             color_labels: Union[pd.Series, np.ndarray],
                             axis_labels: List[str],
                             chart_title: str = "3D Data Visualization",
                             output_filepath: str = "3d_scatter_plot.png",
                             figure_size: Tuple[int, int] = (12, 9),
                             point_size: int = 50,
                             point_alpha: float = 0.7,
                             color_palette: List[str] = None,
                             dpi: int = 300) -> None:
        """
        Create a 3D scatter plot for multidimensional data exploration.
        
        This method generates publication-quality 3D scatter plots with color-coded
        categories, useful for visualizing clustering results, dimensionality reduction
        outputs (PCA, t-SNE), or any three-dimensional dataset analysis.
        
        Parameters
        ----------
        data_array : np.ndarray
            3D data array with shape (n_samples, 3) containing X, Y, Z coordinates
        color_labels : pd.Series or np.ndarray
            Category labels for color-coding points in the scatter plot
        axis_labels : list of str, length=3
            Labels for X, Y, and Z axes respectively
        chart_title : str, default="3D Data Visualization"
            Title to display on the plot
        output_filepath : str, default="3d_scatter_plot.png"
            File path where the plot image will be saved
        figure_size : tuple of int, default=(12, 9)
            Figure dimensions in inches (width, height)
        point_size : int, default=50
            Size of scatter plot points
        point_alpha : float, default=0.7
            Transparency level for points (0=transparent, 1=opaque)
        color_palette : list of str, optional
            Custom colors for categories. If None, uses default matplotlib colors
        dpi : int, default=300
            Resolution for saved image (dots per inch)
            
        Raises
        ------
        ImportError
            If required matplotlib or mpl_toolkits libraries are not available
        ValueError
            If data_array is not 3D or axis_labels length is incorrect
            
        Examples
        --------
        >>> # Visualize PCA results
        >>> pca_data = np.random.rand(100, 3)  # 100 samples, 3 components
        >>> clusters = np.random.choice(['A', 'B', 'C'], 100)
        >>> DataVisualization.create_3d_scatter_plot(
        ...     pca_data, clusters,
        ...     axis_labels=['PC1', 'PC2', 'PC3'],
        ...     chart_title='PCA Results by Cluster',
        ...     output_filepath='pca_visualization.png'
        ... )
        """
        # Input validation
        if not isinstance(data_array, np.ndarray):
            data_array = np.array(data_array)
        
        if data_array.shape[1] != 3:
            raise ValueError("data_array must have exactly 3 columns (X, Y, Z coordinates)")
        
        if len(axis_labels) != 3:
            raise ValueError("axis_labels must contain exactly 3 labels")
        
        if not 0 <= point_alpha <= 1:
            raise ValueError("point_alpha must be between 0 and 1")
        
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D  # Required for 3D projection
        except ImportError:
            raise ImportError(
                "Matplotlib and mpl_toolkits required for 3D plotting. "
                "Install with: pip install matplotlib"
            )
        
        # Setup figure and 3D axis
        fig = plt.figure(figsize=figure_size)
        ax = fig.add_subplot(111, projection='3d')
        
        # Configure axis labels and title
        ax.set_xlabel(axis_labels[0], fontsize=15)
        ax.set_ylabel(axis_labels[1], fontsize=15) 
        ax.set_zlabel(axis_labels[2], fontsize=15)
        ax.set_title(chart_title, fontsize=20)
        
        # Get unique categories for color coding
        unique_categories = np.unique(color_labels)
        
        # Use default color palette if none provided
        if color_palette is None:
            color_palette = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'brown', 'pink']
        
        # Extend color palette if needed
        while len(color_palette) < len(unique_categories):
            color_palette.extend(color_palette)
        
        # Plot each category with distinct color
        legend_labels = []
        for category, color in zip(unique_categories, color_palette):
            # Filter data for current category
            category_mask = (color_labels == category)
            category_data = data_array[category_mask]
            
            if len(category_data) > 0:
                ax.scatter(
                    category_data[:, 0], 
                    category_data[:, 1],
                    category_data[:, 2], 
                    c=color, 
                    s=point_size, 
                    alpha=point_alpha,
                    label=str(category)
                )
                legend_labels.append(str(category))
        
        # Add legend and grid
        if legend_labels:
            ax.legend(legend_labels, loc='best')
        ax.grid(True)
        
        # Save high-quality plot
        plt.tight_layout()
        plt.savefig(output_filepath, format='png', dpi=dpi, bbox_inches='tight')
        plt.close('all')
        
        print(f"3D scatter plot saved to: {output_filepath}")
        print(f"Visualized {len(data_array)} data points across {len(unique_categories)} categories")
    
    @staticmethod  
    def generate_word_cloud_visualization(text_data: Union[pd.Series, pd.DataFrame],
                                        output_filepath: str = "word_cloud.png",
                                        figure_size: Tuple[int, int] = (20, 10),
                                        canvas_width: int = 800,
                                        canvas_height: int = 400,
                                        background_color: str = 'white',
                                        max_words: int = 200,
                                        colormap: str = 'viridis',
                                        include_stopwords: bool = False,
                                        custom_stopwords: List[str] = None) -> None:
        """
        Generate word cloud visualization from text data for content analysis.
        
        This method creates publication-quality word clouds from either raw text
        or pre-computed word frequencies. Useful for exploratory text analysis,
        content summarization, and identifying key themes in textual datasets.
        
        Parameters
        ----------
        text_data : pd.Series or pd.DataFrame
            Text data for word cloud generation:
            - If Series: Raw text strings to be processed
            - If DataFrame: Two columns with words and their frequencies
        output_filepath : str, default="word_cloud.png"
            File path where the word cloud image will be saved
        figure_size : tuple of int, default=(20, 10)
            Figure dimensions in inches (width, height)
        canvas_width : int, default=800
            Word cloud canvas width in pixels
        canvas_height : int, default=400
            Word cloud canvas height in pixels
        background_color : str, default='white'
            Background color for the word cloud ('white', 'black', etc.)
        max_words : int, default=200
            Maximum number of words to include in the visualization
        colormap : str, default='viridis'
            Matplotlib colormap for word coloring
        include_stopwords : bool, default=False
            Whether to include common stopwords in the visualization
        custom_stopwords : list of str, optional
            Additional words to exclude from the word cloud
            
        Raises
        ------
        ImportError
            If required wordcloud or matplotlib libraries are not available
        ValueError
            If text_data format is invalid or contains no processable text
            
        Examples
        --------
        >>> # From raw text data
        >>> reviews = pd.Series([
        ...     "Great product quality amazing",
        ...     "Poor service bad experience", 
        ...     "Excellent quality great value"
        ... ])
        >>> DataVisualization.generate_word_cloud_visualization(
        ...     reviews,
        ...     output_filepath="product_reviews_wordcloud.png",
        ...     max_words=50
        ... )
        
        >>> # From frequency data
        >>> word_freq = pd.DataFrame({
        ...     'words': ['quality', 'great', 'service', 'product'],
        ...     'frequency': [10, 8, 5, 7]
        ... })
        >>> DataVisualization.generate_word_cloud_visualization(
        ...     word_freq,
        ...     output_filepath="frequency_wordcloud.png"
        ... )
        """
        # Input validation
        if text_data is None or (hasattr(text_data, '__len__') and len(text_data) == 0):
            raise ValueError("text_data cannot be None or empty")
        
        try:
            from wordcloud import WordCloud, STOPWORDS
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "WordCloud and matplotlib libraries required. "
                "Install with: pip install wordcloud matplotlib"
            )
        
        # Configure stopwords
        stopwords = set(STOPWORDS) if not include_stopwords else set()
        if custom_stopwords:
            stopwords.update(custom_stopwords)
        
        # Process input data based on type
        if isinstance(text_data, pd.Series):
            # Raw text processing
            if text_data.empty:
                raise ValueError("Text series is empty")
            
            # Combine all text entries
            combined_text = ' '.join(text_data.astype(str).dropna())
            
            if not combined_text.strip():
                raise ValueError("No valid text content found in series")
            
            # Generate word cloud from text
            word_cloud_generator = WordCloud(
                width=canvas_width,
                height=canvas_height,
                background_color=background_color,
                max_words=max_words,
                stopwords=stopwords,
                colormap=colormap
            ).generate(combined_text)
            
        elif isinstance(text_data, pd.DataFrame):
            # Frequency data processing
            if text_data.shape[1] < 2:
                raise ValueError("DataFrame must have at least 2 columns (words and frequencies)")
            
            if text_data.empty:
                raise ValueError("DataFrame is empty")
            
            # Use first two columns as words and frequencies
            word_column = text_data.iloc[:, 0]
            frequency_column = text_data.iloc[:, 1]
            
            # Validate frequency data
            if not pd.api.types.is_numeric_dtype(frequency_column):
                raise ValueError("Second column must contain numeric frequency values")
            
            # Create frequency dictionary
            frequency_dict = dict(zip(
                word_column.astype(str), 
                frequency_column.astype(float)
            ))
            
            # Filter out stopwords from frequency dictionary
            if stopwords:
                frequency_dict = {
                    word: freq for word, freq in frequency_dict.items()
                    if word.lower() not in stopwords
                }
            
            if not frequency_dict:
                raise ValueError("No valid words found after stopword filtering")
            
            # Generate word cloud from frequencies
            word_cloud_generator = WordCloud(
                width=canvas_width,
                height=canvas_height,
                background_color=background_color,
                max_words=max_words,
                colormap=colormap
            ).generate_from_frequencies(frequency_dict)
            
        else:
            raise ValueError("text_data must be pandas Series or DataFrame")
        
        # Create and save visualization
        plt.figure(figsize=figure_size, facecolor=background_color)
        plt.imshow(word_cloud_generator, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout(pad=0)
        
        # Save high-quality image
        plt.savefig(output_filepath, bbox_inches='tight', dpi=300, facecolor=background_color)
        plt.close('all')
        
        print(f"Word cloud visualization saved to: {output_filepath}")
        
        # Provide summary statistics
        if isinstance(text_data, pd.Series):
            word_count = len(combined_text.split())
            print(f"Generated from {len(text_data)} text entries with {word_count} total words")
        else:
            total_frequency = frequency_column.sum()
            print(f"Generated from {len(text_data)} unique words with total frequency: {total_frequency}")

    # Import visualization dependencies at class level
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.graph_objects as go
        import plotly.express as px
        import plotly
        from plotly.subplots import make_subplots
        from statsmodels.tsa.stattools import ccf
        from matplotlib.ticker import MaxNLocator
        import scipy.stats
        import math
        _plotting_available = True
    except ImportError as e:
        _plotting_available = False
        _missing_deps = str(e)

    @staticmethod
    def create_correlation_heatmap(dataframe: pd.DataFrame, 
                                 correlation_method: str = 'kendall',
                                 show_diagonal: bool = True,
                                 config: Optional['PlotConfig'] = None,
                                 **kwargs) -> Tuple[pd.DataFrame, Any]:
        """
        Create a comprehensive correlation heatmap matrix visualization.
        
        This method generates a correlation matrix heatmap with customizable parameters
        for correlation method, masking options, and visual styling. Supports integration
        with seaborn heatmap parameters and provides both correlation matrix and figure.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame with samples as rows and features as columns
        correlation_method : str, default='kendall'
            Correlation method: 'pearson', 'kendall', 'spearman', or callable
            - pearson: Standard correlation coefficient
            - kendall: Kendall Tau correlation coefficient  
            - spearman: Spearman rank correlation
            - callable: Custom function with two 1d arrays returning float
        show_diagonal : bool, default=True
            Whether to show upper triangle (True) or full matrix (False)
        config : PlotConfig, optional
            Configuration object for plot customization
        **kwargs : dict
            Additional arguments passed to pandas.corr() and seaborn.heatmap()
            
        Returns
        -------
        Tuple[pd.DataFrame, matplotlib.figure.Figure]
            - Correlation matrix as DataFrame
            - Matplotlib figure object
            
        Raises
        ------
        ImportError
            If required plotting libraries are not available
        ValueError
            If dataframe is empty or correlation_method is invalid
            
        Examples
        --------
        >>> viz = DataVisualization()
        >>> df = pd.DataFrame(np.random.randn(100, 5), columns=['A','B','C','D','E'])
        >>> corr_matrix, fig = viz.create_correlation_heatmap(df, method='pearson')
        >>> plt.show()
        """
        if not DataVisualization._plotting_available:
            raise ImportError(f"Required plotting libraries not available: {DataVisualization._missing_deps}")
        
        if dataframe.empty:
            raise ValueError("Input dataframe cannot be empty")
        
        if config is None:
            from dataclasses import dataclass, field
            from typing import List, Literal
            
            @dataclass
            class PlotConfig:
                height: int = 800
                width: int = 1200
                colors: List[str] = field(default_factory=lambda: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
                theme: str = 'white'
                title_font_size: int = 24
                axis_font_size: int = 14
                bins: int = 20
                boxpoints: str = 'outliers'
                kde_points: int = 100
                jitter: float = 0.3
                violin_points: bool = True
                first_plot: str = 'box'
            
            config = PlotConfig()
        
        import inspect
        
        # Separate correlation and heatmap arguments
        corr_args = list(inspect.signature(pd.DataFrame.corr).parameters)
        kwargs_corr = {k: kwargs.pop(k) for k in dict(kwargs) if k in corr_args}
        
        heatmap_args = list(inspect.signature(DataVisualization.sns.heatmap).parameters)
        kwargs_heatmap = {k: kwargs.pop(k) for k in dict(kwargs) if k in heatmap_args}
        
        # Calculate correlation matrix
        df_clean = dataframe.dropna(how='any', axis=0).drop_duplicates()
        correlation_matrix = df_clean.corr(method=correlation_method, **kwargs_corr)
        
        # Generate mask for upper triangle if requested
        if show_diagonal:
            mask = np.zeros_like(correlation_matrix)
            mask[np.triu_indices_from(mask)] = True
        else:
            mask = None
        
        # Create figure with configured size
        DataVisualization.plt.figure(figsize=(config.width/100, config.height/100))
        
        # Generate custom diverging colormap
        cmap = DataVisualization.sns.diverging_palette(220, 10, as_cmap=True)
        
        # Create heatmap
        sns_plot = DataVisualization.sns.heatmap(
            correlation_matrix,
            mask=mask,
            cmap=cmap,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .5},
            fmt=".2f",
            annot=True,
            **kwargs_heatmap
        )
        
        # Get figure and configure
        figure = sns_plot.get_figure()
        DataVisualization.plt.title("Correlation Heatmap Matrix", fontsize=config.title_font_size)
        DataVisualization.plt.tight_layout()
        DataVisualization.plt.show()
        DataVisualization.plt.close()
        
        return correlation_matrix, figure

    @staticmethod
    def export_figures_to_html(figures: List[Any], output_filename: str = "dashboard.html") -> None:
        """
        Export multiple Plotly figures to a single HTML dashboard file.
        
        This method combines multiple Plotly figures into a single HTML file for easy sharing
        and presentation. Useful for creating comprehensive analysis dashboards with multiple
        visualizations in one document.
        
        Parameters
        ----------
        figures : List[plotly.graph_objects.Figure]
            List of Plotly figure objects to combine
        output_filename : str, default="dashboard.html"
            Path and filename for the output HTML file
            
        Raises
        ------
        ImportError
            If Plotly is not available
        ValueError
            If figures list is empty
            
        Examples
        --------
        >>> viz = DataVisualization()
        >>> fig1 = px.scatter(df, x='x', y='y')
        >>> fig2 = px.histogram(df, x='value')
        >>> viz.export_figures_to_html([fig1, fig2], "analysis_dashboard.html")
        """
        if not DataVisualization._plotting_available:
            raise ImportError(f"Required plotting libraries not available: {DataVisualization._missing_deps}")
        
        if not figures:
            raise ValueError("figures list cannot be empty")
        
        with open(output_filename, 'w', encoding='utf-8') as dashboard:
            dashboard.write("<html><head><title>Analysis Dashboard</title></head><body>\n")
            
            for i, fig in enumerate(figures):
                try:
                    inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
                    dashboard.write(f"<!-- Figure {i+1} -->\n")
                    dashboard.write(inner_html)
                    dashboard.write("\n")
                except Exception as e:
                    dashboard.write(f"<p>Error rendering figure {i+1}: {str(e)}</p>\n")
            
            dashboard.write("</body></html>\n")

    @staticmethod
    def save_plotly_figure_multiple_formats(figure: Any, 
                                          filename_prefix: str,
                                          image_format: str = "jpg",
                                          include_html: bool = True,
                                          include_json: bool = True) -> List[str]:
        """
        Save a Plotly figure in multiple formats (HTML, JSON, and image).
        
        This method exports a single Plotly figure to multiple file formats for different
        use cases: HTML for interactive sharing, JSON for programmatic access, and
        image formats for presentations and publications.
        
        Parameters
        ----------
        figure : plotly.graph_objects.Figure
            Plotly figure object to save
        filename_prefix : str
            Base filename without extension
        image_format : str, default="jpg"
            Image format for static export ('jpg', 'png', 'pdf', 'svg')
        include_html : bool, default=True
            Whether to save interactive HTML version
        include_json : bool, default=True
            Whether to save JSON version
            
        Returns
        -------
        List[str]
            List of created file paths
            
        Raises
        ------
        ImportError
            If required libraries are not available
        ValueError
            If invalid parameters are provided
            
        Examples
        --------
        >>> viz = DataVisualization()
        >>> fig = px.scatter(df, x='x', y='y', title='My Plot')
        >>> files = viz.save_plotly_figure_multiple_formats(fig, 'scatter_plot', 'png')
        >>> print(f"Created files: {files}")
        """
        if not DataVisualization._plotting_available:
            raise ImportError(f"Required plotting libraries not available: {DataVisualization._missing_deps}")
        
        if not filename_prefix:
            raise ValueError("filename_prefix cannot be empty")
        
        created_files = []
        
        # Determine which files to create
        extensions = []
        if include_html:
            extensions.append("html")
        if include_json:
            extensions.append("json")
        extensions.append(image_format)
        
        file_paths = [f"{filename_prefix}.{ext}" for ext in extensions]
        
        for file_path in file_paths:
            ext = file_path.split(".")[-1].lower()
            
            try:
                if ext == "html":
                    DataVisualization.plotly.offline.plot(
                        figure, filename=file_path, auto_open=False
                    )
                elif ext == "json":
                    DataVisualization.plotly.io.write_json(figure, file_path)
                else:
                    # Image format
                    figure.write_image(file_path, width=2400, height=1400, scale=4)
                
                created_files.append(file_path)
                
            except Exception as e:
                print(f"Warning: Failed to save {file_path}: {str(e)}")
        
        return created_files

    @staticmethod
    def create_categorical_color_mapping(label_series: pd.Series,
                                       color_palette: List[str] = None) -> Tuple[pd.Series, Dict[str, str]]:
        """
        Create consistent color mapping for categorical data visualization.
        
        This method generates a color mapping for categorical variables ensuring
        consistent colors across multiple plots. Supports custom color palettes
        and automatic palette selection based on number of categories.
        
        Parameters
        ----------
        label_series : pd.Series
            Series containing categorical labels
        color_palette : List[str], optional
            Custom color palette. If None, uses Plotly's Alphabet palette
            
        Returns
        -------
        Tuple[pd.Series, Dict[str, str]]
            - Series with color values mapped to original labels
            - Dictionary mapping each unique label to its color
            
        Raises
        ------
        ImportError
            If Plotly is not available
        ValueError
            If label_series is empty or color palette is insufficient
            
        Examples
        --------
        >>> viz = DataVisualization()
        >>> labels = pd.Series(['A', 'B', 'C', 'A', 'B'])
        >>> colors, color_map = viz.create_categorical_color_mapping(labels)
        >>> print(color_map)
        {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c'}
        """
        if not DataVisualization._plotting_available:
            raise ImportError(f"Required plotting libraries not available: {DataVisualization._missing_deps}")
        
        if label_series.empty:
            raise ValueError("label_series cannot be empty")
        
        # Set default color palette if none provided
        if color_palette is None:
            color_palette = DataVisualization.px.colors.qualitative.Alphabet
        
        # Get unique domain values
        unique_labels = label_series.unique()
        
        # Check if we have enough colors
        if len(unique_labels) > len(color_palette):
            print(f"Warning: Number of categories ({len(unique_labels)}) exceeds available colors "
                  f"({len(color_palette)}). Colors will be recycled.")
        
        # Create color mapping
        color_mapping = dict(zip(
            unique_labels, 
            color_palette[:len(unique_labels)] * (len(unique_labels) // len(color_palette) + 1)
        ))
        
        # Map colors to series
        color_series = label_series.map(color_mapping)
        
        return color_series, color_mapping

# =============================================================================
# ADVANCED DATA VISUALIZATION AND ANALYTICS UTILITIES
# =============================================================================
class PartitionAnalyzer:
    """
    Combinatorial partition analysis for systematic data grouping and set theory operations.
    
    This class provides comprehensive partition generation and analysis capabilities
    for exploring all possible ways to group data elements, useful for feature
    grouping, variable selection, and combinatorial optimization problems.
    """
    
    def __init__(self):
        """Initialize the PartitionAnalyzer."""
        self.generated_partitions = {}
        self.partition_statistics = {}
    
    def generate_all_set_partitions(self,
                                   elements: List[Any],
                                   min_subset_count: int = 1,
                                   max_subset_count: Optional[int] = None) -> List[List[List[Any]]]:
        """
        Generate all possible partitions of a set into non-empty subsets.
        
        This method implements a recursive algorithm to generate all possible ways
        to partition a set of elements into non-overlapping, non-empty subsets.
        Useful for feature grouping, variable clustering, and combinatorial analysis.
        
        Parameters:
        -----------
        elements : List[Any]
            List of elements to partition. Elements should be hashable for optimal
            performance but can be any type.
        min_subset_count : int, default=1
            Minimum number of subsets in returned partitions. Must be >= 1.
        max_subset_count : int, optional
            Maximum number of subsets in returned partitions. If None, uses the
            maximum possible (length of elements). Must be >= min_subset_count.
            
        Returns:
        --------
        List[List[List[Any]]]
            List of all possible partitions, where each partition is a list of
            subsets, and each subset is a list of elements.
            
        Raises:
        -------
        ValueError
            If min_subset_count < 1 or max_subset_count < min_subset_count
        TypeError
            If elements is not a list or is empty
            
        Examples:
        --------
        >>> analyzer = PartitionAnalyzer()
        >>> elements = ['A', 'B', 'C']
        >>> partitions = analyzer.generate_all_set_partitions(elements)
        >>> # Returns: [[[A,B,C]], [[A],[B,C]], [[B],[A,C]], [[C],[A,B]], [[A],[B],[C]]]
        
        >>> # Constrain partition size
        >>> partitions = analyzer.generate_all_set_partitions(elements, min_subset_count=2, max_subset_count=2)
        >>> # Returns only partitions with exactly 2 subsets
        """
        # Input validation
        if not isinstance(elements, list):
            raise TypeError("elements must be a list")
            
        if not elements:
            return [[]]
            
        if min_subset_count < 1:
            raise ValueError("min_subset_count must be at least 1")
            
        if max_subset_count is None:
            max_subset_count = len(elements)
        elif max_subset_count < min_subset_count:
            raise ValueError("max_subset_count must be >= min_subset_count")
        
        # Base case: single element
        if len(elements) == 1:
            base_partition = [[elements[0]]]
            if min_subset_count <= 1 <= max_subset_count:
                return [base_partition]
            else:
                return []
        
        # Recursive case
        first_element = elements[0]
        remaining_elements = elements[1:]
        
        # Generate all partitions of remaining elements (no filtering at this level)
        remaining_partitions = self.generate_all_set_partitions(
            remaining_elements, min_subsets=1, max_subsets=None
        )
        
        all_generated_partitions = []
        
        for partition in remaining_partitions:
            # Option 1: Create new subset with first element
            new_partition_isolated = [[first_element]] + partition
            all_generated_partitions.append(new_partition_isolated)
            
            # Option 2: Add first element to each existing subset
            for subset_index in range(len(partition)):
                new_partition_merged = []
                for j, subset in enumerate(partition):
                    if subset_index == j:
                        # Add first element to this subset
                        merged_subset = [first_element] + subset
                        new_partition_merged.append(merged_subset)
                    else:
                        # Keep subset unchanged
                        new_partition_merged.append(subset[:])
                all_generated_partitions.append(new_partition_merged)
        
        # Apply subset count constraints
        filtered_partitions = [
            partition for partition in all_generated_partitions
            if min_subset_count <= len(partition) <= max_subset_count
        ]
        
        # Cache results for potential reuse
        cache_key = (tuple(elements), min_subset_count, max_subset_count)
        self.generated_partitions[cache_key] = filtered_partitions
        
        # Store statistics
        self.partition_statistics[cache_key] = {
            'total_partitions': len(filtered_partitions),
            'element_count': len(elements),
            'min_subsets': min_subset_count,
            'max_subsets': max_subset_count
        }
        
        logger.info(f"Generated {len(filtered_partitions)} partitions for {len(elements)} elements")
        return filtered_partitions
    
    def create_element_grouping_map(self, partition: List[List[Any]]) -> Dict[Any, str]:
        """
        Convert a partition into a mapping dictionary for element grouping.
        
        Creates a dictionary that maps each element to its group identifier,
        facilitating group-based operations and analysis.
        
        Parameters:
        -----------
        partition : List[List[Any]]
            A single partition (list of subsets) to convert to mapping.
            
        Returns:
        --------
        Dict[Any, str]
            Dictionary mapping each element to its group identifier.
            Only elements in subsets with multiple items are included.
            Group identifiers are created by joining sorted element names.
            
        Examples:
        --------
        >>> analyzer = PartitionAnalyzer()
        >>> partition = [['A', 'B'], ['C'], ['D', 'E']]
        >>> mapping = analyzer.create_element_grouping_map(partition)
        >>> # Returns: {'A': 'A_B', 'B': 'A_B', 'D': 'D_E', 'E': 'D_E'}
        """
        if not isinstance(partition, list):
            raise TypeError("partition must be a list of lists")
        
        element_to_group = {}
        
        for subset in partition:
            if len(subset) > 1:  # Only create mappings for multi-element subsets
                # Create group identifier by joining sorted elements
                sorted_elements = sorted([str(element) for element in subset])
                group_identifier = "_".join(sorted_elements)
                
                # Map each element to the group identifier
                for element in subset:
                    element_to_group[element] = group_identifier
        
        return element_to_group

# =============================================================================
# ENHANCED VISUALIZATION AND ANALYSIS CLASSES
# =============================================================================

class ComparativeVisualization:
    """
    Advanced visualization utilities for comparative data analysis.
    
    This class provides methods for creating comparative visualizations including
    Venn diagrams for list comparisons, fuzzy matching analysis, and side-by-side
    data comparison tools commonly used in exploratory data analysis.
    """
    
    @staticmethod
    def create_fuzzy_matching_venn_diagram(list_a: List[Any], 
                                          list_b: List[Any], 
                                          similarity_threshold: float = 60.0,
                                          list_a_name: str = 'List A', 
                                          list_b_name: str = 'List B',
                                          diagram_title: str = 'Fuzzy Matching Comparison', 
                                          output_path: Optional[str] = None) -> Any:
        """
        Create a Venn diagram showing fuzzy matching relationships between two lists.
        
        This method generates publication-quality Venn diagrams that visualize the
        overlap between two datasets using fuzzy string matching. Useful for
        data quality assessment, duplicate detection, and set comparison analysis.
        
        Parameters:
        -----------
        list_a : List[Any]
            First list of items to compare. Items are converted to strings for comparison.
        list_b : List[Any]
            Second list of items to compare. Items are converted to strings for comparison.
        similarity_threshold : float, default=60.0
            Minimum similarity percentage (0-100) for considering items as matches.
            Higher values require closer matches.
        list_a_name : str, default='List A'
            Display name for the first list in the diagram and legend.
        list_b_name : str, default='List B'
            Display name for the second list in the diagram and legend.
        diagram_title : str, default='Fuzzy Matching Comparison'
            Title to display at the top of the Venn diagram.
        output_path : str, optional
            File path to save the diagram. If None, diagram is only displayed.
            
        Returns:
        --------
        matplotlib.figure.Figure
            The matplotlib figure object containing the Venn diagram.
            Can be used for further customization or display.
            
        Raises:
        -------
        ValueError
            If similarity_threshold is not between 0 and 100
        ImportError
            If required matplotlib libraries are not available
            
        Examples:
        --------
        >>> visualizer = ComparativeVisualization()
        >>> list1 = ['Apple Inc.', 'Microsoft Corp', 'Google LLC']
        >>> list2 = ['apple incorporated', 'microsoft corporation', 'amazon.com']
        >>> fig = visualizer.create_fuzzy_matching_venn_diagram(
        ...     list1, list2, similarity_threshold=70,
        ...     list_a_name='Company Names', list_b_name='Cleaned Names',
        ...     output_path='comparison.png'
        ... )
        """
        # Input validation
        if not 0 <= similarity_threshold <= 100:
            raise ValueError("similarity_threshold must be between 0 and 100")
        
        if not list_a or not list_b:
            raise ValueError("Both lists must be non-empty")
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle
            import numpy as np
        except ImportError:
            raise ImportError("matplotlib is required for Venn diagram creation")
        
        # Use existing fuzzy matching functionality from TextProcessor
        text_processor = TextProcessor()
        fuzzy_matches = text_processor.find_fuzzy_matches(
            list_a, list_b, similarity_threshold
        )
        
        # Calculate set relationships
        matched_a = set(fuzzy_matches.keys())
        matched_b = set(match_info['match'] for match_info in fuzzy_matches.values())
        unmatched_a = set(list_a) - matched_a
        unmatched_b = set(list_b) - matched_b
        
        # Count elements in each region
        only_a_count = len(unmatched_a)
        only_b_count = len(unmatched_b)
        both_count = len(fuzzy_matches)
        
        # Create figure with high DPI for quality
        fig, ax = plt.subplots(1, 1, figsize=(12, 10), dpi=100)
        
        # Define circle parameters for optimal visualization
        circle_radius = 1.8
        circle_a_center = (-0.6, 0)
        circle_b_center = (0.6, 0)
        
        # Create aesthetically pleasing circles
        circle_a = Circle(circle_a_center, circle_radius, 
                         alpha=0.4, color='steelblue', 
                         linewidth=2, edgecolor='darkblue')
        circle_b = Circle(circle_b_center, circle_radius, 
                         alpha=0.4, color='lightcoral', 
                         linewidth=2, edgecolor='darkred')
        
        ax.add_patch(circle_a)
        ax.add_patch(circle_b)
        
        # Add count labels with improved positioning
        # Only in A
        ax.text(circle_a_center[0] - 1.0, circle_a_center[1], 
                f'{only_a_count:,}', fontsize=18, ha='center', va='center', 
                weight='bold', color='darkblue')
        
        # Only in B
        ax.text(circle_b_center[0] + 1.0, circle_b_center[1], 
                f'{only_b_count:,}', fontsize=18, ha='center', va='center', 
                weight='bold', color='darkred')
        
        # Intersection (Both)
        ax.text(0, 0, f'{both_count:,}', fontsize=20, ha='center', va='center', 
                weight='bold', color='darkgreen',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        # Add descriptive labels for circles
        ax.text(circle_a_center[0], circle_a_center[1] + 2.3, list_a_name, 
                fontsize=16, ha='center', va='center', weight='bold', color='darkblue')
        ax.text(circle_b_center[0], circle_b_center[1] + 2.3, list_b_name, 
                fontsize=16, ha='center', va='center', weight='bold', color='darkred')
        
        # Create comprehensive information panel
        info_text = (f"📊 COMPARISON SUMMARY\n"
                    f"{'='*40}\n"
                    f"• Unique to {list_a_name}: {only_a_count:,} items\n"
                    f"• Unique to {list_b_name}: {only_b_count:,} items\n"
                    f"• Similar items (≥{similarity_threshold}%): {both_count:,} pairs\n"
                    f"• Total items analyzed: {len(set(list_a) | set(list_b)):,}\n"
                    f"• Similarity threshold: {similarity_threshold}%")
        
        ax.text(-4.2, -3.5, info_text, fontsize=11, ha='left', va='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.9),
                family='monospace')
        
        # Set axis properties for clean appearance
        ax.set_xlim(-4.5, 4.5)
        ax.set_ylim(-4.0, 3.5)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add title with improved formatting
        ax.set_title(diagram_title, fontsize=18, weight='bold', pad=25, color='darkslategray')
        
        # Save diagram if path provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Venn diagram saved to: {output_path}")
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_comprehensive_list_comparison(list_a: List[Any], 
                                           list_b: List[Any],
                                           similarity_threshold: float = 60.0,
                                           include_venn_diagram: bool = False,
                                           list_a_name: str = 'List A', 
                                           list_b_name: str = 'List B',
                                           diagram_title: str = 'List Comparison Analysis',
                                           venn_output_path: Optional[str] = None) -> Tuple[pd.DataFrame, str, Optional[Any]]:
        """
        Create comprehensive comparison analysis between two lists with fuzzy matching.
        
        This method provides detailed tabular analysis of list similarities and differences,
        with optional Venn diagram visualization. Useful for data reconciliation,
        duplicate analysis, and quality assessment workflows.
        
        Parameters:
        -----------
        list_a : List[Any]
            First list for comparison analysis
        list_b : List[Any] 
            Second list for comparison analysis
        similarity_threshold : float, default=60.0
            Minimum similarity percentage for fuzzy matching (0-100)
        include_venn_diagram : bool, default=False
            Whether to generate accompanying Venn diagram visualization
        list_a_name : str, default='List A'
            Descriptive name for first list in outputs
        list_b_name : str, default='List B'
            Descriptive name for second list in outputs  
        diagram_title : str, default='List Comparison Analysis'
            Title for optional Venn diagram
        venn_output_path : str, optional
            Path to save Venn diagram if generated
            
        Returns:
        --------
        Tuple[pd.DataFrame, str, Optional[matplotlib.figure.Figure]]
            - DataFrame with detailed comparison results
            - Formatted text summary of analysis
            - Venn diagram figure if requested, otherwise None
            
        Raises:
        -------
        ValueError
            If similarity_threshold is invalid or lists are empty
            
        Examples:
        --------
        >>> visualizer = ComparativeVisualization()
        >>> companies = ['Apple Inc.', 'Microsoft Corp', 'Google LLC']
        >>> variations = ['apple incorporated', 'microsoft corporation', 'alphabet inc.']
        >>> df, summary, fig = visualizer.create_comprehensive_list_comparison(
        ...     companies, variations, similarity_threshold=70,
        ...     include_venn_diagram=True, 
        ...     list_a_name='Original Names', list_b_name='Variations'
        ... )
        >>> print(summary)
        >>> df.to_csv('comparison_results.csv')
        """
        # Input validation
        if not 0 <= similarity_threshold <= 100:
            raise ValueError("similarity_threshold must be between 0 and 100")
        
        if not list_a or not list_b:
            raise ValueError("Both lists must be non-empty")
        
        # Use TextProcessor for fuzzy matching
        text_processor = TextProcessor()
        fuzzy_matches = text_processor.find_fuzzy_matches(
            list_a, list_b, similarity_threshold
        )
        
        # Identify matched and unmatched elements
        matched_a = set(fuzzy_matches.keys())
        matched_b = set(match_info['match'] for match_info in fuzzy_matches.values())
        unmatched_a = set(list_a) - matched_a
        unmatched_b = set(list_b) - matched_b
        
        # Build comprehensive comparison data
        comparison_data = []
        
        # Add elements unique to list A
        for element in sorted(unmatched_a, key=str):
            comparison_data.append({
                'Element': str(element),
                'Category': f'Only in {list_a_name}',
                'Match': '',
                'Similarity_Percent': '',
                'Match_Quality': 'No Match'
            })
        
        # Add elements unique to list B
        for element in sorted(unmatched_b, key=str):
            comparison_data.append({
                'Element': str(element),
                'Category': f'Only in {list_b_name}',
                'Match': '',
                'Similarity_Percent': '',
                'Match_Quality': 'No Match'
            })
        
        # Add fuzzy matched pairs with quality assessment
        for item_a, match_info in sorted(fuzzy_matches.items(), key=lambda x: x[1]['similarity'], reverse=True):
            similarity = match_info['similarity']
            
            # Determine match quality
            if similarity >= 90:
                quality = 'Excellent'
            elif similarity >= 75:
                quality = 'Good'
            elif similarity >= 60:
                quality = 'Fair'
            else:
                quality = 'Poor'
            
            comparison_data.append({
                'Element': str(item_a),
                'Category': 'Matched',
                'Match': str(match_info['match']),
                'Similarity_Percent': f"{similarity:.1f}%",
                'Match_Quality': quality
            })
        
        # Create comprehensive DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        if not comparison_df.empty:
            # Add index column for easy reference
            comparison_df.insert(0, 'Index', range(1, len(comparison_df) + 1))
            
            # Sort by category priority then by similarity
            category_order = ['Matched', f'Only in {list_a_name}', f'Only in {list_b_name}']
            comparison_df['_sort_key'] = comparison_df['Category'].apply(
                lambda x: category_order.index(x) if x in category_order else 999
            )
            comparison_df = comparison_df.sort_values(['_sort_key', 'Similarity_Percent'], 
                                                    ascending=[True, False]).drop('_sort_key', axis=1)
            comparison_df = comparison_df.reset_index(drop=True)
            comparison_df['Index'] = range(1, len(comparison_df) + 1)
        
        # Generate comprehensive text summary
        summary_text = f"""
{'='*80}
COMPREHENSIVE LIST COMPARISON ANALYSIS
{'='*80}

📋 COMPARISON RESULTS TABLE:
{comparison_df.to_string(index=False) if not comparison_df.empty else 'No data to display'}

📊 STATISTICAL SUMMARY:
{'='*50}
Total unique elements analyzed: {len(set(list_a) | set(list_b)):,}
Elements in {list_a_name}: {len(list_a):,}
Elements in {list_b_name}: {len(list_b):,}

CATEGORY BREAKDOWN:
• Only in {list_a_name}: {len(unmatched_a):,} elements
• Only in {list_b_name}: {len(unmatched_b):,} elements  
• Fuzzy matches found: {len(fuzzy_matches):,} pairs

MATCH QUALITY DISTRIBUTION:
"""
        
        if not comparison_df.empty:
            quality_counts = comparison_df[comparison_df['Match_Quality'] != 'No Match']['Match_Quality'].value_counts()
            for quality, count in quality_counts.items():
                summary_text += f"• {quality}: {count:,} matches\n"
        
        summary_text += f"""
🔍 FUZZY MATCHING CONFIGURATION:
{'='*45}
Similarity threshold: {similarity_threshold}%
Matching algorithm: Sequence-based fuzzy matching
Text normalization: Applied (case, spaces, special chars)

📈 DATA QUALITY INSIGHTS:
"""
        
        if fuzzy_matches:
            similarities = [info['similarity'] for info in fuzzy_matches.values()]
            avg_similarity = np.mean(similarities)
            max_similarity = np.max(similarities)
            min_similarity = np.min(similarities)
            
            summary_text += f"""• Average match similarity: {avg_similarity:.1f}%
• Best match similarity: {max_similarity:.1f}%
• Weakest match similarity: {min_similarity:.1f}%
• Match rate: {len(fuzzy_matches) / max(len(list_a), len(list_b)) * 100:.1f}%

🔗 TOP FUZZY MATCHES:
"""
            # Show top 5 matches
            top_matches = sorted(fuzzy_matches.items(), key=lambda x: x[1]['similarity'], reverse=True)[:5]
            for i, (item_a, match_info) in enumerate(top_matches, 1):
                summary_text += f"{i:2d}. '{item_a}' ↔ '{match_info['match']}' ({match_info['similarity']:.1f}%)\n"
        
        summary_text += f"""
📁 ORIGINAL DATA:
{list_a_name}: {list_a}
{list_b_name}: {list_b}
        """.strip()
        
        # Generate Venn diagram if requested
        venn_figure = None
        if include_venn_diagram:
            try:
                venn_figure = ComparativeVisualization.create_fuzzy_matching_venn_diagram(
                    list_a, list_b, similarity_threshold, 
                    list_a_name, list_b_name, diagram_title, venn_output_path
                )
            except Exception as e:
                logger.warning(f"Could not create Venn diagram: {e}")
        
        return comparison_df, summary_text, venn_figure

class StatisticalUtilities:
    """
    Advanced statistical utilities for data science workflows.
    
    This class provides methods for statistical testing, feature selection,
    hypothesis testing, and advanced statistical analysis commonly used
    in data science and machine learning projects.
    """
    
    @staticmethod
    def perform_comparative_feature_analysis(feature_matrix: np.ndarray,
                                           target_variable: np.ndarray,
                                           analysis_methods: List[str] = None,
                                           feature_names: List[str] = None) -> pd.DataFrame:
        """
        Perform comprehensive comparative analysis using multiple feature selection methods.
        
        This method applies various univariate feature selection techniques to identify
        the most important features in a dataset. Useful for feature engineering,
        dimensionality reduction, and understanding variable relationships.
        
        Parameters:
        -----------
        feature_matrix : np.ndarray
            Feature matrix with shape (n_samples, n_features)
        target_variable : np.ndarray
            Target variable with shape (n_samples,)
        analysis_methods : List[str], optional
            List of feature selection methods to apply. If None, uses default methods:
            ['mutual_info_classif', 'chi2', 'f_classif']
        feature_names : List[str], optional
            Names for features. If None, generates generic names.
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with features as rows and analysis methods as columns,
            containing p-values or scores for each method-feature combination
            
        Raises:
        -------
        ValueError
            If feature_matrix and target_variable have incompatible shapes
        ImportError
            If required scikit-learn components are not available
            
        Examples:
        --------
        >>> stats = StatisticalUtilities()
        >>> X = np.random.rand(100, 5)
        >>> y = np.random.randint(0, 2, 100)
        >>> results = stats.perform_comparative_feature_analysis(
        ...     X, y, feature_names=['feat1', 'feat2', 'feat3', 'feat4', 'feat5']
        ... )
        >>> print(results)
        """
        # Input validation
        if feature_matrix.shape[0] != len(target_variable):
            raise ValueError("feature_matrix and target_variable must have same number of samples")
        
        try:
            from sklearn.feature_selection import (
                SelectKBest, mutual_info_classif, mutual_info_regression,
                chi2, f_classif, SelectFpr, SelectFdr, SelectFwe
            )
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError("scikit-learn, matplotlib, and seaborn are required")
        
        # Set default analysis methods if not provided
        if analysis_methods is None:
            analysis_methods = ['mutual_info_classif', 'chi2', 'f_classif']
        
        # Generate feature names if not provided
        if feature_names is None:
            feature_names = [f'Feature_{i+1}' for i in range(feature_matrix.shape[1])]
        
        # Initialize results array
        results_matrix = np.empty((len(analysis_methods), feature_matrix.shape[1]), dtype=float)
        
        # Apply each analysis method
        for method_idx, method_name in enumerate(analysis_methods):
            logger.info(f"Applying {method_name} analysis...")
            
            try:
                if method_name == 'mutual_info_classif':
                    scores = mutual_info_classif(feature_matrix, target_variable)
                elif method_name == 'mutual_info_regression':
                    scores = mutual_info_regression(feature_matrix, target_variable)
                elif method_name == 'chi2':
                    # Chi2 requires non-negative features
                    if np.any(feature_matrix < 0):
                        logger.warning(f"Chi2 test requires non-negative features. Skipping {method_name}")
                        scores = np.full(feature_matrix.shape[1], np.nan)
                    else:
                        chi2_scores, p_values = chi2(feature_matrix, target_variable)
                        scores = p_values  # Use p-values for chi2
                elif method_name in ['f_classif', 'SelectFpr', 'SelectFdr', 'SelectFwe']:
                    method_func = eval(method_name)
                    if method_name == 'f_classif':
                        f_scores, p_values = f_classif(feature_matrix, target_variable)
                        scores = p_values
                    else:
                        # For selector methods, extract p-values
                        selector = SelectKBest(method_func, k='all')
                        selector.fit(feature_matrix, target_variable)
                        scores = selector.pvalues_
                else:
                    logger.warning(f"Unknown method: {method_name}. Filling with NaN.")
                    scores = np.full(feature_matrix.shape[1], np.nan)
                
                results_matrix[method_idx, :] = scores
                
            except Exception as e:
                logger.error(f"Error applying {method_name}: {e}")
                results_matrix[method_idx, :] = np.full(feature_matrix.shape[1], np.nan)
        
        # Create results DataFrame
        results_df = pd.DataFrame(
            data=results_matrix,
            index=analysis_methods,
            columns=feature_names
        )
        
        # Transpose for better readability (features as rows, methods as columns)
        results_df = results_df.T
        
        # Remove features that are all NaN
        results_df = results_df.dropna(how='all')
        
        logger.info(f"Feature analysis completed for {len(feature_names)} features using {len(analysis_methods)} methods")
        
        return results_df
    
    @staticmethod
    def perform_hypothesis_testing(dataframe: pd.DataFrame,
                                 test_parameter: str,
                                 grouping_column: str,
                                 group_labels: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform hypothesis testing between two groups using t-test analysis.
        
        This method conducts comprehensive two-sample t-tests to compare means
        between groups. Includes descriptive statistics and interpretation
        of results for statistical significance testing.
        
        Parameters:
        -----------
        dataframe : pd.DataFrame
            Input DataFrame containing the data
        test_parameter : str
            Column name for the variable to test
        grouping_column : str
            Column name for group assignment (should be boolean or binary)
        group_labels : List[str]
            Labels for the two groups [group_true, group_false]
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            - Descriptive statistics for both groups
            - Detailed test results with p-values
            - Summary interpretation of results
            
        Raises:
        -------
        ValueError
            If required columns are missing or group_labels not length 2
        ImportError
            If researchpy package is not available
            
        Examples:
        --------
        >>> stats = StatisticalUtilities()
        >>> df = pd.DataFrame({
        ...     'score': np.random.normal(50, 10, 100),
        ...     'treatment': np.random.choice([True, False], 100)
        ... })
        >>> desc, results, summary = stats.perform_hypothesis_testing(
        ...     df, 'score', 'treatment', ['Treated', 'Control']
        ... )
        """
        # Input validation
        if test_parameter not in dataframe.columns:
            raise ValueError(f"Parameter '{test_parameter}' not found in DataFrame")
        
        if grouping_column not in dataframe.columns:
            raise ValueError(f"Grouping column '{grouping_column}' not found in DataFrame")
        
        if len(group_labels) != 2:
            raise ValueError("group_labels must contain exactly 2 labels")
        
        try:
            import researchpy as rp
        except ImportError:
            raise ImportError("researchpy package is required for hypothesis testing")
        
        # Prepare data
        df_work = dataframe.copy()
        df_work[grouping_column] = df_work[grouping_column].astype('bool')
        
        # Split data by groups
        group_1_data = df_work[test_parameter][df_work[grouping_column]]
        group_2_data = df_work[test_parameter][~df_work[grouping_column]]
        
        group_1_name, group_2_name = group_labels[0], group_labels[1]
        
        # Perform t-test
        descriptive_stats, test_results = rp.ttest(
            group_1_data, group_2_data,
            group1_name=group_1_name,
            group2_name=group_2_name,
            equal_variances=False,
            paired=False
        )
        
        # Process results
        test_results = test_results.set_index(test_results.columns[0])
        test_results.columns = [test_parameter]
        
        # Interpret results
        two_sided_p = test_results.loc['Two side test p value = '][0]
        lower_p = test_results.loc['Difference < 0 p value = '][0] 
        higher_p = test_results.loc['Difference > 0 p value = '][0]
        
        if two_sided_p > 0.05:
            interpretation = f"{test_parameter}: No significant difference between {group_1_name} and {group_2_name}"
            summary_code = 'no difference'
        elif two_sided_p <= 0.05 and lower_p <= 0.05:
            interpretation = f"{test_parameter}: {group_1_name} is significantly lower than {group_2_name}"
            summary_code = 'lower'
        elif two_sided_p <= 0.05 and higher_p <= 0.05:
            interpretation = f"{test_parameter}: {group_1_name} is significantly higher than {group_2_name}"
            summary_code = 'higher'
        else:
            interpretation = f"{test_parameter}: Inconclusive results"
            summary_code = 'inconclusive'
        
        # Add interpretation to results
        test_results.loc['Statistical_Interpretation'] = interpretation
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(
            [[summary_code]], 
            index=[test_parameter], 
            columns=[f'{group_1_name}_vs_{group_2_name}']
        )
        
        logger.info(f"Hypothesis testing completed for {test_parameter}")
        
        return descriptive_stats, test_results, summary_df
    
    @staticmethod
    def perform_kruskal_wallis_test(observations: pd.Series, groups: pd.Series) -> Tuple[float, float]:
        """
        Calculate the Kruskal-Wallis H-test for independent samples.
        
        The Kruskal-Wallis test is a non-parametric method for testing whether samples
        originate from the same distribution. It is used for comparing two or more
        independent samples of equal or different sample sizes.
        
        Parameters
        ----------
        observations : pd.Series
            Array of observations/measurements
        groups : pd.Series
            Array of group labels corresponding to each observation
            
        Returns
        -------
        Tuple[float, float]
            H-statistic (float): The Kruskal-Wallis H statistic, corrected for ties
            p-value (float): The p-value for the test using the assumption that H 
                           has a chi square distribution
                           
        Raises
        ------
        ValueError
            If observations and groups have different lengths
        ImportError
            If scipy.stats is not available
            
        Examples
        --------
        >>> stats_util = StatisticalUtilities()
        >>> obs = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])
        >>> grps = pd.Series(['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'])
        >>> h_stat, p_val = stats_util.perform_kruskal_wallis_test(obs, grps)
        >>> print(f"H-statistic: {h_stat:.4f}, p-value: {p_val:.4f}")
        """
        if len(observations) != len(groups):
            raise ValueError("observations and groups must have the same length")
            
        try:
            from scipy import stats
        except ImportError:
            raise ImportError("scipy.stats is required for Kruskal-Wallis test")
        
        # Group observations by category
        grouped_numbers = {}
        for group in groups.unique():
            grouped_numbers[group] = observations.values[groups == group]
        
        # Perform Kruskal-Wallis test
        h_statistic, p_value = stats.mstats.kruskalwallis(*grouped_numbers.values())
        
        return h_statistic, p_value
    
    @staticmethod
    def calculate_point_biserial_correlation(binary_data: str, continuous_data: str, 
                                           dataframe: pd.DataFrame) -> float:
        """
        Compute the point biserial correlation of two pandas DataFrame columns.
        
        Point-biserial correlation is used to measure the strength and direction
        of the association between one continuous variable and one binary variable.
        
        Parameters
        ----------
        binary_data : str
            Name of the binary/dichotomous data column
        continuous_data : str 
            Name of the continuous data column
        dataframe : pd.DataFrame
            DataFrame containing both columns
            
        Returns
        -------
        float
            Point biserial correlation coefficient
            
        Raises
        ------
        ValueError
            If specified columns don't exist in dataframe or binary column
            doesn't have exactly 2 unique values
        KeyError
            If column names are not found in dataframe
            
        Notes
        -----
        This implementation handles missing values by excluding them from calculation.
        The binary variable should have exactly two unique values.
        
        Examples
        --------
        >>> stats_util = StatisticalUtilities()
        >>> df = pd.DataFrame({
        ...     'binary_col': [0, 1, 0, 1, 0, 1],
        ...     'continuous_col': [1.5, 3.2, 2.1, 4.8, 1.9, 3.7]
        ... })
        >>> corr = stats_util.calculate_point_biserial_correlation(
        ...     'binary_col', 'continuous_col', df
        ... )
        """
        if binary_data not in dataframe.columns:
            raise KeyError(f"Column '{binary_data}' not found in dataframe")
        if continuous_data not in dataframe.columns:
            raise KeyError(f"Column '{continuous_data}' not found in dataframe")
        
        # Remove missing values
        df_clean = dataframe[[binary_data, continuous_data]].dropna()
        
        binary_unique = df_clean[binary_data].unique()
        if len(binary_unique) != 2:
            raise ValueError(f"Binary column must have exactly 2 unique values, found {len(binary_unique)}")
        
        # Split continuous data by binary groups
        group_0 = df_clean[df_clean[binary_data] == binary_unique[0]][continuous_data]
        group_1 = df_clean[df_clean[binary_data] == binary_unique[1]][continuous_data]
        
        # Calculate statistics
        s_y = np.std(df_clean[continuous_data])
        n = len(df_clean[binary_data])
        n0 = len(group_0)
        n1 = len(group_1)
        m0 = group_0.mean()
        m1 = group_1.mean()
        
        # Point-biserial correlation formula
        correlation = (m0 - m1) * np.sqrt((n0 * n1) / n**2) / s_y
        
        return correlation
    
    @staticmethod
    def analyze_categorical_relationship(dataframe: pd.DataFrame, 
                                       independent_var: str, 
                                       dependent_var: str, 
                                       alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis on categorical variables.
        
        This method performs chi-square test of independence, calculates Cramer's V
        for effect size, creates contingency table, and provides interpretation
        of results for categorical-categorical variable relationships.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataset containing the variables
        independent_var : str
            Name of the independent variable column
        dependent_var : str
            Name of the dependent variable column
        alpha : float, default=0.05
            Significance level for hypothesis testing
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - contingency_table: pd.DataFrame with cross-tabulation
            - chi_square_statistic: float, chi-square test statistic
            - p_value: float, significance probability
            - degrees_of_freedom: int, degrees of freedom
            - cramers_v: float, effect size measure
            - significant: bool, whether result is statistically significant
            - interpretation: str, text interpretation of results
            
        Raises
        ------
        ValueError
            If variables don't exist in dataframe
        ImportError
            If required dependencies are not available
            
        Examples
        --------
        >>> stats_util = StatisticalUtilities()
        >>> df = pd.DataFrame({
        ...     'gender': ['M', 'F', 'M', 'F'] * 25,
        ...     'preference': ['A', 'B', 'A', 'B'] * 25
        ... })
        >>> results = stats_util.analyze_categorical_relationship(
        ...     df, 'gender', 'preference'
        ... )
        >>> print(f"Chi-square: {results['chi_square_statistic']:.4f}")
        >>> print(f"Significant: {results['significant']}")
        """
        if independent_var not in dataframe.columns:
            raise ValueError(f"Column '{independent_var}' not found in DataFrame")
        if dependent_var not in dataframe.columns:
            raise ValueError(f"Column '{dependent_var}' not found in DataFrame")
            
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.stats import chi2_contingency
        except ImportError:
            raise ImportError("matplotlib, seaborn, and scipy are required for this analysis")
        
        # Create contingency table
        contingency_table = pd.crosstab(dataframe[independent_var], dataframe[dependent_var])
        
        # Perform Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        
        # Calculate Cramer's V (effect size)
        n = contingency_table.sum().sum()
        min_dim = min(contingency_table.shape) - 1
        cramer_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        # Interpret results
        interpretation = StatisticalUtilities._interpret_categorical_results(
            p_value, cramer_v, alpha
        )
        
        # Compile results
        results = {
            'contingency_table': contingency_table,
            'chi_square_statistic': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'cramers_v': cramer_v,
            'significant': p_value < alpha,
            'interpretation': interpretation
        }
        
        return results
    
    @staticmethod
    def _interpret_categorical_results(p_value: float, cramers_v: float, alpha: float) -> str:
        """
        Interpret the statistical test results for categorical analysis.
        
        Parameters
        ----------
        p_value : float
            P-value from the chi-square test
        cramers_v : float
            Cramer's V effect size measure
        alpha : float
            Significance level
            
        Returns
        -------
        str
            Interpretation of results
        """
        interpretation = []
        
        # Chi-square test interpretation
        if p_value < alpha:
            interpretation.append(
                f"There is a statistically significant relationship between the variables "
                f"(p-value = {p_value:.4f} < {alpha})."
            )
        else:
            interpretation.append(
                f"There is no statistically significant relationship between the variables "
                f"(p-value = {p_value:.4f} > {alpha})."
            )
        
        # Cramer's V interpretation
        if cramers_v < 0.1:
            strength = "negligible"
        elif cramers_v < 0.3:
            strength = "weak"
        elif cramers_v < 0.5:
            strength = "moderate"
        else:
            strength = "strong"
        
        interpretation.append(
            f"The strength of the association is {strength} (Cramer's V = {cramers_v:.3f})."
        )
        
        return " ".join(interpretation)
    
    @staticmethod
    def analyze_categorical_vs_numeric(dataframe: pd.DataFrame,
                                     categorical_var: str,
                                     numeric_var: str,
                                     alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis for categorical vs numeric variables.
        
        This method automatically selects and performs appropriate statistical tests
        based on data characteristics (normality, homoscedasticity, number of groups).
        It includes assumption testing and provides detailed results interpretation.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            The dataset containing both variables
        categorical_var : str
            Name of the categorical (independent) variable column
        numeric_var : str
            Name of the numeric (dependent) variable column
        alpha : float, default=0.05
            Significance level for statistical tests
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - test_type: str, which test was performed
            - test_statistic: float, the test statistic value
            - p_value: float, the p-value from the test
            - significant: bool, whether result is significant
            - descriptive_stats: dict, descriptive statistics by group
            - assumption_tests: dict, results of assumption tests
            - effect_size: float, effect size measure when applicable
            
        Raises
        ------
        TypeError
            If input data is not a pandas DataFrame
        ValueError
            If specified variables don't exist or have insufficient data
        ImportError
            If required scipy.stats is not available
            
        Examples
        --------
        >>> stats_util = StatisticalUtilities()
        >>> df = pd.DataFrame({
        ...     'group': ['A', 'B', 'C'] * 20,
        ...     'score': np.random.normal(50, 10, 60)
        ... })
        >>> results = stats_util.analyze_categorical_vs_numeric(df, 'group', 'score')
        >>> print(f"Test: {results['test_type']}")
        >>> print(f"Significant: {results['significant']}")
        """
        # Input validation
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame")
        if categorical_var not in dataframe.columns:
            raise ValueError(f"Column '{categorical_var}' not found in DataFrame")
        if numeric_var not in dataframe.columns:
            raise ValueError(f"Column '{numeric_var}' not found in DataFrame")
        
        try:
            from scipy import stats
        except ImportError:
            raise ImportError("scipy.stats is required for this analysis")
        
        # Clean data and get groups
        df_clean = dataframe[[categorical_var, numeric_var]].dropna()
        categories = df_clean[categorical_var].unique()
        n_groups = len(categories)
        
        if n_groups < 2:
            raise ValueError("Need at least 2 groups for comparison")
        
        # Create groups for analysis
        groups = [df_clean[df_clean[categorical_var] == cat][numeric_var] for cat in categories]
        
        # Calculate descriptive statistics
        descriptive_stats = df_clean.groupby(categorical_var)[numeric_var].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(3).to_dict('index')
        
        # Test assumptions
        assumption_results = StatisticalUtilities._test_assumptions(groups, categories, alpha)
        
        # Select and perform appropriate test
        test_results = StatisticalUtilities._perform_appropriate_test(
            groups, n_groups, assumption_results, alpha
        )
        
        # Compile final results
        results = {
            'test_type': test_results['test_type'],
            'test_statistic': test_results['statistic'],
            'p_value': test_results['p_value'],
            'significant': test_results['p_value'] < alpha,
            'descriptive_stats': descriptive_stats,
            'assumption_tests': assumption_results
        }
        
        return results
    
    @staticmethod
    def _test_assumptions(groups: List[pd.Series], categories: List[str], alpha: float) -> Dict[str, Any]:
        """Test statistical assumptions for group comparison tests."""
        from scipy import stats
        
        # Test normality for each group
        normality_tests = {}
        for cat, group in zip(categories, groups):
            if len(group) >= 3:  # Shapiro-Wilk requires at least 3 samples
                stat, p_val = stats.shapiro(group)
                normality_tests[cat] = {
                    'statistic': stat,
                    'p_value': p_val,
                    'normal': p_val > alpha
                }
        
        # Test homogeneity of variances (Levene's test)
        levene_stat, levene_p = stats.levene(*groups)
        
        return {
            'normality': normality_tests,
            'homogeneity_of_variance': {
                'statistic': levene_stat,
                'p_value': levene_p,
                'equal_variances': levene_p > alpha
            }
        }
    
    @staticmethod
    def _perform_appropriate_test(groups: List[pd.Series], n_groups: int, 
                                assumption_results: Dict[str, Any], alpha: float) -> Dict[str, Any]:
        """Select and perform the most appropriate statistical test."""
        from scipy import stats
        
        # Check assumptions
        all_normal = all(
            test.get('normal', False) 
            for test in assumption_results['normality'].values()
        )
        equal_variances = assumption_results['homogeneity_of_variance']['equal_variances']
        
        if n_groups == 2:
            # Two-group comparison
            if all_normal:
                stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=equal_variances)
                test_type = "Independent t-test"
                if not equal_variances:
                    test_type += " with Welch's correction"
            else:
                stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                test_type = "Mann-Whitney U test"
        else:
            # Multi-group comparison
            if all_normal and equal_variances:
                stat, p_value = stats.f_oneway(*groups)
                test_type = "One-way ANOVA"
            else:
                stat, p_value = stats.kruskal(*groups)
                test_type = "Kruskal-Wallis H test"
        
        return {
            'test_type': test_type,
            'statistic': stat,
            'p_value': p_value
        }

class DataFrameMerger:
    """
    Advanced DataFrame merging utilities for data analysis workflows.
    
    This class provides specialized merging operations including interval-based
    joins, flexible string matching, and data preprocessing utilities commonly
    needed in machine learning and data analysis pipelines.
    
    Examples
    --------
    >>> merger = DataFrameMerger()
    >>> result = merger.merge_dataframes_between_dates(df1, df2, 'group_col')
    """
    
    def merge_dataframes_between_dates(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                                     group_col: str, date_col: str = 'Date',
                                     start_col: str = 'Start', end_col: str = 'End',
                                     closed: str = "both") -> pd.DataFrame:
        """
        Merge two DataFrames based on date intervals using IntervalIndex.
        
        This method merges df1 with df2 by checking if dates in df1 fall within
        date ranges defined in df2. The merge is performed per group.
        
        Parameters
        ----------
        df1 : pd.DataFrame
            Source DataFrame containing dates to be matched
        df2 : pd.DataFrame  
            Reference DataFrame containing date ranges (Start/End columns)
        group_col : str
            Column name for grouping both DataFrames
        date_col : str, default 'Date'
            Column name containing dates in df1
        start_col : str, default 'Start'
            Column name containing start dates in df2
        end_col : str, default 'End'
            Column name containing end dates in df2
        closed : {'both', 'left', 'right', 'neither'}, default 'both'
            Whether intervals are closed on left, right, both or neither side
            
        Returns
        -------
        pd.DataFrame
            Merged DataFrame with additional 'Index_no' column indicating 
            interval index matches
            
        Examples
        --------
        >>> merger = DataFrameMerger()
        >>> df_merged = merger.merge_dataframes_between_dates(
        ...     df1=vessel_data, df2=cases_data, 
        ...     group_col='Vessel', date_col='Date'
        ... )
        
        Raises
        ------
        ValueError
            If required columns are missing from input DataFrames
        KeyError
            If specified column names don't exist in DataFrames
        """
        # Input validation
        required_cols_df1 = [group_col, date_col]
        required_cols_df2 = [group_col, start_col, end_col]
        
        missing_df1 = [col for col in required_cols_df1 if col not in df1.columns]
        missing_df2 = [col for col in required_cols_df2 if col not in df2.columns]
        
        if missing_df1:
            raise ValueError(f"Missing columns in df1: {missing_df1}")
        if missing_df2:
            raise ValueError(f"Missing columns in df2: {missing_df2}")
            
        # Initialize output DataFrame
        output_columns = df1.columns.tolist() + ['Index_no']
        df_out = pd.DataFrame(columns=output_columns)
        
        # Process each group separately
        for group_name, group_df in df1.groupby([group_col]):
            # Get corresponding group from df2
            df2_subset = df2.loc[df2[group_col] == group_name].copy()
            
            if df2_subset.empty:
                # No matching intervals for this group
                group_df_copy = group_df.copy()
                group_df_copy['Index_no'] = -1  # No match indicator
                df_out = pd.concat([df_out, group_df_copy], axis=0, ignore_index=True)
                continue
                
            # Create IntervalIndex from date ranges
            try:
                interval_index = pd.IntervalIndex.from_arrays(
                    df2_subset[start_col],
                    df2_subset[end_col], 
                    closed=closed
                )
                
                # Find which interval each date belongs to
                group_df_copy = group_df.copy()
                group_df_copy['Index_no'] = interval_index.get_indexer(
                    group_df_copy[date_col]
                )
                
                df_out = pd.concat([df_out, group_df_copy], axis=0, ignore_index=True)
                
            except Exception as e:
                print(f"Warning: Error processing group {group_name}: {str(e)}")
                group_df_copy = group_df.copy()
                group_df_copy['Index_no'] = -1
                df_out = pd.concat([df_out, group_df_copy], axis=0, ignore_index=True)
                
        return df_out
    
    def flexible_join(self, left_df: pd.DataFrame, right_df: pd.DataFrame, 
                     left_on: str = None, right_on: str = None, on: str = None, 
                     how: str = 'inner', **kwargs) -> pd.DataFrame:
        """
        Join two DataFrames with flexible string matching.
        
        This method handles differences in spaces, underscores, special characters,
        and letter case during the join operation by normalizing join keys.
        
        Parameters
        ----------
        left_df, right_df : pd.DataFrame
            DataFrames to join
        left_on, right_on : str or list of str, optional
            Column(s) to use as join key(s)
        on : str or list of str, optional
            Column name(s) if identical in both DataFrames
        how : {'inner', 'left', 'right', 'outer'}, default 'inner'
            Type of join to perform
        **kwargs : dict
            Additional arguments passed to pd.merge()
            
        Returns
        -------
        pd.DataFrame
            Joined DataFrame
            
        Examples
        --------
        >>> merger = DataFrameMerger()
        >>> result = merger.flexible_join(
        ...     df1, df2, left_on='company_name', right_on='Company Name'
        ... )
        
        Raises
        ------
        ValueError
            If join column specifications are invalid
        """
        # Create copies to avoid modifying originals
        left_copy = left_df.copy()
        right_copy = right_df.copy()
        
        # Handle column specification
        if on is not None:
            left_on = right_on = on
            
        if left_on is None or right_on is None:
            raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")
            
        # Convert to lists for uniform handling
        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]
            
        if len(left_on) != len(right_on):
            raise ValueError("Length of 'left_on' must equal length of 'right_on'")
            
        # Create normalized columns
        left_norm_cols, right_norm_cols = [], []
        
        for lcol, rcol in zip(left_on, right_on):
            left_norm_col = f"_normalized_left_{lcol}"
            right_norm_col = f"_normalized_right_{rcol}"
            
            left_norm_cols.append(left_norm_col)
            right_norm_cols.append(right_norm_col)
            
            # Apply string normalization
            left_copy[left_norm_col] = left_copy[lcol].apply(self._normalize_string)
            right_copy[right_norm_col] = right_copy[rcol].apply(self._normalize_string)
            
        # Perform join on normalized keys
        result = pd.merge(
            left_copy, right_copy,
            left_on=left_norm_cols,
            right_on=right_norm_cols,
            how=how,
            **kwargs
        )
        
        # Remove temporary normalized columns
        result = result.drop(columns=left_norm_cols + right_norm_cols)
        
        return result
        
    @staticmethod
    def _normalize_string(s):
        """
        Normalize string for flexible matching.
        
        Parameters
        ----------
        s : str or any
            Input to normalize
            
        Returns
        -------
        str
            Normalized string (lowercase, alphanumeric only)
        """
        if pd.isna(s):
            return ''
        
        import re
        # Convert to string, lowercase, keep only alphanumeric
        normalized = re.sub(r'[^a-zA-Z0-9]', '', str(s).lower())
        return normalized


class DataFrameComparator:
    """
    Comprehensive DataFrame comparison utilities.
    
    This class provides advanced functionality for comparing DataFrames including
    column-level analysis, data type compatibility, memory usage comparison,
    and value commonality analysis.
    
    Attributes
    ----------
    df1, df2 : pd.DataFrame
        The DataFrames being compared
    df1_name, df2_name : str
        Display names for the DataFrames
    all_columns : list
        Sorted list of all unique columns across both DataFrames
    common_columns : set
        Columns that exist in both DataFrames
    df1_only, df2_only : list
        Columns that exist only in df1 or df2 respectively
        
    Examples
    --------
    >>> comparator = DataFrameComparator(df1, df2, "Train Data", "Test Data")
    >>> comparison_table, summary, types = comparator.compare()
    """
    
    def __init__(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                 df1_name: str = "DataFrame 1", df2_name: str = "DataFrame 2"):
        """
        Initialize the comparator with two DataFrames.
        
        Parameters:
        -----------
        df1, df2 : pd.DataFrame
            DataFrames to compare
        df1_name, df2_name : str
            Names for the DataFrames (for display purposes)
        """
        self.df1 = df1
        self.df2 = df2
        self.df1_name = df1_name
        self.df2_name = df2_name
        self.all_columns = sorted(set(df1.columns) | set(df2.columns))
        self.common_columns = set(df1.columns) & set(df2.columns)
        self.df1_only = list(set(df1.columns) - set(df2.columns))
        self.df2_only = list(set(df2.columns) - set(df1.columns))
    
    def compare(self, display: bool = True) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
        """
        Compare columns between the two DataFrames.
        
        Parameters:
        -----------
        display : bool, default True
            Whether to display comparison results
            
        Returns:
        --------
        tuple : (comparison_table, summary_dict, type_summary)
        """
        
        if display:
            self._print_header()
        
        # Build comparison data
        comparison_data = [self._build_column_comparison(col) for col in self.all_columns]
        
        # Create and sort comparison table
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = self._sort_comparison_table(comparison_df)
        
        # Calculate summary statistics
        type_matches = sum(1 for col in self.common_columns if self.df1[col].dtype == self.df2[col].dtype)
        type_match_pct = (type_matches / len(self.common_columns) * 100) if self.common_columns else 0
        
        # Display results
        if display:
            self._display_results(comparison_df, type_matches, type_match_pct)
        
        # Prepare outputs
        summary_dict = self._create_summary_dict(comparison_df, type_matches, type_match_pct)
        type_summary = self._get_type_summary()
        
        return comparison_df, summary_dict, type_summary
    
    def _print_header(self):
        """Print comparison header."""
        print(f"{'='*80}\nDATAFRAME COLUMN COMPARISON: {self.df1_name} vs {self.df2_name}\n{'='*80}")
        print(f"\n📊 BASIC INFORMATION\n{'-'*50}")
        print(f"{self.df1_name}: {self.df1.shape[0]:,} rows × {self.df1.shape[1]:,} columns")
        print(f"{self.df2_name}: {self.df2.shape[0]:,} rows × {self.df2.shape[1]:,} columns")
    
    def _build_column_comparison(self, col: str) -> dict:
        """Build comparison data for a single column."""
        in_df1, in_df2 = col in self.df1.columns, col in self.df2.columns
        
        row = {
            'Column': col,
            'In_DF1': '✓' if in_df1 else '✗',
            'In_DF2': '✓' if in_df2 else '✗',
            'Type_Match': self._get_type_match(col, in_df1, in_df2),
            'Value_Commonality_Pct': self._calculate_value_commonality(
                self.df1[col], self.df2[col]) if in_df1 and in_df2 else 'N/A'
        }
        
        # Add DF1 stats
        if in_df1:
            row.update(self._get_column_stats(self.df1[col], 'DF1'))
        else:
            row.update(self._get_empty_stats('DF1'))
        
        # Add DF2 stats
        if in_df2:
            row.update(self._get_column_stats(self.df2[col], 'DF2'))
        else:
            row.update(self._get_empty_stats('DF2'))
        
        return row
    
    def _get_type_match(self, col: str, in_df1: bool, in_df2: bool) -> str:
        """Get type match indicator for a column."""
        if in_df1 and in_df2:
            return '✓' if self.df1[col].dtype == self.df2[col].dtype else '✗'
        return 'N/A'
    
    def _get_column_stats(self, series: pd.Series, prefix: str) -> dict:
        """Get statistics for a column."""
        return {
            f'{prefix}_Type': str(series.dtype),
            f'{prefix}_Memory_MB': series.memory_usage(deep=True) / 1024**2,
            f'{prefix}_Missing_Count': series.isnull().sum(),
            f'{prefix}_Missing_Pct': (series.isnull().sum() / len(series)) * 100
        }
    
    def _get_empty_stats(self, prefix: str) -> dict:
        """Get empty statistics for non-existent column."""
        return {
            f'{prefix}_Type': 'N/A',
            f'{prefix}_Memory_MB': 0,
            f'{prefix}_Missing_Count': 'N/A',
            f'{prefix}_Missing_Pct': 'N/A'
        }
    
    def _calculate_value_commonality(self, series1: pd.Series, series2: pd.Series) -> float:
        """Calculate percentage of matching values between two series."""
        min_length = min(len(series1), len(series2))
        if min_length == 0:
            return 0.0
        
        s1 = series1.iloc[:min_length].reset_index(drop=True)
        s2 = series2.iloc[:min_length].reset_index(drop=True)
        
        try:
            if pd.api.types.is_numeric_dtype(s1) and pd.api.types.is_numeric_dtype(s2):
                s1_num, s2_num = pd.to_numeric(s1, errors='coerce'), pd.to_numeric(s2, errors='coerce')
                both_nan = s1_num.isna() & s2_num.isna()
                both_valid = ~s1_num.isna() & ~s2_num.isna()
                matches = both_nan | (np.isclose(s1_num, s2_num, equal_nan=False) & both_valid)
            else:
                s1_str, s2_str = s1.astype(str), s2.astype(str)
                matches = (s1_str == s2_str) | ((s1_str == 'nan') & (s2_str == 'nan'))
            
            return round((matches.sum() / min_length) * 100, 1)
        except:
            return 0.0
    
    def _sort_comparison_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sort table by commonality (desc) then alphabetically."""
        def sort_key(row):
            commonality = -1 if row['Value_Commonality_Pct'] == 'N/A' else float(row['Value_Commonality_Pct'])
            return (-commonality, str(row['Column']).lower())
        
        sorted_indices = sorted(range(len(df)), key=lambda i: sort_key(df.iloc[i]))
        return df.iloc[sorted_indices].reset_index(drop=True)
    
    def _display_results(self, comparison_df: pd.DataFrame, type_matches: int, type_match_pct: float):
        """Display formatted comparison results."""
        
        # Format display table
        display_df = self._format_display_table(comparison_df)
        
        print(f"\n📋 DETAILED COLUMN COMPARISON\n{display_df.to_string(index=False)}")
        
        # Summary stats
        self._print_summary_stats(type_matches, type_match_pct, comparison_df)
        
        # Memory summary
        self._print_memory_summary(comparison_df)
    
    def _format_display_table(self, comparison_df: pd.DataFrame) -> pd.DataFrame:
        """Format the comparison table for display."""
        display_df = comparison_df.copy()
        
        # Rename columns
        display_df.columns = ['Column', 'In DF1', 'In DF2', 'Type Match', 'Value Match %', 
                             'DF1 Type', 'DF1 Memory (MB)', 'DF1 Missing Count', 'DF1 Missing %',
                             'DF2 Type', 'DF2 Memory (MB)', 'DF2 Missing Count', 'DF2 Missing %']
        
        # Format numbers
        for col in ['DF1 Memory (MB)', 'DF2 Memory (MB)']:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.4f}" if isinstance(x, (int, float)) and x != 0 else str(x))
        
        for col in ['DF1 Missing %', 'DF2 Missing %', 'Value Match %']:
            display_df[col] = display_df[col].apply(
                lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else str(x))
        
        return display_df
    
    def _print_summary_stats(self, type_matches: int, type_match_pct: float, comparison_df: pd.DataFrame):
        """Print summary statistics."""
        print(f"\n📈 COLUMN SUMMARY")
        print(f"Total unique columns: {len(comparison_df)}")
        print(f"Common columns: {len(self.common_columns)}")
        print(f"Only in {self.df1_name}: {len(self.df1_only)}")
        print(f"Only in {self.df2_name}: {len(self.df2_only)}")
        
        if self.common_columns:
            print(f"Common columns with matching types: {type_matches}/{len(self.common_columns)} ({type_match_pct:.1f}%)")
            
            # Average value commonality
            value_commonalities = [float(row['Value_Commonality_Pct']) for _, row in comparison_df.iterrows() 
                                  if row['Value_Commonality_Pct'] != 'N/A']
            if value_commonalities:
                avg_commonality = sum(value_commonalities) / len(value_commonalities)
                print(f"Average value commonality: {avg_commonality:.1f}%")
        
        if self.df1_only:
            print(f"\nColumns only in {self.df1_name}: {sorted(self.df1_only)}")
        if self.df2_only:
            print(f"Columns only in {self.df2_name}: {sorted(self.df2_only)}")
    
    def _print_memory_summary(self, comparison_df: pd.DataFrame):
        """Print memory usage summary."""
        mem1 = comparison_df[comparison_df['DF1_Memory_MB'] != 0]['DF1_Memory_MB'].sum()
        mem2 = comparison_df[comparison_df['DF2_Memory_MB'] != 0]['DF2_Memory_MB'].sum()
        
        print(f"\n🎯 MEMORY USAGE SUMMARY\n{'='*50}")
        print(f"{self.df1_name} total memory: {mem1:.3f} MB")
        print(f"{self.df2_name} total memory: {mem2:.3f} MB")
        
        if mem1 > 0 and mem2 > 0:
            diff_pct = ((mem2 - mem1) / mem1) * 100
            print(f"Memory difference: {diff_pct:+.1f}% ({mem2 - mem1:+.3f} MB)")
    
    def _create_summary_dict(self, comparison_df: pd.DataFrame, type_matches: int, type_match_pct: float) -> dict:
        """Create summary dictionary."""
        return {
            'basic_info': {
                f'{self.df1_name}_shape': self.df1.shape,
                f'{self.df2_name}_shape': self.df2.shape
            },
            'common_columns': list(self.common_columns),
            'df1_only_columns': self.df1_only,
            'df2_only_columns': self.df2_only,
            'type_matches': type_matches,
            'type_match_percentage': type_match_pct,
            'total_memory_df1_mb': comparison_df[comparison_df['DF1_Memory_MB'] != 0]['DF1_Memory_MB'].sum(),
            'total_memory_df2_mb': comparison_df[comparison_df['DF2_Memory_MB'] != 0]['DF2_Memory_MB'].sum()
        }
    
    def _get_type_summary(self) -> pd.DataFrame:
        """Get summary of data types in both DataFrames."""
        df1_types = self.df1.dtypes.value_counts().to_dict()
        df2_types = self.df2.dtypes.value_counts().to_dict()
        all_types = sorted(set(list(df1_types.keys()) + list(df2_types.keys())), key=str)
        
        return pd.DataFrame([
            {
                'Data_Type': str(dtype),
                f'{self.df1_name}_Count': df1_types.get(dtype, 0),
                f'{self.df2_name}_Count': df2_types.get(dtype, 0)
            }
            for dtype in all_types
        ])

# Convenience function for backward compatibility
    def compare_dataframes_columns(df1: pd.DataFrame, df2: pd.DataFrame, 
                                  df1_name: str = "DataFrame 1", df2_name: str = "DataFrame 2",
                                  display: bool = True) -> Tuple[pd.DataFrame, dict, pd.DataFrame]:
        """
        Convenience method for comparing DataFrame columns.
        
        This is a static method that provides a quick interface to DataFrame
        column comparison without explicitly creating a DataFrameComparator instance.
        
        Parameters
        ----------
        df1, df2 : pd.DataFrame
            DataFrames to compare
        df1_name, df2_name : str, optional
            Display names for the DataFrames
        display : bool, default True
            Whether to print comparison results to console
            
        Returns
        -------
        tuple[pd.DataFrame, dict, pd.DataFrame]
            - comparison_table: Detailed column comparison results
            - summary_dict: Summary statistics and metrics
            - type_summary: Data type distribution comparison
            
        Examples
        --------
        >>> results = DataFrameComparator.compare_dataframes_columns(
        ...     train_df, test_df, "Training", "Testing"
        ... )
        >>> comparison_table, summary, type_dist = results
        """
        comparator = DataFrameComparator(df1, df2, df1_name, df2_name)
        return comparator.compare(display)
        

class DataPreprocessor:
    """
    Data preprocessing utilities for machine learning and analysis workflows.
    
    This class provides comprehensive data preprocessing functionality including
    categorical encoding, column unification, data type conversion, and other
    common preprocessing tasks needed in ML pipelines.
    
    Examples
    --------
    >>> preprocessor = DataPreprocessor()
    >>> encoded_df = preprocessor.encode_categorical_features(df, method='cat_codes')
    >>> unified_df1, unified_df2 = preprocessor.unify_dataframe_columns(df1, df2)
    """
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  method: str = 'cat_codes',
                                  handle_missing: str = 'None') -> pd.DataFrame:
        """
        Convert categorical and object features to numeric representations.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with categorical features to encode
        method : {'cat_codes', 'one_hot', 'label_encode'}, default 'cat_codes'
            Encoding method to use:
            - 'cat_codes': Convert to pandas category codes
            - 'one_hot': Use OneHotEncoder from sklearn
            - 'label_encode': Use LabelEncoder from sklearn
        handle_missing : str, default 'None'
            How to handle missing values before encoding
            
        Returns
        -------
        pd.DataFrame
            DataFrame with encoded categorical features
            
        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> df_encoded = preprocessor.encode_categorical_features(
        ...     df, method='cat_codes'
        ... )
        
        Notes
        -----
        This method identifies categorical columns using both 'object' and 'category'
        dtypes. The original function name was 'cat2no' for backward compatibility.
        """
        df_copy = df.copy()
        
        # Identify categorical columns
        cat_columns = (df_copy.select_dtypes(include=['object']).columns.tolist() + 
                      df_copy.select_dtypes(include=['category']).columns.tolist())
        
        if len(cat_columns) == 0:
            return df_copy
            
        print(f'Categorical columns found: {cat_columns}')
        
        if method == 'cat_codes':
            # Convert to category codes
            df_copy[cat_columns] = df_copy[cat_columns].apply(
                lambda x: x.astype('category').cat.codes
            )
        elif method == 'one_hot':
            # Use OneHotEncoder
            df_copy = self._apply_one_hot_encoding(df_copy, cat_columns, handle_missing)
        elif method == 'label_encode':
            # Use LabelEncoder 
            try:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                for col in cat_columns:
                    df_copy[col] = df_copy[col].fillna(handle_missing)
                    df_copy[col] = le.fit_transform(df_copy[col])
            except ImportError:
                print("Warning: sklearn not available, falling back to cat_codes method")
                df_copy = self.encode_categorical_features(df_copy, method='cat_codes')
        else:
            raise ValueError(f"Unknown encoding method: {method}")
            
        return df_copy
    
    def _apply_one_hot_encoding(self, df: pd.DataFrame, cat_columns: list, 
                               handle_missing: str) -> pd.DataFrame:
        """Apply one-hot encoding to categorical columns."""
        df[cat_columns] = df[cat_columns].fillna(handle_missing).astype('category')
        
        try:
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse=False)
            encoded_array = encoder.fit_transform(df[cat_columns])
            
            # Create feature names
            feature_names = encoder.get_feature_names_out(cat_columns)
            encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
            
            # Combine with non-categorical columns
            result_df = pd.concat([df.drop(cat_columns, axis=1), encoded_df], axis=1)
            return result_df
            
        except ImportError:
            print("Warning: sklearn not available, falling back to cat_codes method")
            return self.encode_categorical_features(df, method='cat_codes')
    
    def unify_dataframe_columns(self, df1: pd.DataFrame, df2: pd.DataFrame,
                               df1_name: str = "DataFrame1", df2_name: str = "DataFrame2",
                               fill_value: any = 0) -> tuple:
        """
        Unify column structure between two DataFrames by adding missing columns.
        
        This method ensures both DataFrames have the same columns by adding missing
        columns filled with a specified value. Useful for model training/testing
        where feature sets must match.
        
        Parameters
        ----------
        df1, df2 : pd.DataFrame
            DataFrames to unify
        df1_name, df2_name : str, optional
            Names for logging purposes
        fill_value : any, default 0
            Value to use for newly created columns
            
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame]
            Unified DataFrames with matching column structures
            
        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> train_unified, test_unified = preprocessor.unify_dataframe_columns(
        ...     train_df, test_df, "Training", "Testing"
        ... )
        """
        df1_copy = df1.copy()
        df2_copy = df2.copy()
        
        # Align indices if needed
        if len(df1_copy) == len(df2_copy):
            df1_copy.index = df2_copy.index
        
        def _add_missing_columns(source_df, target_df, source_name, target_name):
            """Add columns that exist in source but not in target."""
            missing_cols = np.setdiff1d(source_df.columns, target_df.columns)
            
            if len(missing_cols) > 0:
                print(f'Adding {len(missing_cols)} missing columns to {target_name} '
                     f'from {source_name}: {list(missing_cols)}')
                
                # Add missing columns with fill_value
                missing_data = pd.DataFrame(
                    fill_value, 
                    index=target_df.index, 
                    columns=missing_cols
                )
                target_df = pd.concat([target_df, missing_data], axis=1)
                
                # Reorder columns to match source
                target_df = target_df[source_df.columns]
                
            return target_df
        
        # Unify in both directions
        df2_unified = _add_missing_columns(df1_copy, df2_copy, df1_name, df2_name)
        df1_unified = _add_missing_columns(df2_unified, df1_copy, df2_name, df1_name)
        
        return df1_unified, df2_unified
    
    @staticmethod
    def convert_dates_to_numeric(dataframe: pd.DataFrame, 
                               date_columns: List[str],
                               time_unit: str = 'M') -> pd.DataFrame:
        """
        Convert date columns to numeric representation based on elapsed time.
        
        This method converts datetime columns to numeric values representing time
        elapsed from the minimum date in the dataset. Useful for machine learning
        models that require numeric input features.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame containing date columns
        date_columns : List[str]
            List of column names containing date values
        time_unit : str, default='M'
            Time unit for conversion ('M' for months, 'D' for days, 'Y' for years)
            
        Returns
        -------
        pd.DataFrame
            DataFrame with date columns converted to numeric values
            
        Raises
        ------
        ValueError
            If specified date columns are not found in the dataframe
            
        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> df_numeric = preprocessor.convert_dates_to_numeric(
        ...     df, ['start_date', 'end_date'], time_unit='M'
        ... )
        
        Notes
        -----
        Missing values are handled by first converting to -1, then back to NaN.
        The conversion is based on the minimum date across all specified columns.
        """
        df_copy = dataframe.copy()
        
        # Find columns that match the date_columns pattern
        matching_columns, _ = inWithReg(date_columns, df_copy.columns.values)
        
        if not matching_columns:
            logger.warning(f"No matching date columns found for pattern: {date_columns}")
            return df_copy
        
        # Convert to datetime
        df_copy[matching_columns] = df_copy[matching_columns].astype('datetime64[ns]')
        
        # Find minimum date across all date columns
        min_date = df_copy[matching_columns].min().min()
        logger.info(f"Converting date features to numeric ({time_unit}): baseline date = {min_date}")
        
        # Convert to numeric based on time difference
        for col in matching_columns:
            time_diff = df_copy[col] - min_date
            
            if time_unit == 'M':
                df_copy[col] = time_diff / np.timedelta64(1, 'M')
            elif time_unit == 'D':
                df_copy[col] = time_diff / np.timedelta64(1, 'D')
            elif time_unit == 'Y':
                df_copy[col] = time_diff / np.timedelta64(1, 'Y')
            else:
                raise ValueError(f"Unsupported time_unit: {time_unit}")
            
            # Handle missing values: convert to NaN, then back to proper format
            df_copy[col] = df_copy[col].fillna(-1).round(2).astype('float64')
            df_copy[col] = df_copy[col].replace(-1, np.nan)
        
        return df_copy
    
    @staticmethod
    def identify_low_variance_features(dataframe: pd.DataFrame, 
                                     variance_threshold: float = 0.0) -> List[str]:
        """
        Identify features with variance below a specified threshold.
        
        Low variance features typically provide little information for machine learning
        models and can be candidates for removal during feature selection.
        
        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame with numeric features
        variance_threshold : float, default=0.0
            Minimum variance threshold. Features with variance <= threshold are flagged
            
        Returns
        -------
        List[str]
            List of column names with low variance
            
        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> low_var_features = preprocessor.identify_low_variance_features(df, 0.01)
        >>> print(f"Low variance features: {low_var_features}")
        
        Notes
        -----
        This method only considers numeric columns and skips missing values in
        variance calculation.
        """
        # Calculate variance for numeric columns only
        numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
        
        if numeric_columns.empty:
            logger.warning("No numeric columns found for variance analysis")
            return []
        
        variance_series = dataframe[numeric_columns].var(skipna=True)
        low_variance_features = list(variance_series[variance_series <= variance_threshold].index)
        
        logger.info(f"Identified {len(low_variance_features)} features with variance <= {variance_threshold}")
        
        return low_variance_features
    
    @staticmethod
    def find_highly_correlated_features(correlation_matrix: pd.DataFrame,
                                      feature_scores: pd.Series,
                                      correlation_threshold: float = 0.9) -> Tuple[List[str], List[str], Dict[str, str]]:
        """
        Identify highly correlated feature pairs and suggest features to drop.
        
        This method finds pairs of features with correlation above a threshold and
        suggests which features to drop based on feature importance scores.
        
        Parameters
        ----------
        correlation_matrix : pd.DataFrame
            Correlation matrix of features
        feature_scores : pd.Series
            Feature importance scores (higher is better)
        correlation_threshold : float, default=0.9
            Correlation threshold above which features are considered highly correlated
            
        Returns
        -------
        Tuple[List[str], List[str], Dict[str, str]]
            - Features recommended for dropping
            - All features involved in high correlations
            - Dictionary mapping correlated feature pairs
            
        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> corr_matrix = df.corr()
        >>> scores = pd.Series([0.1, 0.8, 0.3, 0.9], index=df.columns)
        >>> to_drop, indices, pairs = preprocessor.find_highly_correlated_features(
        ...     corr_matrix, scores, 0.8
        ... )
        
        Notes
        -----
        The method recommends dropping the feature with lower importance score
        from each highly correlated pair.
        """
        if correlation_matrix.empty or feature_scores.empty:
            return [], [], {}
        
        # Get absolute correlation values and upper triangle
        abs_corr_matrix = correlation_matrix.abs()
        upper_triangle = abs_corr_matrix.where(
            np.triu(np.ones(abs_corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation greater than threshold
        high_corr_cols = [col for col in upper_triangle.columns 
                         if any(upper_triangle[col] > correlation_threshold)]
        high_corr_rows = [idx for idx in upper_triangle.index 
                         if any(upper_triangle.loc[idx] > correlation_threshold)]
        
        # Determine which features to drop based on feature scores
        features_to_drop = []
        correlation_indices = []
        
        for col, row in zip(high_corr_cols, high_corr_rows):
            # Compare feature scores and drop the one with lower score
            if col in feature_scores.index and row in feature_scores.index:
                if feature_scores[col] > feature_scores[row]:
                    features_to_drop.append(row)
                else:
                    features_to_drop.append(col)
            else:
                # If scores not available, default to dropping first feature
                features_to_drop.append(col)
            
            correlation_indices.extend([col, row])
        
        # Create mapping of correlated pairs
        high_correlation_pairs = dict(zip(high_corr_cols, high_corr_rows))
        
        # Remove duplicates
        features_to_drop = list(set(features_to_drop))
        correlation_indices = list(set(correlation_indices))
        
        logger.info(f"Found {len(high_correlation_pairs)} highly correlated pairs")
        logger.info(f"Recommending {len(features_to_drop)} features for removal")
        
        return features_to_drop, correlation_indices, high_correlation_pairs
    
    @staticmethod
    def discretize_continuous_features(series: pd.Series,
                                     target_series: pd.Series = None,
                                     method: str = 'cut',
                                     bins: Union[int, List[str]] = 4,
                                     bin_labels: List[str] = None) -> Tuple[pd.Series, List[float]]:
        """
        Discretize continuous variables into categorical bins.
        
        This method converts continuous features into discrete categories using
        various binning strategies. Useful for creating interpretable features
        or meeting algorithm requirements.
        
        Parameters
        ----------
        series : pd.Series
            Continuous series to discretize
        target_series : pd.Series, optional
            Target variable for supervised discretization methods
        method : {'cut', 'qcut', 'tree1', 'tree2'}, default='cut'
            Discretization method:
            - 'cut': Equal-width bins using pd.cut with normalization
            - 'qcut': Equal-frequency bins using pd.qcut  
            - 'tree1': Decision tree with max_depth=1 (requires target_series)
            - 'tree2': Decision tree with max_depth=2 (requires target_series)
        bins : int or List[str], default=4
            Number of bins (for cut/qcut) or bin labels
        bin_labels : List[str], optional
            Custom labels for bins. If None, uses default quartile labels
            
        Returns
        -------
        Tuple[pd.Series, List[float]]
            - Discretized series with bin labels
            - List of bin edges/thresholds used for discretization
            
        Raises
        ------
        ValueError
            If invalid method or insufficient data for tree-based methods
        ImportError
            If scikit-learn is not available for tree-based methods
            
        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> discretized, bins = preprocessor.discretize_continuous_features(
        ...     df['continuous_var'], method='cut', bins=3, 
        ...     bin_labels=['Low', 'Medium', 'High']
        ... )
        
        Notes
        -----
        Tree-based methods use the target variable to find optimal split points.
        The method includes jittering for edge cases with qcut method.
        """
        if bin_labels is None:
            bin_labels = ["Q1", "Q2", "Q3", "Q4"][:bins] if isinstance(bins, int) else ["Q1", "Q2", "Q3", "Q4"]
        
        if method == 'cut':
            try:
                from sklearn.preprocessing import MinMaxScaler
            except ImportError:
                raise ImportError("scikit-learn is required for 'cut' method with normalization")
            
            # Normalize data and apply equal-width binning
            scaler = MinMaxScaler()
            normalized_data = scaler.fit_transform(series.values.reshape(-1, 1)).reshape(-1)
            discretized_series, bin_edges = pd.cut(
                normalized_data, 
                bins=[-0.1, 0.33, 0.66, 1.1], 
                labels=bin_labels[:3], 
                retbins=True
            )
        
        elif method in ['tree1', 'tree2']:
            if target_series is None:
                raise ValueError(f"target_series is required for method '{method}'")
            
            try:
                from sklearn.tree import DecisionTreeClassifier
            except ImportError:
                raise ImportError("scikit-learn is required for tree-based discretization")
            
            max_depth = 1 if method == 'tree1' else 2
            clf = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
            clf.fit(series.to_frame(), target_series)
            
            # Extract split thresholds
            thresholds = clf.tree_.threshold[clf.tree_.threshold != -2]
            
            if len(thresholds) == 0:
                logger.warning("No splits found by decision tree, using quantile-based fallback")
                discretized_series, bin_edges = pd.qcut(
                    series.rank(method='first'), 
                    q=3, 
                    labels=bin_labels[:3], 
                    retbins=True
                )
            else:
                unique_thresholds = np.unique(thresholds)
                bin_edges = np.sort(np.append(
                    unique_thresholds, 
                    [series.max() + 1e-6, series.min() - 1e-6]
                ))
                
                discretized_series = pd.cut(
                    series, 
                    bins=bin_edges, 
                    labels=bin_labels[:len(bin_edges)-1]
                )
        
        elif method == 'qcut':
            # Handle potential duplicates with jittering
            try:
                discretized_series, bin_edges = pd.qcut(
                    series.rank(method='first'), 
                    q=bins if isinstance(bins, int) else len(bin_labels), 
                    labels=bin_labels, 
                    retbins=True
                )
            except ValueError:
                # Fallback with jittering for duplicate edges
                logger.warning("Duplicate bin edges detected, applying jittering")
                jittered_series = series + DataPreprocessor.add_jitter_to_series(series)
                discretized_series, bin_edges = pd.qcut(
                    jittered_series, 
                    q=bins if isinstance(bins, int) else len(bin_labels), 
                    labels=bin_labels, 
                    retbins=True
                )
        
        else:
            raise ValueError(f"Unsupported discretization method: {method}")
        
        # Convert bin edges to float and log results
        bin_edges = [float(x) for x in bin_edges]
        logger.info(f"Discretized {series.name} using {method} method:")
        logger.info(f"Bin edges: {bin_edges}")
        
        return discretized_series, bin_edges
    
    @staticmethod  
    def add_jitter_to_series(series: pd.Series, 
                           noise_reduction: int = 1000000) -> pd.Series:
        """
        Add small random noise (jitter) to a numeric series.
        
        This method adds minimal random variation to break ties and handle edge
        cases in binning operations, particularly useful for qcut operations with
        duplicate values.
        
        Parameters
        ----------
        series : pd.Series
            Input numeric series to add jitter to
        noise_reduction : int, default=1000000
            Factor to reduce noise magnitude. Higher values = less noise
            
        Returns
        -------
        pd.Series
            Series with small random noise added
            
        Examples
        --------
        >>> preprocessor = DataPreprocessor()
        >>> jittered = preprocessor.add_jitter_to_series(df['values'], 500000)
        
        Notes
        -----
        The jitter is scaled based on the series standard deviation to maintain
        the original data distribution while breaking ties.
        """
        if series.empty:
            return series
        
        std_dev = series.std()
        if std_dev == 0:
            return series
        
        # Generate random noise scaled by standard deviation
        noise = (np.random.random(len(series)) * std_dev / noise_reduction) - (std_dev / (2 * noise_reduction))
        
        return series + noise

class StatisticalAnalyzer:
    """
    Statistical analysis utilities for data science and research workflows.
    
    This class provides comprehensive statistical testing functionality including
    hypothesis testing, group comparisons, and other statistical analysis methods
    commonly used in data analysis and research.
    
    Examples
    --------
    >>> analyzer = StatisticalAnalyzer()
    >>> des, res, summary = analyzer.hypothesis_test(df, 'parameter', 'group', ['A', 'B'])
    >>> batch_stats = analyzer.hypothesis_test_batch(df, ['p1', 'p2'], 'group', ['A', 'B'])
    """
    
    def hypothesis_test(self, df: pd.DataFrame, parameter: str, group_column: str, 
                       group_names: list, alpha: float = 0.05, 
                       equal_variances: bool = False, paired: bool = False) -> tuple:
        """
        Perform hypothesis testing between two groups using t-test.
        
        This method compares a continuous parameter between two groups defined
        by a binary grouping variable using statistical t-testing.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing the data
        parameter : str
            Name of the continuous variable to test
        group_column : str
            Name of the binary grouping variable
        group_names : list of str
            Names for the two groups [group1_name, group2_name]
        alpha : float, default 0.05
            Significance level for the test
        equal_variances : bool, default False
            Whether to assume equal variances in the t-test
        paired : bool, default False
            Whether to perform paired t-test
            
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            - Descriptive statistics for both groups
            - Detailed test results including p-values
            - Summary of test conclusions
            
        Examples
        --------
        >>> analyzer = StatisticalAnalyzer()
        >>> desc, results, summary = analyzer.hypothesis_test(
        ...     df, 'score', 'treatment', ['Treatment', 'Control']
        ... )
        
        Raises
        ------
        ImportError
            If researchpy library is not available
        ValueError
            If specified columns don't exist or groups are invalid
        """
        try:
            import researchpy as rp
        except ImportError:
            raise ImportError("researchpy library is required for hypothesis testing. "
                            "Install with: pip install researchpy")
        
        # Input validation
        if parameter not in df.columns:
            raise ValueError(f"Parameter column '{parameter}' not found in DataFrame")
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found in DataFrame")
        if len(group_names) != 2:
            raise ValueError("Exactly two group names must be provided")
            
        # Prepare data
        df_work = df.copy()
        df_work[group_column] = df_work[group_column].astype('bool')
        
        # Split data by groups
        group1_data = df_work[parameter][df_work[group_column]]
        group2_data = df_work[parameter][~df_work[group_column]]
        
        if len(group1_data) == 0 or len(group2_data) == 0:
            raise ValueError("One or both groups are empty")
        
        group1_name, group2_name = group_names[0], group_names[1]
        
        # Perform t-test
        descriptive_stats, test_results = rp.ttest(
            group1_data, group2_data,
            group1_name=group1_name,
            group2_name=group2_name,
            equal_variances=equal_variances,
            paired=paired
        )
        
        # Process results
        test_results = test_results.set_index(test_results.columns[0])
        test_results.columns = [parameter]
        
        # Interpret results
        p_value = test_results.loc['Two side test p value = '][0]
        
        if p_value > alpha:
            conclusion = f"{parameter}: No significant difference between {group1_name} and {group2_name}"
            summary_text = 'no difference'
        elif (test_results.loc['Two side test p value = '][0] <= alpha and 
              test_results.loc['Difference < 0 p value = '][0] <= alpha):
            conclusion = f"{parameter}: {group1_name} is significantly lower than {group2_name}"
            summary_text = 'lower'
        elif (test_results.loc['Two side test p value = '][0] <= alpha and 
              test_results.loc['Difference > 0 p value = '][0] <= alpha):
            conclusion = f"{parameter}: {group1_name} is significantly higher than {group2_name}"
            summary_text = 'higher'
        else:
            conclusion = f"{parameter}: Inconclusive results"
            summary_text = 'inconclusive'
        
        # Add summary to results
        test_results.loc['summary'] = conclusion
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_text, index=[parameter], columns=[group1_name])
        
        return descriptive_stats, test_results, summary_df
    
    def hypothesis_test_batch(self, df: pd.DataFrame, parameters: list, group_column: str, 
                             group_names: list, alpha: float = 0.05) -> tuple:
        """
        Perform hypothesis testing on multiple parameters simultaneously.
        
        This method applies hypothesis testing to multiple continuous variables
        against the same grouping variable, providing batch processing for
        comparative analysis.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing the data
        parameters : list of str
            List of continuous variable names to test
        group_column : str
            Name of the binary grouping variable
        group_names : list of str
            Names for the two groups [group1_name, group2_name]
        alpha : float, default 0.05
            Significance level for all tests
            
        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            - Combined descriptive statistics for all parameters
            - Combined test results for all parameters
            - Combined summary conclusions for all parameters
            
        Examples
        --------
        >>> analyzer = StatisticalAnalyzer()
        >>> stats, tests, summaries = analyzer.hypothesis_test_batch(
        ...     df, ['score1', 'score2', 'score3'], 'treatment', ['Treated', 'Control']
        ... )
        
        Notes
        -----
        This method does not apply multiple testing corrections. Consider using
        methods like Bonferroni correction for multiple comparisons.
        """
        # Initialize result containers
        all_stats = pd.DataFrame()
        all_tests = pd.DataFrame()
        all_summaries = pd.DataFrame()
        
        # Process each parameter
        for param in parameters:
            try:
                param_stats, param_test, param_summary = self.hypothesis_test(
                    df, parameter=param, group_column=group_column, 
                    group_names=group_names, alpha=alpha
                )
                
                # Accumulate results
                all_stats = pd.concat([all_stats, param_stats], axis=0)
                all_tests = pd.concat([all_tests, param_test], axis=1)
                all_summaries = pd.concat([all_summaries, param_summary], axis=0)
                
            except Exception as e:
                print(f"Warning: Error processing parameter '{param}': {str(e)}")
                continue
                
        return all_stats, all_tests, all_summaries


class DataAggregator:
    """
    Data aggregation utilities for grouped analysis and reporting.
    
    This class provides specialized aggregation functions for calculating
    percentages, grouping operations, and other summary statistics commonly
    used in data analysis workflows.
    
    Examples
    --------
    >>> aggregator = DataAggregator()
    >>> percent_df = aggregator.calculate_percentage_aggregation(df, 'group1', 'group2', 'value')
    """
    
    def calculate_percentage_aggregation(self, df: pd.DataFrame, groupby_col1: str, 
                                       groupby_col2: str, sum_column: str) -> pd.DataFrame:
        """
        Calculate percentage distribution within grouped data.
        
        This method computes the percentage contribution of each subgroup within
        larger groups, useful for composition analysis and reporting.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with data to aggregate
        groupby_col1 : str
            Primary grouping column (subgroups)
        groupby_col2 : str  
            Secondary grouping column (major groups)
        sum_column : str
            Column to sum and calculate percentages for
            
        Returns
        -------
        pd.DataFrame
            DataFrame with percentage calculations and original values
            
        Examples
        --------
        >>> aggregator = DataAggregator()
        >>> result = aggregator.calculate_percentage_aggregation(
        ...     sales_df, 'product', 'region', 'revenue'
        ... )
        
        Notes
        -----
        The original function name was 'percent_agg' for backward compatibility.
        """
        # Validate inputs
        required_cols = [groupby_col1, groupby_col2, sum_column]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
            
        # Calculate grouped sums
        grouped_sums = df.groupby(groupby_col1)[sum_column].sum()
        
        # Calculate percentages within each major group
        percent_result = (grouped_sums.groupby(level=groupby_col2)
                         .apply(lambda x: 100 * x / float(x.sum()))
                         .reset_index())
        
        # Rename percentage column
        percent_result.rename(columns={sum_column: f"{sum_column}_percent"}, inplace=True)
        
        # Filter out zero percentages
        percent_result = percent_result[percent_result[f"{sum_column}_percent"] != 0]
        
        # Add original values back
        percent_result[sum_column] = pd.Series(df.groupby(groupby_col1)[sum_column].sum().values)
        
        return percent_result


class DataEncoder:
    """
    Advanced data encoding utilities for machine learning preprocessing.
    
    This class provides various encoding methods for categorical data, including
    one-hot encoding, label encoding, and other transformation techniques
    commonly used in ML pipelines.
    
    Examples
    --------
    >>> encoder = DataEncoder()
    >>> encoded_df = encoder.encode_categorical_data(df, method='OneHotEncoder')
    """
    
    def encode_categorical_data(self, df: pd.DataFrame, 
                               encoding_method: str = 'OneHotEncoder') -> pd.DataFrame:
        """
        Convert categorical data to numerical representations with advanced preprocessing.
        
        This method provides comprehensive categorical encoding with data cleaning,
        normalization, and multiple encoding strategies for ML workflows.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with categorical features
        encoding_method : {'OneHotEncoder', 'LabelEncoder', 'cat_codes'}, default 'OneHotEncoder'
            Encoding strategy to apply:
            - 'OneHotEncoder': Creates binary columns for each category
            - 'LabelEncoder': Assigns integer labels to categories  
            - 'cat_codes': Uses pandas categorical codes
            
        Returns
        -------
        pd.DataFrame
            DataFrame with encoded categorical features
            
        Examples
        --------
        >>> encoder = DataEncoder()
        >>> df_encoded = encoder.encode_categorical_data(df, 'OneHotEncoder')
        
        Notes
        -----
        This method performs string normalization (lowercase, strip whitespace)
        before encoding. The original function name was 'cat2num'.
        """
        df_copy = df.copy()
        
        # Identify categorical columns
        cat_columns = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(cat_columns) == 0:
            print("No categorical columns found")
            return df_copy
            
        # Normalize categorical data (lowercase, strip whitespace)
        df_copy[cat_columns] = df_copy[cat_columns].applymap(
            lambda x: str(x).lower().strip() if not pd.isnull(x) else x
        )
        
        print('Categorical columns found:')
        unique_counts = df_copy[cat_columns].nunique()
        unique_counts.sort_values(inplace=True, ascending=False)
        print(unique_counts)
        
        if encoding_method == 'OneHotEncoder':
            df_copy = self._apply_onehot_encoding(df_copy, cat_columns)
        elif encoding_method == 'LabelEncoder':
            df_copy = self._apply_label_encoding(df_copy, cat_columns)
        elif encoding_method == 'cat_codes':
            df_copy[cat_columns] = df_copy[cat_columns].apply(
                lambda x: x.astype('category').cat.codes
            )
        else:
            raise ValueError(f"Unknown encoding method: {encoding_method}")
            
        return df_copy
    
    def _apply_onehot_encoding(self, df: pd.DataFrame, cat_columns: list) -> pd.DataFrame:
        """Apply one-hot encoding with error handling."""
        # Fill NaN values and convert to category
        df[cat_columns] = df[cat_columns].fillna('None').astype('category')
        
        try:
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse=False)
            encoded_array = encoder.fit_transform(df[cat_columns])
            
            # Get feature names (handle sklearn version differences)
            try:
                feature_names = encoder.get_feature_names_out(cat_columns)
            except AttributeError:
                # Fallback for older sklearn versions
                feature_names = encoder.get_feature_names(cat_columns)
                
            encoded_df = pd.DataFrame(encoded_array, columns=feature_names, index=df.index)
            result = pd.concat([df.drop(cat_columns, axis=1), encoded_df], axis=1)
            
            return result
            
        except ImportError:
            print("Warning: sklearn not available, falling back to cat_codes method")
            return self.encode_categorical_data(df, encoding_method='cat_codes')
    
    def _apply_label_encoding(self, df: pd.DataFrame, cat_columns: list) -> pd.DataFrame:
        """Apply label encoding with error handling."""
        try:
            from sklearn.preprocessing import LabelEncoder
            
            for col in cat_columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].fillna('None'))
                
            return df
            
        except ImportError:
            print("Warning: sklearn not available, falling back to cat_codes method")
            return self.encode_categorical_data(df, encoding_method='cat_codes')


class StringProcessor:
    """
    String processing utilities for data analysis and text preprocessing.
    
    This class provides various string manipulation and encoding functions
    commonly used in data cleaning and text analysis workflows.
    
    Examples
    --------
    >>> processor = StringProcessor()
    >>> encoded = processor.run_length_encode("aaabbbcccaaa")
    """
    
    @staticmethod
    def run_length_encode(data: str) -> str:
        """
        Apply run-length encoding to compress repetitive character sequences.
        
        This method converts consecutive identical characters into a compressed
        format showing character and count, useful for data compression and
        pattern analysis.
        
        Parameters
        ----------
        data : str
            Input string to encode
            
        Returns
        -------
        str
            Run-length encoded string in format 'char(count);char(count);...'
            
        Examples
        --------
        >>> processor = StringProcessor()
        >>> encoded = processor.run_length_encode("aaabbbcccaaa")
        >>> print(encoded)  # Output: 'a(3);b(3);c(3);a(3);'
        
        Notes
        -----
        Reference: https://stackabuse.com/run-length-encoding/
        The original function name was 'rle_encode' for backward compatibility.
        """
        if not data:
            return ''
            
        encoding = ''
        prev_char = ''
        count = 1
        
        for char in data:
            if char != prev_char:
                if prev_char:
                    encoding += f"{prev_char}({count});"
                count = 1
                prev_char = char
            else:
                count += 1
        else:
            # Add the last sequence
            encoding += f"{prev_char}({count});"
            
        return encoding

# -------------------------------------------------------------------------
# Backward compatibility aliases will be added at the end of the file


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def retrieve_variable_name(variable) -> str:
    """
    Retrieve the name of a variable (for debugging purposes).
    
    Parameters
    ----------
    variable : Any
        The variable whose name to retrieve
        
    Returns
    -------
    str
        Variable name as string, or '<unknown>' if not determinable
        
    Examples
    --------
    >>> data = [1, 2, 3]
    >>> retrieve_variable_name(data)  # May return 'data' in some contexts
    '<unknown>'
    """
    import inspect
    
    try:
        frame = inspect.currentframe().f_back
        for name, value in frame.f_locals.items():
            if value is variable:
                return name
    except:
        pass
    
    return '<unknown>'

def join_non_zero_values(values: List[Union[str, int, float]], 
                        separator: str = ', ') -> str:
    """
    Join non-zero values with specified separator.
    
    Parameters
    ----------
    values : list
        List of values to filter and join
    separator : str, default=', '
        String separator for joining values
        
    Returns
    -------
    str
        Joined string of non-zero values
        
    Examples
    --------
    >>> join_non_zero_values([1, 0, 3, 0, 5])
    '1, 3, 5'
    """
    non_zero = [str(val) for val in values if val != 0 and val != '' and val is not None]
    return separator.join(non_zero)

def fetch_azure_key_vault_secret(key_vault_name: str, secret_name: str,
                                platform: str = 'databricks',
                                local_access_config: Optional[Dict] = None) -> str:
    """
    Fetch secret from Azure Key Vault with environment-specific authentication.
    
    This function provides cross-platform Azure Key Vault secret retrieval
    with automatic authentication method selection based on the execution
    environment (Databricks vs local/other platforms).
    
    Parameters
    ----------
    key_vault_name : str
        Name of the Azure Key Vault
    secret_name : str
        Name of the secret to retrieve
    platform : str, default='databricks'
        Execution platform ('databricks' or other)
    local_access_config : dict, optional
        Local access configuration with 'tenant_id', 'client_id', 'client_secret'
        
    Returns
    -------
    str
        Retrieved secret value
        
    Raises
    ------
    ImportError
        If required Azure SDK libraries are not installed
    Exception
        If authentication fails or secret cannot be retrieved
        
    Examples
    --------
    >>> # Databricks environment
    >>> secret = fetch_azure_key_vault_secret('my-vault', 'db-password')
    
    >>> # Local environment with service principal
    >>> config = {
    ...     'tenant_id': 'your-tenant-id',
    ...     'client_id': 'your-client-id', 
    ...     'client_secret': 'your-client-secret'
    ... }
    >>> secret = fetch_azure_key_vault_secret(
    ...     'my-vault', 'db-password', 'local', config
    ... )
    """
    try:
        if platform == 'databricks':
            import IPython
            dbutils = IPython.get_ipython().user_ns["dbutils"]
            return dbutils.secrets.get(scope=key_vault_name, key=secret_name)
        else:
            try:
                from azure.keyvault.secrets import SecretClient
                from azure.identity import DefaultAzureCredential
            except ImportError:
                raise ImportError(
                    "Azure SDK libraries required. Install with: "
                    "pip install azure-keyvault-secrets azure-identity"
                )
            
            try:
                credential = DefaultAzureCredential()
                vault_url = f"https://{key_vault_name}.vault.azure.net/"
                client = SecretClient(vault_url=vault_url, credential=credential)
                secret = client.get_secret(secret_name)
                return secret.value
            except Exception as e:
                if not local_access_config:
                    raise Exception(
                        "Set AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET "
                        "environment variables"
                    )
                
                from azure.identity import ClientSecretCredential
                
                credential = ClientSecretCredential(
                    tenant_id=local_access_config['tenant_id'],
                    client_id=local_access_config['client_id'],
                    client_secret=local_access_config['client_secret']
                )
                
                vault_url = f"https://{key_vault_name}.vault.azure.net/"
                client = SecretClient(vault_url=vault_url, credential=credential)
                secret = client.get_secret(secret_name)
                return secret.value
    except Exception as e:
        raise Exception(f"Error fetching secret from Key Vault: {str(e)}")

# =============================================================================
# FUNCTION MAPPING DOCUMENTATION
# =============================================================================

FUNCTION_MAPPING = {
    # Text Processing Functions
    'normalize_text': 'TextProcessor.normalize_text()',
    'sanitize_filename': 'TextProcessor.sanitize_filename()',
    'clean_column_names': 'TextProcessor.clean_column_names()',
    'find_fuzzy_matches': 'TextProcessor.find_fuzzy_matches()',
    'clean_sql_query': 'TextProcessor.clean_sql_query()',
    
    # List Utilities
    'inWithReg': 'ListUtilities.search_with_regex()',
    'flattenList': 'ListUtilities.flatten_nested_list()', 
    'unique_list': 'ListUtilities.get_unique_ordered()',
    'remove_extra_none': 'ListUtilities.remove_nested_none_values()',
    
    # SQL Processing (NEW)
    'valid_path': 'SQLProcessor.validate_file_path()',
    'split_sql_expressions_sub': 'SQLProcessor.split_sql_statements()',
    'parse_sql_file_sub': 'SQLProcessor.parse_sql_file()',
    
    # Encoding Utilities (NEW)
    'get_dummies2': 'EncodingUtilities.create_optimized_dummy_encoding()',
    'sparseLabel': 'EncodingUtilities.create_sparse_label_encoding()',
    'fill_with_colnames': 'EncodingUtilities.fill_dataframe_with_column_names()',
    'get_dummies2_sub': '[Internal function - now handled within class]',
    'sparseLabel_sub': '[Internal function - now handled within class]',
    
    # Product Utilities (NEW)
    'joinNonZero': 'ProductUtilities.join_non_zero_values()',
    'prodDesc_clean': 'ProductUtilities.clean_product_descriptions()',
    'condense_cols': 'ProductUtilities.condense_dataframe_columns()',
    'current_prds': 'ProductUtilities.extract_current_products()',
    'sortPrds': 'ProductUtilities.sort_prediction_probabilities()',
    
    # DataFrame Operations
    'movecol': 'DataFrameUtilities.reorder_columns()',
    'cellWeight': 'DataFrameUtilities.calculate_cell_proportions()',
    'reduce_mem_usage': 'DataFrameUtilities.reduce_memory_usage()',
    'null_per_column': 'DataFrameUtilities.analyze_missing_values()',
    'unify_columns': 'DataFrameUtilities.unify_columns()',
    
    # Date & Time Operations
    'check_timestamps': 'DateTimeUtilities.validate_timestamp_format()',
    'readableTime': 'DateTimeUtilities.convert_seconds_to_readable()',
    'datesList': 'DateTimeUtilities.generate_date_range()',
    
    # File System Operations
    'check_path': 'FileSystemUtilities.validate_path_exists()',
    'setOutputFolder': 'FileSystemUtilities.setup_output_directory()',
    'extract_zip_archive': 'FileSystemUtilities.extract_zip_archive()',
    'filter_files_for_extraction': 'FileSystemUtilities.filter_files_for_extraction()',
    
    # Visualization Functions
    'sankey': 'DataVisualization.create_sankey_flow_diagram()',
    'plot3D': 'DataVisualization.create_3d_scatter_plot()',
    'worldCloud_graph': 'DataVisualization.generate_word_cloud_visualization()',
    
    # Utility Functions
    'retrieve_name': 'retrieve_variable_name()',
    'joinNonZero': 'join_non_zero_values()',
}

def print_function_mapping():
    """
    Print complete mapping of old function names to new class-based methods.
    
    This function displays the migration guide showing how to update from
    deprecated procedural functions to the new object-oriented class structure.
    """
    print("\n" + "=" * 80)
    print("COMMON FUNCTIONS MIGRATION GUIDE")
    print("=" * 80)
    print("Old Function Name → New Class-Based Method")
    print("-" * 80)
    
    categories = {
        'Text Processing': ['normalize_text', 'sanitize_filename', 'clean_column_names', 'find_fuzzy_matches'],
        'List Operations': ['inWithReg', 'flattenList', 'unique_list', 'remove_extra_none'], 
        'SQL Processing': ['valid_path', 'split_sql_expressions_sub', 'parse_sql_file_sub'],
        'Data Encoding': ['get_dummies2', 'sparseLabel', 'fill_with_colnames', 'get_dummies2_sub', 'sparseLabel_sub'],
        'Product Utilities': ['joinNonZero', 'prodDesc_clean', 'condense_cols', 'current_prds', 'sortPrds'],
        'DataFrame Operations': ['movecol', 'cellWeight', 'reduce_mem_usage', 'null_per_column'],
        'Date & Time': ['check_timestamps', 'readableTime', 'datesList'],
        'File System': ['check_path', 'setOutputFolder', 'extract_zip_archive', 'filter_files_for_extraction'],
        'Visualization': ['sankey', 'plot3D', 'worldCloud_graph'],
        'Utilities': ['retrieve_name']
    }
    
    for category, functions in categories.items():
        print(f"\n📁 {category}:")
        for func in functions:
            if func in FUNCTION_MAPPING:
                print(f"  {func:<30} → {FUNCTION_MAPPING[func]}")
    
    print("\n" + "=" * 80)
    print("USAGE EXAMPLE:")
    print("-" * 80)
    print("# Old way (deprecated)")
    print("result = normalize_text('Hello World!')")
    print("\n# New way (recommended)")
    print("processor = TextProcessor()")
    print("result = processor.normalize_text('Hello World!')")
    print("\n# Or using static method")
    print("result = TextProcessor.normalize_text('Hello World!')")
    print("=" * 80)

# =============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS (DEPRECATED)
# =============================================================================

# Text Processing Functions
def normalize_text(text, remove_spaces=True, lowercase=True, special_chars=r'[^a-zA-Z0-9\s]',
                   replace_with='', max_length=None, fallback_text='unnamed'):
    """DEPRECATED: Use TextProcessor.normalize_text() instead."""
    warnings.warn("normalize_text() is deprecated. Use TextProcessor.normalize_text() instead.", DeprecationWarning, stacklevel=2)
    return TextProcessor.normalize_text(text, remove_spaces, lowercase, special_chars,
                                       replace_with, max_length, fallback_text)

def sanitize_filename(filename, max_length=100):
    """DEPRECATED: Use TextProcessor.sanitize_filename() instead."""
    warnings.warn("sanitize_filename() is deprecated. Use TextProcessor.sanitize_filename() instead.", DeprecationWarning, stacklevel=2)
    return TextProcessor.sanitize_filename(filename, max_length)

def clean_column_names(column_list, replacements=None, lowercase=False):
    """DEPRECATED: Use TextProcessor.clean_column_names() instead."""
    warnings.warn("clean_column_names() is deprecated. Use TextProcessor.clean_column_names() instead.", DeprecationWarning, stacklevel=2)
    if replacements is None:
        replacements = {}
    return TextProcessor.clean_column_names(column_list, replacements, lowercase)

def find_fuzzy_matches(listA, listB, threshold=60):
    """DEPRECATED: Use TextProcessor.find_fuzzy_matches() instead."""
    warnings.warn("find_fuzzy_matches() is deprecated. Use TextProcessor.find_fuzzy_matches() instead.", DeprecationWarning, stacklevel=2)
    return TextProcessor.find_fuzzy_matches(listA, listB, threshold)

def clean_sql_query(query, start_time=None, end_time=None):
    """DEPRECATED: Use TextProcessor.clean_sql_query() instead."""
    warnings.warn("clean_sql_query() is deprecated. Use TextProcessor.clean_sql_query() instead.", DeprecationWarning, stacklevel=2)
    return TextProcessor.clean_sql_query(query, start_time, end_time)

def generate_all_partitions(elements, min_subsets=1, max_subsets=None):
    """
    Backward compatibility wrapper for partition generation.
    
    DEPRECATED: Please use PartitionAnalyzer.generate_all_set_partitions() instead.
    This function will be removed in a future version.
    """
    warnings.warn(
        "generate_all_partitions() is deprecated. Use PartitionAnalyzer.generate_all_set_partitions() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    analyzer = PartitionAnalyzer()
    return analyzer.generate_all_set_partitions(elements, min_subsets, max_subsets)

def partition_to_mapping(partition):
    """
    Backward compatibility wrapper for partition to mapping conversion.
    
    DEPRECATED: Please use PartitionAnalyzer.create_element_grouping_map() instead.
    This function will be removed in a future version.
    """
    warnings.warn(
        "partition_to_mapping() is deprecated. Use PartitionAnalyzer.create_element_grouping_map() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    analyzer = PartitionAnalyzer()
    return analyzer.create_element_grouping_map(partition)

# List Operations
def inWithReg(regLst, LstAll):
    """DEPRECATED: Use ListUtilities.search_with_regex() instead."""
    warnings.warn("inWithReg() is deprecated. Use ListUtilities.search_with_regex() instead.", DeprecationWarning, stacklevel=2)
    return ListUtilities.search_with_regex(regLst, LstAll)

def flattenList(ulist):
    """DEPRECATED: Use ListUtilities.flatten_nested_list() instead."""
    warnings.warn("flattenList() is deprecated. Use ListUtilities.flatten_nested_list() instead.", DeprecationWarning, stacklevel=2)
    return ListUtilities.flatten_nested_list(ulist)

def unique_list(seq):
    """DEPRECATED: Use ListUtilities.get_unique_ordered() instead."""
    warnings.warn("unique_list() is deprecated. Use ListUtilities.get_unique_ordered() instead.", DeprecationWarning, stacklevel=2)
    return ListUtilities.get_unique_ordered(seq)

def remove_extra_none(nested_lst):
    """DEPRECATED: Use ListUtilities.remove_nested_none_values() instead."""
    warnings.warn("remove_extra_none() is deprecated. Use ListUtilities.remove_nested_none_values() instead.", DeprecationWarning, stacklevel=2)
    return ListUtilities.remove_nested_none_values(nested_lst)

# SQL Processing Functions
def valid_path(path):
    """DEPRECATED: Use SQLProcessor.validate_file_path() instead."""
    warnings.warn("valid_path() is deprecated. Use SQLProcessor.validate_file_path() instead.", DeprecationWarning, stacklevel=2)
    return SQLProcessor.validate_file_path(path)

def split_sql_expressions_sub(text):
    """DEPRECATED: Use SQLProcessor.split_sql_statements() instead."""
    warnings.warn("split_sql_expressions_sub() is deprecated. Use SQLProcessor.split_sql_statements() instead.", DeprecationWarning, stacklevel=2)
    return SQLProcessor.split_sql_statements(text)

def parse_sql_file_sub(file_name):
    """DEPRECATED: Use SQLProcessor.parse_sql_file() instead."""
    warnings.warn("parse_sql_file_sub() is deprecated. Use SQLProcessor.parse_sql_file() instead.", DeprecationWarning, stacklevel=2)
    return SQLProcessor.parse_sql_file(file_name)

def get_dummies2(df, splitter='; '):
    """DEPRECATED: Use EncodingUtilities.create_optimized_dummy_encoding() instead."""
    warnings.warn("get_dummies2() is deprecated. Use EncodingUtilities.create_optimized_dummy_encoding() instead.", DeprecationWarning, stacklevel=2)
    return EncodingUtilities.create_optimized_dummy_encoding(df, splitter)

def sparseLabel(df, prodCol, priceCol, splitter="; "):
    """DEPRECATED: Use EncodingUtilities.create_sparse_label_encoding() instead."""
    warnings.warn("sparseLabel() is deprecated. Use EncodingUtilities.create_sparse_label_encoding() instead.", DeprecationWarning, stacklevel=2)
    return EncodingUtilities.create_sparse_label_encoding(df, prodCol, priceCol, splitter)

def fill_with_colnames(udata):
    """DEPRECATED: Use EncodingUtilities.fill_dataframe_with_column_names() instead."""
    warnings.warn("fill_with_colnames() is deprecated. Use EncodingUtilities.fill_dataframe_with_column_names() instead.", DeprecationWarning, stacklevel=2)
    return EncodingUtilities.fill_dataframe_with_column_names(udata)

# Product Utility Functions  
def joinNonZero(x, sep=', '):
    """DEPRECATED: Use ProductUtilities.join_non_zero_values() instead."""
    warnings.warn("joinNonZero() is deprecated. Use ProductUtilities.join_non_zero_values() instead.", DeprecationWarning, stacklevel=2)
    return ProductUtilities.join_non_zero_values(x, sep)

def prodDesc_clean(prodDesc, df):
    """DEPRECATED: Use ProductUtilities.clean_product_descriptions() instead."""
    warnings.warn("prodDesc_clean() is deprecated. Use ProductUtilities.clean_product_descriptions() instead.", DeprecationWarning, stacklevel=2)
    return ProductUtilities.clean_product_descriptions(prodDesc, df)

def condense_cols(df, remove_prefix, umap):
    """DEPRECATED: Use ProductUtilities.condense_dataframe_columns() instead."""
    warnings.warn("condense_cols() is deprecated. Use ProductUtilities.condense_dataframe_columns() instead.", DeprecationWarning, stacklevel=2)
    return ProductUtilities.condense_dataframe_columns(df, remove_prefix, umap)

def current_prds(uData, lob):
    """DEPRECATED: Use ProductUtilities.extract_current_products() instead."""
    warnings.warn("current_prds() is deprecated. Use ProductUtilities.extract_current_products() instead.", DeprecationWarning, stacklevel=2)
    return ProductUtilities.extract_current_products(uData, lob)

def sortPrds(x, y):
    """DEPRECATED: Use ProductUtilities.sort_prediction_probabilities() instead."""
    warnings.warn("sortPrds() is deprecated. Use ProductUtilities.sort_prediction_probabilities() instead.", DeprecationWarning, stacklevel=2)
    return ProductUtilities.sort_prediction_probabilities(x, y)

# DataFrame Operations  
def movecol(df, cols_to_move=None, ref_col='', place='After'):
    """DEPRECATED: Use DataFrameUtilities.reorder_columns() instead."""
    warnings.warn("movecol() is deprecated. Use DataFrameUtilities.reorder_columns() instead.", DeprecationWarning, stacklevel=2)
    if cols_to_move is None:
        cols_to_move = []
    return DataFrameUtilities.reorder_columns(df, cols_to_move, ref_col, place)

def cellWeight(df, axis=0):
    """DEPRECATED: Use DataFrameUtilities.calculate_cell_proportions() instead."""
    warnings.warn("cellWeight() is deprecated. Use DataFrameUtilities.calculate_cell_proportions() instead.", DeprecationWarning, stacklevel=2)
    return DataFrameUtilities.calculate_cell_proportions(df, axis)

def reduce_mem_usage(df, obj2str_cols='all_columns', str2cat_cols='all_columns', verbose=True):
    """DEPRECATED: Use DataFrameUtilities.reduce_memory_usage() instead."""
    warnings.warn("reduce_mem_usage() is deprecated. Use DataFrameUtilities.reduce_memory_usage() instead.", DeprecationWarning, stacklevel=2)
    return DataFrameUtilities.reduce_memory_usage(df, obj2str_cols, str2cat_cols, verbose)

def null_per_column(df):
    """DEPRECATED: Use DataFrameUtilities.analyze_missing_values() instead."""
    warnings.warn("null_per_column() is deprecated. Use DataFrameUtilities.analyze_missing_values() instead.", DeprecationWarning, stacklevel=2)
    return DataFrameUtilities.analyze_missing_values(df)

# Date & Time Operations
def check_timestamps(start, end, format_required='%Y-%m-%d'):
    """DEPRECATED: Use DateTimeUtilities.validate_timestamp_format() instead."""
    warnings.warn("check_timestamps() is deprecated. Use DateTimeUtilities.validate_timestamp_format() instead.", DeprecationWarning, stacklevel=2)
    return DateTimeUtilities.validate_timestamp_format(start, end, format_required)

def readableTime(time):
    """DEPRECATED: Use DateTimeUtilities.convert_seconds_to_readable() instead."""
    warnings.warn("readableTime() is deprecated. Use DateTimeUtilities.convert_seconds_to_readable() instead.", DeprecationWarning, stacklevel=2)
    return DateTimeUtilities.convert_seconds_to_readable(time)

def datesList(range_date__year=None, firstDate=None, lastDate=None, month_step=1):
    """DEPRECATED: Use DateTimeUtilities.generate_date_range() instead."""
    warnings.warn("datesList() is deprecated. Use DateTimeUtilities.generate_date_range() instead.", DeprecationWarning, stacklevel=2)
    if range_date__year is None:
        range_date__year = [2018, 2099]
    end_year = range_date__year[1] if len(range_date__year) > 1 else 2099
    return DateTimeUtilities.generate_date_range(
        range_date__year[0], end_year, firstDate, lastDate, month_step
    )

# File System Operations
def check_path(path):
    """DEPRECATED: Use FileSystemUtilities.validate_path_exists() instead."""
    warnings.warn("check_path() is deprecated. Use FileSystemUtilities.validate_path_exists() instead.", DeprecationWarning, stacklevel=2)
    return FileSystemUtilities.validate_path_exists(path)

def setOutputFolder(outputFolder, uFiles, overWrite):
    """DEPRECATED: Use FileSystemUtilities.setup_output_directory() instead."""
    warnings.warn("setOutputFolder() is deprecated. Use FileSystemUtilities.setup_output_directory() instead.", DeprecationWarning, stacklevel=2)
    return FileSystemUtilities.setup_output_directory(outputFolder, uFiles, overWrite)

def extract_zip_archive(download_directory, zip_filename, extract_folders=None, exclude_folders=None):
    """DEPRECATED: Use FileSystemUtilities.extract_zip_archive() instead."""
    warnings.warn("extract_zip_archive() is deprecated. Use FileSystemUtilities.extract_zip_archive() instead.", DeprecationWarning, stacklevel=2)
    return FileSystemUtilities.extract_zip_archive(download_directory, zip_filename, extract_folders, exclude_folders)

def filter_files_for_extraction(all_files, extract_folders=None, exclude_folders=None):
    """DEPRECATED: Use FileSystemUtilities.filter_files_for_extraction() instead."""
    warnings.warn("filter_files_for_extraction() is deprecated. Use FileSystemUtilities.filter_files_for_extraction() instead.", DeprecationWarning, stacklevel=2)
    return FileSystemUtilities.filter_files_for_extraction(all_files, extract_folders, exclude_folders)

# Utility Functions
def retrieve_name(var):
    """DEPRECATED: Use retrieve_variable_name() instead."""
    warnings.warn("retrieve_name() is deprecated. Use retrieve_variable_name() instead.", DeprecationWarning, stacklevel=2)
    return retrieve_variable_name(var)

# Visualization Functions
def sankey(left, right, value, thershold, utitle, filename):
    """DEPRECATED: Use DataVisualization.create_sankey_flow_diagram() instead."""
    warnings.warn(
        "sankey() is deprecated. Use DataVisualization.create_sankey_flow_diagram() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return DataVisualization.create_sankey_flow_diagram(
        source_series=left,
        target_series=right,
        flow_values=value,
        minimum_flow_threshold=thershold,
        chart_title=utitle,
        output_filepath=filename
    )

def worldCloud_graph(txtSeries_df, outputFile):
    """DEPRECATED: Use DataVisualization.generate_word_cloud_visualization() instead."""
    warnings.warn(
        "worldCloud_graph() is deprecated. Use DataVisualization.generate_word_cloud_visualization() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    DataVisualization.generate_word_cloud_visualization(
        text_data=txtSeries_df,
        output_filepath=outputFile
    )

def retrieve_name(var):
    """
    Get the name of a variable as a string.
    
    **DEPRECATED**: This function has been moved to the utilities section
    with enhanced functionality. Please use:
    `retrieve_variable_name()` instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "retrieve_name() is deprecated and will be removed in a future version. "
        "Please use retrieve_variable_name() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return retrieve_variable_name(var)

def create_venn_diagram(listA, listB, similarity_threshold=60, listA_name='List A', listB_name='List B', utitle='Venn Diagram - List Comparison with Fuzzy Matching', save_path=None):
    """
    Create a Venn diagram showing the relationship between two lists.
    
    **DEPRECATED**: This function has been moved to the ComparativeVisualization class
    with enhanced functionality. Please use:
    `ComparativeVisualization.create_fuzzy_matching_venn_diagram()` instead.
    
    This function will be removed in a future version.
    """
    warnings.warn(
        "create_venn_diagram() is deprecated and will be removed in a future version. "
        "Please use ComparativeVisualization.create_fuzzy_matching_venn_diagram() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    visualizer = ComparativeVisualization()
    return visualizer.create_fuzzy_matching_venn_diagram(
        listA, listB, similarity_threshold, 
        listA_name, listB_name, utitle, save_path
    )

def merge_between(df1, df2, groupCol, closed="both"):
    """
    DEPRECATED: Use DataFrameMerger.merge_dataframes_between_dates() instead.
    
    Merge DataFrames based on date intervals using IntervalIndex.
    """
    import warnings
    warnings.warn(
        "merge_between() is deprecated. Use DataFrameMerger.merge_dataframes_between_dates() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    merger = DataFrameMerger()
    return merger.merge_dataframes_between_dates(df1, df2, group_col=groupCol, closed=closed)

def cat2no(df):
    """DEPRECATED: Use DataPreprocessor.encode_categorical_features() instead."""
    import warnings
    warnings.warn(
        "cat2no() is deprecated. Use DataPreprocessor.encode_categorical_features() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    preprocessor = DataPreprocessor()
    return preprocessor.encode_categorical_features(df, method='cat_codes')

def unify_cols(df1, df2, df1_name, df2_name):
    """DEPRECATED: Use DataPreprocessor.unify_dataframe_columns() instead."""
    import warnings
    warnings.warn(
        "unify_cols() is deprecated. Use DataPreprocessor.unify_dataframe_columns() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    preprocessor = DataPreprocessor()
    return preprocessor.unify_dataframe_columns(df1, df2, df1_name, df2_name)

def hypothesis_test(df, par, group, group_names):
    """DEPRECATED: Use StatisticalAnalyzer.hypothesis_test() instead."""
    import warnings
    warnings.warn(
        "hypothesis_test() is deprecated. Use StatisticalAnalyzer.hypothesis_test() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    analyzer = StatisticalAnalyzer()
    return analyzer.hypothesis_test(df, par, group, group_names)

def hypothesis_test_batch_pars(df, pars, group, group_names):
    """DEPRECATED: Use StatisticalAnalyzer.hypothesis_test_batch() instead."""
    import warnings
    warnings.warn(
        "hypothesis_test_batch_pars() is deprecated. Use StatisticalAnalyzer.hypothesis_test_batch() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    analyzer = StatisticalAnalyzer()
    return analyzer.hypothesis_test_batch(df, pars, group, group_names)

def percent_agg(df, grpby1, grpby2, sumCol):
    """DEPRECATED: Use DataAggregator.calculate_percentage_aggregation() instead."""
    import warnings
    warnings.warn(
        "percent_agg() is deprecated. Use DataAggregator.calculate_percentage_aggregation() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    aggregator = DataAggregator()
    return aggregator.calculate_percentage_aggregation(df, grpby1, grpby2, sumCol)

def rle_encode(data):
    """DEPRECATED: Use StringProcessor.run_length_encode() instead."""
    import warnings
    warnings.warn(
        "rle_encode() is deprecated. Use StringProcessor.run_length_encode() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return StringProcessor.run_length_encode(data)

def cat2num(df, cat_decoder='OneHotEncoder'):
    """DEPRECATED: Use DataEncoder.encode_categorical_data() instead."""
    import warnings
    warnings.warn(
        "cat2num() is deprecated. Use DataEncoder.encode_categorical_data() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    encoder = DataEncoder()
    return encoder.encode_categorical_data(df, cat_decoder)

def flexible_join(left_df, right_df, left_on=None, right_on=None, on=None, how='inner', **kwargs):
    """DEPRECATED: Use DataFrameMerger.flexible_join() instead."""
    import warnings
    warnings.warn(
        "flexible_join() is deprecated. Use DataFrameMerger.flexible_join() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    merger = DataFrameMerger()
    return merger.flexible_join(left_df, right_df, left_on, right_on, on, how, **kwargs)


  
# =============================================================================
# TESTING AND EXAMPLES
# =============================================================================

if __name__ == "__main__":
    # Comprehensive testing
    print("=" * 60)
    print("COMMON FUNCTIONS REFACTORED - COMPREHENSIVE TESTING")
    print("=" * 60)
    
    # Test TextProcessor
    print("\n1. TextProcessor Tests:")
    print("-" * 30)
    processor = TextProcessor()
    test_text = "Hello World! @#$%"
    print(f"Original: '{test_text}'")
    print(f"Normalized: '{processor.normalize_text(test_text)}'")
    
    columns = ['First Name', 'Email@Domain', '2nd_Score']
    print(f"Column cleaning: {processor.clean_column_names(columns)}")
    
    # Test ListUtilities  
    print("\n2. ListUtilities Tests:")
    print("-" * 30)
    nested_list = [1, [2, None, 3], [4, [5, 6]], None]
    flattened = ListUtilities.flatten_nested_list(nested_list)
    print(f"Flattened: {flattened}")
    
    # Test SQLProcessor (NEW)
    print("\n3. SQLProcessor Tests:")
    print("-" * 30)
    test_sql = "SELECT * FROM users; -- Comment\nINSERT INTO logs VALUES ('test');"
    statements = SQLProcessor.split_sql_statements(test_sql)
    print(f"SQL statements parsed: {len(statements)}")
    
    # Test EncodingUtilities (NEW)
    print("\n4. EncodingUtilities Tests:")
    print("-" * 30)
    test_data = pd.Series(['cat1; cat2', 'cat2; cat3', 'cat1'])
    encoded = EncodingUtilities.create_optimized_dummy_encoding(test_data)
    print(f"Dummy encoding shape: {encoded.shape}")
    
    # Test ProductUtilities (NEW)
    print("\n5. ProductUtilities Tests:")
    print("-" * 30)
    test_values = [1, 0, 'test', '', 5]
    joined = ProductUtilities.join_non_zero_values(test_values)
    print(f"Joined non-zero: '{joined}'")
    
    # Test DataFrameUtilities
    print("\n6. DataFrameUtilities Tests:")
    print("-" * 30)
    test_df = pd.DataFrame({
        'A': [1, 2, None, 4, 5],
        'B': [10.0, 20.0, 30.0, 40.0, 50.0],  
        'C': ['x', 'y', 'z', 'x', 'y']
    })
    
    print(f"Original DataFrame shape: {test_df.shape}")
    missing_analysis = DataFrameUtilities.analyze_missing_values(test_df)
    print(f"Missing analysis shape: {missing_analysis.shape}")
    
    # Test DateTimeUtilities
    print("\n7. DateTimeUtilities Tests:")
    print("-" * 30)
    is_valid = DateTimeUtilities.validate_timestamp_format('2023-01-01', '2023-12-31')
    print(f"Timestamp validation: {is_valid}")
    
    # Test backward compatibility
    print("\n8. Backward Compatibility Tests:")
    print("-" * 30)
    old_result = movecol(test_df, ['C'], 'A', 'After')
    new_result = DataFrameUtilities.reorder_columns(test_df, ['C'], 'A', 'After')
    print(f"Results identical: {list(old_result.columns) == list(new_result.columns)}")
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    
    print("\n📋 ENHANCED CLASSES SUMMARY:")
    print("-" * 40)
    print("✅ TextProcessor: Text normalization, cleaning, fuzzy matching")  
    print("✅ ListUtilities: List operations, flattening, regex search")
    print("✅ SQLProcessor: SQL parsing, statement splitting, file processing [NEW]")
    print("✅ EncodingUtilities: Advanced dummy encoding, sparse labels [NEW]") 
    print("✅ ProductUtilities: Product analysis, business logic processing [NEW]")
    print("✅ DataFrameUtilities: DataFrame manipulation, optimization, analysis")
    print("✅ DateTimeUtilities: Date validation, time conversion, range generation")
    print("✅ FileSystemUtilities: Path validation, directory setup")
    print("✅ DataVisualization: Sankey diagrams, 3D plots, word clouds")
    print("✅ Complete backward compatibility with deprecation warnings")
    print("✅ Enhanced error handling and comprehensive documentation")
    print("✅ 40+ new and refactored functions organized into logical classes")
    
    # Print function mapping
    print_function_mapping()