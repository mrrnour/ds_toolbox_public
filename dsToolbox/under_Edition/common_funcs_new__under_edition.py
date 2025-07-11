"""
Common Functions Module for Data Science Toolbox

This module provides utility functions and classes for common data science tasks including:
- Text processing and normalization
- Data manipulation and analysis
- Statistical analysis
- Visualization
- File operations
- Logging utilities
- Web scraping

Author: Reza Nourzadeh - reza.nourzadeh@gmail.com
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
from typing import List, Tuple, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass

# Plotting imports
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns

# Statistical analysis imports
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import chi2, f_classif
from difflib import SequenceMatcher
import scipy.stats as stats
import scipy


class TextProcessor:
    """
    A class for text processing and normalization operations.
    
    This class provides methods for cleaning, normalizing, and comparing text data,
    including fuzzy matching capabilities.
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
        Flexible text normalization function.
        
        Args:
            text: Text to normalize
            remove_spaces: Remove spaces from text
            lowercase: Convert to lowercase
            special_chars: Regex pattern for characters to match
            replace_with: String to replace matched characters with
            max_length: Maximum length of the output text
            fallback_text: Text to return if result is empty
            
        Returns:
            Normalized text string
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
        
        # Trim text if max_length is specified
        if max_length is not None and len(text) > max_length:
            text = text[:max_length].strip()
        
        # Handle empty or whitespace-only results
        if not text or text.isspace():
            text = fallback_text
        
        return text

    @staticmethod
    def sanitize_filename(filename: str, max_length: int = 100) -> str:
        """
        Sanitize a filename using the normalize_text function.
        
        Args:
            filename: Original filename
            max_length: Maximum length for the filename
            
        Returns:
            Sanitized filename safe for filesystem use
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
        Clean a list of strings to be suitable for use as column names.
        
        Args:
            column_list: List of strings to clean
            replacements: Dictionary of string replacements to apply
            lowercase: Whether to convert to lowercase
            
        Returns:
            List of cleaned column names
        """
        if replacements is None:
            replacements = {}
            
        cleaned_columns = []
        
        for col in column_list:
            col = str(col)
            
            # Apply custom replacements first
            for old, new in replacements.items():
                col = col.replace(old, new)
            
            # Use normalize_text for most of the work
            col = TextProcessor.normalize_text(
                col,
                remove_spaces=True,
                lowercase=lowercase,
                special_chars=r'[^a-zA-Z0-9_]',
                replace_with='_',
                fallback_text='unnamed_column'
            )
            
            # Collapse multiple underscores and strip leading/trailing ones
            col = re.sub(r'_+', '_', col).strip('_')
            
            # Ensure column doesn't start with a number
            if col and col[0].isdigit():
                col = 'col_' + col
            
            # Final fallback
            if not col:
                col = 'unnamed_column'
            
            cleaned_columns.append(col)
        
        return cleaned_columns

    @staticmethod
    def find_fuzzy_matches(listA: List[str], listB: List[str], 
                          threshold: float = 60) -> Dict[str, Dict[str, Any]]:
        """
        Find fuzzy matches between two lists based on normalized text similarity.
        
        Args:
            listA: First list of data
            listB: Second list of data
            threshold: Similarity threshold percentage
            
        Returns:
            Dictionary with match information
        """
        matches = {}
        used_b_indices = set()
        
        for i, item_a in enumerate(listA):
            normalized_a = TextProcessor.normalize_text(item_a)
            best_match = None
            best_similarity = 0
            best_index = -1
            
            for j, item_b in enumerate(listB):
                if j in used_b_indices:
                    continue
                    
                normalized_b = TextProcessor.normalize_text(item_b)
                similarity = SequenceMatcher(None, normalized_a, normalized_b).ratio() * 100
                
                if similarity >= threshold and similarity > best_similarity:
                    best_match = item_b
                    best_similarity = similarity
                    best_index = j
            
            if best_match:
                matches[item_a] = {
                    'match': best_match,
                    'similarity': best_similarity,
                    'normalized_a': normalized_a,
                    'normalized_b': TextProcessor.normalize_text(best_match)
                }
                used_b_indices.add(best_index)
        
        return matches


class ListProcessor:
    """
    A class for list and array processing operations.
    
    This class provides methods for manipulating lists, arrays, and sequences
    including flattening, deduplication, and filtering operations.
    """
    
    @staticmethod
    def in_with_regex(reg_list: Union[str, List[str]], 
                     list_all: List[str]) -> Tuple[List[str], np.ndarray]:
        """
        Search regular expression list in a list.
        
        Args:
            reg_list: List of strings with regex patterns or single regex string
            list_all: List of strings to be searched
            
        Returns:
            Tuple of (matched_items, boolean_indices)
            
        Example:
            reg_list = ['.vol_flag$', 'fefefre', '_date']
            list_all = ['bi_alt_account_id', 'snapshot_date', 'snapshot_year', 
                       'tv_vol_flag', 'phone_vol_flag']
            out, ind = in_with_regex(reg_list, list_all)
            # out = ['tv_vol_flag', 'phone_vol_flag', 'snapshot_date']
            # ind = [False, True, False, True, True]
        """
        out = []
        if not isinstance(reg_list, list):
            reg_list = [reg_list]
            
        for pattern in reg_list:
            tmp = list(filter(re.compile(pattern).search, list_all))
            out.extend(tmp)
            
        ind = np.in1d(list_all, out)
        return out, ind

    @staticmethod
    def flatten_list(nested_list: List[Any]) -> List[Any]:
        """
        Make a flat list out of list of lists.
        
        Args:
            nested_list: A list of nested lists
            
        Returns:
            Flattened list
        """
        results = []
        for item in nested_list:
            if isinstance(item, list):
                results.extend(item)
                results = ListProcessor.flatten_list(results)
            else:
                results.append(item)
        return results

    @staticmethod
    def unique_list(seq: List[Any]) -> List[Any]:
        """
        Get unique values from a list while preserving order.
        
        Args:
            seq: A list with duplicate elements
            
        Returns:
            List with unique elements in original order
        """
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    @staticmethod
    def remove_extra_none(nested_list: List[Any]) -> List[Any]:
        """
        Remove extra 'None' values from a list, keeping only one if it's the only element.
        
        Args:
            nested_list: List that may contain 'None' values
            
        Returns:
            Cleaned list with extra 'None' values removed
        """
        items = list(dict.fromkeys(nested_list))
        if ('None' in items) and (len(items) > 1):
            items.remove('None')
        return items


class DataFrameProcessor:
    """
    A class for DataFrame processing and manipulation operations.
    
    This class provides methods for DataFrame operations including column manipulation,
    memory optimization, and data type conversions.
    """
    
    @staticmethod
    def move_columns(df: pd.DataFrame, 
                    cols_to_move: List[str] = None, 
                    ref_col: str = '', 
                    place: str = 'After') -> pd.DataFrame:
        """
        Reorder DataFrame columns by moving specified columns relative to a reference column.
        
        Args:
            df: Input DataFrame
            cols_to_move: List of columns to move
            ref_col: Reference column name
            place: 'After' or 'Before' - where to place the moved columns
            
        Returns:
            DataFrame with reordered columns
        """
        if cols_to_move is None:
            cols_to_move = []
            
        cols = df.columns.tolist()
        
        if place == 'After':
            seg1 = cols[:list(cols).index(ref_col) + 1]
            seg2 = cols_to_move
        elif place == 'Before':
            seg1 = cols[:list(cols).index(ref_col)]
            seg2 = cols_to_move + [ref_col]
        else:
            raise ValueError("place must be 'After' or 'Before'")
        
        seg1 = [i for i in seg1 if i not in seg2]
        seg3 = [i for i in cols if i not in seg1 + seg2]
        
        return df[seg1 + seg2 + seg3]

    @staticmethod
    def reduce_memory_usage(df: pd.DataFrame, 
                           obj2str_cols: Union[str, List[str]] = 'all_columns',
                           str2cat_cols: Union[str, List[str]] = 'all_columns',
                           use_float16: bool = False, 
                           verbose: bool = False) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage by converting data types.
        
        Args:
            df: Input DataFrame
            obj2str_cols: Columns to convert from object to string
            str2cat_cols: Columns to convert from string to category
            use_float16: Whether to use float16 (can cause precision loss)
            verbose: Whether to print conversion details
            
        Returns:
            DataFrame with optimized memory usage
        """
        from pandas.api.types import is_datetime64_any_dtype as is_datetime

        start_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f'Memory usage of dataframe is {start_mem:.2f} MB')
        
        for col in df.columns:
            col_type = df[col].dtype
            
            # Skip datetime columns
            if is_datetime(df[col]):
                continue
                
            # Handle numeric columns
            if pd.api.types.is_numeric_dtype(df[col]):
                if verbose: 
                    print(f"{col}: compressing numeric column from {col_type}")
                
                c_min = df[col].min()
                c_max = df[col].max()
                
                # Handle integer columns
                if pd.api.types.is_integer_dtype(df[col]):
                    if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                        
                # Handle float columns
                elif pd.api.types.is_float_dtype(df[col]):
                    if np.isfinite(df[col]).all():
                        if use_float16 and c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                            if np.abs(c_max - c_min) < 65504:  # float16 max value
                                df[col] = df[col].astype(np.float16)
                        elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                            df[col] = df[col].astype(np.float32)
            
            # Handle object columns
            elif col_type == object:
                obj2str = False
                if (col in obj2str_cols) or (obj2str_cols == 'all_columns'):
                    df[col] = df[col].astype('string')
                    obj2str = True
                    if verbose:
                        print(f"{col}: object --> string")
                
                # Convert to category if specified
                if ((df[col].dtype == 'string' or obj2str) and 
                    ((col in str2cat_cols) or (str2cat_cols == 'all_columns'))):
                    df[col] = df[col].astype('category')
                    if verbose:
                        if obj2str:
                            print(f"{col}: object --> string --> category")
                        else:
                            print(f"{col}: string --> category")
            
            # Handle string columns that weren't converted from object
            elif pd.api.types.is_string_dtype(df[col]):
                if (col in str2cat_cols) or (str2cat_cols == 'all_columns'):
                    df[col] = df[col].astype('category')
                    if verbose:
                        print(f"{col}: string --> category")
        
        end_mem = df.memory_usage(deep=True).sum() / 1024**2
        print(f'Memory usage after optimization is: {end_mem:.2f} MB')
        reduction = 100 * (start_mem - end_mem) / start_mem
        print(f'Decreased by {reduction:.1f}%')
        
        return df

    @staticmethod
    def categorical_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all categorical and object features to numeric codes.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with categorical columns converted to numeric codes
        """
        cat_columns = (df.select_dtypes(include=['object']).columns.tolist() + 
                      df.select_dtypes(include=['category']).columns.tolist())
        
        if len(cat_columns) != 0:
            print(f'Categorical columns: {cat_columns}')
            df[cat_columns] = df[cat_columns].apply(
                lambda x: x.astype('category').cat.codes)
        
        return df

    @staticmethod
    def null_percentage_per_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate null percentage for each column in DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with null percentages sorted by percentage
        """
        null_per = df.isnull().sum() / df.shape[0] * 100
        null_per = pd.DataFrame(null_per, columns=["null_percent"])
        null_per = (null_per.reset_index()
                   .sort_values(by=["null_percent", "index"], ascending=False)
                   .set_index("index"))
        return null_per


class StatisticalAnalyzer:
    """
    A class for statistical analysis operations.
    
    This class provides methods for hypothesis testing, correlation analysis,
    and other statistical computations.
    """
    
    @staticmethod
    def hypothesis_test(df: pd.DataFrame, 
                       parameter: str, 
                       group: str, 
                       group_names: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Perform hypothesis test between two groups.
        
        Args:
            df: Input DataFrame
            parameter: Column name for the parameter to test
            group: Column name for the grouping variable
            group_names: List of two group names
            
        Returns:
            Tuple of (descriptive_stats, test_results, summary)
        """
        try:
            import researchpy as rp
        except ImportError:
            raise ImportError("researchpy package is required for hypothesis testing")
        
        df[group] = df[group].astype('bool')
        X1 = df[parameter][df[group]]
        X2 = df[parameter][~df[group]]
        
        group1_name, group2_name = group_names[0], group_names[1]
        des, res = rp.ttest(X1, X2,
                           group1_name=group1_name,
                           group2_name=group2_name,
                           equal_variances=False,
                           paired=False)
        
        res = res.set_index(res.columns[0])
        res.columns = [parameter]

        # Interpret results
        if res.loc['Two side test p value = '][0] != 0:
            txt = f"{parameter}: There is no difference between {group1_name} and {group2_name}"
            txt2 = 'no difference'
        elif (res.loc['Two side test p value = '][0] == 0) & (res.loc['Difference < 0 p value = '][0] == 0):
            txt = f"{parameter}: {group1_name} is lower"
            txt2 = 'lower'
        elif (res.loc['Two side test p value = '][0] == 0) & (res.loc['Difference > 0 p value = '][0] == 0):
            txt = f"{parameter}: {group1_name} is higher"
            txt2 = 'higher'
        else:
            txt2 = txt = ''
            
        res.loc['summary'] = txt
        summary = pd.DataFrame(txt2, index=[parameter], columns=[group1_name])
        
        return des, res, summary

    @staticmethod
    def kruskal_wallis_test(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        """
        Calculate the Kruskal-Wallis H-test for independent samples.
        
        Args:
            x: Array of observations
            y: Array of groups
            
        Returns:
            Tuple of (H-statistic, p-value)
        """
        grouped_numbers = {}
        for grp in y.unique():
            grouped_numbers[grp] = x.values[y == grp]
        
        args = grouped_numbers.values()
        return stats.mstats.kruskalwallis(*args)

    @staticmethod
    def chi2_contingency_test(x: pd.Series, y: pd.Series) -> Optional[float]:
        """
        Chi-square test of independence of variables in a contingency table.
        
        Args:
            x: First categorical variable
            y: Second categorical variable
            
        Returns:
            P-value of the test, or None if test cannot be performed
        """
        xtab = pd.crosstab(x, y)
        if xtab.size == 0:
            return None
        
        try:
            _, pval, _, _ = stats.chi2_contingency(xtab)
            return pval
        except Exception:
            return 0.0

    @staticmethod
    def correlation_heatmap(df: pd.DataFrame, 
                           method: str = 'kendall', 
                           diagonal_plot: bool = True, 
                           **kwargs) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Plot a correlation heatmap matrix.
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'kendall', 'spearman')
            diagonal_plot: Whether to show diagonal elements
            **kwargs: Additional parameters for corr() and heatmap()
            
        Returns:
            Tuple of (correlation_matrix, figure)
        """
        import inspect
        
        corr_args = list(inspect.signature(pd.DataFrame.corr).parameters)
        kwargs_corr = {k: kwargs.pop(k) for k in dict(kwargs) if k in corr_args}
        
        heatmap_args = list(inspect.signature(sns.heatmap).parameters)
        kwargs_heatmap = {k: kwargs.pop(k) for k in dict(kwargs) if k in heatmap_args}
        
        corr = df.dropna(how='any', axis=0).drop_duplicates().corr(method=method, **kwargs_corr)
        
        # Generate a mask for the upper triangle
        if diagonal_plot:
            mask = np.zeros_like(corr)
            mask[np.triu_indices_from(mask)] = True
        else:
            mask = None

        plt.figure(figsize=(30, 20))
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns_plot = sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .5},
            fmt=".1f",
            annot=True,
            **kwargs_heatmap,
        )
        figure = sns_plot.get_figure()
        plt.show()
        plt.close()
        
        return corr, figure


class DateTimeProcessor:
    """
    A class for date and time processing operations.
    
    This class provides methods for date manipulation, validation, and formatting.
    """
    
    @staticmethod
    def readable_time(time_seconds: float) -> Tuple[int, int, int, int]:
        """
        Convert time in seconds to readable format (days, hours, minutes, seconds).
        
        Args:
            time_seconds: Time in seconds
            
        Returns:
            Tuple of (days, hours, minutes, seconds)
        """
        day = time_seconds // (24 * 3600)
        time_seconds = time_seconds % (24 * 3600)
        hour = time_seconds // 3600
        time_seconds %= 3600
        minutes = time_seconds // 60
        time_seconds %= 60
        seconds = time_seconds
        return int(day), int(hour), int(minutes), int(seconds)

    @staticmethod
    def check_timestamps(start: str, end: str, format_required: str = '%Y-%m-%d') -> bool:
        """
        Validate the format required for timestamp strings.
        
        Args:
            start: Start date string
            end: End date string
            format_required: Expected date format
            
        Returns:
            True if both timestamps are valid, False otherwise
        """
        try:
            check_start = type(time.strptime(start, format_required))
            check_end = type(time.strptime(end, format_required))
            return (check_start.__name__ == 'struct_time') & (check_end.__name__ == 'struct_time')
        except ValueError as e:
            print(e)
            return False

    @staticmethod
    def generate_date_list(range_date_year: List[int] = None,
                          first_date: Optional[Union[str, dt.date]] = None,
                          last_date: Union[str, dt.date] = None,
                          month_step: int = 1) -> List[str]:
        """
        Generate a list of first dates of months within a given range of years.
        
        Args:
            range_date_year: List of [first_year, last_year+1]
            first_date: Starting date (if None, uses first day of range_date_year[0])
            last_date: Ending date (if None, uses current date)
            month_step: Step size for months
            
        Returns:
            List of date strings in 'YYYY-MM-DD' format
        """
        import itertools
        
        if range_date_year is None:
            range_date_year = [2018, 2099]
        if last_date is None:
            last_date = dt.datetime.now().date()

        yrs = [str(i) for i in range(range_date_year[0], range_date_year[1])]
        months = [str(i).zfill(2) for i in range(1, 13, month_step)]
        udates = ['-'.join(udate) for udate in itertools.product(yrs, months, ['01'])]

        # Handle date conversions
        if isinstance(first_date, str):
            first_date = dt.datetime.strptime(first_date, "%Y-%m-%d").date()
        elif isinstance(first_date, pd._libs.tslibs.timestamps.Timestamp):
            first_date = first_date.date()
        
        if isinstance(last_date, str):
            last_date = dt.datetime.strptime(last_date, "%Y-%m-%d").date()
        elif isinstance(last_date, pd._libs.tslibs.timestamps.Timestamp):
            last_date = last_date.date()

        if first_date is None:
            first_date = dt.datetime.strptime(udates[0], '%Y-%m-%d').date()

        # Filter dates within range
        udates = [date_str for date_str in udates 
                 if (dt.datetime.strptime(date_str, '%Y-%m-%d').date() >= first_date) and
                    (dt.datetime.strptime(date_str, '%Y-%m-%d').date() <= last_date)]
        
        return udates


class FileProcessor:
    """
    A class for file and path processing operations.
    
    This class provides methods for file validation, directory operations,
    and file system utilities.
    """
    
    @staticmethod
    def check_path(path: str) -> str:
        """
        Validate that a file path exists.
        
        Args:
            path: File path to check
            
        Returns:
            Expanded path if valid
            
        Raises:
            FileNotFoundError: If path doesn't exist
        """
        if '~' in path:
            path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File ({path}) not found!")
        return path

    @staticmethod
    def setup_output_folder(output_folder: str, 
                           files_to_copy: List[str] = None, 
                           overwrite: bool = False) -> str:
        """
        Create output directory and copy template files.
        
        Args:
            output_folder: Path of the output folder
            files_to_copy: List of files to copy to output folder
            overwrite: Whether to overwrite existing files
            
        Returns:
            Absolute path of the output folder
        """
        import shutil
        
        if files_to_copy is None:
            files_to_copy = []

        # Convert to absolute path
        if len(output_folder.split('/')) == 1:
            output_folder = os.path.abspath(os.path.join(os.getcwd(), output_folder))
        else:
            output_folder = os.path.abspath(output_folder)

        # Check if directory exists and handle overwrite
        if os.path.exists(output_folder) and not overwrite:
            raise FileExistsError(
                "Program terminated: overwrite is not allowed and the output directory exists")
        elif not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Copy files if overwrite is allowed
        if overwrite:
            for file_path in files_to_copy:
                shutil.copyfile(
                    file_path,
                    os.path.join(output_folder, os.path.basename(file_path)))
        
        return output_folder

    @staticmethod
    def parse_sql_file(file_name: str) -> List[str]:
        """
        Read SQL queries from a file and split them by semicolon.
        
        Args:
            file_name: Path to SQL file
            
        Returns:
            List of SQL statements
        """
        FileProcessor.check_path(file_name)
        sql_statements = []
        
        with open(file_name, 'r') as f:
            content = f.read()
            statements = FileProcessor._split_sql_expressions(content)
            sql_statements.extend(statements)
        
        return sql_statements

    @staticmethod
    def _split_sql_expressions(text: str) -> List[str]:
        """
        Split SQL queries based on semicolon delimiter while handling comments and strings.
        
        Args:
            text: SQL text content
            
        Returns:
            List of individual SQL statements
        """
        results = []
        current = ''
        state = None
        
        for c in text:
            if state is None:  # default state, outside of special entity
                current += c
                if c in '"\'':
                    state = c  # quoted string
                elif c == '-':
                    state = '-'  # probably "--" comment
                elif c == '/':
                    state = '/'  # probably '/*' comment
                elif c == ';':
                    # remove it from the statement
                    current = current[:-1].strip()
                    # and save current stmt unless empty
                    if current:
                        results.append(current)
                    current = ''
            elif state == '-':
                if c != '-':
                    state = None  # not a comment
                    current += c
                    continue
                current = current[:-1]  # remove first minus
                state = '--'  # comment until end of line
            elif state == '--':
                if c == '\n':
                    current += c  # include this newline
                    state = None  # end of comment
                # else just ignore
            elif state == '/':
                if c != '*':
                    state = None
                    current += c
                    continue
                current = current[:-1]  # remove starting slash
                state = '/*'  # multiline comment
            elif state == '/*':
                if c == '*':
                    state = '/**'  # probably end of comment
            elif state == '/**':
                if c == '/':
                    state = None
                else:
                    state = '/*'  # not an end
            elif state[0] in '"\'':
                current += c
                if state.endswith('\\'):
                    state = state[0]  # prev was backslash, revert to regular state
                    continue
                elif c == '\\':
                    state += '\\'  # don't check next char
                    continue
                elif c == state[0]:
                    state = None  # end of quoted string
            else:
                raise Exception(f'Illegal state {state}')

        if current:
            current = current.rstrip(';').strip()
            if current:
                results.append(current)
        
        return results


class Logger:
    """
    A class for logging operations and utilities.
    
    This class provides methods for creating loggers, redirecting output,
    and managing log files.
    """
    
    @staticmethod
    def create_logger(log_file: str, 
                     name: str, 
                     log_level: int = logging.INFO) -> logging.Logger:
        """
        Create a logger with file and console handlers.
        
        Args:
            log_file: Path to log file
            name: Logger name
            log_level: Logging level (default: INFO)
            
        Returns:
            Configured logger instance
        """
        # Configure formatters
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-3s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        console_formatter = logging.Formatter("%(message)s")

        # Configure handlers
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(console_formatter)

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        if not len(logger.handlers):
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        
        return logger

    @staticmethod
    def setup_logger(log_file: str) -> logging.Logger:
        """
        Set up logger to write to console and file.
        
        Args:
            log_file: Path to the log file
            
        Returns:
            Configured logger instance
        """
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        logger = logging.getLogger('data_processor')
        logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        if logger.hasHandlers():
            logger.handlers.clear()
        
        # Create formatters
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_formatter = logging.Formatter('%(message)s')
        
        # File handler
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file at {log_file}: {str(e)}")
            print("Continuing with console logging only...")
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger

    @staticmethod
    def custom_print(message: str, logger: Optional[logging.Logger] = None) -> None:
        """
        Print message to console and optionally to logger.
        
        Args:
            message: Message to print
            logger: Optional logger instance
        """
        print(message)
        if logger:
            logger.info(message)


class WebScraper:
    """
    A class for web scraping and content downloading operations.
    
    This class provides methods for downloading files and content from web sources,
    including intranet sites with authentication.
    """
    
    @staticmethod
    def download_intranet_files(url: str, 
                               save_location: str, 
                               username: str, 
                               password: str, 
                               file_extensions: List[str], 
                               chunk_size: int = 8192, 
                               logger: Optional[logging.Logger] = None) -> None:
        """
        Download all files from specified intranet URL using NTLM authentication.
        
        Args:
            url: URL to download files from
            save_location: Directory to save files
            username: Username for authentication
            password: Password for authentication
            file_extensions: List of file extensions to download
            chunk_size: Size of chunks for downloading
            logger: Optional logger instance
        """
        try:
            import requests
            from requests_ntlm import HttpNtlmAuth
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin
            from tqdm import tqdm
            import urllib3
        except ImportError as e:
            raise ImportError(f"Required packages not installed: {e}")
        
        # Create save directory if it doesn't exist
        os.makedirs(save_location, exist_ok=True)
        
        # Create session with NTLM authentication and disable SSL verification
        session = requests.Session()
        session.verify = False
        
        # Suppress SSL verification warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        if username and password:
            session.auth = HttpNtlmAuth(username, password)
        else:
            session.auth = HttpNtlmAuth('', '')

        try:
            if logger:
                logger.info(f"Accessing {url}...")
            else:
                print(f"Accessing {url}...")
                
            response = session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            
            downloaded_files = []
            failed_files = []
            
            for link in links:
                href = link.get('href')
                if not any(href.lower().endswith(ext) for ext in file_extensions):
                    continue
                    
                file_url = urljoin(url, href)
                filename = os.path.basename(href)
                save_path = os.path.join(save_location, filename)
                
                try:
                    if logger:
                        logger.info(f"Downloading: {filename}")
                    else:
                        print(f"Downloading: {filename}")
                    
                    response = session.get(file_url, stream=True)
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(save_path, 'wb') as f, tqdm(
                        desc=filename,
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                    ) as pbar:
                        for data in response.iter_content(chunk_size=chunk_size):
                            size = f.write(data)
                            pbar.update(size)
                            
                    downloaded_files.append(filename)
                    if logger:
                        logger.info(f"Saved: {save_path}")
                    else:
                        print(f"Saved: {save_path}")
                    
                except Exception as e:
                    error_msg = f"Failed to download {filename}: {str(e)}"
                    if logger:
                        logger.error(error_msg)
                    else:
                        print(error_msg)
                    failed_files.append(filename)
                    continue
                    
            # Print summary
            summary_msg = f"\nDownload Summary:\nSuccessfully downloaded ({len(downloaded_files)}):"
            if logger:
                logger.info(summary_msg)
            else:
                print(summary_msg)
                
            for file in downloaded_files:
                file_msg = f"  ✓ {file}"
                if logger:
                    logger.info(file_msg)
                else:
                    print(file_msg)
                
            if failed_files:
                failed_msg = f"\nFailed downloads ({len(failed_files)}):"
                if logger:
                    logger.warning(failed_msg)
                else:
                    print(failed_msg)
                for file in failed_files:
                    fail_msg = f"  ✗ {file}"
                    if logger:
                        logger.warning(fail_msg)
                    else:
                        print(fail_msg)
                
        except Exception as e:
            error_msg = f"Error occurred: {str(e)}"
            if logger:
                logger.error(error_msg)
            else:
                print(error_msg)
        finally:
            session.close()


# Backward compatibility functions - these maintain the original function names
# for existing code that might be using them

def inWithReg(regLst, LstAll):
    """Backward compatibility wrapper for ListProcessor.in_with_regex"""
    return ListProcessor.in_with_regex(regLst, LstAll)

def flattenList(ulist):
    """Backward compatibility wrapper for ListProcessor.flatten_list"""
    return ListProcessor.flatten_list(ulist)

def unique_list(seq):
    """Backward compatibility wrapper for ListProcessor.unique_list"""
    return ListProcessor.unique_list(seq)

def normalize_text(text, **kwargs):
    """Backward compatibility wrapper for TextProcessor.normalize_text"""
    return TextProcessor.normalize_text(text, **kwargs)

def sanitize_filename(filename, max_length=100):
    """Backward compatibility wrapper for TextProcessor.sanitize_filename"""
    return TextProcessor.sanitize_filename(filename, max_length)

def clean_column_names(column_list, replacements=None, lowercase=False):
    """Backward compatibility wrapper for TextProcessor.clean_column_names"""
    return TextProcessor.clean_column_names(column_list, replacements, lowercase)

def movecol(df, cols_to_move=None, ref_col='', place='After'):
    """Backward compatibility wrapper for DataFrameProcessor.move_columns"""
    return DataFrameProcessor.move_columns(df, cols_to_move, ref_col, place)

def reduce_mem_usage(df, **kwargs):
    """Backward compatibility wrapper for DataFrameProcessor.reduce_memory_usage"""
    return DataFrameProcessor.reduce_memory_usage(df, **kwargs)

def cat2no(df):
    """Backward compatibility wrapper for DataFrameProcessor.categorical_to_numeric"""
    return DataFrameProcessor.categorical_to_numeric(df)

def null_per_column(df):
    """Backward compatibility wrapper for DataFrameProcessor.null_percentage_per_column"""
    return DataFrameProcessor.null_percentage_per_column(df)

def readableTime(time_seconds):
    """Backward compatibility wrapper for DateTimeProcessor.readable_time"""
    return DateTimeProcessor.readable_time(time_seconds)

def check_timestamps(start, end, format_required='%Y-%m-%d'):
    """Backward compatibility wrapper for DateTimeProcessor.check_timestamps"""
    return DateTimeProcessor.check_timestamps(start, end, format_required)

def datesList(**kwargs):
    """Backward compatibility wrapper for DateTimeProcessor.generate_date_list"""
    return DateTimeProcessor.generate_date_list(**kwargs)

def check_path(path):
    """Backward compatibility wrapper for FileProcessor.check_path"""
    return FileProcessor.check_path(path)

def setOutputFolder(outputFolder, uFiles=None, overWrite=False):
    """Backward compatibility wrapper for FileProcessor.setup_output_folder"""
    return FileProcessor.setup_output_folder(outputFolder, uFiles, overWrite)

def parse_sql_file(file_name):
    """Backward compatibility wrapper for FileProcessor.parse_sql_file"""
    return FileProcessor.parse_sql_file(file_name)

def logmaker(uFile, name, logLevel=logging.INFO):
    """Backward compatibility wrapper for Logger.create_logger"""
    return Logger.create_logger(uFile, name, logLevel)

def setup_logger(log_file):
    """Backward compatibility wrapper for Logger.setup_logger"""
    return Logger.setup_logger(log_file)

def custom_print(message, logger=None):
    """Backward compatibility wrapper for Logger.custom_print"""
    return Logger.custom_print(message, logger)

def hypothesis_test(df, par, group, group_names):
    """Backward compatibility wrapper for StatisticalAnalyzer.hypothesis_test"""
    return StatisticalAnalyzer.hypothesis_test(df, par, group, group_names)

def kruskalwallis2(x, y):
    """Backward compatibility wrapper for StatisticalAnalyzer.kruskal_wallis_test"""
    return StatisticalAnalyzer.kruskal_wallis_test(x, y)

def chi2_contingency(x, y):
    """Backward compatibility wrapper for StatisticalAnalyzer.chi2_contingency_test"""
    return StatisticalAnalyzer.chi2_contingency_test(x, y)

def corrmap(df0, method='kendall', diagonal_plot=True, **kwargs):
    """Backward compatibility wrapper for StatisticalAnalyzer.correlation_heatmap"""
    return StatisticalAnalyzer.correlation_heatmap(df0, method, diagonal_plot, **kwargs)

def find_fuzzy_matches(listA, listB, threshold=60):
    """Backward compatibility wrapper for TextProcessor.find_fuzzy_matches"""
    return TextProcessor.find_fuzzy_matches(listA, listB, threshold)

def download_intranet_files(url, save_location, username, password, file_extensions, chunk_size=8192, logger=None):
    """Backward compatibility wrapper for WebScraper.download_intranet_files"""
    return WebScraper.download_intranet_files(url, save_location, username, password, file_extensions, chunk_size, logger)
