import numpy as np
import pandas as pd
import logging
import time
import re

import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
import plotly.express as px

#import pydotplus
import pylab as pl
import seaborn as sns
import os,sys
import datetime as dt
import math

import plotly.express as px

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------Basic functions-------------------------------
def inWithReg(regLst, LstAll):
    """ search regualr expression list in a list

    Parameters:
    ----------
    regLst (list of strings with regx)

    LstAll (list of strings to be searched for regLst

    returns:
    -------
    out: the subset of LstAll which met contains any subset of regLst
    
    ind: a list of True/False values of existence LstAll in any subset of regLst

    Example:
    ---------
        regLst=['.vol_flag$','fefefre','_date']
        LstAll=['bi_alt_account_id', 'snapshot_date', 'snapshot_year','tv_vol_flag', 'phone_vol_flag']
        out,ind=inWithReg(regLst,LstAll)
        out=['tv_vol_flag', 'phone_vol_flag', 'snapshot_date']
        ind=[False,  True, False,  True,  True]

    -------
    Author: Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """
    out = []
    if type(regLst) != list:
        regLst = [regLst]
    for i in regLst:
        tmp = list(filter(re.compile(i).search, LstAll))
        out = out + tmp
    ind = np.in1d(LstAll, out)
    return out, ind

import pandas as pd
import re
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import numpy as np

def normalize_text(text, 
                  remove_spaces=True, 
                  lowercase=True, 
                  special_chars=r'[^a-zA-Z0-9\s]',
                  replace_with='',
                  max_length=None,
                  fallback_text='unnamed'):
    """
    Flexible text normalization function.
    
    Parameters:
    text (str): Text to normalize
    remove_spaces (bool): Remove spaces from text
    lowercase (bool): Convert to lowercase
    special_chars (str): Regex pattern for characters to match (default matches all except letters, numbers, and spaces)
    replace_with (str): String to replace matched characters with (default: '', removes them)
    max_length (int): Maximum length of the output text (default: None, no limit)
    fallback_text (str): Text to return if result is empty or only whitespace (default: 'unnamed')
    
    Returns:
    str: Normalized text
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
    
    # Trim text if max_length is specified and text exceeds it
    if max_length is not None and len(text) > max_length:
        text = text[:max_length]
        # Strip again after truncation in case we cut mid-word
        text = text.strip()
    
    # Handle empty or whitespace-only results
    if not text or text.isspace():
        text = fallback_text
    
    return text

###TODO: retire it:
def sanitize_filename(filename, max_length=100):
    """Sanitize a filename using the flexible normalize_text function."""
    return normalize_text(
        filename,
        remove_spaces=False,
        lowercase=True,
        special_chars=r'[\\/*?:"<>|\r\n\t]',
        replace_with='_',
        max_length=max_length,
        fallback_text='unnamed'
    )

def clean_column_names(column_list, replacements={}, lowercase=False):
    """
    Clean a list of strings to be suitable for use as column names.
    
    Parameters:
    column_list (list): List of strings to clean
    replacements (dict): Dictionary of string replacements to apply (default: {})
    
    Returns:
    list: Cleaned column names
    """
    cleaned_columns = []
    
    for col in column_list:
        # Convert to string if not already
        col = str(col)
        
        # Apply custom replacements first
        for old, new in replacements.items():
            col = col.replace(old, new)
        
        # Use normalize_text for most of the work
        col = normalize_text(
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

def find_fuzzy_matches(listA, listB, threshold=60):
    """
    Find fuzzy matches between two lists based on normalized text similarity.
    
    Parameters:
    listA (list): First list of data
    listB (list): Second list of data
    threshold (float): Similarity threshold percentage (default 60)
    
    Returns:
    dict: Dictionary with match information
    """
    matches = {}
    used_b_indices = set()
    
    for i, item_a in enumerate(listA):
        normalized_a = normalize_text(item_a)
        best_match = None
        best_similarity = 0
        best_index = -1
        
        for j, item_b in enumerate(listB):
            if j in used_b_indices:
                continue
                
            normalized_b = normalize_text(item_b)
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
                'normalized_b': normalize_text(best_match)
            }
            used_b_indices.add(best_index)
    
    return matches

def create_venn_diagram(listA, listB, similarity_threshold=60, listA_name='List A', listB_name='List B', utitle='Venn Diagram - List Comparison with Fuzzy Matching', save_path=None):
    """
    Create a Venn diagram showing the relationship between two lists.
    
    Parameters:
    listA (list): First list of data
    listB (list): Second list of data
    similarity_threshold (float): Similarity threshold percentage (default 60)
    listA_name (str): Name for first list (default 'List A')
    listB_name (str): Name for second list (default 'List B')
    utitle (str): Title for the diagram (default 'Venn Diagram - List Comparison with Fuzzy Matching')
    save_path (str): Path to save the diagram (optional)
    
    Returns:
    matplotlib.figure.Figure: The figure object
    """
    # Find fuzzy matches using the existing function
    fuzzy_matches = find_fuzzy_matches(listA, listB, similarity_threshold)
    # Calculate counts
    matched_a = set(fuzzy_matches.keys())
    matched_b = set(match_info['match'] for match_info in fuzzy_matches.values())
    unmatched_a = set(listA) - matched_a
    unmatched_b = set(listB) - matched_b
    
    only_a_count = len(unmatched_a)
    only_b_count = len(unmatched_b)
    both_count = len(fuzzy_matches)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Define circle parameters
    circle_radius = 1.5
    circle_a_center = (-0.5, 0)
    circle_b_center = (0.5, 0)
    
    # Create circles
    circle_a = Circle(circle_a_center, circle_radius, alpha=0.3, color='blue', label=listA_name)
    circle_b = Circle(circle_b_center, circle_radius, alpha=0.3, color='red', label=listB_name)
    
    ax.add_patch(circle_a)
    ax.add_patch(circle_b)
    
    # Add text labels with counts
    # Only A
    ax.text(circle_a_center[0] - 0.8, circle_a_center[1], f'{only_a_count}', 
            fontsize=16, ha='center', va='center', weight='bold')
    
    # Only B
    ax.text(circle_b_center[0] + 0.8, circle_b_center[1], f'{only_b_count}', 
            fontsize=16, ha='center', va='center', weight='bold')
    
    # Both (intersection)
    ax.text(0, 0, f'{both_count}', fontsize=16, ha='center', va='center', weight='bold')
    
    # Add circle labels
    ax.text(circle_a_center[0], circle_a_center[1] + 2, listA_name, 
            fontsize=14, ha='center', va='center', weight='bold', color='blue')
    ax.text(circle_b_center[0], circle_b_center[1] + 2, listB_name, 
            fontsize=14, ha='center', va='center', weight='bold', color='red')
    
    # Add detailed breakdown text
    breakdown_text = f"Breakdown:\n"
    breakdown_text += f"• Only in {listA_name}: {only_a_count} items\n"
    breakdown_text += f"• Only in {listB_name}: {only_b_count} items\n"
    breakdown_text += f"• Similar items (≥{similarity_threshold}%): {both_count} pairs\n"
    breakdown_text += f"• Total unique items: {only_a_count + only_b_count + both_count}"
    
    ax.text(-3, -3, breakdown_text, fontsize=10, ha='left', va='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    # Set axis properties
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    ax.set_title(utitle, 
                fontsize=16, weight='bold', pad=20)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def compare_lists(listA, listB, similarity_threshold=60, create_venn=False, listA_name='List A', listB_name='List B', utitle='Venn Diagram - List Comparison with Fuzzy Matching', save_venn_path=None):
    """
    Create a table showing unique elements from two lists with fuzzy matching and appropriate group identification.
    Elements with similarity above threshold are marked as 'Both' with similarity percentage.
    
    Parameters:
    listA (list): First list of data
    listB (list): Second list of data
    similarity_threshold (float): Similarity threshold percentage (default 60)
    create_venn (bool): Whether to create a Venn diagram (default False)
    listA_name (str): Name for first list in Venn diagram (default 'List A')
    listB_name (str): Name for second list in Venn diagram (default 'List B')
    save_venn_path (str): Path to save Venn diagram (optional)
    
    Returns:
    tuple: (pd.DataFrame, str, matplotlib.figure.Figure or None) - Table with unique elements, formatted summary, and optional Venn diagram
    """
    # Find fuzzy matches
    fuzzy_matches = find_fuzzy_matches(listA, listB, similarity_threshold)
    
    # Create sets for tracking matched items
    matched_a = set(fuzzy_matches.keys())
    matched_b = set(match_info['match'] for match_info in fuzzy_matches.values())
    
    # Find unmatched items
    unmatched_a = set(listA) - matched_a
    unmatched_b = set(listB) - matched_b
    
    # Create list of dictionaries for DataFrame
    data = []
    
    # Add elements only in listA (unmatched)
    for element in sorted(unmatched_a):
        data.append({
            'Element': element,
            'Group': listA_name,
            'Match': '',
            'Similarity': ''
        })
    
    # Add elements only in listB (unmatched)
    for element in sorted(unmatched_b):
        data.append({
            'Element': element,
            'Group': listB_name,
            'Match': '',
            'Similarity': ''
        })
    
    # Add fuzzy matched elements
    for item_a, match_info in sorted(fuzzy_matches.items()):
        data.append({
            'Element': item_a,
            'Group': 'Both',
            'Match': match_info['match'],
            'Similarity': f"{match_info['similarity']:.1f}%"
        })
    
    # Create DataFrame
    result_df = pd.DataFrame(data)
    
    # Sort by Group first, then by Element
    if not result_df.empty:
        result_df = result_df.sort_values(['Group', 'Element'], ascending=[True, True])
        result_df = result_df.reset_index(drop=True)
        result_df.insert(0, 'Index', range(1, len(result_df) + 1))
    
    # Create formatted summary
    summary_text = "Unique Elements Table with Fuzzy Matching:\n"
    summary_text += "=" * 60 + "\n"
    summary_text += result_df.to_string(index=False) + "\n"
    summary_text += "\nGroup Summary:\n"
    summary_text += "=" * 20 + "\n"
    
    if not result_df.empty:
        group_counts = result_df['Group'].value_counts()
        for group, count in group_counts.items():
            summary_text += f"{group}: {count} elements\n"
    
    summary_text += f"\nFuzzy Matching Details:\n"
    summary_text += "=" * 25 + "\n"
    summary_text += f"Similarity threshold: {similarity_threshold}%\n"
    summary_text += f"Fuzzy matches found: {len(fuzzy_matches)}\n"
    
    if fuzzy_matches:
        summary_text += "\nDetailed Matches:\n"
        for item_a, match_info in sorted(fuzzy_matches.items()):
            summary_text += f"  '{item_a}' ↔ '{match_info['match']}' ({match_info['similarity']:.1f}%)\n"
            summary_text += f"    Normalized: '{match_info['normalized_a']}' ↔ '{match_info['normalized_b']}'\n"
    
    summary_text += f"\nOriginal Data:\n"
    summary_text += f"listA: {listA}\n"
    summary_text += f"listB: {listB}"
    
    # Create Venn diagram if requested
    venn_fig = None
    if create_venn:
        venn_fig = create_venn_diagram(listA, listB, similarity_threshold, listA_name, listB_name, utitle, save_venn_path)
    
    return result_df, summary_text, venn_fig

def retrieve_name(var):
    import inspect
    """Getting the name of a variable as a string
    Ref: https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string 
    """
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]

def flattenList(ulist):
    """ makes a flat list out of list of lists

    Parameters:
    ----------
    ulist: a list of nested lists
    
    returns:
    -------
    results  ulist: a list of flatten ulist

    -------
    Author: Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """
    results = []
    for rec in ulist:
        if isinstance(rec, list):
            results.extend(rec)
            results = flattenList(results)
        else:
            results.append(rec)
    return results

def unique_list(seq):
    """ Get unique values from a list seq with saving the order of list elements
    Parameters:
    ----------
    seq: a list with duplicates elements
    -------
    out  (list): 
    a list with unqiue elements
    -------
    Author: Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """
    seen = set()
    seen_add = seen.add
    out=[x for x in seq if not (x in seen or seen_add(x))]
    return out
  
def check_timestamps(start, end, format_required='%Y-%m-%d'):
    '''validate the format required for the query'''
    try:
        check_start = type(time.strptime(start, format_required))
        check_end = type(time.strptime(end, format_required))
        if (check_start.__name__=='struct_time') & (check_end.__name__=='struct_time'):
            return True
    except ValueError as e:
        print(e)

def check_path(path):
    """Raise exception if the file path doesn't exist."""
    import argparse
    if '~' in path:
        path = os.path.expanduser(path)
    if not os.path.exists(path):
        msg = "File (%s) not found!" % path
        raise argparse.ArgumentTypeError(msg)
    return path

def pass_days(start_date, end_date):
    ##TODO: add comment
    import pandas as pd

    # if (start_date is None)|(end_date is None)| (pd.isnull(start_date))| (pd.isnull(end_date)):
    #   return None
    month_year_index = (
        pd.date_range(start=start_date, end=end_date, freq="D").to_period("Q").unique()
    )
    # print(start_date, end_date, month_year_index)

    pass_days_dict = {}
    for month_year in month_year_index:
        days_in_month = (
            min(end_date, month_year.end_time)
            - max(start_date, (month_year.start_time))
        ).days + 1
        pass_days_dict[month_year] = days_in_month

    result_series = pd.Series(pass_days_dict)
    qs = "Q" + result_series.index.quarter.astype(str)
    result_series = result_series.groupby(qs).sum()
    # print(result_series)
    return result_series.fillna(0)

def copy_ymls(dsToolbox, platform='databricks', destination=None):
  ##TODO: add comments:  
  import sys, os
  from io_funcs import io_funcs
  upath=dsToolbox.__file__
  if destination==None:
    destination=os.getcwd()
  for ufile in ['config.yml', 'sql_template.yml']:
    ufile_src=os.path.join(os.path.dirname(upath), ufile)
    ufile_desc=os.path.join(destination, ufile)
    ufile_desc_tmp=os.path.join(destination, f'.{ufile}.crc')
    print(f"copying {ufile_src} ---> {ufile_desc}")
    if platform=='databricks':
      dbutils=io_funcs.get_dbutils()
      dbutils.fs.cp(f'file://{ufile_src}', f'file://{ufile_desc}')
      dbutils.fs.rm(f'file://{ufile_desc_tmp}')

def remove_extra_none(nested_lst):
    items=list(dict.fromkeys(nested_lst)) 
    if (('None' in items) & (len(items)>1)):
        items.remove('None')  
    # print(items)
    return items

def setOutputFolder(outputFolder, uFiles, overWrite):
    """ creates output directory and copies template file(s)

    Parameters:
    ----------
    outputFolder (string): path of the outputFolder

    uFiles (a list of strings): of files that shoud be copied in outputFolder

    overWrite (True/False): if it True , it copies uFiles to outputFolder,even they exist;

    returns:
    -------
    outputFolder string
    """
    import os, shutil, sys
    
    # Setting output directory and copying template file(s)
    if len(outputFolder.split('/')) == 1:
        outputFolder = os.path.abspath(os.path.join(os.getcwd(), outputFolder))
    else:
        outputFolder = os.path.abspath(outputFolder)

    if os.path.exists(outputFolder) & (not overWrite):
        sys.exit(
            "Program terminated: overwrite is not allowed and the output directory exists")
    # elif os.path.exists(outputFolder):
    #     shutil.rmtree(outputFolder)
    #     os.makedirs(outputFolder)
    # else:
    #     os.makedirs(outputFolder)
    elif not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    # pdb.set_trace()
    if (overWrite):
        for uFile in uFiles:
            shutil.copyfile(
                uFile,
                os.path.join(
                    outputFolder,
                    os.path.basename(uFile)))
    return (outputFolder)

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# --------------------------------Date time -------------------------------
def readableTime(time):
    """ convert time to day, hour, minutes, seconds

    -------
    Author: Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """
    day = time // (24 * 3600)
    time = time % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    time %= 60
    seconds = time
    return (day, hour, minutes, seconds)

def datesList(range_date__year=[2018,2099],
              firstDate=None,
              lastDate=dt.datetime.now().date(),
              month_step=1):
  """generates a list of first dates of months within a given range of years
    Params:       
      range_date__year(list):  list of first and the last year+1                               
      firstDate (string or datetime): if it is not given, it will be the first day and month of range_date__year[0]
      lastDate (string or datetime): if it is not given,  it will be the current date
    Returns: python list 
  """ 
  import itertools  
  import datetime as dt

  # print(firstDate)
  # print(lastDate)
  yrs=[str(i) for i in range(range_date__year[0], range_date__year[1])]
  months=[str(i).zfill(2) for i in range(1,13, month_step)]
  udates=['-'.join(udate) for udate in itertools.product(yrs,months,['01'])]

  if isinstance(firstDate, str):
    print(firstDate)
    firstDate   = dt.datetime.strptime(firstDate, "%Y-%m-%d").date()
  elif isinstance(firstDate,pd._libs.tslibs.timestamps.Timestamp):
    firstDate   =firstDate.date()
  if isinstance(lastDate, str):
    lastDate     = dt.datetime.strptime(lastDate, "%Y-%m-%d").date()  
  elif isinstance(lastDate,pd._libs.tslibs.timestamps.Timestamp):
    lastDate   =lastDate.date()

  if firstDate is None:
    firstDate=dt.datetime.strptime(udates[0], '%Y-%m-%d').date() 
  if lastDate is None:
    lastDate=dt.datetime.strptime(udates[-1], '%Y-%m-%d').date() 

  if udates[-1]!=lastDate:
    udates.append(lastDate.strftime("%Y-%m-%d"))
  if udates[0]!=firstDate:
    udates[0]=firstDate.strftime("%Y-%m-%d")
    # udates.insert(0,firstDate.strftime("%Y-%m-%d"))
  
  udates=[ii for ii in udates if (dt.datetime.strptime(ii, '%Y-%m-%d').date()>=firstDate)&\
                                 (dt.datetime.strptime(ii, '%Y-%m-%d').date()<=lastDate)]
  # print(udates)
  return udates

##TODO:remove it 
# def extract_start_end(udates, ii):
#   import datetime as dt
#   start_date=udates[ii]
#   end_date=(dt.datetime.strptime(udates[ii+1], '%Y-%m-%d').date()- dt.timedelta(days=1)).strftime("%Y-%m-%d")
# #   print(start_date,' to ',end_date ,":")  
#   return start_date, end_date

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------Panadas functions------------------------------- 
def movecol(df, cols_to_move=[], ref_col='', place='After'):
    """ Reorders a panda dataframe columns
    Parameters:
    ----------
    df (pandas dataframe) 
    cols_to_move:                          list of columns to move
    ref_col(string):                       name of a specific column to move  cols_to_move columns to after/before it
    place(string) [options:"After","Before"]: cols_to_move columns will be move before/after it
    
    returns:
    -------
    df (pandas dataframe)  reordered dataframe

    -------
    Author:    https://towardsdatascience.com/reordering-pandas-dataframe-columns-thumbs-down-on-standard-solutions-1ff0bc2941d5
    """ 
    cols = df.columns.tolist()
    if place == 'After':
        seg1 = cols[:list(cols).index(ref_col) + 1]
        seg2 = cols_to_move
    if place == 'Before':
        seg1 = cols[:list(cols).index(ref_col)]
        seg2 = cols_to_move + [ref_col]
    
    seg1 = [i for i in seg1 if i not in seg2]
    seg3 = [i for i in cols if i not in seg1 + seg2]
    
    return(df[seg1 + seg2 + seg3]) 

def merge_between(df1, df2, groupCol, closed="both"):
    #   df1=df_pi_dic_wide
    #   df2=df_cases_edited
    #   groupCol='Vessel'

  df_out=pd.DataFrame(columns=df1.columns.tolist()+['Index_no'])
  for name, group_df in df1.groupby([groupCol]):
    print(name)
    df2_sub=df2.loc[df2[groupCol]==name]

    #     https://stackoverflow.com/questions/68792511/efficient-way-to-merge-large-pandas-dataframes-between-two-dates
    #     https://stackoverflow.com/questions/31328014/merging-dataframes-based-on-date-range
    #     https://stackoverflow.com/questions/69824730/check-if-value-in-pandas-dataframe-is-within-any-two-values-of-two-other-columns
    #     https://stackoverflow.com/questions/43593554/merging-two-dataframes-based-on-a-date-between-two-other-dates-without-a-common
    #     https://pandas.pydata.org/docs/reference/api/pandas.IntervalIndex.from_arrays.html
    i = pd.IntervalIndex.from_arrays(df2_sub['Start'],
                                     df2_sub['End'], 
                                     closed=closed
                                               )
    group_df['Index_no']=i.get_indexer(group_df['Date'])

    df_out=pd.concat([group_df, df_out],axis=0)
  
  return df_out

def cellWeight(df, axis=0):
  if axis==0:
    out=df.div(df.sum(axis=0), axis=1)
  else:
    out=df.div(df.sum(axis=1), axis=0)
  return out

import pandas as pd
import numpy as np
from typing import List, Tuple

class DataFrameColumnComparator:
    """A class for comparing columns between two DataFrames."""
    
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
    Compare columns between two DataFrames.
    
    Returns:
    --------
    tuple : (comparison_table, summary_dict, type_summary)
    """
    comparator = DataFrameColumnComparator(df1, df2, df1_name, df2_name)
    return comparator.compare(display)

def cat2no(df):
    """ converts all categorical and object features of a dataframe (df) to cat.code

    Parameters:
    ----------
    df(pandas dataframe) with [sample*features] format

    thresh(number): thersold value for variance of features

    -------
    Author: Reza Nourzadeh- reza.nourzadeh@gmail.com 
    """

    cat_columns = df.select_dtypes(include=['object']).columns.tolist()+df.select_dtypes(include=['category']).columns.tolist()
    if len(cat_columns) != 0:
        print('Categorical columns...\n'+cat_columns)
        df[cat_columns] = df[cat_columns].apply(
            lambda x: x.astype('category').cat.codes)
    return df

def reduce_mem_usage(df, obj2str_cols='all_columns', str2cat_cols='all_columns', 
                    use_float16=False, verbose=False):
    """ 
    Iterate through all columns of a dataframe and modify data types to reduce memory usage.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe to optimize
    obj2str_cols : list or 'all_columns'
        Columns to convert from object to string before categorization
    str2cat_cols : list or 'all_columns' 
        Columns to convert from string to category
    use_float16 : bool, default False
        Whether to use float16 (can cause precision loss)
    verbose : bool, default False
        Whether to print conversion details
        
    Returns:
    --------
    pandas.DataFrame
        Optimized dataframe with reduced memory usage
    """

    import pandas as pd
    import numpy as np
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
                # int64 is already the largest, no need to convert
                    
            # Handle float columns
            elif pd.api.types.is_float_dtype(df[col]):
                # Check if all values are finite (no inf/-inf)
                if np.isfinite(df[col]).all():
                    if use_float16 and c_min >= np.finfo(np.float16).min and c_max <= np.finfo(np.float16).max:
                        # Additional check for float16 precision
                        if np.abs(c_max - c_min) < 65504:  # float16 max value
                            df[col] = df[col].astype(np.float16)
                    elif c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    # float64 is default, no need to convert
        
        # Handle object columns
        elif col_type == object:
            obj2str = False
            if (col in obj2str_cols) or (obj2str_cols == 'all_columns'):
                df[col] = df[col].astype('string')  # Use pandas string dtype
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

def null_per_column(df):
    null_per = df.isnull().sum() / df.shape[0] * 100
    null_per = pd.DataFrame(null_per, columns=["null_percent"])
    null_per = (
        null_per.reset_index()
        .sort_values(by=["null_percent", "index"], ascending=False)
        .set_index("index")
    )
    return null_per

def unify_cols(df1, df2, df1_name, df2_name):
    df1.index=df2.index
    def unify_cols__sub(df1, df2, df1_name, df2_name):
        diff1=np.setdiff1d(df1.columns, df2.columns)
        if diff1.size!=0:
            print(f'Adding following columns to {df2_name} as there are in {df1_name}:\n {diff1}')
            df2=pd.concat([df2,
                  pd.DataFrame(0, index=df2.index, columns=diff1)], axis=1)
            df2=df2[df1.columns]
        return df2
    df2=unify_cols__sub(df1, df2, df1_name, df2_name)
    df1=unify_cols__sub(df2, df1, df2_name, df1_name)
    return df1, df2

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------EDA, Statisitcal analysis-----------------------     
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import f_classif

def compare_univar_fea(X, y, univar_fea_lst ):   
    ##TODO: refactor it: 
    arr = np.empty((0,X.shape[1]), float)
    for univar in univar_fea_lst:
        print(univar)
        uFunc=eval(univar)
        score = uFunc(X, y)
        if univar in ['mutual_info_classif','mutual_info_regression']:
          score=score
        elif univar in ['chi2']:
          score=score[1]
        elif univar in ['SelectFdr','SelectFdr','SelectFwe','f_classif']:
          ###TODO: correct it:
          selector = SelectKBest(uFunc,k='all')
          selector.fit(X,y)
          score = (selector.pvalues_)
    #     cols = selector.get_support(indices=True)

        if (score is  None):
            score=np.empty(X.shape[1]) 
            score[:]=np.nan
        arr = np.append(arr,[score] , axis=0)
    scores=pd.DataFrame(data=arr,index=univar_fea_lst,columns=X.columns)
    scores_long = pd.melt(scores.T.dropna(how='all').reset_index().rename(columns={"index": "feature"}),id_vars=['feature'],value_name='P_value', var_name='Feature_selection_Method')

    fig, ax = plt.subplots(figsize = (25,15))
    uplot   = sns.scatterplot(y="feature",
                          x="P_value",
                          hue="Feature_selection_Method",
                          style='Feature_selection_Method',
                          size='Feature_selection_Method',
                          data=scores_long,
                          ax=ax
                          ) 
    cutter_values=[.05]
    for con,xl in enumerate(cutter_values):
      ax.axvline(x=xl, color='red', linestyle='--')
      ax.text(xl, con+5, f'P_value={xl}',rotation=90, size=10)

    ## plt.xticks(rotation=90)
    plt.show()
    plt.close()

    return scores.T

def hypothesis_test(
                    df,
                    par,
                    group,
                    group_names,
                   ):
  import researchpy as rp
  
  df[group]=df[group].astype('bool')
  X1=df[par][df[group]]
  X2=df[par][~df[group]]
  
  group1_name, group2_name= group_names[0], group_names[1]
  des, res =rp.ttest(X1, X2,
                     group1_name= group1_name,
                     group2_name= group2_name,
                     equal_variances= False,
                     paired= False,
                     #correction= None
                    )
  res=res.set_index(res.columns[0])
  res.columns=[par]

  if res.loc['Two side test p value = '][0]!=0:
    txt=f"{par}: There is no difference between {group1_name} and {group2_name}"
    txt2='no difference'
  elif (res.loc['Two side test p value = '][0]==0) & (res.loc['Difference < 0 p value = '][0]==0):
    txt=f"{par}: {group1_name} is lower " #than {group2_name}"
    txt2='lower'
  elif  (res.loc['Two side test p value = '][0]==0) & (res.loc['Difference > 0 p value = '][0]==0):
    txt=f"{par}: {group1_name} is higher" #than {group2_name}"
    txt2='higher'
  else:
     txt2=txt=''
      
  res.loc['summary']=txt
  #   print(txt)

  summary=pd.DataFrame(txt2,index=[par],columns=[group1_name])
  #   print(summary)
  return des, res, summary

def hypothesis_test_batch_pars(df,
                              pars ,
                              group,
                              group_names):
  tsts=pd.DataFrame()
  stats=pd.DataFrame()
  summary=pd.DataFrame()
  for par in pars:
    par1_stats_tmp, par1_test_tmp, summary_tmp= hypothesis_test(df,
                                                                par=par,
                                                                group=group,
                                                                group_names=group_names
                                                               )
    stats=pd.concat([stats,
                    par1_stats_tmp],axis=0,
                    # keys=[par]
                    )
    
    tsts=pd.concat([tsts,
                    par1_test_tmp],axis=1)

    summary=pd.concat([summary,
                    summary_tmp],axis=0,
    #                     keys=[par]
                    )
  return stats, tsts, summary

def percent_agg(df, grpby1, grpby2, sumCol):
  agg1=df.groupby(grpby1)[sumCol].sum().reset_index()
  agg2=df.groupby(grpby2)[sumCol].sum().reset_index()

  agg1 = df.groupby(grpby1)[sumCol].sum()
  agg1 = agg1.groupby(level=grpby2).apply(lambda x:100 * x / float(x.sum())).reset_index()
  agg1.rename(columns={sumCol:f"{sumCol}_percent"}, inplace=True)
  
  agg1=agg1[agg1[f"{sumCol}_percent"]!=0]
  #   agg1=agg1.merge(agg2,on=grpby2)
  #   agg1[f'{sumCol}_percent']=np.round(agg1[f'{sumCol}_x']/agg1[f'{sumCol}_y']*100,0)
  #   agg1=agg1[agg1[outCol]!=0]

    ##NOTE:
  #   agg1.div(agg2, level=grpby2) * 100  doesnot work

  ##print(agg1.groupby(grpby2)[f"{sumCol}_percent"].sum())
  agg1[f"{sumCol}"]= pd.Series(df.groupby(grpby1)[sumCol].sum().values)
  
  return agg1

def rle_encode(data):
    #Ref:https://stackabuse.com/run-length-encoding/  
    encoding = ''
    prev_char = ''
    count = 1

    if not data: return ''

    for char in data:
        if char != prev_char:
            if prev_char:
                encoding += prev_char+"("+str(count) +");" 
            count = 1
            prev_char = char
        else:
            count += 1
    else:
        encoding += prev_char+"("+str(count) +");" 
        return encoding

# ----------------------------Feature Eng. Functions ----------------------
def get_dummies2_sub(x, allLabels):
    """ a sub-function for get_dummies2 function

    -------
    Author: Reza Nourzadeh 
    """
    ind = np.in1d(allLabels, x) * 1
    return ind

def sparseLabel_sub(x, splitter="; "):
    """ a sub-function for sparseLabel function

    -------
    Author: Reza Nourzadeh 
    """
    # TODO: how we can apply in two mat in parallel
    try:
        tmp = x[1].split(splitter)
        x2 = x[2:]
        x2[np.where(x2)] = np.fromstring(x[0], dtype=int, sep=splitter)
    except Exception :
        print("error in sparseLabel_sub")
        print(x)
        pdb.set_trace()

    # if len(set(tmp))!=len(tmp):
    #     tmp3=pd.Series(data=np.fromstring(x[0], dtype=int, sep=splitter)  , index=tmp)
    #     tmp3=tmp3.groupby(tmp3.index).sum()
    #     x2[np.where(x2)]=tmp3
    #     # try:
    #     #     x2[np.where(x2)]=tmp3
    #     # except Exception:
    #     #     print("stp1")
    #     #     pdb.set_trace()
    # else:
    #     x2[np.where(x2)]=np.fromstring(x[0], dtype=int, sep=splitter)
    #     # try:
    #     #     x2[np.where(x2)]=np.fromstring(x[0], dtype=int, sep=splitter)
    #     # except Exception :
    #     #     print("stp2")
    #     #     pdb.set_trace()
    return x2

def get_dummies2(df, splitter='; '):
    """ a function like panda.get_dummies, however, get_dummies is very slow and has out of memory issue for large data frame
        Convert categorical variable into dummy/indicator variables.

    Parameters:
    ----------
    df : array-like, Series, or DataFrame Data of which to get dummy indicators.

    splitter (string) default ‘; ’
    String or regular expression to split on.

    returns:
    -------
    DataFrame
    Dummy-coded data.

    -------
    Author: Reza Nourzadeh 
    """

    df = df.str.split(splitter, expand=True)
    tmp = df.values.flatten()
    tmp = tmp[~(pd.isnull(tmp))]
    allLabels = np.sort(np.unique(tmp))
    tmp1 = np.apply_along_axis(
                                get_dummies2_sub,
                                1,
                                df.values,
                                allLabels=allLabels)
    tmp1 = pd.DataFrame(tmp1, columns=allLabels, index=df.index)
    return tmp1

def sparseLabel(df, prodCol, priceCol, splitter="; "):
    """create a Dummy-coded data of column prodCol of dataframe df which fills with value of column priceCol

    Parameters:
    ----------
    df(pandas dataframe) with [sample*features] format

    prodCol (a list of strings): column names

    priceCol (a list of strings): the value of columns

    splitter (string), default ‘; ’
    String or regular expression to split on.

    returns:
    ----------
    DataFrame
    Dummy-coded data of column prodCol which fills with value of priceCol

    ###NOTE: prodCol and priceCol should have same order
    Example1: prodCol=['INT10, SPN, TMN, INT20'], priceCol="10,20,30,5"
        output= INT10, SPN, TMN, INT50
                10   , 20 , 30,  5

    -------
    Author: Reza Nourzadeh 
    """
    #---debugging:
    # prodCol='PRODS_B4' 
    # priceCol='MRRS_B4'
    # splitter="; "

    tmp1 = get_dummies2(df[prodCol], splitter)
    print("dummies generated")

    tmp2 = pd.concat([df[priceCol], df[prodCol], tmp1], axis=1)
    out0 = np.apply_along_axis(sparseLabel_sub, 1, tmp2, splitter=splitter)

    out = pd.DataFrame(
        out0,
        dtype=np.int16,
        columns=tmp1.columns.str.upper(),
        index=df.index)
    return out

def fill_with_colnames(udata):
    """fills non zero elements of a dataframe with their column names

    Parameters:
    ----------
    udata(pandas dataframe) with [sample*features] format

    returns:
    ----------
    udata(pandas dataframe) with [sample*features] format and column names as new value for nonzero elements

    -------
    Author: Reza Nourzadeh 
    """

    tmp = np.tile(udata.columns, [len(udata.index), 1])
    tmp2 = pd.DataFrame(
        np.where(
            udata.astype(int),
            tmp,
            0),
        columns=udata.columns,
        index=udata.index)
    # tmp2 = tmp2.replace(0, "")
    return (tmp2)

def joinNonZero(x, sep=', '):
    """Concatenate a list or tuple of non zero vlaues with intervening occurrences of sep
    ###TODO: it should be corrected xx!=0???

    Parameters:
    ----------
    x (a list of strings)

    sep (string): default value=', '
    String or regular expression to split on

    returns:
    ----------
    y string

    -------
    Author: Reza Nourzadeh 
    """
    y = sep.join(list(filter(lambda xx: xx != 0, x)))
    return(y)

def prodDesc_clean(prodDesc, df):
    """removes duplicted prodDesc values and keeps only rows in prodDesc that their  product_id  match with product_id in df

    Parameters:
    ----------
    prodDesc(pandas dataframe) with [prodcuts*features] format: a product dataframe which consists of ['PRODUCT_ID','LOB'] columns

    udata(pandas dataframe) with [sample*features] format

    returns:
    ----------
    prodDesc2(pandas dataframe) with [sample*features] format

    -------
    Author: Reza Nourzadeh 
    """

    prodDesc2 = prodDesc[['PRODUCT_ID', 'LOB']].drop_duplicates()
    prodDesc2['PRODUCT_ID'] = prodDesc2['PRODUCT_ID'].str.upper()
    prodDesc2['LOB'] = prodDesc2['LOB'].str.upper()

    tmp = (pd.DataFrame(df.columns.str.upper(), columns=['PRODUCT_ID']))
    prodDesc2 = tmp.merge(prodDesc2, on='PRODUCT_ID', how='left')
    # prodDesc2=prodDesc2.loc[~prodDesc2['LOB'].isna(),:]

    if prodDesc2.shape[0] != df.shape[1]:
        print(prodDesc2)
        print(df.columns)
        # pdb.set_trace()
        sys.exit("something is wrong with prodDesc2")

    return prodDesc2

def condense_cols(df, remove_prefix, umap):
    """joins all columns in a row to a single string that are seprated with ", "

    Parameters:
    ----------
    df(pandas dataframe) with [sample*features] format

    remove_prefix (True/False): 
    if it is Ture,prefrix with format *_ will be deleted

    umap (dictionary):
    a dictionary to map old value to new values before joining ß

    returns:
    ----------
    df2 (pandas dataframe) with [sample*features] format

    -------
    Author: Reza Nourzadeh 
    """

    if remove_prefix:
        df.columns = pd.Series(["_".join(x.split('_')[1:])
                                for x in df.columns])
    # pdb.set_trace()
    if len(umap) != 0:
        df.rename(columns=umap, inplace=True)
    df2 = fill_with_colnames(df)
    df2 = df2.apply(joinNonZero, sep=', ', axis=1)
    return df2

def sortPrds(x, y):
    """sort values of classifier.predict_proba(x) based on probability

    Parameters:
    ----------
    x : a row of classifier.predict_proba(x)
    y : column names of classifier.predict_proba(x); inother words classes of y

    returns:
    ----------
    y and x , sorted by x


    -------
    Author: Reza Nourzadeh 
    """
    x = x.values
    y = y.values
    # pdb.set_trace()
    idx = np.argsort(-x)
    x = x[idx]
    y = y[idx]

    idx = np.argwhere(x == 0)
    # pdb.set_trace()
    y[idx] = ''

    return y.tolist() + x.tolist()

def cat2num(df,cat_decoder='OneHotEncoder'):
    # -------Conversion cat to numerical
    #TODO:add to functions or use existing lib
    cat_columns = df.select_dtypes(include=['object','category']).columns.tolist()
    if len(cat_columns) != 0:
        df[cat_columns] = df[cat_columns].applymap(lambda x: str(x).lower().strip() if not pd.isnull(x) else x) #lower case
        print('Categorical columns...')
        tmp=df[cat_columns].nunique()
        tmp.sort_values(inplace=True,ascending=False)
        print(tmp)
        ### debugging:         
        # df=df0.copy()
        # cat_columns = df.select_dtypes(include=['object']).columns.tolist()+df.select_dtypes(include=['category']).columns.tolist()

        if cat_decoder=='OneHotEncoder':
            df[cat_columns]=df[cat_columns].fillna('None').astype('category')
            from sklearn.preprocessing import OneHotEncoder  
            enc = OneHotEncoder()
            tmp = enc.fit_transform(df[cat_columns]) 
            
            tmp2=pd.DataFrame(tmp.todense(),columns=enc.get_feature_names(cat_columns ),index=df.index)  
            df=pd.concat([df.drop(cat_columns,axis=1),tmp2],axis=1)
        else:
            df[cat_columns] = df[cat_columns].apply(lambda x: x.astype('category').cat.codes)
    
    return df

def flexible_join(left_df, right_df, left_on=None, right_on=None, on=None, how='inner', **kwargs):
    """
    Join two DataFrames with flexible string matching that handles differences in:
    - spaces, underscores, and other special characters (/, -, etc.)
    - letter case (upper/lower)
    
    Parameters:
    -----------
    left_df : pandas DataFrame
        Left DataFrame to join
    right_df : pandas DataFrame
        Right DataFrame to join
    left_on : str or list of str, optional
        Column(s) from left_df to use as join key(s)
    right_on : str or list of str, optional
        Column(s) from right_df to use as join key(s)
    on : str or list of str, optional
        Column name(s) to join on if column names are identical in both DataFrames
    how : str, default 'inner'
        Type of join to perform ('inner', 'left', 'right', 'outer')
    **kwargs : 
        Additional keyword arguments to pass to pd.merge()
        
    Returns:
    --------
    pandas DataFrame
        Joined DataFrame
    """
    # Create copies to avoid modifying the original DataFrames
    left_copy = left_df.copy()
    right_copy = right_df.copy()
    
    # Handle the case where 'on' is specified
    if on is not None:
        left_on = right_on = on
    
    # Convert single column to list
    if isinstance(left_on, str):
        left_on = [left_on]
    if isinstance(right_on, str):
        right_on = [right_on]
    
    # Make sure we have valid join columns
    if left_on is None or right_on is None:
        raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")
    
    # Make sure the lengths match
    if len(left_on) != len(right_on):
        raise ValueError("Length of 'left_on' must equal length of 'right_on'")
    
    # Create normalized versions of each join column
    left_norm_cols = []
    right_norm_cols = []
    
    for lcol, rcol in zip(left_on, right_on):
        # Create normalized column names that include the original column names
        left_norm_col = f"_normalized_left_{lcol}"
        right_norm_col = f"_normalized_right_{rcol}"
        
        # Add to our lists of normalized columns
        left_norm_cols.append(left_norm_col)
        right_norm_cols.append(right_norm_col)
        
        # Create the normalized columns
        left_copy[left_norm_col] = left_copy[lcol].apply(normalize_string)
        right_copy[right_norm_col] = right_copy[rcol].apply(normalize_string)
    
    # Perform the join on the normalized keys
    result = pd.merge(
        left_copy, right_copy,
        left_on=left_norm_cols,
        right_on=right_norm_cols,
        how=how,
        **kwargs
    )
    
    # Drop the temporary normalized key columns
    result = result.drop(columns=left_norm_cols + right_norm_cols)
    
    return result

# -------------------------------------------------------------------------
# ---------------------------Graph and plot functions----------------------
def corrmap(df0, method='kendall', diagonal_plot=True, **kwargs):
    """ plot a correlation heatmap matrix

    Parameters:
    ----------
    uData : (pandas dataframe) with [sample*features] format

    method : {‘pearson’, ‘kendall’, ‘spearman’} or callable
                pearson : standard correlation coefficient
                kendall : Kendall Tau correlation coefficient
                spearman : Spearman rank correlation
                callable: callable with input two 1d ndarray and returning a float.

    **kwargs: parameter of corr and seaborn heatmap: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html

    -------
    Author: Reza Nourzadeh- reza.nourzadeh@gmail.com
    """
    import inspect
    
    corr_args = list(inspect.signature(pd.DataFrame.corr).parameters)
    kwargs_corr = {k: kwargs.pop(k) for k in dict(kwargs) if k in corr_args}
    
    heatmap_args = list(inspect.signature(sns.heatmap).parameters)
    kwargs_heatmap = {k: kwargs.pop(k) for k in dict(kwargs) if k in heatmap_args}
    
    corr = df0.dropna(how='any',axis=0).drop_duplicates().corr(method=method,**kwargs_corr)
    # Generate a mask for the upper triangle
    
    if diagonal_plot:
      mask = np.zeros_like(corr)
      mask[np.triu_indices_from(mask)] = True
    else:
      mask=None

    plt.figure(figsize = (30,20))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    snsPlot = sns.heatmap(                                    
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
    figure = snsPlot.get_figure()
    # figure.savefig(os.path.join(outputFolder,"corr_map.png"), bbox_inches='tight')
    plt.show()
    plt.close()
    
    return corr, figure

def sankey(left, right, value, thershold, utitle, filename):
    """ create and plot a simplified sankey graph with only one source (left) and one target(right)

    Parameters:
    ----------
    left (array, shape) [n_samples]
    the label of left column

    right (array, shape) [n_samples]
    the label of right column

    value (array, shape) [n_samples]
    the value of transaction

    thershold (float)
    to filter those transactions which have less value than thershold

    utitle (string):
    the title of the plot

    filename (string):
    the location of the plot

    returns:
    -------
    tranactions(array, shape) [filtered n_samples*3]
    tranactions has three columns: left, right, value

    -------
    
    """

    tranactions0 = pd.concat(
        [left.rename('left'), right.rename('right'), value.rename('value')], axis=1)
    tranactions = tranactions0.groupby(
        ['left', 'right'], as_index=False).agg('sum')
    counts = tranactions0.groupby(
        ['left', 'right'], as_index=False).agg('count')
    tranactions = tranactions.loc[counts['value'] > thershold, :]
    tranactions.sort_values(['value'], ascending=[False], inplace=True)
    left = tranactions['left']
    right = tranactions['right']
    values = tranactions['value']

    #import chart_studio.plotly as py
    import plotly

    lbLeft = list(pd.unique(left))
    lbRight = list(pd.unique(right))

    # label=lbLeft+lbRight
    source = []
    target = []
    value = []
    for i in list(range(left.shape[0])):
        # if i==3:
        #     pdb.set_trace()
        tmpSource = np.where(
            np.asarray(lbLeft) == np.asarray(
                left.iloc[i]))[0].tolist()
        source = source + tmpSource

        tmpTarget = np.where(
            np.asarray(lbRight) == np.asarray(
                right.iloc[i]))[0].tolist()
        target = target + tmpTarget

        tmpValue = [values.iloc[i]]
        value = value + tmpValue

    target = [x + len(lbLeft) for x in target]

    data = dict(
        type='sankey',
        node=dict(
            pad=15,
            thickness=20,
            line=dict(
                color="black",
                width=0.5
            ),
            label=list(pd.unique(left)) + list(pd.unique(right))
            # color = ["blue", "blue", "blue", "blue", "blue", "blue"]
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        ))

    layout = dict(
        title=utitle,
        font=dict(
            size=10
        )
    )
    fig = dict(data=[data], layout=layout)
    plotly.offline.plot(fig, filename=filename)
    return tranactions

def worldCloud_graph(txtSeries_df,outputFile):  
    ##TODO: document it  
    # txtSeries_df=tmp['prizm_68_2019']
    # outputFile=os.path.join(outputFolder,'enviroPostal_wordCloud2.png')

    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    
    if  type(txtSeries_df)==pd.core.series.Series:
        txt=' '.join(txtSeries_df.astype('str'))
        worldCloud_instance = WordCloud(width=800, height=400).generate(txt)
    else:
        worldCloud_instance = WordCloud(width=800, height=400).generate_from_frequencies(dict(zip(txtSeries_df.iloc[:,0] ,txtSeries_df.iloc[:,1])))

    ## Generate plot
    plt.figure(figsize=(20,10), facecolor='k')
    plt.imshow(worldCloud_instance)
    plt.axis("off")
    plt.savefig(outputFile, bbox_inches='tight')
    plt.close('all')
    # print("it was saved in "+os.path.join(outputFile))

def plot3D(udata, uY, xyzLabels, utitle, outPutFile):
    """ plot3d of udata

    parameters:
    ----------
    udata : (pandas dataframe) with [sample*3] format

    uY (string):
    the name of columns which is used to color dot plot

    xyzLabels: the list of string
    labels of Axis X,Y,Z

    utitle (string):
    the title of graph

    outputFile (string):
    the location of the plot

    -------
    
    """

    # %matplotlib notebook
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(xyzLabels[0], fontsize=15)
    ax.set_ylabel(xyzLabels[1], fontsize=15)
    ax.set_zlabel(xyzLabels[2], fontsize=15)
    ax.set_title(utitle, fontsize=20)
    targets = pd.unique(uY)
    colors = ['r', 'g', 'b', 'y']
    for target, color in list(zip(targets, colors)):
        indicesToKeep = uY.squeeze() == target
        ax.scatter(udata[indicesToKeep, 0], udata[indicesToKeep, 1],
                   udata[indicesToKeep, 2], c=color, s=50, alpha=.5)
    ax.legend(pd.unique(uY).astype('str'))
    ax.grid()

    plt.tight_layout()
    plt.savefig(outPutFile, format='png', dpi=300, bbox_inches='tight')
    plt.close('all')
    print('Plot saved in ' + outPutFile)
    # for angle in list(range(0, 360,60)):
    #     ax.view_init(30, angle)
    #     # plt.draw()
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(outputFolder,fileName.split('.')[0]+str(angle)+'.'+fileName.split('.')[1]) , format='png', dpi=300, bbox_inches='tight')
    #     plt.pause(.001)

def lag_plot(x, y=None, nlags=24):
  # x=x.dropna()
  # print(f"data size after removing nulls({plant}):", x.shape )
  with sns.plotting_context("paper"):
    fig, ax = plt.subplots(nrows=math.ceil((nlags)/4), ncols=4, figsize=[15, 10])
    
    if y is None:
      fig.suptitle(f'Auto correlation plot {x.name}', fontsize=30)
      for i, ax_ in enumerate(ax.flatten()):
          
          pd.plotting.lag_plot(x, lag=i + 1, ax=ax_)
          # ax_.set_title(f"Lag {i+1}")
          ax_.ticklabel_format(style="sci", scilimits=(0, 0))
          ax_.set_ylabel(f"{x.name}$_t$")
          ax_.set_xlabel(f"{x.name}$_{{t-{i}}}$")
    else:
      fig.suptitle(f'Cross correlation plot {x.name} vs {y.name}', fontsize=30)
      for i, ax_ in enumerate(ax.flatten()):
          ax_.scatter(y=y, x=x.shift(periods=i), s=10)
          ax_.set_ylabel(f"{y.name}$_{{t}}$")
          ax_.set_xlabel(f"{x.name}$_{{t-{i}}}$")

    # plt.tight_layout()

def plot_ccf(x, y, lags,  ax=None, title="Cross-correlation", **kwargs):
  from statsmodels.tsa.stattools import ccf
  from matplotlib.ticker import MaxNLocator
  # Compute CCF and confidence interval
  cross_corrs = ccf(x, y, **kwargs)
  ci = 2 / np.sqrt(len(y))
  # Create plot
  if ax is None:
    fig, ax = plt.subplots(figsize=[10, 5])
  ax.stem(range(0, lags + 1), cross_corrs[: lags + 1])
  ax.fill_between(range(0, lags + 1), ci, y2=-ci, alpha=0.2)
  ax.set_title(title)
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  return ax, cross_corrs

def figures_to_html(figs, filename="dashboard.html"):
    """save a list of figures all to a single HTML file.
    """
    ##from https://stackoverflow.com/questions/45577255/plot-multiple-figures-as-subplots
    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for fig in figs:
            inner_html = fig.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")
        
def save_plotly_fig(fig, fname_prefix, image_format="jpg"):
    import plotly      
    uFiles = [f"{fname_prefix}.{ext}" for ext in ["html", "json", image_format]]
    for uFile in uFiles:
        ext = uFile.split(".")[-1]
        if ext == "html":
            plotly.offline.plot(fig, filename=uFile, auto_open=False)
        elif ext == "json":
            plotly.io.write_json(fig, uFile)
        else:
            fig.write_image(uFile, width=2400, height=1400, scale=4)
    return uFiles

def cat2color_plotly(label_col, color_palette=None):
  import plotly.express as px
  # color_palette=px.colors.qualitative.Antique
  if color_palette is None:
    ## https://plotly.com/python/discrete-color/
    color_palette=px.colors.qualitative.Alphabet  ##Light24  ##Plotly
  domain=label_col.unique()
  if len(domain)>len(color_palette):
    print(f"number of available colors({len(color_palette)}) is more than categorizes({len(domain)}), change the palette")
  color_map = dict(zip(domain, color_palette[:len(domain)+2])) 
  c=label_col.map(color_map)
  return c, color_map

def plotly_group_stack(df_plot,
                        col2grp,
                        col2stack,
                        col2c,
                        date_col,
                        title,
                        color_palette=px.colors.qualitative.Light24,
                        patterns=['','/']
											):
  import plotly.graph_objects as go
  import plotly.express as px
  import pandas as pd

  x = [list(df_plot[date_col].dt.date.values),list(df_plot[col2grp].values)]

  colors,color_map=cat2color_plotly(df_plot[col2grp], color_palette=color_palette)

  fig = go.Figure()

  for shift_name, upattern in zip(df_plot[col2stack].unique(), patterns):
    df_tmp = df_plot.mask(df_plot[col2stack]!=shift_name, pd.NA)
    for Machine in df_plot[col2grp].unique():
      y = df_tmp[col2c].mask(df_plot[col2grp]!=Machine, pd.NA)
      fig.add_bar(
                  x=x,
                  y=y,
                  name=f"{Machine} - {shift_name}", 
                  hovertext = df_plot[col2stack],
                  marker_color=color_map[Machine],
                  marker_pattern_shape=upattern,
                  legendgroup=shift_name,
                  legendgrouptitle_text=shift_name,
                  hovertemplate="Date: %{x[0]}<br>"+
                                "Machine: %{x[1]}<br>"+
                                "Efficiency: %{y}<br>"+
                                "DayNight: %{hovertext}<br>",
                )	

  fig.update_layout(
                    barmode="relative",
                    xaxis_title="Date",
                    yaxis_title=col2c,
                    legend_title_text='',
                    title=title
                    )
  return fig

def subplot_plExpress(figs, sub_titles, main_title):
  from plotly.subplots import make_subplots
  figure_traces=[]
  for con, fig_sub in enumerate(figs):
    figure_traces_sub=[]
    for trace in range(len(fig_sub["data"])):
        if con>0: 
          fig_sub["data"][trace]['showlegend'] = False 
        figure_traces_sub.append(fig_sub["data"][trace])
    figure_traces.append(figure_traces_sub)
  figure = make_subplots(rows = 3, cols = 1, subplot_titles =sub_titles)
  figure.update_layout(height = 500, width = 1200, title_text =main_title, title_font_size = 25)
  for con, figure_traces_sub in enumerate(figure_traces):
    for traces in figure_traces_sub:
        figure.append_trace(traces, row = con+1, col = 1)
  return figure

from typing import Optional, Dict, Any, Tuple, Literal
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import scipy
from dataclasses import dataclass
from typing import List

@dataclass
class PlotConfig:
    """Configuration class for plot customization"""
    height: int = 800
    width: int = 1200
    colors: List[str] = None
    theme: str = 'white'  # or 'dark'
    title_font_size: int = 24
    axis_font_size: int = 14
    bins: int = 20
    boxpoints: Literal['all', 'outliers', False] = 'outliers'
    kde_points: int = 100
    jitter: float = 0.3
    violin_points: bool = True
    first_plot: Literal['box', 'violin'] = 'box'
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        if self.first_plot not in ['box', 'violin']:
            raise ValueError("first_plot must be either 'box' or 'violin'")

def validate_input(data: pd.DataFrame, dependent_var: str, group_var: str) -> None:
    """Validate input data and variables"""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input 'data' must be a pandas DataFrame")
        
    if dependent_var not in data.columns:
        raise ValueError(f"Column '{dependent_var}' not found in DataFrame")
        
    if group_var not in data.columns:
        raise ValueError(f"Column '{group_var}' not found in DataFrame")
        
    if not pd.api.types.is_numeric_dtype(data[dependent_var]):
        raise ValueError(f"Column '{dependent_var}' must be numeric")
        
    if len(data[group_var].unique()) > 10:
        raise ValueError("Too many groups (>10) for meaningful visualization")

def plot_distributions(
    data: pd.DataFrame,
    dependent_var: str,
    group_var: str,
    config: Optional[PlotConfig] = None
) -> go.Figure:
    """
    Create interactive distribution plots for each group using plotly
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input DataFrame containing the data
    dependent_var : str
        Name of the column containing the dependent variable
    group_var : str
        Name of the column containing the grouping variable
    config : PlotConfig, optional
        Configuration object for customizing the plots
        
    Returns:
    --------
    plotly figure object
    """
    # Input validation
    validate_input(data, dependent_var, group_var)
    
    # Use default config if none provided
    if config is None:
        config = PlotConfig()

    # Create subplot figure with 4 subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'{config.first_plot.capitalize()} Plot', 
            'Histogram with KDE',
            'Empirical CDF',
            'Q-Q Plot'
        )
    )
    
    groups = sorted(data[group_var].unique())
    
    # 1. Box or Violin plot (top-left)
    for i, group in enumerate(groups):
        group_data = data[data[group_var] == group][dependent_var]
        if config.first_plot == 'box':
            fig.add_trace(
                go.Box(
                    y=group_data,
                    name=group,
                    boxpoints=config.boxpoints,
                    jitter=config.jitter,
                    pointpos=-1.8,
                    marker_color=config.colors[i % len(config.colors)],
                    marker=dict(size=6),
                    legendgroup=group,
                    showlegend=True
                ),
                row=1, col=1
            )
        else:  # violin plot
            fig.add_trace(
                go.Violin(
                    y=group_data,
                    name=group,
                    box_visible=True,
                    meanline_visible=True,
                    points="all" if config.violin_points else None,
                    marker_color=config.colors[i % len(config.colors)],
                    legendgroup=group,
                    showlegend=True
                ),
                row=1, col=1
            )

    # 2. Histogram with KDE (top-right)
    for i, group in enumerate(groups):
        group_data = data[data[group_var] == group][dependent_var]
        
        # Calculate KDE with proper scaling
        kde_x = np.linspace(group_data.min(), group_data.max(), config.kde_points)
        kde = scipy.stats.gaussian_kde(group_data)
        kde_y = kde(kde_x)
        
        # Scale KDE to match histogram height
        hist, bin_edges = np.histogram(group_data, bins=config.bins)
        scaling_factor = max(hist) / max(kde_y)
        kde_y = kde_y * scaling_factor
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=group_data,
                name=group,
                opacity=0.7,
                nbinsx=config.bins,
                marker_color=config.colors[i % len(config.colors)],
                legendgroup=group,
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add KDE line
        fig.add_trace(
            go.Scatter(
                x=kde_x,
                y=kde_y,
                name=group,
                line=dict(color=config.colors[i % len(config.colors)]),
                legendgroup=group,
                showlegend=False
            ),
            row=1, col=2
        )

    # 3. ECDF (bottom-left)
    for i, group in enumerate(groups):
        group_data = data[data[group_var] == group][dependent_var]
        
        # Calculate ECDF
        sorted_data = np.sort(group_data)
        n = len(sorted_data)
        ecdf = np.arange(1, n + 1) / n
        
        fig.add_trace(
            go.Scatter(
                x=sorted_data,
                y=ecdf,
                name=group,
                mode='lines',
                line=dict(color=config.colors[i % len(config.colors)]),
                legendgroup=group,
                showlegend=False
            ),
            row=2, col=1
        )

    # 4. Q-Q Plot (bottom-right)
    for i, group in enumerate(groups):
        group_data = data[data[group_var] == group][dependent_var]
        qq = scipy.stats.probplot(group_data, dist="norm")
        
        # Add Q-Q points
        fig.add_trace(
            go.Scatter(
                x=qq[0][0],
                y=qq[0][1],
                mode='markers',
                name=group,
                marker=dict(color=config.colors[i % len(config.colors)]),
                legendgroup=group,
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add Q-Q line
        z = np.polyfit(qq[0][0], qq[0][1], 1)
        p = np.poly1d(z)
        fig.add_trace(
            go.Scatter(
                x=qq[0][0],
                y=p(qq[0][0]),
                mode='lines',
                name=group,
                line=dict(color=config.colors[i % len(config.colors)], dash='dash'),
                legendgroup=group,
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=config.height,
        width=config.width,
        title=dict(
            text="Distribution Analysis",
            x=0.5,
            font=dict(size=config.title_font_size)
        ),
        template=f"plotly_{config.theme}",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        boxmode='group',
        violinmode='group'
    )
    
    # Update axes labels
    fig.update_xaxes(title_text=group_var, row=1, col=1, title_font=dict(size=config.axis_font_size))
    fig.update_xaxes(title_text=dependent_var, row=1, col=2, title_font=dict(size=config.axis_font_size))
    fig.update_xaxes(title_text=dependent_var, row=2, col=1, title_font=dict(size=config.axis_font_size))
    fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=2, title_font=dict(size=config.axis_font_size))
    
    fig.update_yaxes(title_text=dependent_var, row=1, col=1, title_font=dict(size=config.axis_font_size))
    fig.update_yaxes(title_text="Count", row=1, col=2, title_font=dict(size=config.axis_font_size))
    fig.update_yaxes(title_text="Cumulative Probability", row=2, col=1, title_font=dict(size=config.axis_font_size))
    fig.update_yaxes(title_text="Sample Quantiles", row=2, col=2, title_font=dict(size=config.axis_font_size))
    
    return fig

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# --------------------------Reading a SQL file/string----------------------
def split_sql_expressions_sub(text):
    """ split sql queries based on ";"

    Parameters:
    ----------
    text (string): (sql queries)

    returns:
    a list of queries
    --------

    -------
    
    """
    # from riskmodelPipeline.py
    results = []
    current = ''
    state = None
    for c in text:
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
            raise Exception('Illegal state %s' % state)

    if current:
        current = current.rstrip(';').strip()
        if current:
            results.append(current)
    return results

def parse_sql_file(file_name):
    """ read sql queries in a file

    Parameters:
    ----------
    file (string): the location of sql file

    returns:
    a list of queries
    --------

    -------
    
    """
    # from riskmodelPipeline.py
    check_path(file_name)
    sql_statements = []
    with open(file_name, 'r') as f:
        for sql_statement in split_sql_expressions_sub(f.read()):
            sql_statements.append(sql_statement)
    return(sql_statements)

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------EDA, Statisitcal analysis-----------------------
def date2Num(df, dateCols):
    """ returns a new dataframe after deducing dates columns from the min dates of columns

    Parameters:
    ----------
    df (pandas dataframe) with [sample*features] format

    datecols  (a list of strings):: name of columns with date values

    returns:
    -------
    df[dateCols]-eval(df[dateCols].min()[0])

    -------
    
    """

    tmploc, _ = inWithReg(dateCols, df.columns.values)
        
    if len(tmploc) != 0:
        
        df[tmploc] = df[tmploc].astype('datetime64[ns]')

        print("conversion date features to number (month): date - " + str(df[tmploc].min()[0]))
        tmp3 = df[tmploc].apply(lambda x: x - df[tmploc].min()[0])
        tmp3 = tmp3.apply(lambda x: x / np.timedelta64(1, 'M'))
        df[tmploc] = tmp3.fillna(-1).round(0).astype('int64')
        df[tmploc]=df[tmploc].replace(-1,np.nan)

    return(df)

def find_low_variance(df, thresh=0.0):
    """ returns name of features  with variance less than thersold

    Parameters:
    ----------
    df(pandas dataframe) with [sample*features] format

    thresh(number): thersold value for variance of features

    -------
    
    """

    variance = df.var(skipna=True)
    low_variance = list(variance[variance <= thresh].index)
    return low_variance

def kruskalwallis2(x, y):
    """ calculate the Kruskal-Wallis H-test for independent samples

    Parameters:
    ----------
    x  array of observations
    y  array of groups

    returns:
    -------
    H-statistic  (float)
    The Kruskal-Wallis H statistic, corrected for ties
    p-value  (float)
    The p-value for the test using the assumption that H has a chi square distribution
    """
    groupednumbers = {}
    from scipy import stats
    for grp in y.unique():
        groupednumbers[grp] = x.values[y == grp]
    args = groupednumbers.values()
    tmp = stats.mstats.kruskalwallis(*args)
    # pdb.set_trace()
    return (tmp)

def chi2_contingency(x, y):
    """ Chi-square test of independence of variables in a contingency table.
    This function computes the chi-square statistic and p-value for the hypothesis test of independence of the observed frequencies in the contingency table.

    Parameters:
    ----------
    x  array of sample1

    y  array of sample2

    returns:
    -------
    p (float) :The p-value of the test

    Example:
    ---------

    -------
    
    """
    from scipy import stats
    xtab = pd.crosstab(x, y)
    pval = None
    if xtab.size != 0:
        try:
            _, pval, _, _ = stats.chi2_contingency(xtab)
        except Exception:
            pval = 0
    return pval

def corr_pointbiserial(binary_data, continuous_data, data):
    # TODO: correct it with nan
    """ computes the point biserial correlation of two pandas data frame columns

    Parameters:
    ----------
    binary_data :list : name of dichotomous data column

    continuous_data: list : name of dichotomous data column

    data (pandas dataframe) with [sample*feature] format: dataframe where above columns come from

    returns:
    -------
    out :  Point Biserial Correlation
    -------
    
    """

    import math
    bd_unique = data[binary_data].unique()
    
    g0 = data[data[binary_data] == bd_unique[0]][continuous_data]
    g1 = data[data[binary_data] == bd_unique[1]][continuous_data]
    
    s_y = np.std(data[continuous_data])
    n = len(data[binary_data])
    n0 = len(g0)
    n1 = len(g1)
    m0 = g0.mean()
    m1 = g1.mean()
    out=(m0-m1)*math.sqrt((n0*n1)/n**2)/s_y
    return out

def highcorr_finder(corrMat,df_scores,thershold):
    #TODO: add comment 
    corr_matrix = corrMat.abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find features with correlation greater than x
    drop_cols = [x for x in upper.columns if any(upper[x] >thershold)]
    drops_rows = [x for x in upper.index   if any(upper.loc[x] >thershold)]

    to_drop = list()
    mat_ind=list()
    for x,y in zip (drop_cols,drops_rows):
        tmp= y if df_scores[x]>df_scores[x] else x
        to_drop.append(tmp)
        mat_ind.append(x)
        mat_ind.append(y)

    high_corrs = dict(zip(drop_cols, drops_rows))

    return to_drop,mat_ind,high_corrs

def discretizer(x,y,labels=["Q1", "Q2", "Q3","Q4"],method='cut'): 
    # print('discretizer:'+method)

    if method=='cut':
        from sklearn import preprocessing
        min_max_scaler = preprocessing.MinMaxScaler()        
        out,bins=pd.cut(min_max_scaler.fit_transform(x.values.reshape(-1,1)).reshape(-1),bins=[-.1,.33,.66,1.1],labels=labels,retbins=True)

    elif method=='tree1':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion = 'entropy',max_depth = 1)
        clf.fit(x.to_frame(),y)

        # if len(np.unique(clf.tree_.threshold[clf.tree_.threshold!=-2]))==1:
        #     # print ("max_depth increased")
        #     clf = DecisionTreeClassifier(criterion = 'entropy',max_depth = 2)
        #     clf.fit(x.to_frame(),y)

        bins =  np.sort(np.append(np.unique(clf.tree_.threshold[clf.tree_.threshold!=-2]),[x.max()+1/1e6,x.min()-1/1e6])).tolist()
        out=pd.cut(x,bins=bins,labels=labels[:(len(bins)-1)])

    elif method=='tree2':
        from sklearn.tree import DecisionTreeClassifier
        clf = DecisionTreeClassifier(criterion = 'entropy',max_depth = 2)
        clf.fit(x.to_frame(),y)

        bins =  np.sort(np.append(np.unique(clf.tree_.threshold[clf.tree_.threshold!=-2]),[x.max()+1/1e6,x.min()-1/1e6])).tolist()
        out=pd.cut(x,bins=bins,labels=labels[:(len(bins)-1)])
    
    else:
        ##TODO: it needs to be corrected:
        out,bins=pd.qcut(x.rank(method='first'),q=3,labels=labels,retbins=True)
        # out,bins=pd.qcut(x,q=3,labels=labels,retbins=True, duplicates='drop')
        # out,bins=pd.qcut(x+ jitter(x),q=3,labels=labels,retbins=True)

    bins=[np.float(x) for x in bins]
    print(x.name+':\n'+str(bins))
    return (out ,bins)

def jitter(a_series, noise_reduction=1000000):
    # https://stackoverflow.com/questions/20158597/how-to-qcut-with-non-unique-bin-edges
    #TODO: add docs
    return (np.random.random(len(a_series))*a_series.std()/noise_reduction)-(a_series.std()/(2*noise_reduction))

def extract_equation(results_pars):
  vars=results_pars.reset_index()
  vars[0]=np.round(vars[0],2).astype(str)
  vars['ploys']=vars['index'].str.extract('np.power\((.+?),')
  vars['power']=np.where(vars['ploys'].isnull(),np.nan, vars['index'].str[-2:-1])
  vars['index']=np.where(vars['ploys'].isnull(),vars['index'],vars['ploys']+'**'+vars['power'])
  equation=""
  for row in vars.iterrows():
    sign='' if (np.sign(float(row[1][0]))==-1) or (row[0]==0) else '+'
    tmp=f"{sign}{row[1][0]}" if row[1]['index']=='Intercept' else f"{sign}{row[1][0]}*{row[1]['index']}"
    equation+=tmp
  return equation

def analyze_categorical_data(data, independent_var, dependent_var, alpha=0.05):
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2_contingency
    import matplotlib.pyplot as plt
    import seaborn as sns

    """
    Perform statistical analysis on categorical variables.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing the variables
    independent_var : str
        Name of the independent variable column
    dependent_var : str
        Name of the dependent variable column
    alpha : float, optional
        Significance level for hypothesis testing (default is 0.05)
    
    Returns:
    --------
    dict
        Dictionary containing test results, contingency table, and visualization
    """
    # Create contingency table
    contingency_table = pd.crosstab(data[independent_var], data[dependent_var])
    
    # Perform Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Calculate Cramer's V
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    cramer_v = np.sqrt(chi2 / (n * min_dim))
    
    # Create result dictionary
    results = {
        'contingency_table': contingency_table,
        'chi_square_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'cramers_v': cramer_v,
        'significant': p_value < alpha
    }
    
    # Add interpretation
    results['interpretation'] = interpret_results_analyze_categorical(results, alpha)
    
    # Create visualization
    plt.figure(figsize=(10, 6))
    sns.heatmap(contingency_table, annot=True, cmap='YlOrRd', fmt='d')
    plt.title(f'Contingency Table: {independent_var} vs {dependent_var}')
    plt.tight_layout()
    
    return results

def interpret_results_analyze_categorical(results, alpha):
    """
    Interpret the statistical test results.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing test results
    alpha : float
        Significance level
    
    Returns:
    --------
    str
        Interpretation of results
    """
    interpretation = []
    
    # Chi-square test interpretation
    if results['p_value'] < alpha:
        interpretation.append(f"There is a statistically significant relationship between the variables (p-value = {results['p_value']:.4f} < {alpha}).")
    else:
        interpretation.append(f"There is no statistically significant relationship between the variables (p-value = {results['p_value']:.4f} > {alpha}).")
    
    # Cramer's V interpretation
    cramers_v = results['cramers_v']
    if cramers_v < 0.1:
        strength = "negligible"
    elif cramers_v < 0.3:
        strength = "weak"
    elif cramers_v < 0.5:
        strength = "moderate"
    else:
        strength = "strong"
    
    interpretation.append(f"The strength of the association is {strength} (Cramer's V = {cramers_v:.3f}).")
    
    return " ".join(interpretation)

import numpy as np
from scipy import stats
import pandas as pd
from typing import Tuple, Dict, Union, List
import matplotlib.pyplot as plt
import seaborn as sns

#TODO:make it better
def analyze_cat_num(
    data: pd.DataFrame,
    categorical_var: str,
    numeric_var: str,
    alpha: float = 0.05
) -> Dict[str, Union[str, float, Dict]]:
    """
    Performs statistical analysis for categorical independent variable and numeric dependent variable.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        The dataset containing both variables
    categorical_var : str
        Name of the categorical (independent) variable column
    numeric_var : str
        Name of the numeric (dependent) variable column
    alpha : float, optional
        Significance level for statistical tests (default is 0.05)
        
    Returns:
    --------
    Dict containing:
        - test_type: string indicating which test was performed
        - test_statistic: the test statistic value
        - p_value: the p-value from the test
        - significant: boolean indicating if result is significant
        - descriptive_stats: dictionary of descriptive statistics by group
        - assumption_tests: dictionary of assumption test results
    """
    # Validate input data
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if categorical_var not in data.columns or numeric_var not in data.columns:
        raise ValueError("Specified variables must exist in the DataFrame")
    
    # Get unique categories and check number of groups
    categories = data[categorical_var].unique()
    n_groups = len(categories)
    
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for comparison")
    
    # Create groups for analysis
    groups = [data[data[categorical_var] == cat][numeric_var].dropna() for cat in categories]
    
    # Calculate descriptive statistics
    descriptive_stats = data.groupby(categorical_var)[numeric_var].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3).to_dict('index')
    
    # Test for normality in each group
    normality_tests = {}
    for cat, group in zip(categories, groups):
        if len(group) >= 3:  # Shapiro-Wilk test requires at least 3 samples
            stat, p_val = stats.shapiro(group)
            normality_tests[cat] = {
                'statistic': stat,
                'p_value': p_val,
                'normal': p_val > alpha
            }
    
    # Test for homogeneity of variances
    levene_stat, levene_p = stats.levene(*groups)
    
    # Perform appropriate statistical test based on number of groups
    if n_groups == 2:
        # Perform t-test if normal, Mann-Whitney U test if not
        all_normal = all(test['normal'] for test in normality_tests.values())
        equal_var = levene_p > alpha
        
        if all_normal:
            stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=equal_var)
            test_type = "Independent t-test"
            if not equal_var:
                test_type += " with Welch's correction"
        else:
            stat, p_value = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
            test_type = "Mann-Whitney U test"
    else:
        # Perform one-way ANOVA if normal, Kruskal-Wallis if not
        all_normal = all(test['normal'] for test in normality_tests.values())
        equal_var = levene_p > alpha
        
        if all_normal and equal_var:
            stat, p_value = stats.f_oneway(*groups)
            test_type = "One-way ANOVA"
        else:
            stat, p_value = stats.kruskal(*groups)
            test_type = "Kruskal-Wallis H test"
    
    # Compile results
    results = {
        'test_type': test_type,
        'test_statistic': stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'descriptive_stats': descriptive_stats,
        'assumption_tests': {
            'normality': normality_tests,
            'homogeneity_of_variance': {
                'statistic': levene_stat,
                'p_value': levene_p,
                'equal_variances': levene_p > alpha
            }
        }
    }
    
    return results

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# --------------------------------loggers----------------------------------
def logmaker(uFile, name, logLevel=logging.INFO):
    """ create a logger

    Parameters:
    ----------
    uFile (string): path of logger file

    name (string): name of logger

    loggingLevel integer:
    The numeric values of logging levels are given in the following table. These are primarily of interest if you want to define your own levels, and need them to have specific values relative to the predefined levels. If you define a level with the same numeric value, it overwrites the predefined value; the predefined name is lost.
    
    -------
    #numerical logging level:
                # CRITICAL	50
                # ERROR	    40
                # WARNING	30
                # INFO	    20
                # DEBUG	    10
                # NOTSET	0

    returns:
    -------
    logger (logger type):  created logger

    -------
    Author: Reza Nourzadeh 
    """

    #   logging.basicConfig(filemode='w')

    # configure log formatter
    logFormatter1 = logging.Formatter(
        '%(asctime)s | %(levelname)-3s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logFormatter2 = logging.Formatter("%(message)s")
    #   logFormatter2 = logFormatter1

    # configure file handler
    fileHandler = logging.FileHandler(uFile, 'w')
    fileHandler.setFormatter(logFormatter1)

    # configure stream handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter2)

    # get the logger instance
    logger = logging.getLogger(name)

    # set the logging level
    logger.setLevel(logLevel)

    if not len(logger.handlers):
        logger.addHandler(fileHandler)
        logger.addHandler(consoleHandler)
    return logger

def custom_print(message, logger=None):
    print(message)
    if logger:
        logger.info(message)

class loggerWriter_sub:
    """ sub class used for loggerWriter2
    ###---------- Redirect stderr and stdout to log --------------
    ### https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python

    -------
    Author: Reza Nourzadeh 
    """

    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)

def loggerWriter2(stdoutLvl, stderrLvl):
    """ Redirect stderr and stdout to log

    Parameters:
    ----------
    stdoutLvl (string): the level of stdout

    stderrLvl (string): the level of stderrLvl

    returns:
    -------
    old_stdout,old_stderr: stdout and stderr before Redirecting to log

    -------
    Author: Reza Nourzadeh 
    """

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = loggerWriter_sub(stdoutLvl)
    sys.stderr = loggerWriter_sub(stderrLvl)
    return(old_stdout, old_stderr)

import importlib.util

def setup_logger(log_file):
    """
    Set up logger to write to console and file
    Args:
        log_file (str): Path to the log file
    Returns:
        logging.Logger: Configured logger instance
    """
    import logging
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:  # Only create directory if log_file includes a path
        os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('intranet_downloader')
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

def load_config(config_file, logger):
    """
    Load sensitive credentials from config.yml
    Args:
        config_file (str): Path to the config file
        logger: Logger instance for logging messages
    Returns:
        dict: Dictionary containing credentials
    """
    import yaml
    
    try:
        if not os.path.exists(config_file):
            logger.error(f"Config file {config_file} not found")
            return {'username': '', 'password': ''}
            
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
            
        if not config or 'internet_credentials' not in config:
            logger.error("Invalid config file format: missing 'internet_credentials' section")
            return {'username': '', 'password': ''}
            
        return config['internet_credentials']
    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_file}: {str(e)}")
        return {'username': '', 'password': ''}
    except Exception as e:
        logger.error(f"Error loading {config_file}: {str(e)}")
        return {'username': '', 'password': ''}

def load_params(param_file):
    """
    Load parameters from the params file
    """
    try:
        spec = importlib.util.spec_from_file_location("params", param_file)
        params = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(params)
        return params
    except Exception as e:
        print(f"Error loading params file: {str(e)}")
        sys.exit(1)

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# --------------------------------Web Scraping and Crawling -------------------
def download_intranet_files(url, save_location, username, password, file_extensions, chunk_size, logger):
    """
    Download all files from specified intranet URL using NTLM authentication
    """
    import requests
    from requests_ntlm import HttpNtlmAuth
    from bs4 import BeautifulSoup
    from urllib.parse import urljoin
    from tqdm import tqdm
    import urllib3
    
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
        logger.info(f"Accessing {url}...")
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
                logger.info(f"\nDownloading: {filename}")
                
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
                logger.info(f"Saved: {save_path}")
                
            except Exception as e:
                logger.error(f"Failed to download {filename}: {str(e)}")
                failed_files.append(filename)
                continue
                
        # Print summary
        logger.info("\nDownload Summary:")
        logger.info(f"Successfully downloaded ({len(downloaded_files)}):")
        for file in downloaded_files:
            logger.info(f"  ✓ {file}")
            
        if failed_files:
            logger.warning(f"\nFailed downloads ({len(failed_files)}):")
            for file in failed_files:
                logger.warning(f"  ✗ {file}")
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred: {str(e)}")
    finally:
        session.close()

def download_webpage_content(url, save_location, username, password, logger):
    """Download content of a specified webpage using NTLM authentication"""
    import requests
    from requests_ntlm import HttpNtlmAuth
    from bs4 import BeautifulSoup
    import urllib3
    import os

    # Create save directory if it doesn't exist
    os.makedirs(save_location, exist_ok=True)

    # Create session with NTLM authentication and disable SSL verification
    session = requests.Session()
    session.verify = False
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    if username and password:
        session.auth = HttpNtlmAuth(username, password)
    else:
        session.auth = HttpNtlmAuth('', '')

    try:
        logger.info(f"Accessing {url}...")
        response = session.get(url)
        response.raise_for_status()

        # Save the webpage content
        filename = 'webpage_content.html'
        save_path = os.path.join(save_location, filename)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
        logger.info(f"Successfully saved webpage content to: {save_path}")

    except requests.exceptions.RequestException as e:
        logger.error(f"Error occurred: {str(e)}")
    finally:
        session.close()