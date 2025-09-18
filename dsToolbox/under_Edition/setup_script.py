###### pdoc3
import glob, os, sys
upath = 'D:\\Dropbox\\codes\\ds_toolbox_public'
udesc_path = os.path.join(upath, 'doc')

ufiles=['ml_funcs.py', 'io_funcs.py', 'io_funcs_msql_local.py', 'nlp_llm_funcs.py', 'spark_funcs.py', 'common_funcs.py']
os.chdir(upath)
sys.path.insert(1, upath)
for ufile in  ufiles: ##glob.glob("*.py"):
    file=os.path.join(upath, "dsToolbox", ufile)
    print(file)
    desc_file = os.path.join(udesc_path, ufile.replace('.py', '.html'))
    if os.path.exists(desc_file):
        os.remove(desc_file)
    os.system(f"pdoc3 --html --force -o {udesc_path} {file}")  
    print("*"*150)
    print("*"*150)

#####local system:
# conda activate dstoolbox 
# echo Y | pip uninstall dsToolbox  
# cd C:\Users\rnourza\main_folder\codes\dsToolbox,
# rmdir /s /q dist build dsToolbox.egg-info 

# pip install .
# python -m build
# cd dist
# pip install dsToolbox-0.2.0-py3-none-any.whl  --force-reinstall 

##if dstoolbox installed by cluster , it will be :
# '/databricks/python/lib/python3.8/site-packages/dsToolbox/sql_template.yml'
# '/databricks/python/lib/python3.8/site-packages/dsToolbox/config.yml'
##if it is install manually:
# /local_disk0/.ephemeral_nfs/envs/pythonEnv-c08bc406-031b-47d6-80a3-aa89d65fc06e/lib/python3.8/site-packages

##/dbfs/FileStore/dsToolbox/dsToolbox-0.1.0-py3-none-any.whl
# cd C:\Users\rnourza\Anaconda3\envs\dsToolbox\Lib\site-packages\dsToolbox

#########Draft:
###method1:
# pip install .
# python -m build

###method2:
# python setup.py bdist_wheel
# check-wheel-contents C:\Users\rnourza\main_folder\codes\ds_toolbox\dist
# rmdir /s /q dist build dsToolbox.egg-info 


# COMMAND ----------
# import sys
# sys.modules.keys()
# import types
# def imports():
#     for name, val in globals().items():
#         if isinstance(val, types.ModuleType):
#             yield val.__name__

# COMMAND ----------
# import sys
# modulenames = set(sys.modules) & set(globals())
# allmodules = [sys.modules[name] for name in modulenames]
# allmodules