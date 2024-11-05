###### pdoc3
import os
os.getcwd()
dbutils.fs.rm('dbfs:/tmp/',True)
!pdoc3 --html --force -o /dbfs/tmp/ ../io_funcs.py
io_funcs.dbfs2blob('/dbfs/tmp/io_funcs.html',
                   blob_dict={'container':'knifegatever2', 
                              'blob': f'io_funcs.html',
                              'storage_account':'knifegateprod'}
                   )

#####local system:
conda activate dstoolbox 
echo Y | pip uninstall dsToolbox  
cd C:\Users\rnourza\main_folder\codes\dsToolbox
rmdir /s /q dist build dsToolbox.egg-info 

# pip install .
python -m build
cd dist
pip install dsToolbox-0.2.0-py3-none-any.whl  --force-reinstall 

##if dstoolbox installed by cluster , it will be :
# '/databricks/python/lib/python3.8/site-packages/dsToolbox/sql_template.yml'
# '/databricks/python/lib/python3.8/site-packages/dsToolbox/config.yml'
##if it is install manually:
# /local_disk0/.ephemeral_nfs/envs/pythonEnv-c08bc406-031b-47d6-80a3-aa89d65fc06e/lib/python3.8/site-packages

##/dbfs/FileStore/dsToolbox/dsToolbox-0.1.0-py3-none-any.whl
# cd C:\Users\rnourza\Anaconda3\envs\dsToolbox\Lib\site-packages\dsToolbox

cd \
ipython
from ds_toolbox.dsToolbox import io_funcs


# get workspace
ws = Workspace.from_config()

# get compute target
target = ws.compute_targets['target-name']

# get curated environment
curated_env_name = 'AzureML-PyTorch-1.6-GPU'
env = Environment.get(workspace=ws, name=curated_env_name)

# get/create experiment
exp = Experiment(ws, 'experiment_name')

# distributed job configuration
distributed_job_config = MpiConfiguration(process_count_per_node=4, node_count=2)

# set up script run configuration
config = ScriptRunConfig(
    source_directory='.',
    script='script.py',
    compute_target=target,
    environment=env,
    distributed_job_config=distributed_job_config,
)

# submit script to AML
run = exp.submit(config)
print(run.get_portal_url()) # link to ml.azure.com
run.wait_for_completion(show_output=True)


###method1:
pip install .
python -m build

###method2:
python setup.py bdist_wheel
check-wheel-contents C:\Users\rnourza\main_folder\codes\ds_toolbox\dist
rmdir /s /q dist build dsToolbox.egg-info 


# COMMAND ----------
# import sys
# sys.modules.keys()
import types
def imports():
    for name, val in globals().items():
        if isinstance(val, types.ModuleType):
            yield val.__name__

# COMMAND ----------

import sys
modulenames = set(sys.modules) & set(globals())
allmodules = [sys.modules[name] for name in modulenames]
allmodules

# COMMAND ----------

import functions as mls

# COMMAND ----------