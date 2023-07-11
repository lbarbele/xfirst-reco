#!/bin/bash

# *
# * helper functions
# *

function redirect_std()
{
  tty -s || {
    mkdir -p "${LOGS_DIR}"
    local -r base_path="${LOGS_DIR}/${SLURM_JOB_NAME}_job_${SLURM_JOB_ID:-0}${SLURM_STEP_ID:+_step_${SLURM_STEP_ID}_proc_${SLURM_PROCID}}"
    exec 1> "${base_path}.out"
    exec 2> "${base_path}.err"
  }
}
export -f redirect_std

function set_job_name() {
  local jobName="${1}"
  [ -z "${jobName}" ] && return 0

  scontrol update job "${SLURM_JOB_ID}" JobName="${jobName}"
  export SLURM_JOB_NAME="${jobName}"
}
export -f set_job_name

# *
# * directories
# *

export HOME_DIR='/scratch/astroparti/luan.arbeletche'
export BASE_DIR="${HOME_DIR}/ml-conex-xfirst"

export LOGS_DIR="${HOME_DIR}/joblogs/ml-conex-xfirst"
export DATA_DIR="${BASE_DIR}/data"
export MODELS_DIR="${BASE_DIR}/models"
export SCRIPTS_DIR="${BASE_DIR}/scripts"

# *
# * configuration affecting results
# *

# filter to apply when loading the datasets

export LOAD_FILTER='lgE >= 18'

# number of showers to load from conex files

export LOAD_TRAIN=1606308
export LOAD_VAL=1606308
export LOAD_TEST=803154

# number of showers to use on traning and evaluation

export NTRAIN=1000000
export NVAL=1000000
export NTEST=500000

# *
# * environment setup
# *

# rename job

if [ -n "${RENAME_JOB}" ]; then
  set_job_name "${RENAME_JOB}"
fi

# redirect output to the joblogs dir

redirect_std

# load modules

module load cmake/3.17.3
module load binutils/2.32
module load cudnn/8.2_cuda-11.4
module load gcc/11.1

# python version

export PYTHON_VERSION=3.11.3
export PYTHON_MAJOR=3

# python directories

export PYTHON_TARGET=${SCRATCH}/programas/Python-${PYTHON_VERSION}
export PYTHON_SRC=${PYTHON_TARGET}-src
export PYTHON_PACKAGES=${PYTHON_TARGET}-packages
export PYTHON_DEP=${PYTHON_TARGET}-dep

# openssl

export OPENSSL_TARGET=${PYTHON_DEP}/openssl-3.1.1
export PATH=${OPENSSL_TARGET}/bin:${PATH}
export LD_LIBRARY_PATH=${OPENSSL_TARGET}/lib64:${LD_LIBRARY_PATH}

# root

source /scratch/astroparti/luan.arbeletche/programas/root-6.28.02-python3.11.3/bin/thisroot.sh

# update paths

export PATH=${PYTHON_TARGET}/bin:${PATH}
export LD_LIBRARY_PATH=${PYTHON_TARGET}/lib:${LD_LIBRARY_PATH}
export PYTHONPATH="${BASE_DIR}:${PYTHON_PACKAGES}:${PYTHONPATH}"

# aliases

alias python=python3
alias pip=pip3
