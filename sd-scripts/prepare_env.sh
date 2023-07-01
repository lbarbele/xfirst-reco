#!/bin/bash

# helper functions

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

# directories

export HOME_DIR='/scratch/astroparti/luan.arbeletche'
export BASE_DIR="${HOME_DIR}/ml-conex-xfirst"

export LOGS_DIR="${HOME_DIR}/joblogs/ml-conex-xfirst"
export DATA_DIR="${BASE_DIR}/data"
export SCRIPTS_DIR="${BASE_DIR}/scripts"
export MODELS_DIR="${BASE_DIR}/models"
export CONEX_DATA_DIR="${DATA_DIR}/conex"
export PROFILES_DATA_DIR="${DATA_DIR}/profiles"
export XFIRST_DATA_DIR="${DATA_DIR}/xfirst"
export FITS_DATA_DIR="${DATA_DIR}/fits/range-${MIN_DEPTH}-${MAX_DEPTH}"

export CONEX_JSON="${DATA_DIR}/conex.json"

# number of showers to use

export MAX_TRAIN=1800000
export MAX_VAL=1800000
export MAX_TEST=40000

# rename job

if [ -n "${RENAME_JOB}" ]; then
  set_job_name "${RENAME_JOB}"
fi

# redirect output to the joblogs dir

redirect_std

# environment

module load deepl/deeplearn-py3.7
export PYTHONPATH="${BASE_DIR}:${PYTHONPATH}:${HOME_DIR}/programas/python-packages"
source "${HOME_DIR}/programas/root-6.28.02-python3.7/bin/thisroot.sh"
