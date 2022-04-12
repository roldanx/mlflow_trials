#!/usr/bin/env bash

if [[ -z "${REGISTRY_URI}" ]] || [[ -z "${MLRUNS_DIRECTORY}" ]]; then
  echo "Essential variables are not defined. Please check env_file from https://github.com/roldanx/mlflow_trials to see an example and use --env-file docker flag when running this container."
  exit 1
fi
