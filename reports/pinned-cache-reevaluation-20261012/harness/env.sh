export CUCASCADE_PR_ENV=/home/kkristensen/Code/cuCascade/.pixi/envs/default
export PATH="$CUCASCADE_PR_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CUCASCADE_PR_ENV/lib:$CUCASCADE_PR_ENV/targets/sbsa-linux/lib:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$CUCASCADE_PR_ENV;$CUCASCADE_PR_ENV/aarch64-conda-linux-gnu/sysroot/usr"
export CXX="$CUCASCADE_PR_ENV/bin/aarch64-conda-linux-gnu-c++"
export CUDACXX="$CUCASCADE_PR_ENV/bin/nvcc"
export CUDAARCHS=121
