#!/bin/bash -l

#SBATCH -J test_parallel_interfaces_ray
#SBATCH -o /scratch2/11358/yashchennawar5555/logs/nested/test_parallel_interfaces_ray.%j.o
#SBATCH -e /scratch2/11358/yashchennawar5555/logs/nested/test_parallel_interfaces_ray.%j.e
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --partition=rtx
#SBATCH --mem=80G
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --mail-user=yc1376@scarletmail.rutgers.edu
#SBATCH --mail-type=ALL

set -euo pipefail
set -x

mkdir -p "$SCRATCH/logs/nested"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

module purge
module load cuda/12.2
module load intel/23.1.0

source /work2/11358/yashchennawar5555/frontera/miniconda3/etc/profile.d/conda.sh
conda activate eiann7
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

cd $HOME/nested

export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
mkdir -p "$RAY_TMPDIR"

# 1. Get node list and head node IP.
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

port=6379
ip_head="$head_node_ip:$port"
export ip_head
export RAY_ADDRESS="$ip_head"

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"

# 2. Start Ray head.
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
	ray start --head --node-ip-address="$head_node_ip" --port="$port" --num-cpus=16 --num-gpus=4 \
	--temp-dir "$RAY_TMPDIR" --disable-usage-stats &

# 3. Start Ray workers on the other nodes.
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i=1; i<=worker_num; i++)); do
	node_i=${nodes_array[$i]}
	echo "Starting worker node on $node_i"
	srun --overlap --nodes=1 --ntasks=1 -w "$node_i" \
		ray start --address "$ip_head" --num-cpus=16 --num-gpus=4 --disable-usage-stats &
done

cleanup_ray() {
	set +e
	for node in "${nodes_array[@]}"; do
		stop_output=$(srun --overlap --nodes=1 --ntasks=1 -w "$node" ray stop --force 2>&1) || true
		echo "$stop_output" | grep -E "Stopped all [0-9]+ Ray processes|No active Ray processes" || true
	done
}
trap cleanup_ray EXIT

# 4. Wait for cluster to come up and verify status.
sleep 20
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" ray status --address "$ip_head"

# 5. Run the ray interface test from the head node.
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
	python tests/test_parallel_interfaces.py --framework=ray "${@:1}"

# sbatch jobscripts/batch_test_parallel_interfaces_ray.sh