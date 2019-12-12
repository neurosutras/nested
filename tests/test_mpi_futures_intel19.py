import time, sys
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
num_workers = size - 1


def report(id):
    print('Rank: %i, size: %i, task id: %i' % (rank, size, id))
    sys.stdout.flush()
    time.sleep(0.1)
    return id


def main():

    executor = MPIPoolExecutor()
    futures = []
    for id in range(1, size):
        futures.append(executor.submit(report, id))
    report(0)
    results = [future.result() for future in futures]
    num_returned = len(set(results))
    print('%i workers completed %i tasks' % (num_workers, num_returned))
    sys.stdout.flush()
    time.sleep(0.1)
    executor.shutdown()


if __name__ == "__main__":
    main()