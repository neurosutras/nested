from mpi4py import MPI
import h5py
rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
print "Hello World (from process %d)" % rank
f = h5py.File('parallel_test.hdf5', 'w', driver='mpio', comm=MPI.COMM_WORLD)
dset = f.create_dataset('test', (4,), dtype='i')
dset[rank] = rank
f.close()
