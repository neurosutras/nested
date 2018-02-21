from mpi4py import MPI

comm = MPI.COMM_WORLD

print 'Rank: %i; size: %i' % (comm.rank, comm.size)

import neuron

print neuron.__file__
