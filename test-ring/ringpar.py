from neuron import h
h.load_file('nrngui.hoc')
pc = h.ParallelContext()
rank = int(pc.id())
nhost = int(pc.nhost())

from cell import BallStick

# Network Creation

NCELL = 20

cells = []
nclist = []

def mkring(ncell):
  mkcells(ncell)
  connectcells()

def mkcells(ncell):
  global cells, rank, nhost
  cells = []
  for i in range(rank, ncell, nhost):
    cell = BallStick()
    cells.append(cell)
    pc.set_gid2node(i, rank)
    nc = cell.connect2target(None)
    pc.cell(i, nc)

def connectcells():
  global cells, nclist, rank, nhost, NCELL
  nclist = []
  # not efficient but demonstrates use of pc.gid_exists
  for i in range(NCELL):
    targid = (i+1)%NCELL
    if pc.gid_exists(targid):
      target = pc.gid2cell(targid)
      syn = target.synlist[0]
      nc = pc.gid_connect(i, syn)
      nclist.append(nc)
      nc.delay = 1
      nc.weight[0] = 0.01

mkring(NCELL)

#Instrumentation - stimulation and recording

def mkstim():
  ''' stimulate gid 0 with NetStim to start ring '''
  global stim, ncstim
  if not pc.gid_exists(0):
    return
  stim = h.NetStim()
  stim.number = 1
  stim.start = 0
  ncstim = h.NetCon(stim, pc.gid2cell(0).synlist[0])
  ncstim.delay = 0
  ncstim.weight[0] = 0.01

mkstim()

def spike_record():
  ''' record spikes from all gids '''
  global tvec, idvec
  tvec = h.Vector()
  idvec = h.Vector()
  pc.spike_record(-1, tvec, idvec)

def prun(tstop):
  ''' simulation control '''
  pc.set_maxstep(10)
  h.stdinit()
  pc.psolve(tstop)


def spikeout():
  ''' report simulation results to stdout '''
  global rank, tvec, idvec
  pc.barrier()
  for i in range(nhost):
    if i == rank:
      for i in range(len(tvec)):
        print('%g %d' % (tvec.x[i], int(idvec.x[i])))
    pc.barrier()


def finish():
  ''' proper exit '''
  pc.runworker()
  pc.done()
  h.quit()

if __name__ == '__main__':
  spike_record()
  prun(100)
  spikeout()
  if (nhost > 1):
    finish()

