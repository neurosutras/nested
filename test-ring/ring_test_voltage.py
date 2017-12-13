from mpi4py import MPI
import sys
from neuron import h
h.load_file('nrngui.hoc')
from cell import BallStick

class Ring(object):

  def __init__(self, ncell, delay, this_pc, dt=None):
    #spiking script uses dt = 0.02
    global pc
    pc = this_pc
    global rank
    rank = int(pc.id())
    global nhost
    nhost = int(pc.nhost())
    self.delay = delay
    self.ncell = int(ncell)
    self.mkring(self.ncell)
    self.mkstim()
    self.voltage_record(dt)
    self.spike_record()
    self.pydicts = {}

  """
  def __del__(self):
    pc.gid_clear()
    #print "delete ", self
  """

  def mkring(self, ncell):
    self.mkcells(ncell)
    self.connectcells(ncell)

  def mkcells(self, ncell):
    global rank, nhost
    self.cells = []
    self.gids = []
    for i in range(rank, ncell, nhost):
      cell = BallStick()
      self.cells.append(cell)
      self.gids.append(i)
      pc.set_gid2node(i, rank)
      nc = cell.connect2target(None)
      pc.cell(i, nc)

  def connectcells(self, ncell):
    global rank, nhost
    self.nclist = []
    # not efficient but demonstrates use of pc.gid_exists
    for i in [0]: #connect only cell 0 to cell 1
      targid = (i+1)%ncell
      if pc.gid_exists(targid):
        target = pc.gid2cell(targid)
        syn = target.synlist[0]
        nc = pc.gid_connect(i, syn)
        self.nclist.append(nc)
        nc.delay = self.delay
        nc.weight[0] = 0.01


  #Instrumentation - stimulation and recording
  def mkstim(self):
    if not pc.gid_exists(0):
      return
    self.stim = h.NetStim()
    self.stim.number = 1
    self.stim.start = 0
    self.ncstim = h.NetCon(self.stim, pc.gid2cell(0).synlist[0])
    self.ncstim.delay = 0
    self.ncstim.weight[0] = 0.01

  def update_stim_weight(self, new_weight):
    if 0 in self.gids:
      self.ncstim.weight[0] = new_weight

  def update_syn_weight(self, new_weight):
    if self.nclist:
      self.nclist[0].weight[0] = new_weight

  def spike_record(self):
    self.spike_tvec = {}
    self.spike_idvec = {}
    for i, gid in enumerate(self.gids):
      tvec = h.Vector()
      idvec = h.Vector()
      nc = self.cells[i].connect2target(None)
      pc.spike_record(nc.srcgid(), tvec, idvec)
      #Alternatively, could use nc.record(tvec)
      self.spike_tvec[gid] = tvec
      self.spike_idvec[gid] = idvec

  def voltage_record(self, dt=None):
    self.voltage_tvec = {}
    self.voltage_recvec = {}
    if dt is None:
      self.dt = h.dt
    else:
      self.dt = dt
    for i, cell in enumerate(self.cells):
      tvec = h.Vector()
      tvec.record(h._ref_t) #dt is not accepted as an argument to this function in the PC environment -- may need to turn on cvode?
      rec = h.Vector()
      rec.record(getattr(cell.soma(0), '_ref_v')) #dt is not accepted as an argument
      self.voltage_tvec[self.gids[i]] = tvec
      self.voltage_recvec[self.gids[i]] = rec

  def vecdict_to_pydict(self, vecdict, name):
    self.pydicts[name] = {}
    for key, value in vecdict.iteritems():
      self.pydicts[name][key] = value.to_python()


def runring(ring, ncell=5, delay=1, tstop=100):
  pc.set_maxstep(10)
  h.stdinit()
  pc.psolve(tstop)
  ring.vecdict_to_pydict(ring.voltage_tvec, 't')
  ring.vecdict_to_pydict(ring.voltage_recvec, 'rec')
  all_dicts = pc.py_alltoall([ring.pydicts for i in range(nhost)])
  t = {key: value for dict in all_dicts for key, value in dict['t'].iteritems()}
  rec = {key: value for dict in all_dicts for key, value in dict['rec'].iteritems()}
  return {'t': t, 'rec': rec}