from neuron import h
h.load_file('nrngui.hoc')
from cell import BallStick

class Ring(object):

  def __init__(self, ncell, delay, this_pc, dt = None):
    #print "construct ", self
    global pc
    pc = this_pc
    global rank
    rank = int(pc.id())
    global nhost
    nhost = int(pc.nhost())
    self.delay = delay
    if dt is None:
      self.dt = h.dt
    else:
      self.dt = dt
    self.tvec = h.Vector()
    self.tvec.record(h._ref_t, self.dt)
    self.ncell = int(2)
    self.mkring(self.ncell)
    self.mkstim()
    self.voltage_record()
    #self.spike_record()

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
    for i in range(rank, ncell, nhost):
      cell = BallStick()
      self.cells.append(cell)
      pc.set_gid2node(i, rank)
      nc = cell.connect2target(None)
      pc.cell(i, nc)
    print self.cells

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

  """
  def spike_record(self):
    self.tvec = h.Vector()
    self.idvec = h.Vector()
    for i in range(len(self.cells)):
      nc = self.cells[i].connect2target(None)
      pc.spike_record(nc.srcgid(), self.tvec, self.idvec)
  """

  def voltage_record(self):
    self.vvec = h.Vector()
    if not pc.gid_exists(1):
      return
    self.vvec.record(getattr(pc.gid2cell(1).soma(0), '_ref_v'), self.dt)



def runring(ring, ncell=5, delay=1, tstop=100):
  pc.set_maxstep(10)
  h.stdinit()
  pc.psolve(tstop)
  max_vm = pc.allreduce(ring.vvec, 2)
  return {'max_vm': max_vm, 'id_world': pc.id_world()}