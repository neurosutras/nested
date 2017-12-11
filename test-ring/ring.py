from neuron import h
h.load_file('nrngui.hoc')
from cell import BallStick

class Ring(object):

  def __init__(self, ncell, delay, this_pc):
    #print "construct ", self
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
    self.spike_record()

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

  def connectcells(self, ncell):
    global rank, nhost
    self.nclist = []
    # not efficient but demonstrates use of pc.gid_exists
    for i in range(ncell):
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

  def spike_record(self):
    self.tvec_dict = {}
    self.idvec_dict = {}
    for i in range(len(self.cells)):
      tvec = h.Vector()
      idvec = h.Vector()
      nc = self.cells[i].connect2target(None)
      pc.spike_record(nc.srcgid(), tvec, idvec)
      # nc.record(tvec)
      self.tvec_dict[i] = tvec
      self.idvec_dict[i] = idvec

  def convert_to_py(self, tvec_dict):
    self.t_dict = {}
    for key, value in self.tvec_dict.items():
      self.t_dict[key] = value.to_python()


def runring(ring, ncell=5, delay=1, tstop=100):
  pc.set_maxstep(10)
  h.stdinit()
  pc.psolve(tstop)
  #spkcnt = pc.allreduce(len(ring.tvec), 1)
  #tmax = pc.allreduce(ring.tvec.x[-1], 2)
  #tt = h.Vector()
  #idv = h.Vector()
  #pc.allgather(ring.tvec.x[-1], tt)
  #pc.allgather(ring.idvec.x[-1], idv)
  #idmax = int(idv.x[int(tt.max_ind())])
  ring.convert_to_py(ring.tvec_dict)
  tt = pc.py_alltoall([ring.t_dict for i in range(nhost)])
  print [element.items() for element in tt]
  #return (int(spkcnt), tmax, idmax, (ncell, delay, tstop, (pc.id_world(), pc.nhost_world())))
  #return {'spkcnt': int(spkcnt), 'tmax': tmax, 'tt': tt, 'id_world': pc.id_world()}
  return {'tt': tt}