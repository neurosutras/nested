from neuron import h
pc = h.ParallelContext()
pc.subworlds(1)

s = 'world ({0}, {1}) bbs ({2}, {3}) net ({4}, {5})'.format(
    pc.id_world(), pc.nhost_world(), pc.id_bbs(), pc.nhost_bbs(), pc.id(), pc.nhost()
)

print(s)

from ring import runring

pc.runworker()

for ncell in range(5, 10):
  pc.submit(runring, ncell, 1, 100)

while (pc.working()):
  print(pc.pyret())  

pc.done()
h.quit()

