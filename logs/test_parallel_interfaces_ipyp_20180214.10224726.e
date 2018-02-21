ModuleCmd_Switch.c(179):ERROR:152: Module 'PrgEnv-intel' is currently not loaded
+ cd /global/homes/a/aaronmil/python_modules/nested
++ date +%Y%m%d_%H%M%S
+ export DATE=20180214_105531
+ DATE=20180214_105531
+ cluster_id=test_parallel_interfaces_ipyp_20180214_105531
+ sleep 1
+ srun -N 1 -n 1 -c 2 --cpu_bind=cores ipcontroller '--ip=*' --nodb --cluster-id=test_parallel_interfaces_ipyp_20180214_105531
+ sleep 45
2018-02-14 10:55:46.493 [IPControllerApp] Hub listening on tcp://*:59520 for registration.
2018-02-14 10:55:46.495 [IPControllerApp] Hub using DB backend: 'NoDB'
2018-02-14 10:55:46.790 [IPControllerApp] hub::created hub
2018-02-14 10:55:46.790 [IPControllerApp] writing connection info to /global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-client.json
2018-02-14 10:55:46.801 [IPControllerApp] writing connection info to /global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json
2018-02-14 10:55:46.816 [IPControllerApp] task::using Python leastload Task scheduler
2018-02-14 10:55:46.817 [IPControllerApp] Heartmonitor started
2018-02-14 10:55:46.847 [scheduler] Scheduler started [leastload]
2018-02-14 10:55:46.851 [IPControllerApp] Creating pid file: /global/u1/a/aaronmil/.ipython/profile_default/pid/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531.pid
2018-02-14 10:55:46.855 [IPControllerApp] client::client '\x00k\x8bEg' requested u'connection_request'
2018-02-14 10:55:46.855 [IPControllerApp] client::client ['\x00k\x8bEg'] connected
+ sleep 1
+ srun -N 1 -n 32 -c 2 --cpu_bind=cores ipengine --mpi=mpi4py --cluster-id=test_parallel_interfaces_ipyp_20180214_105531
+ sleep 180
2018-02-14 10:56:54.830 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.830 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.830 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.830 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.831 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.831 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.831 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.831 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.831 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.831 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.831 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.831 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.831 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.832 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.832 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.832 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.832 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.832 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.832 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.832 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.832 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.832 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.832 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.832 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.832 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.832 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.833 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.833 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.833 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.833 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.833 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.833 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.833 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.833 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.833 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.833 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.833 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.833 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.833 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.833 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.834 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.833 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.834 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.834 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.834 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.834 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.834 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.834 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.834 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.834 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.834 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.834 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.834 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.834 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.834 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.834 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.834 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.834 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.835 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.835 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.835 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.835 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:54.835 [IPEngineApp] Initializing MPI:
2018-02-14 10:56:54.835 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.349 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180214_105531-engine.json'
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.564 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.565 [IPEngineApp] Registering with controller at tcp://10.128.2.158:59520
2018-02-14 10:56:55.576 [IPControllerApp] client::client 'f334d07a-730ada3b88430ef0578a9f87' requested u'registration_request'
2018-02-14 10:56:55.577 [IPControllerApp] client::client 'bed6985c-3215497151876954d8963ddf' requested u'registration_request'
2018-02-14 10:56:55.578 [IPControllerApp] client::client '42c04187-e2f6c448a2b417f8d174af8c' requested u'registration_request'
2018-02-14 10:56:55.579 [IPControllerApp] client::client '6a2b4f66-a26f54414cb77ad41afb906f' requested u'registration_request'
2018-02-14 10:56:55.579 [IPControllerApp] client::client '0a5a0ac3-aa1ade2b4fccd62edc2dbb30' requested u'registration_request'
2018-02-14 10:56:55.580 [IPControllerApp] client::client '6467d772-fcee1955fb02f34e2cb623d2' requested u'registration_request'
2018-02-14 10:56:55.582 [IPControllerApp] client::client '4f3acd2d-56b4b84bb80f3089c33912d4' requested u'registration_request'
2018-02-14 10:56:55.583 [IPControllerApp] client::client 'e961e28f-f7ebde017006795517d77f01' requested u'registration_request'
2018-02-14 10:56:55.584 [IPControllerApp] client::client 'cc65615d-44d51b2a9fabbab3f53a2448' requested u'registration_request'
2018-02-14 10:56:55.585 [IPControllerApp] client::client '7f5748bf-54692250c714919da5f6f376' requested u'registration_request'
2018-02-14 10:56:55.587 [IPControllerApp] client::client '6b2d4f86-bb2f96e8e086b3325a748886' requested u'registration_request'
2018-02-14 10:56:55.589 [IPControllerApp] client::client '833a735c-15bd041db4439522afdc85fa' requested u'registration_request'
2018-02-14 10:56:55.590 [IPControllerApp] client::client '9cf02ad8-05cd6796c1a2b89fe201930b' requested u'registration_request'
2018-02-14 10:56:55.591 [IPControllerApp] client::client '498ff39b-421611898896a4ae920cd0de' requested u'registration_request'
2018-02-14 10:56:55.593 [IPControllerApp] client::client 'c2b43a3e-20a1ba90554460626c4e2777' requested u'registration_request'
2018-02-14 10:56:55.594 [IPControllerApp] client::client '65d23c32-bd536b9cd4a01c075d4dea0a' requested u'registration_request'
2018-02-14 10:56:55.595 [IPControllerApp] client::client 'b90e6de0-f69b777c1f766878bf567e3b' requested u'registration_request'
2018-02-14 10:56:55.597 [IPControllerApp] client::client 'd14c1807-f2ad7270b06dd7c56bcb46be' requested u'registration_request'
2018-02-14 10:56:55.598 [IPControllerApp] client::client '512eb5b6-0cdf5b84e7f9e1ce6d6262fc' requested u'registration_request'
2018-02-14 10:56:55.599 [IPControllerApp] client::client '13866dcf-c4b098bdfb70103965cc946d' requested u'registration_request'
2018-02-14 10:56:55.601 [IPControllerApp] client::client '36aed55e-9322517f279e6f6074346677' requested u'registration_request'
2018-02-14 10:56:55.602 [IPControllerApp] client::client '041bfe76-1ad168f7b2106c15bea4eab6' requested u'registration_request'
2018-02-14 10:56:55.603 [IPControllerApp] client::client 'facb6dbd-49b447634f7d303ed3bf78cb' requested u'registration_request'
2018-02-14 10:56:55.605 [IPControllerApp] client::client 'da03216b-ac376d9902701fb3acb8c4db' requested u'registration_request'
2018-02-14 10:56:55.606 [IPControllerApp] client::client 'eb7f0c74-540383ede406c32eb912412d' requested u'registration_request'
2018-02-14 10:56:55.607 [IPControllerApp] client::client '41f440b9-750741600340dfdcee5241ff' requested u'registration_request'
2018-02-14 10:56:55.608 [IPControllerApp] client::client 'ef1151bd-84566cfac512bdb470c17baf' requested u'registration_request'
2018-02-14 10:56:55.610 [IPControllerApp] client::client '89685963-454a06138c0307bbf472ac3d' requested u'registration_request'
2018-02-14 10:56:55.611 [IPControllerApp] client::client '8de863d2-0d173e3628379df65dad6366' requested u'registration_request'
2018-02-14 10:56:55.612 [IPControllerApp] client::client '06b5e371-749b904f263451838f31695a' requested u'registration_request'
2018-02-14 10:56:55.614 [IPControllerApp] client::client '54213f75-dfb4b91c579043aa667e543e' requested u'registration_request'
2018-02-14 10:56:55.615 [IPControllerApp] client::client 'f94550c1-0741c73574b36b6b399c3a28' requested u'registration_request'
2018-02-14 10:56:55.749 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.749 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.749 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.749 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.749 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.749 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.749 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.749 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.749 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.749 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.749 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.749 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.749 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.750 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.751 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.751 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-02-14 10:56:55.772 [IPEngineApp] Completed registration with id 23
2018-02-14 10:56:55.774 [IPEngineApp] Completed registration with id 18
2018-02-14 10:56:55.774 [IPEngineApp] Completed registration with id 27
2018-02-14 10:56:55.775 [IPEngineApp] Completed registration with id 24
2018-02-14 10:56:55.775 [IPEngineApp] Completed registration with id 10
2018-02-14 10:56:55.776 [IPEngineApp] Completed registration with id 4
2018-02-14 10:56:55.778 [IPEngineApp] Completed registration with id 3
2018-02-14 10:56:55.778 [IPEngineApp] Completed registration with id 26
2018-02-14 10:56:55.779 [IPEngineApp] Completed registration with id 20
2018-02-14 10:56:55.779 [IPEngineApp] Completed registration with id 0
2018-02-14 10:56:55.779 [IPEngineApp] Completed registration with id 19
2018-02-14 10:56:55.780 [IPEngineApp] Completed registration with id 30
2018-02-14 10:56:55.780 [IPEngineApp] Completed registration with id 16
2018-02-14 10:56:55.780 [IPEngineApp] Completed registration with id 7
2018-02-14 10:56:55.780 [IPEngineApp] Completed registration with id 14
2018-02-14 10:56:55.780 [IPEngineApp] Completed registration with id 17
2018-02-14 10:56:55.781 [IPEngineApp] Completed registration with id 2
2018-02-14 10:56:55.781 [IPEngineApp] Completed registration with id 29
2018-02-14 10:56:55.781 [IPEngineApp] Completed registration with id 6
2018-02-14 10:56:55.782 [IPEngineApp] Completed registration with id 28
2018-02-14 10:56:55.782 [IPEngineApp] Completed registration with id 5
2018-02-14 10:56:55.782 [IPEngineApp] Completed registration with id 22
2018-02-14 10:56:55.782 [IPEngineApp] Completed registration with id 12
2018-02-14 10:56:55.782 [IPEngineApp] Completed registration with id 8
2018-02-14 10:56:55.782 [IPEngineApp] Completed registration with id 1
2018-02-14 10:56:55.782 [IPEngineApp] Completed registration with id 11
2018-02-14 10:56:55.782 [IPEngineApp] Completed registration with id 13
2018-02-14 10:56:55.782 [IPEngineApp] Completed registration with id 9
2018-02-14 10:56:55.782 [IPEngineApp] Completed registration with id 15
2018-02-14 10:56:55.782 [IPEngineApp] Completed registration with id 21
2018-02-14 10:56:55.782 [IPEngineApp] Completed registration with id 25
2018-02-14 10:56:55.782 [IPEngineApp] Completed registration with id 31
2018-02-14 10:56:58.819 [IPControllerApp] registration::finished registering engine 18:512eb5b6-0cdf5b84e7f9e1ce6d6262fc
2018-02-14 10:56:58.820 [IPControllerApp] engine::Engine Connected: 18
2018-02-14 10:56:58.944 [IPControllerApp] registration::finished registering engine 29:06b5e371-749b904f263451838f31695a
2018-02-14 10:56:58.944 [IPControllerApp] engine::Engine Connected: 29
2018-02-14 10:56:58.947 [IPControllerApp] registration::finished registering engine 13:498ff39b-421611898896a4ae920cd0de
2018-02-14 10:56:58.947 [IPControllerApp] engine::Engine Connected: 13
2018-02-14 10:56:58.949 [IPControllerApp] registration::finished registering engine 27:89685963-454a06138c0307bbf472ac3d
2018-02-14 10:56:58.950 [IPControllerApp] engine::Engine Connected: 27
2018-02-14 10:56:58.952 [IPControllerApp] registration::finished registering engine 3:6a2b4f66-a26f54414cb77ad41afb906f
2018-02-14 10:56:58.952 [IPControllerApp] engine::Engine Connected: 3
2018-02-14 10:56:58.955 [IPControllerApp] registration::finished registering engine 12:9cf02ad8-05cd6796c1a2b89fe201930b
2018-02-14 10:56:58.955 [IPControllerApp] engine::Engine Connected: 12
2018-02-14 10:56:58.957 [IPControllerApp] registration::finished registering engine 8:cc65615d-44d51b2a9fabbab3f53a2448
2018-02-14 10:56:58.957 [IPControllerApp] engine::Engine Connected: 8
2018-02-14 10:56:58.960 [IPControllerApp] registration::finished registering engine 4:0a5a0ac3-aa1ade2b4fccd62edc2dbb30
2018-02-14 10:56:58.960 [IPControllerApp] engine::Engine Connected: 4
2018-02-14 10:56:58.962 [IPControllerApp] registration::finished registering engine 11:833a735c-15bd041db4439522afdc85fa
2018-02-14 10:56:58.962 [IPControllerApp] engine::Engine Connected: 11
2018-02-14 10:56:58.964 [IPControllerApp] registration::finished registering engine 25:41f440b9-750741600340dfdcee5241ff
2018-02-14 10:56:58.965 [IPControllerApp] engine::Engine Connected: 25
2018-02-14 10:56:58.970 [IPControllerApp] registration::finished registering engine 15:65d23c32-bd536b9cd4a01c075d4dea0a
2018-02-14 10:56:58.970 [IPControllerApp] engine::Engine Connected: 15
2018-02-14 10:56:58.973 [IPControllerApp] registration::finished registering engine 14:c2b43a3e-20a1ba90554460626c4e2777
2018-02-14 10:56:58.973 [IPControllerApp] engine::Engine Connected: 14
2018-02-14 10:56:58.977 [IPControllerApp] registration::finished registering engine 17:d14c1807-f2ad7270b06dd7c56bcb46be
2018-02-14 10:56:58.977 [IPControllerApp] engine::Engine Connected: 17
2018-02-14 10:56:58.981 [IPControllerApp] registration::finished registering engine 21:041bfe76-1ad168f7b2106c15bea4eab6
2018-02-14 10:56:58.981 [IPControllerApp] engine::Engine Connected: 21
2018-02-14 10:56:58.984 [IPControllerApp] registration::finished registering engine 0:f334d07a-730ada3b88430ef0578a9f87
2018-02-14 10:56:58.984 [IPControllerApp] engine::Engine Connected: 0
2018-02-14 10:56:58.988 [IPControllerApp] registration::finished registering engine 23:da03216b-ac376d9902701fb3acb8c4db
2018-02-14 10:56:58.988 [IPControllerApp] engine::Engine Connected: 23
2018-02-14 10:56:58.992 [IPControllerApp] registration::finished registering engine 24:eb7f0c74-540383ede406c32eb912412d
2018-02-14 10:56:58.992 [IPControllerApp] engine::Engine Connected: 24
2018-02-14 10:56:58.995 [IPControllerApp] registration::finished registering engine 9:7f5748bf-54692250c714919da5f6f376
2018-02-14 10:56:58.995 [IPControllerApp] engine::Engine Connected: 9
2018-02-14 10:56:58.999 [IPControllerApp] registration::finished registering engine 7:e961e28f-f7ebde017006795517d77f01
2018-02-14 10:56:58.999 [IPControllerApp] engine::Engine Connected: 7
2018-02-14 10:56:59.002 [IPControllerApp] registration::finished registering engine 28:8de863d2-0d173e3628379df65dad6366
2018-02-14 10:56:59.003 [IPControllerApp] engine::Engine Connected: 28
2018-02-14 10:56:59.006 [IPControllerApp] registration::finished registering engine 10:6b2d4f86-bb2f96e8e086b3325a748886
2018-02-14 10:56:59.006 [IPControllerApp] engine::Engine Connected: 10
2018-02-14 10:56:59.010 [IPControllerApp] registration::finished registering engine 6:4f3acd2d-56b4b84bb80f3089c33912d4
2018-02-14 10:56:59.010 [IPControllerApp] engine::Engine Connected: 6
2018-02-14 10:56:59.013 [IPControllerApp] registration::finished registering engine 20:36aed55e-9322517f279e6f6074346677
2018-02-14 10:56:59.013 [IPControllerApp] engine::Engine Connected: 20
2018-02-14 10:56:59.016 [IPControllerApp] registration::finished registering engine 2:42c04187-e2f6c448a2b417f8d174af8c
2018-02-14 10:56:59.017 [IPControllerApp] engine::Engine Connected: 2
2018-02-14 10:56:59.020 [IPControllerApp] registration::finished registering engine 5:6467d772-fcee1955fb02f34e2cb623d2
2018-02-14 10:56:59.020 [IPControllerApp] engine::Engine Connected: 5
2018-02-14 10:56:59.024 [IPControllerApp] registration::finished registering engine 31:f94550c1-0741c73574b36b6b399c3a28
2018-02-14 10:56:59.024 [IPControllerApp] engine::Engine Connected: 31
2018-02-14 10:56:59.028 [IPControllerApp] registration::finished registering engine 22:facb6dbd-49b447634f7d303ed3bf78cb
2018-02-14 10:56:59.028 [IPControllerApp] engine::Engine Connected: 22
2018-02-14 10:56:59.031 [IPControllerApp] registration::finished registering engine 30:54213f75-dfb4b91c579043aa667e543e
2018-02-14 10:56:59.032 [IPControllerApp] engine::Engine Connected: 30
2018-02-14 10:56:59.035 [IPControllerApp] registration::finished registering engine 1:bed6985c-3215497151876954d8963ddf
2018-02-14 10:56:59.035 [IPControllerApp] engine::Engine Connected: 1
2018-02-14 10:56:59.039 [IPControllerApp] registration::finished registering engine 16:b90e6de0-f69b777c1f766878bf567e3b
2018-02-14 10:56:59.039 [IPControllerApp] engine::Engine Connected: 16
2018-02-14 10:56:59.042 [IPControllerApp] registration::finished registering engine 26:ef1151bd-84566cfac512bdb470c17baf
2018-02-14 10:56:59.043 [IPControllerApp] engine::Engine Connected: 26
2018-02-14 10:56:59.046 [IPControllerApp] registration::finished registering engine 19:13866dcf-c4b098bdfb70103965cc946d
2018-02-14 10:56:59.046 [IPControllerApp] engine::Engine Connected: 19
+ srun -N 1 -n 1 -c 2 --cpu_bind=cores python test_parallel_interfaces.py --cluster-id=test_parallel_interfaces_ipyp_20180214_105531 --framework=ipyp
2018-02-14 10:59:25.890 [IPControllerApp] client::client '\x00k\x8bEh' requested u'connection_request'
2018-02-14 10:59:25.890 [IPControllerApp] client::client ['\x00k\x8bEh'] connected
2018-02-14 10:59:25.896 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'b2e32b23-862cb1a50b60a0f6cf01606f' to 0
2018-02-14 10:59:25.897 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'4466fdba-b7a50b8c699b710ab010de10' to 1
2018-02-14 10:59:25.898 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'2eab04ea-4213f55896205d94211b80bb' to 2
2018-02-14 10:59:25.898 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'cb315521-dd7187aff548cad146f43419' to 3
2018-02-14 10:59:25.899 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'3e732975-c6fe83e21702f011876a7368' to 4
2018-02-14 10:59:25.899 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c0c8a749-18df2e2852222d85ad71659e' to 5
2018-02-14 10:59:25.900 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'dad27594-03499459e614371a0df36163' to 6
2018-02-14 10:59:25.901 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c1ec2577-642d0cf74fa3af4699881339' to 7
2018-02-14 10:59:25.901 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'9eb020b2-45e6a5cb1313a1dee3f8612d' to 8
2018-02-14 10:59:25.902 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'f83da0a7-916300f9febdeb58f0a53afd' to 9
2018-02-14 10:59:25.902 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'0d239863-ef618f7aca78eb40e086e476' to 10
2018-02-14 10:59:25.903 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'4be04021-a693c0f427894c6cdfdb31bc' to 11
2018-02-14 10:59:25.903 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'd0441e1c-0590017c7afb82dc8831fcc6' to 12
2018-02-14 10:59:25.904 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'49927065-c84978c5e02c4ff00819f251' to 13
2018-02-14 10:59:25.904 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'440792fa-1593f11e66ca9ab85334db43' to 14
2018-02-14 10:59:25.905 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'ddca11fc-d57db95a150592baf5b3bed0' to 15
2018-02-14 10:59:25.905 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'd2ee18d5-b5589d1eb2c71d3174a3f502' to 16
2018-02-14 10:59:25.906 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'91b4453d-d193617e624d2505e4a69665' to 17
2018-02-14 10:59:25.906 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'abcd0093-89285e6c9c0fe33989be0af5' to 18
2018-02-14 10:59:25.907 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'70815454-6274497beb6f2e22c29ea533' to 19
2018-02-14 10:59:25.908 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'2a3796b9-0b41ee99e33146b01b985c59' to 20
2018-02-14 10:59:25.909 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'34fe9c68-dc9d89126db1fd243eff2f78' to 21
2018-02-14 10:59:25.909 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'efcdce88-764e508cc86a0b68b9ae7a11' to 22
2018-02-14 10:59:25.910 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'48b70fa2-924b8c4e26a76cb04b5d1305' to 23
2018-02-14 10:59:25.910 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'96582554-3762b58c2b66c1d2163ad84c' to 24
2018-02-14 10:59:25.911 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'7a8665eb-272597edf2feff16e399d339' to 25
2018-02-14 10:59:25.911 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'3422323b-697ef44ea6114a39e09a7958' to 26
2018-02-14 10:59:25.912 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a7198eea-ed7213738f89b74472eed37f' to 27
2018-02-14 10:59:25.912 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'67f7a18c-b01efa3ebfe650f44e8c306e' to 28
2018-02-14 10:59:25.913 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'de0a655d-aeb36cb3c61b013423b3196c' to 29
2018-02-14 10:59:25.913 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'90c5d238-024548aa38dada477dfdaa79' to 30
2018-02-14 10:59:25.914 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a92ad691-d4fb513a5ede824fc7e19f72' to 31
2018-02-14 10:59:33.560 [IPControllerApp] queue::request u'96582554-3762b58c2b66c1d2163ad84c' completed on 24
2018-02-14 10:59:33.561 [IPControllerApp] queue::request u'0d239863-ef618f7aca78eb40e086e476' completed on 10
2018-02-14 10:59:33.563 [IPControllerApp] queue::request u'3422323b-697ef44ea6114a39e09a7958' completed on 26
2018-02-14 10:59:33.564 [IPControllerApp] queue::request u'67f7a18c-b01efa3ebfe650f44e8c306e' completed on 28
2018-02-14 10:59:33.565 [IPControllerApp] queue::request u'4be04021-a693c0f427894c6cdfdb31bc' completed on 11
2018-02-14 10:59:33.566 [IPControllerApp] queue::request u'4466fdba-b7a50b8c699b710ab010de10' completed on 1
2018-02-14 10:59:33.567 [IPControllerApp] queue::request u'efcdce88-764e508cc86a0b68b9ae7a11' completed on 22
2018-02-14 10:59:33.569 [IPControllerApp] queue::request u'49927065-c84978c5e02c4ff00819f251' completed on 13
2018-02-14 10:59:33.570 [IPControllerApp] queue::request u'2eab04ea-4213f55896205d94211b80bb' completed on 2
2018-02-14 10:59:33.571 [IPControllerApp] queue::request u'd2ee18d5-b5589d1eb2c71d3174a3f502' completed on 16
2018-02-14 10:59:33.572 [IPControllerApp] queue::request u'70815454-6274497beb6f2e22c29ea533' completed on 19
2018-02-14 10:59:33.573 [IPControllerApp] queue::request u'34fe9c68-dc9d89126db1fd243eff2f78' completed on 21
2018-02-14 10:59:33.575 [IPControllerApp] queue::request u'90c5d238-024548aa38dada477dfdaa79' completed on 30
2018-02-14 10:59:33.576 [IPControllerApp] queue::request u'de0a655d-aeb36cb3c61b013423b3196c' completed on 29
2018-02-14 10:59:33.577 [IPControllerApp] queue::request u'440792fa-1593f11e66ca9ab85334db43' completed on 14
2018-02-14 10:59:33.578 [IPControllerApp] queue::request u'48b70fa2-924b8c4e26a76cb04b5d1305' completed on 23
2018-02-14 10:59:33.579 [IPControllerApp] queue::request u'ddca11fc-d57db95a150592baf5b3bed0' completed on 15
2018-02-14 10:59:33.580 [IPControllerApp] queue::request u'abcd0093-89285e6c9c0fe33989be0af5' completed on 18
2018-02-14 10:59:33.582 [IPControllerApp] queue::request u'7a8665eb-272597edf2feff16e399d339' completed on 25
2018-02-14 10:59:33.583 [IPControllerApp] queue::request u'3e732975-c6fe83e21702f011876a7368' completed on 4
2018-02-14 10:59:33.584 [IPControllerApp] queue::request u'cb315521-dd7187aff548cad146f43419' completed on 3
2018-02-14 10:59:33.585 [IPControllerApp] queue::request u'91b4453d-d193617e624d2505e4a69665' completed on 17
2018-02-14 10:59:33.586 [IPControllerApp] queue::request u'd0441e1c-0590017c7afb82dc8831fcc6' completed on 12
2018-02-14 10:59:33.587 [IPControllerApp] queue::request u'c0c8a749-18df2e2852222d85ad71659e' completed on 5
2018-02-14 10:59:33.589 [IPControllerApp] queue::request u'9eb020b2-45e6a5cb1313a1dee3f8612d' completed on 8
2018-02-14 10:59:33.590 [IPControllerApp] queue::request u'f83da0a7-916300f9febdeb58f0a53afd' completed on 9
2018-02-14 10:59:33.591 [IPControllerApp] queue::request u'c1ec2577-642d0cf74fa3af4699881339' completed on 7
2018-02-14 10:59:33.592 [IPControllerApp] queue::request u'a7198eea-ed7213738f89b74472eed37f' completed on 27
2018-02-14 10:59:33.593 [IPControllerApp] queue::request u'2a3796b9-0b41ee99e33146b01b985c59' completed on 20
2018-02-14 10:59:33.594 [IPControllerApp] queue::request u'b2e32b23-862cb1a50b60a0f6cf01606f' completed on 0
2018-02-14 10:59:33.596 [IPControllerApp] queue::request u'a92ad691-d4fb513a5ede824fc7e19f72' completed on 31
2018-02-14 10:59:33.596 [IPControllerApp] queue::request u'dad27594-03499459e614371a0df36163' completed on 6
2018-02-14 10:59:33.597 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'ba00cbc2-b2fb2cb09c67494e16667044' to 0
2018-02-14 10:59:33.597 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'e6ebdcf3-ea840c32f849f9400b8adbde' to 1
2018-02-14 10:59:33.598 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'f4916a18-6f5da0fad61f0a5344739d7f' to 2
2018-02-14 10:59:33.599 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c446b9e7-f716d05e4f6cbc740e33b9de' to 3
2018-02-14 10:59:33.600 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'09c6f097-a632f7a64b4f066109426fd6' to 4
2018-02-14 10:59:33.601 [IPControllerApp] queue::request u'ba00cbc2-b2fb2cb09c67494e16667044' completed on 0
2018-02-14 10:59:33.602 [IPControllerApp] queue::request u'e6ebdcf3-ea840c32f849f9400b8adbde' completed on 1
2018-02-14 10:59:33.604 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'797c3e8b-c0d8cb3f3b36e9440a1c378d' to 5
2018-02-14 10:59:33.605 [IPControllerApp] queue::request u'f4916a18-6f5da0fad61f0a5344739d7f' completed on 2
2018-02-14 10:59:33.606 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'9a2fcf54-5413bab4b1c5318bce618362' to 6
2018-02-14 10:59:33.608 [IPControllerApp] queue::request u'c446b9e7-f716d05e4f6cbc740e33b9de' completed on 3
2018-02-14 10:59:33.609 [IPControllerApp] queue::request u'09c6f097-a632f7a64b4f066109426fd6' completed on 4
2018-02-14 10:59:33.611 [IPControllerApp] queue::request u'797c3e8b-c0d8cb3f3b36e9440a1c378d' completed on 5
2018-02-14 10:59:33.612 [IPControllerApp] queue::request u'9a2fcf54-5413bab4b1c5318bce618362' completed on 6
2018-02-14 10:59:33.614 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'9a7a1d5e-4c9984cd435a3f0acaaadd88' to 7
2018-02-14 10:59:33.615 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'0a229a01-f756ea66bb4d950272f365aa' to 8
2018-02-14 10:59:33.616 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'e7edee08-13424c6707eac2b5002dbe63' to 9
2018-02-14 10:59:33.617 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'1f85f197-0bbd48fcb6401f734f7d7bcc' to 10
2018-02-14 10:59:33.618 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'4352d64d-06f25179ab5f4e39b3b20847' to 11
2018-02-14 10:59:33.620 [IPControllerApp] queue::request u'9a7a1d5e-4c9984cd435a3f0acaaadd88' completed on 7
2018-02-14 10:59:33.621 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'0361e7ab-7758fe02800f30502df256a1' to 12
2018-02-14 10:59:33.622 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'fadf22a0-f3750f2da6c1fa64504a0f84' to 13
2018-02-14 10:59:33.623 [IPControllerApp] queue::request u'0a229a01-f756ea66bb4d950272f365aa' completed on 8
2018-02-14 10:59:33.624 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'309ee3e8-0f1245031e06b587b00e4317' to 14
2018-02-14 10:59:33.624 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'90dc4880-0c90f62e159aca117204c0b8' to 15
2018-02-14 10:59:33.625 [IPControllerApp] queue::request u'e7edee08-13424c6707eac2b5002dbe63' completed on 9
2018-02-14 10:59:33.626 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c7f65923-8124486f2c2c474bf03f5f8d' to 16
2018-02-14 10:59:33.627 [IPControllerApp] queue::request u'1f85f197-0bbd48fcb6401f734f7d7bcc' completed on 10
2018-02-14 10:59:33.628 [IPControllerApp] queue::request u'4352d64d-06f25179ab5f4e39b3b20847' completed on 11
2018-02-14 10:59:33.629 [IPControllerApp] queue::request u'0361e7ab-7758fe02800f30502df256a1' completed on 12
2018-02-14 10:59:33.631 [IPControllerApp] queue::request u'fadf22a0-f3750f2da6c1fa64504a0f84' completed on 13
2018-02-14 10:59:33.632 [IPControllerApp] queue::request u'309ee3e8-0f1245031e06b587b00e4317' completed on 14
2018-02-14 10:59:33.633 [IPControllerApp] queue::request u'90dc4880-0c90f62e159aca117204c0b8' completed on 15
2018-02-14 10:59:33.634 [IPControllerApp] queue::request u'c7f65923-8124486f2c2c474bf03f5f8d' completed on 16
2018-02-14 10:59:33.635 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a221fa54-9ae7557874aa126bf72de7df' to 17
2018-02-14 10:59:33.636 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'02ce53a2-f81003141266fdc1957b8527' to 18
2018-02-14 10:59:33.636 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'440131d2-3290b7494d1fe82e0c2e94d0' to 19
2018-02-14 10:59:33.637 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'f66631c9-fc44053289b3a9445bff13cf' to 20
2018-02-14 10:59:33.638 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'73009e55-383b9eb7d3756b9b4a608863' to 21
2018-02-14 10:59:33.639 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'752c5cfc-32f378feaf97bc016e2e9db4' to 22
2018-02-14 10:59:33.640 [IPControllerApp] queue::request u'a221fa54-9ae7557874aa126bf72de7df' completed on 17
2018-02-14 10:59:33.641 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'40ad3854-87f983249ac5126f4e02b5ac' to 23
2018-02-14 10:59:33.642 [IPControllerApp] queue::request u'02ce53a2-f81003141266fdc1957b8527' completed on 18
2018-02-14 10:59:33.643 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'acd3d94c-ec6e10c7020584bd121b5425' to 24
2018-02-14 10:59:33.644 [IPControllerApp] queue::request u'440131d2-3290b7494d1fe82e0c2e94d0' completed on 19
2018-02-14 10:59:33.645 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'1d77af57-b30ca691aeafc0d4df8a4e99' to 25
2018-02-14 10:59:33.646 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'78afcced-c2a3d9e1476d3b1d1cce0673' to 26
2018-02-14 10:59:33.646 [IPControllerApp] queue::request u'f66631c9-fc44053289b3a9445bff13cf' completed on 20
2018-02-14 10:59:33.648 [IPControllerApp] queue::request u'73009e55-383b9eb7d3756b9b4a608863' completed on 21
2018-02-14 10:59:33.649 [IPControllerApp] queue::request u'752c5cfc-32f378feaf97bc016e2e9db4' completed on 22
2018-02-14 10:59:33.650 [IPControllerApp] queue::request u'40ad3854-87f983249ac5126f4e02b5ac' completed on 23
2018-02-14 10:59:33.651 [IPControllerApp] queue::request u'acd3d94c-ec6e10c7020584bd121b5425' completed on 24
2018-02-14 10:59:33.652 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'71dcde37-7abecbd205394e1915dd79b0' to 27
2018-02-14 10:59:33.653 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'8dc52ab3-8f8b416f639859e847332c20' to 28
2018-02-14 10:59:33.654 [IPControllerApp] queue::request u'1d77af57-b30ca691aeafc0d4df8a4e99' completed on 25
2018-02-14 10:59:33.655 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'ff028ada-f83fc1ade13218b7af008ff7' to 29
2018-02-14 10:59:33.655 [IPControllerApp] queue::request u'78afcced-c2a3d9e1476d3b1d1cce0673' completed on 26
2018-02-14 10:59:33.656 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'2d7e4453-6ea0ec1791a2c090038bb943' to 30
2018-02-14 10:59:33.657 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'93f6f06e-6ca99c61ddca0612bd2f9e23' to 31
2018-02-14 10:59:33.658 [IPControllerApp] queue::request u'71dcde37-7abecbd205394e1915dd79b0' completed on 27
2018-02-14 10:59:33.659 [IPControllerApp] queue::request u'ff028ada-f83fc1ade13218b7af008ff7' completed on 29
2018-02-14 10:59:33.660 [IPControllerApp] queue::request u'8dc52ab3-8f8b416f639859e847332c20' completed on 28
2018-02-14 10:59:33.661 [IPControllerApp] queue::request u'2d7e4453-6ea0ec1791a2c090038bb943' completed on 30
2018-02-14 10:59:33.662 [IPControllerApp] queue::request u'93f6f06e-6ca99c61ddca0612bd2f9e23' completed on 31
2018-02-14 10:59:33.918 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'0be71a95-b2c16df385f8264d82227861' to 0
2018-02-14 10:59:33.919 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'7b1c9fcc-8e42c6b59ad596ec347f41ae' to 1
2018-02-14 10:59:33.920 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a33c8c2b-319424437698fb4042342472' to 2
2018-02-14 10:59:33.921 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c9afae95-19c434e94d7e3cc1494e30dc' to 3
2018-02-14 10:59:33.923 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'b801cae3-db1d23fcd5120505eb7dec49' to 4
2018-02-14 10:59:33.923 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a91e0950-304f091f90800b124b43525d' to 5
2018-02-14 10:59:33.924 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c3d8b2ae-fab57380d199d2822caba966' to 6
2018-02-14 10:59:33.925 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'86d3678e-a2a859fdb416865b5b70b6ca' to 7
2018-02-14 10:59:33.926 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a83d0459-f7661dc4524e5c2209fbb43d' to 8
2018-02-14 10:59:33.927 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'048abf0f-e05fc322dbbf1bddc22f584c' to 9
2018-02-14 10:59:33.928 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'b2f51f75-b46bb1e0ea70a1ebf3c64877' to 10
2018-02-14 10:59:33.929 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'49b8576c-0c744b57aa51f5542871c0b9' to 11
2018-02-14 10:59:33.930 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'b0d81470-4d074d68a2b6b6b2caba70f7' to 12
2018-02-14 10:59:33.930 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'ab93a4ca-fa0d06570eadd59238f374a2' to 13
2018-02-14 10:59:33.931 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'1947e1b1-604e5b123d0ba02847fec549' to 14
2018-02-14 10:59:33.932 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'554a1d50-e786d29aebb9e429b5c41ade' to 15
2018-02-14 10:59:33.933 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'1e1932b9-1573fd36425a196048089098' to 16
2018-02-14 10:59:33.935 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'369c2d1c-e523328dd08c787f7688aee5' to 17
2018-02-14 10:59:33.936 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'099640f7-24a716faa6d287e3334db927' to 18
2018-02-14 10:59:33.937 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'39d37a9f-db8cb9abe89d6a506fb41fae' to 19
2018-02-14 10:59:33.938 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'64344349-fbdf5da69ff7036f60e1b1dc' to 20
2018-02-14 10:59:33.939 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'dc3c3bd4-2acffcfae09e96b012bf3098' to 21
2018-02-14 10:59:33.940 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'96a4a22a-3620d9af0cb38bf77f17f82e' to 22
2018-02-14 10:59:33.941 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c67a1dc5-8e1af2441610f8d016fa527f' to 23
2018-02-14 10:59:33.942 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'896f8991-4ee8155cc53acb3ad3558cbb' to 24
2018-02-14 10:59:33.943 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'3e955e72-4226c383144fe94513be4228' to 25
2018-02-14 10:59:33.943 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a805e7bc-635fd0f1f14610ab1d4f286b' to 26
2018-02-14 10:59:33.944 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'398a2eaa-909f5b49e61fa717ee41657e' to 27
2018-02-14 10:59:33.945 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'3d8e3f17-f92a9b336c333cae126318c4' to 28
2018-02-14 10:59:33.946 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'0a958179-445a6e0a45b951b41da50442' to 29
2018-02-14 10:59:33.947 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'51543a09-84078dabe0540464154d7c93' to 30
2018-02-14 10:59:33.947 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'02000890-898b696a558a68f17bab84f0' to 31
2018-02-14 10:59:34.121 [IPControllerApp] queue::request u'0be71a95-b2c16df385f8264d82227861' completed on 0
2018-02-14 10:59:34.123 [IPControllerApp] queue::request u'7b1c9fcc-8e42c6b59ad596ec347f41ae' completed on 1
2018-02-14 10:59:34.124 [IPControllerApp] queue::request u'a33c8c2b-319424437698fb4042342472' completed on 2
2018-02-14 10:59:34.125 [IPControllerApp] queue::request u'c9afae95-19c434e94d7e3cc1494e30dc' completed on 3
2018-02-14 10:59:34.126 [IPControllerApp] queue::request u'b801cae3-db1d23fcd5120505eb7dec49' completed on 4
2018-02-14 10:59:34.128 [IPControllerApp] queue::request u'a91e0950-304f091f90800b124b43525d' completed on 5
2018-02-14 10:59:34.129 [IPControllerApp] queue::request u'c3d8b2ae-fab57380d199d2822caba966' completed on 6
2018-02-14 10:59:34.130 [IPControllerApp] queue::request u'86d3678e-a2a859fdb416865b5b70b6ca' completed on 7
2018-02-14 10:59:34.131 [IPControllerApp] queue::request u'a83d0459-f7661dc4524e5c2209fbb43d' completed on 8
2018-02-14 10:59:34.132 [IPControllerApp] queue::request u'048abf0f-e05fc322dbbf1bddc22f584c' completed on 9
2018-02-14 10:59:34.134 [IPControllerApp] queue::request u'b2f51f75-b46bb1e0ea70a1ebf3c64877' completed on 10
2018-02-14 10:59:34.135 [IPControllerApp] queue::request u'49b8576c-0c744b57aa51f5542871c0b9' completed on 11
2018-02-14 10:59:34.136 [IPControllerApp] queue::request u'b0d81470-4d074d68a2b6b6b2caba70f7' completed on 12
2018-02-14 10:59:34.137 [IPControllerApp] queue::request u'ab93a4ca-fa0d06570eadd59238f374a2' completed on 13
2018-02-14 10:59:34.139 [IPControllerApp] queue::request u'1947e1b1-604e5b123d0ba02847fec549' completed on 14
2018-02-14 10:59:34.140 [IPControllerApp] queue::request u'554a1d50-e786d29aebb9e429b5c41ade' completed on 15
2018-02-14 10:59:34.141 [IPControllerApp] queue::request u'1e1932b9-1573fd36425a196048089098' completed on 16
2018-02-14 10:59:34.142 [IPControllerApp] queue::request u'369c2d1c-e523328dd08c787f7688aee5' completed on 17
2018-02-14 10:59:34.143 [IPControllerApp] queue::request u'099640f7-24a716faa6d287e3334db927' completed on 18
2018-02-14 10:59:34.144 [IPControllerApp] queue::request u'39d37a9f-db8cb9abe89d6a506fb41fae' completed on 19
2018-02-14 10:59:34.145 [IPControllerApp] queue::request u'64344349-fbdf5da69ff7036f60e1b1dc' completed on 20
2018-02-14 10:59:34.147 [IPControllerApp] queue::request u'dc3c3bd4-2acffcfae09e96b012bf3098' completed on 21
2018-02-14 10:59:34.148 [IPControllerApp] queue::request u'96a4a22a-3620d9af0cb38bf77f17f82e' completed on 22
2018-02-14 10:59:34.149 [IPControllerApp] queue::request u'c67a1dc5-8e1af2441610f8d016fa527f' completed on 23
2018-02-14 10:59:34.150 [IPControllerApp] queue::request u'896f8991-4ee8155cc53acb3ad3558cbb' completed on 24
2018-02-14 10:59:34.151 [IPControllerApp] queue::request u'3e955e72-4226c383144fe94513be4228' completed on 25
2018-02-14 10:59:34.152 [IPControllerApp] queue::request u'a805e7bc-635fd0f1f14610ab1d4f286b' completed on 26
2018-02-14 10:59:34.153 [IPControllerApp] queue::request u'398a2eaa-909f5b49e61fa717ee41657e' completed on 27
2018-02-14 10:59:34.154 [IPControllerApp] queue::request u'3d8e3f17-f92a9b336c333cae126318c4' completed on 28
2018-02-14 10:59:34.155 [IPControllerApp] queue::request u'0a958179-445a6e0a45b951b41da50442' completed on 29
2018-02-14 10:59:34.156 [IPControllerApp] queue::request u'02000890-898b696a558a68f17bab84f0' completed on 31
2018-02-14 10:59:34.157 [IPControllerApp] queue::request u'51543a09-84078dabe0540464154d7c93' completed on 30
2018-02-14 10:59:34.238 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a64207c4-1f2529e8c54dd3a2f4c95e92' to 0
2018-02-14 10:59:34.240 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'4f50506f-1bc08f112310ab8c9c65abc4' to 1
2018-02-14 10:59:34.240 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'e523c27e-e858bac2efece46668f5b700' to 2
2018-02-14 10:59:34.241 [IPControllerApp] queue::request u'a64207c4-1f2529e8c54dd3a2f4c95e92' completed on 0
2018-02-14 10:59:34.242 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'cafc9438-d9397d7975ed7afd2c8216f3' to 3
2018-02-14 10:59:34.243 [IPControllerApp] queue::request u'4f50506f-1bc08f112310ab8c9c65abc4' completed on 1
2018-02-14 10:59:34.245 [IPControllerApp] queue::request u'e523c27e-e858bac2efece46668f5b700' completed on 2
2018-02-14 10:59:34.246 [IPControllerApp] queue::request u'cafc9438-d9397d7975ed7afd2c8216f3' completed on 3
2018-02-14 10:59:34.247 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'8958e35c-7597f80a119db45c3e2fc80a' to 4
2018-02-14 10:59:34.248 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'84a24598-b8e6ea2ebc00c8b337af1ecc' to 5
2018-02-14 10:59:34.249 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'95dfff81-1c8018fb1b257a4cb8e15c02' to 6
2018-02-14 10:59:34.250 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'e615c272-9c394b6293f42113cdf27787' to 7
2018-02-14 10:59:34.251 [IPControllerApp] queue::request u'8958e35c-7597f80a119db45c3e2fc80a' completed on 4
2018-02-14 10:59:34.252 [IPControllerApp] queue::request u'84a24598-b8e6ea2ebc00c8b337af1ecc' completed on 5
2018-02-14 10:59:34.253 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'1256b109-a8fb22319f008a620e278542' to 8
2018-02-14 10:59:34.254 [IPControllerApp] queue::request u'95dfff81-1c8018fb1b257a4cb8e15c02' completed on 6
2018-02-14 10:59:34.255 [IPControllerApp] queue::request u'e615c272-9c394b6293f42113cdf27787' completed on 7
2018-02-14 10:59:34.256 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'559773be-4bf3bec825f54fd81c490d07' to 9
2018-02-14 10:59:34.257 [IPControllerApp] queue::request u'1256b109-a8fb22319f008a620e278542' completed on 8
2018-02-14 10:59:34.258 [IPControllerApp] queue::request u'559773be-4bf3bec825f54fd81c490d07' completed on 9
2018-02-14 10:59:34.259 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c892d080-763c6b4996956087bb146f40' to 10
2018-02-14 10:59:34.260 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'ae6bd598-f15987c36e43540944c9fbd4' to 11
2018-02-14 10:59:34.261 [IPControllerApp] queue::request u'c892d080-763c6b4996956087bb146f40' completed on 10
2018-02-14 10:59:34.262 [IPControllerApp] queue::request u'ae6bd598-f15987c36e43540944c9fbd4' completed on 11
2018-02-14 10:59:34.275 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'3a796975-536c20e6dc8372bd249330bf' to 12
2018-02-14 10:59:34.276 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'e2da0ff4-5d409d4d1ba026f327907086' to 13
2018-02-14 10:59:34.278 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'6e8f7ac4-ced48c8b99c80094937fec63' to 14
2018-02-14 10:59:34.279 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'91907a2c-0cd2fed6b3fa7d2d65c0df3f' to 15
2018-02-14 10:59:34.280 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'e7a31b5f-9237da544328ce0a14761726' to 16
2018-02-14 10:59:34.281 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'8f0e28e0-5037b073bffada4497e57b78' to 17
2018-02-14 10:59:34.282 [IPControllerApp] queue::request u'3a796975-536c20e6dc8372bd249330bf' completed on 12
2018-02-14 10:59:34.284 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'104cc37c-024e86bd6f03444c648f93f2' to 18
2018-02-14 10:59:34.285 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'fbe93d63-213058dc5ef239a1eba26274' to 19
2018-02-14 10:59:34.285 [IPControllerApp] queue::request u'e2da0ff4-5d409d4d1ba026f327907086' completed on 13
2018-02-14 10:59:34.286 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'079ec62a-aa7a344b3c11346b03077892' to 20
2018-02-14 10:59:34.287 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'41a06274-52a57b0f4f09b1b5f0e9aba4' to 21
2018-02-14 10:59:34.288 [IPControllerApp] queue::request u'6e8f7ac4-ced48c8b99c80094937fec63' completed on 14
2018-02-14 10:59:34.289 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'3d0f4ea7-8e7575a65e5f38156ea4a04b' to 22
2018-02-14 10:59:34.290 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'da1e8052-25cfd8046b179305ffa1e30b' to 23
2018-02-14 10:59:34.291 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'5a9f9555-219dcbf412273d5333d3a3c0' to 24
2018-02-14 10:59:34.292 [IPControllerApp] queue::request u'91907a2c-0cd2fed6b3fa7d2d65c0df3f' completed on 15
2018-02-14 10:59:34.293 [IPControllerApp] queue::request u'e7a31b5f-9237da544328ce0a14761726' completed on 16
2018-02-14 10:59:34.294 [IPControllerApp] queue::request u'8f0e28e0-5037b073bffada4497e57b78' completed on 17
2018-02-14 10:59:34.295 [IPControllerApp] queue::request u'104cc37c-024e86bd6f03444c648f93f2' completed on 18
2018-02-14 10:59:34.296 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'd685439b-729fdec8d285627cf0d2b632' to 25
2018-02-14 10:59:34.297 [IPControllerApp] queue::request u'fbe93d63-213058dc5ef239a1eba26274' completed on 19
2018-02-14 10:59:34.298 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'430c2a52-da4e63909caf2c0713d2c224' to 26
2018-02-14 10:59:34.299 [IPControllerApp] queue::request u'079ec62a-aa7a344b3c11346b03077892' completed on 20
2018-02-14 10:59:34.300 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c3c56735-7368f4562553d9273c508c27' to 27
2018-02-14 10:59:34.300 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a6d21af1-0311a21e474891596735aec6' to 28
2018-02-14 10:59:34.301 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'89617d6a-b9abede0716a81524dfd1de2' to 29
2018-02-14 10:59:34.302 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'93627943-5bc56fce995fb418104fea2c' to 30
2018-02-14 10:59:34.303 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'41be6820-84db91515433ba536ebd8f97' to 31
2018-02-14 10:59:34.304 [IPControllerApp] queue::request u'41a06274-52a57b0f4f09b1b5f0e9aba4' completed on 21
2018-02-14 10:59:34.305 [IPControllerApp] queue::request u'3d0f4ea7-8e7575a65e5f38156ea4a04b' completed on 22
2018-02-14 10:59:34.306 [IPControllerApp] queue::request u'da1e8052-25cfd8046b179305ffa1e30b' completed on 23
2018-02-14 10:59:34.307 [IPControllerApp] queue::request u'5a9f9555-219dcbf412273d5333d3a3c0' completed on 24
2018-02-14 10:59:34.308 [IPControllerApp] queue::request u'430c2a52-da4e63909caf2c0713d2c224' completed on 26
2018-02-14 10:59:34.309 [IPControllerApp] queue::request u'd685439b-729fdec8d285627cf0d2b632' completed on 25
2018-02-14 10:59:34.310 [IPControllerApp] queue::request u'c3c56735-7368f4562553d9273c508c27' completed on 27
2018-02-14 10:59:34.311 [IPControllerApp] queue::request u'a6d21af1-0311a21e474891596735aec6' completed on 28
2018-02-14 10:59:34.312 [IPControllerApp] queue::request u'89617d6a-b9abede0716a81524dfd1de2' completed on 29
2018-02-14 10:59:34.314 [IPControllerApp] queue::request u'93627943-5bc56fce995fb418104fea2c' completed on 30
2018-02-14 10:59:34.315 [IPControllerApp] queue::request u'41be6820-84db91515433ba536ebd8f97' completed on 31
2018-02-14 10:59:34.316 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'cc7a2001-60d40e58ac2f2177a8e46504' to 0
2018-02-14 10:59:34.317 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'8e463714-2adb0400a402f64d8284e96d' to 1
2018-02-14 10:59:34.319 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'b9939645-0aa5b465c676b5c3c2fe8ac3' to 2
2018-02-14 10:59:34.321 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'6bb5161e-adc6b9384d23516cc5129099' to 3
2018-02-14 10:59:34.322 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'ba9f4f3c-5efb88b3e9b6db968b66fc35' to 4
2018-02-14 10:59:34.324 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'df087d18-70631fb2303c9c4989b575b3' to 5
2018-02-14 10:59:34.325 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'4344f2d2-8e84b3fcc40c67226ecfcffd' to 6
2018-02-14 10:59:34.325 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'768e5c92-7753dac841c2f16087795bc5' to 7
2018-02-14 10:59:34.326 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'735bd491-612efbf2c536231ed86d874b' to 8
2018-02-14 10:59:34.327 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'0e8fef89-965fd766cef1cac94f9f6bdb' to 9
2018-02-14 10:59:34.518 [IPControllerApp] queue::request u'cc7a2001-60d40e58ac2f2177a8e46504' completed on 0
2018-02-14 10:59:34.520 [IPControllerApp] queue::request u'8e463714-2adb0400a402f64d8284e96d' completed on 1
2018-02-14 10:59:34.523 [IPControllerApp] queue::request u'b9939645-0aa5b465c676b5c3c2fe8ac3' completed on 2
2018-02-14 10:59:34.523 [IPControllerApp] queue::request u'6bb5161e-adc6b9384d23516cc5129099' completed on 3
2018-02-14 10:59:34.524 [IPControllerApp] queue::request u'ba9f4f3c-5efb88b3e9b6db968b66fc35' completed on 4
2018-02-14 10:59:34.528 [IPControllerApp] queue::request u'df087d18-70631fb2303c9c4989b575b3' completed on 5
2018-02-14 10:59:34.528 [IPControllerApp] queue::request u'4344f2d2-8e84b3fcc40c67226ecfcffd' completed on 6
2018-02-14 10:59:34.530 [IPControllerApp] queue::request u'768e5c92-7753dac841c2f16087795bc5' completed on 7
2018-02-14 10:59:34.531 [IPControllerApp] queue::request u'735bd491-612efbf2c536231ed86d874b' completed on 8
2018-02-14 10:59:34.532 [IPControllerApp] queue::request u'0e8fef89-965fd766cef1cac94f9f6bdb' completed on 9
2018-02-14 10:59:34.642 [IPControllerApp] task::task u'63188747-c25ecb06a39ce18da9999ab8' arrived on 19
2018-02-14 10:59:34.645 [IPControllerApp] task::task u'71f122fc-501517a857d707ba0dfa3f81' arrived on 26
2018-02-14 10:59:34.647 [IPControllerApp] task::task u'8bdf0847-8a0a95a5dc9364d7ee654678' arrived on 16
2018-02-14 10:59:34.650 [IPControllerApp] task::task u'8ff0ed7c-024353ea26298bac07b64450' arrived on 1
2018-02-14 10:59:34.654 [IPControllerApp] task::task u'53ee986b-f9ffb55ef990882c67890739' arrived on 30
2018-02-14 10:59:34.655 [IPControllerApp] task::task u'7b75de72-2e7de4c16a53aaef6bd701c1' arrived on 22
2018-02-14 10:59:34.656 [IPControllerApp] task::task u'21576d19-3d1d0914aec7170f192c9eed' arrived on 31
2018-02-14 10:59:34.657 [IPControllerApp] task::task u'e65100de-40e6659b33f0ccd137ec8441' arrived on 5
2018-02-14 10:59:34.658 [IPControllerApp] task::task u'71f6592f-d378e51c319d3887d1eee1bd' arrived on 2
2018-02-14 10:59:34.660 [IPControllerApp] task::task u'7325b5e9-ac1502da8adfce7d6c6f7c70' arrived on 20
2018-02-14 10:59:34.847 [IPControllerApp] task::task u'63188747-c25ecb06a39ce18da9999ab8' finished on 19
2018-02-14 10:59:34.848 [IPControllerApp] task::task u'71f122fc-501517a857d707ba0dfa3f81' finished on 26
2018-02-14 10:59:34.851 [IPControllerApp] task::task u'8bdf0847-8a0a95a5dc9364d7ee654678' finished on 16
2018-02-14 10:59:34.853 [IPControllerApp] task::task u'8ff0ed7c-024353ea26298bac07b64450' finished on 1
2018-02-14 10:59:34.860 [IPControllerApp] task::task u'53ee986b-f9ffb55ef990882c67890739' finished on 30
2018-02-14 10:59:34.862 [IPControllerApp] task::task u'7b75de72-2e7de4c16a53aaef6bd701c1' finished on 22
2018-02-14 10:59:34.864 [IPControllerApp] task::task u'e65100de-40e6659b33f0ccd137ec8441' finished on 5
2018-02-14 10:59:34.865 [IPControllerApp] task::task u'21576d19-3d1d0914aec7170f192c9eed' finished on 31
2018-02-14 10:59:34.866 [IPControllerApp] task::task u'71f6592f-d378e51c319d3887d1eee1bd' finished on 2
2018-02-14 10:59:34.867 [IPControllerApp] task::task u'7325b5e9-ac1502da8adfce7d6c6f7c70' finished on 20
2018-02-14 10:59:34.938 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'34a1cf94-2e6a9bc8ff5b3dd26a5e4360' to 0
2018-02-14 10:59:34.939 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'1523135b-6ccb3973917239c6177371d7' to 1
2018-02-14 10:59:34.939 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c2c1b9c2-1d6de134ae68268764ebbcbb' to 2
2018-02-14 10:59:34.941 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'5bb936ec-ca163c1bee9f9dba8664d595' to 3
2018-02-14 10:59:34.942 [IPControllerApp] queue::request u'34a1cf94-2e6a9bc8ff5b3dd26a5e4360' completed on 0
2018-02-14 10:59:34.943 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'bc8c2567-bf1af4d92449f48bcb86c193' to 4
2018-02-14 10:59:34.944 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'38608481-60c923d3ec1db6ce8033a556' to 5
2018-02-14 10:59:34.945 [IPControllerApp] queue::request u'1523135b-6ccb3973917239c6177371d7' completed on 1
2018-02-14 10:59:34.946 [IPControllerApp] queue::request u'c2c1b9c2-1d6de134ae68268764ebbcbb' completed on 2
2018-02-14 10:59:34.948 [IPControllerApp] queue::request u'5bb936ec-ca163c1bee9f9dba8664d595' completed on 3
2018-02-14 10:59:34.949 [IPControllerApp] queue::request u'bc8c2567-bf1af4d92449f48bcb86c193' completed on 4
2018-02-14 10:59:34.951 [IPControllerApp] queue::request u'38608481-60c923d3ec1db6ce8033a556' completed on 5
2018-02-14 10:59:34.952 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'04e0133b-26daebdcec43d2273c3e2470' to 6
2018-02-14 10:59:34.953 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'0f5d2f33-502796d81a06127c49e573cf' to 7
2018-02-14 10:59:34.954 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'2cd23aa9-2a419338636c81efbc8aaeef' to 8
2018-02-14 10:59:34.955 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'164c81a6-7f512533449c16f8154a0a45' to 9
2018-02-14 10:59:34.956 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'9b744eb7-ff5d28c03d9e524629c730d0' to 10
2018-02-14 10:59:34.958 [IPControllerApp] queue::request u'04e0133b-26daebdcec43d2273c3e2470' completed on 6
2018-02-14 10:59:34.959 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'6685f6ea-24c885966c123ea0ac99bae7' to 11
2018-02-14 10:59:34.960 [IPControllerApp] queue::request u'0f5d2f33-502796d81a06127c49e573cf' completed on 7
2018-02-14 10:59:34.962 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'552389cf-d7f364afc9d671c49ad7b78d' to 12
2018-02-14 10:59:34.963 [IPControllerApp] queue::request u'2cd23aa9-2a419338636c81efbc8aaeef' completed on 8
2018-02-14 10:59:34.963 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'70001d12-f9b1fe94086aeaa80f51c6a1' to 13
2018-02-14 10:59:34.964 [IPControllerApp] queue::request u'164c81a6-7f512533449c16f8154a0a45' completed on 9
2018-02-14 10:59:34.965 [IPControllerApp] queue::request u'9b744eb7-ff5d28c03d9e524629c730d0' completed on 10
2018-02-14 10:59:34.967 [IPControllerApp] queue::request u'6685f6ea-24c885966c123ea0ac99bae7' completed on 11
2018-02-14 10:59:34.968 [IPControllerApp] queue::request u'552389cf-d7f364afc9d671c49ad7b78d' completed on 12
2018-02-14 10:59:34.969 [IPControllerApp] queue::request u'70001d12-f9b1fe94086aeaa80f51c6a1' completed on 13
2018-02-14 10:59:34.970 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'33182802-73141ef41ec4ebbd627ebcfe' to 14
2018-02-14 10:59:34.970 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'680ffd7b-987ea450274b32feda797cdf' to 15
2018-02-14 10:59:34.971 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'8c05caeb-5596fac3111b2b7125443792' to 16
2018-02-14 10:59:34.972 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'8b62d066-1c63ac3df3fe7b34dda04234' to 17
2018-02-14 10:59:34.973 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'74061dbc-54d64659baf72b51623bad44' to 18
2018-02-14 10:59:34.974 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'1cad1202-af842b5265d7044f2bc8c9fb' to 19
2018-02-14 10:59:34.975 [IPControllerApp] queue::request u'33182802-73141ef41ec4ebbd627ebcfe' completed on 14
2018-02-14 10:59:34.975 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'4e142f11-6ac2e2ebd0b7844b0bd91f6e' to 20
2018-02-14 10:59:34.976 [IPControllerApp] queue::request u'680ffd7b-987ea450274b32feda797cdf' completed on 15
2018-02-14 10:59:34.977 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'5ae7bf63-8333172c9349da9536ad01b8' to 21
2018-02-14 10:59:34.978 [IPControllerApp] queue::request u'8c05caeb-5596fac3111b2b7125443792' completed on 16
2018-02-14 10:59:34.979 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'41fda217-da8f1d9f834b170a6fcadf94' to 22
2018-02-14 10:59:34.980 [IPControllerApp] queue::request u'8b62d066-1c63ac3df3fe7b34dda04234' completed on 17
2018-02-14 10:59:34.981 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'57ac90a0-44f86d50c8022469e0e730f9' to 23
2018-02-14 10:59:34.982 [IPControllerApp] queue::request u'74061dbc-54d64659baf72b51623bad44' completed on 18
2018-02-14 10:59:34.983 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'9675ad81-ea14e4666a1ae5d0c9f40ce4' to 24
2018-02-14 10:59:34.984 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'833a39b2-3f23d6c70e7c5a87766562cb' to 25
2018-02-14 10:59:34.984 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'd889d63a-7a91d3fe606523df0d41ba9c' to 26
2018-02-14 10:59:34.985 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'86966d56-c84c1d914f9bb778f5105410' to 27
2018-02-14 10:59:34.986 [IPControllerApp] queue::request u'1cad1202-af842b5265d7044f2bc8c9fb' completed on 19
2018-02-14 10:59:34.987 [IPControllerApp] queue::request u'4e142f11-6ac2e2ebd0b7844b0bd91f6e' completed on 20
2018-02-14 10:59:34.988 [IPControllerApp] queue::request u'5ae7bf63-8333172c9349da9536ad01b8' completed on 21
2018-02-14 10:59:34.989 [IPControllerApp] queue::request u'41fda217-da8f1d9f834b170a6fcadf94' completed on 22
2018-02-14 10:59:34.990 [IPControllerApp] queue::request u'57ac90a0-44f86d50c8022469e0e730f9' completed on 23
2018-02-14 10:59:34.991 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'cca9bb53-41544fc1e96df341585d2f57' to 28
2018-02-14 10:59:34.992 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'182460a5-f86b4b05cbe34f675aeeb710' to 29
2018-02-14 10:59:34.993 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'49674b86-b7e1d23cd92a9aab2598a02b' to 30
2018-02-14 10:59:34.994 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a06da91d-656263347cf21ec09b2cc5c4' to 31
2018-02-14 10:59:34.995 [IPControllerApp] queue::request u'9675ad81-ea14e4666a1ae5d0c9f40ce4' completed on 24
2018-02-14 10:59:34.996 [IPControllerApp] queue::request u'833a39b2-3f23d6c70e7c5a87766562cb' completed on 25
2018-02-14 10:59:34.997 [IPControllerApp] queue::request u'd889d63a-7a91d3fe606523df0d41ba9c' completed on 26
2018-02-14 10:59:34.998 [IPControllerApp] queue::request u'86966d56-c84c1d914f9bb778f5105410' completed on 27
2018-02-14 10:59:34.999 [IPControllerApp] queue::request u'cca9bb53-41544fc1e96df341585d2f57' completed on 28
2018-02-14 10:59:35.000 [IPControllerApp] queue::request u'182460a5-f86b4b05cbe34f675aeeb710' completed on 29
2018-02-14 10:59:35.001 [IPControllerApp] queue::request u'49674b86-b7e1d23cd92a9aab2598a02b' completed on 30
2018-02-14 10:59:35.002 [IPControllerApp] queue::request u'a06da91d-656263347cf21ec09b2cc5c4' completed on 31
