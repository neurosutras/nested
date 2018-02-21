ModuleCmd_Switch.c(179):ERROR:152: Module 'PrgEnv-intel' is currently not loaded
+ cd /global/homes/a/aaronmil/python_modules/nested
++ date +%Y%m%d_%H%M%S
+ export DATE=20180131_111933
+ DATE=20180131_111933
+ cluster_id=test_parallel_interfaces_ipyp_20180131_111933
+ sleep 1
+ srun -N 1 -n 1 -c 2 --cpu_bind=cores ipcontroller '--ip=*' --nodb --cluster-id=test_parallel_interfaces_ipyp_20180131_111933
+ sleep 45
2018-01-31 11:19:45.131 [IPControllerApp] Hub listening on tcp://*:51865 for registration.
2018-01-31 11:19:45.133 [IPControllerApp] Hub using DB backend: 'NoDB'
2018-01-31 11:19:45.436 [IPControllerApp] hub::created hub
2018-01-31 11:19:45.436 [IPControllerApp] writing connection info to /global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-client.json
2018-01-31 11:19:45.450 [IPControllerApp] writing connection info to /global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json
2018-01-31 11:19:45.465 [IPControllerApp] task::using Python leastload Task scheduler
2018-01-31 11:19:45.465 [IPControllerApp] Heartmonitor started
2018-01-31 11:19:45.500 [scheduler] Scheduler started [leastload]
2018-01-31 11:19:45.507 [IPControllerApp] Creating pid file: /global/u1/a/aaronmil/.ipython/profile_default/pid/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933.pid
2018-01-31 11:19:45.513 [IPControllerApp] client::client '\x00k\x8bEg' requested u'connection_request'
2018-01-31 11:19:45.513 [IPControllerApp] client::client ['\x00k\x8bEg'] connected
+ sleep 1
+ srun -N 1 -n 32 -c 2 --cpu_bind=cores ipengine --mpi=mpi4py --cluster-id=test_parallel_interfaces_ipyp_20180131_111933
+ sleep 180
2018-01-31 11:20:54.662 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.662 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.664 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.664 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.665 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.665 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.665 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.666 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.666 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.666 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.666 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.666 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.668 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.668 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.668 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.668 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.669 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.669 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.669 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.670 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.671 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.671 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.671 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.671 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.671 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.672 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.672 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.672 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.672 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.672 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.673 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.673 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.673 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.673 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.674 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.674 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.674 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.674 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.675 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.675 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.675 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.675 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.678 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.678 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.679 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.679 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.682 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.682 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.683 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.683 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.692 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.692 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.696 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.696 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.696 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.696 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.696 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.696 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.698 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.698 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.698 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.698 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:54.702 [IPEngineApp] Initializing MPI:
2018-01-31 11:20:54.703 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-31 11:20:55.198 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.198 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.198 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.198 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.198 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.198 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.198 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.198 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.198 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.199 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-test_parallel_interfaces_ipyp_20180131_111933-engine.json'
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.415 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.416 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.416 [IPEngineApp] Registering with controller at tcp://10.128.49.247:51865
2018-01-31 11:20:55.421 [IPControllerApp] client::client '221de75c-042096b89bf178921f102e29' requested u'registration_request'
2018-01-31 11:20:55.423 [IPControllerApp] client::client '1a98bfd4-02b7dc27dd867adf20b92181' requested u'registration_request'
2018-01-31 11:20:55.423 [IPControllerApp] client::client '79443941-1f0c32cd17345fa6a6d00947' requested u'registration_request'
2018-01-31 11:20:55.424 [IPControllerApp] client::client '06ebd41f-f9c9a93a985c7ee52537930b' requested u'registration_request'
2018-01-31 11:20:55.425 [IPControllerApp] client::client 'dfbec0de-5ef2d3752ef990d441b9d835' requested u'registration_request'
2018-01-31 11:20:55.426 [IPControllerApp] client::client 'd02ae21a-27d739b023df6bc03d4d5979' requested u'registration_request'
2018-01-31 11:20:55.427 [IPControllerApp] client::client '63a26a5f-4f544d9bed69852196885af6' requested u'registration_request'
2018-01-31 11:20:55.429 [IPControllerApp] client::client '11f300f0-2e95edc3a793dc778dd15f37' requested u'registration_request'
2018-01-31 11:20:55.430 [IPControllerApp] client::client 'f9c57525-ca556eb391ccaec5ac5ddc45' requested u'registration_request'
2018-01-31 11:20:55.431 [IPControllerApp] client::client '059f8fab-473a7f6dbb7ac638c757ae8b' requested u'registration_request'
2018-01-31 11:20:55.432 [IPControllerApp] client::client '25522ddf-1aed48cd5774b31fef4dee6f' requested u'registration_request'
2018-01-31 11:20:55.433 [IPControllerApp] client::client 'e821959d-d61ebe34116944c96992df53' requested u'registration_request'
2018-01-31 11:20:55.434 [IPControllerApp] client::client '22de0ac9-6d225c3e7cc01994b865b4c8' requested u'registration_request'
2018-01-31 11:20:55.436 [IPControllerApp] client::client '4aca9272-8e85dc33f8b5891a9a1f5c83' requested u'registration_request'
2018-01-31 11:20:55.437 [IPControllerApp] client::client 'e96e768d-d1bceaed8580d5776cf01171' requested u'registration_request'
2018-01-31 11:20:55.438 [IPControllerApp] client::client 'dec8abb6-de5460ec134308d708679f7e' requested u'registration_request'
2018-01-31 11:20:55.439 [IPControllerApp] client::client '8fb6efac-2f33413659d5ec891b25e61a' requested u'registration_request'
2018-01-31 11:20:55.440 [IPControllerApp] client::client 'd3922901-adb463e16595be82346d6bb4' requested u'registration_request'
2018-01-31 11:20:55.441 [IPControllerApp] client::client '9bb2f7e2-e6e9a2aa3b6f1decd60a5023' requested u'registration_request'
2018-01-31 11:20:55.442 [IPControllerApp] client::client '93acde55-16764335b26cc694bea1c9bd' requested u'registration_request'
2018-01-31 11:20:55.444 [IPControllerApp] client::client '35480e5d-0eef0a0d9e59900433f1a717' requested u'registration_request'
2018-01-31 11:20:55.445 [IPControllerApp] client::client '4a5c669b-c0961cd0bde3321fe0e6627a' requested u'registration_request'
2018-01-31 11:20:55.446 [IPControllerApp] client::client 'de074122-c02623f43923adf72ef7542e' requested u'registration_request'
2018-01-31 11:20:55.447 [IPControllerApp] client::client '0bf17959-dfc9dad4afe0d9da8e1a3ccf' requested u'registration_request'
2018-01-31 11:20:55.448 [IPControllerApp] client::client '025e535b-acbfb7ad125d872c801620a9' requested u'registration_request'
2018-01-31 11:20:55.450 [IPControllerApp] client::client '85c909f5-2c9336f58a91ad9bec2ff470' requested u'registration_request'
2018-01-31 11:20:55.451 [IPControllerApp] client::client '63f0cd2d-d2225129818a82119664ebdf' requested u'registration_request'
2018-01-31 11:20:55.452 [IPControllerApp] client::client 'db10c4fc-95ad35025a3b91842e9e877f' requested u'registration_request'
2018-01-31 11:20:55.453 [IPControllerApp] client::client '66af1237-59a732fa3867f3466ca0ec07' requested u'registration_request'
2018-01-31 11:20:55.454 [IPControllerApp] client::client 'e68fcf6a-014bd2d87f1c4771185db4e8' requested u'registration_request'
2018-01-31 11:20:55.455 [IPControllerApp] client::client 'd29c1668-26d1fa96be2d1d225b63a076' requested u'registration_request'
2018-01-31 11:20:55.456 [IPControllerApp] client::client '757047ac-5a7faacae130985f0fd617a8' requested u'registration_request'
2018-01-31 11:20:55.621 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.621 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.621 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.621 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.621 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.621 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.621 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.621 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.621 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.621 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.621 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.621 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.621 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.622 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.623 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-31 11:20:55.647 [IPEngineApp] Completed registration with id 23
2018-01-31 11:20:55.647 [IPEngineApp] Completed registration with id 14
2018-01-31 11:20:55.648 [IPEngineApp] Completed registration with id 10
2018-01-31 11:20:55.648 [IPEngineApp] Completed registration with id 26
2018-01-31 11:20:55.649 [IPEngineApp] Completed registration with id 3
2018-01-31 11:20:55.652 [IPEngineApp] Completed registration with id 2
2018-01-31 11:20:55.653 [IPEngineApp] Completed registration with id 30
2018-01-31 11:20:55.653 [IPEngineApp] Completed registration with id 13
2018-01-31 11:20:55.653 [IPEngineApp] Completed registration with id 29
2018-01-31 11:20:55.654 [IPEngineApp] Completed registration with id 0
2018-01-31 11:20:55.655 [IPEngineApp] Completed registration with id 9
2018-01-31 11:20:55.656 [IPEngineApp] Completed registration with id 21
2018-01-31 11:20:55.656 [IPEngineApp] Completed registration with id 20
2018-01-31 11:20:55.657 [IPEngineApp] Completed registration with id 25
2018-01-31 11:20:55.657 [IPEngineApp] Completed registration with id 16
2018-01-31 11:20:55.657 [IPEngineApp] Completed registration with id 11
2018-01-31 11:20:55.657 [IPEngineApp] Completed registration with id 27
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 17
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 28
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 31
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 5
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 8
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 22
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 4
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 15
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 6
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 19
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 7
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 24
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 1
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 12
2018-01-31 11:20:55.658 [IPEngineApp] Completed registration with id 18
2018-01-31 11:21:00.467 [IPControllerApp] registration::finished registering engine 2:79443941-1f0c32cd17345fa6a6d00947
2018-01-31 11:21:00.468 [IPControllerApp] engine::Engine Connected: 2
2018-01-31 11:21:00.508 [IPControllerApp] registration::finished registering engine 23:0bf17959-dfc9dad4afe0d9da8e1a3ccf
2018-01-31 11:21:00.508 [IPControllerApp] engine::Engine Connected: 23
2018-01-31 11:21:00.511 [IPControllerApp] registration::finished registering engine 5:d02ae21a-27d739b023df6bc03d4d5979
2018-01-31 11:21:00.511 [IPControllerApp] engine::Engine Connected: 5
2018-01-31 11:21:00.513 [IPControllerApp] registration::finished registering engine 27:db10c4fc-95ad35025a3b91842e9e877f
2018-01-31 11:21:00.514 [IPControllerApp] engine::Engine Connected: 27
2018-01-31 11:21:00.516 [IPControllerApp] registration::finished registering engine 7:11f300f0-2e95edc3a793dc778dd15f37
2018-01-31 11:21:00.516 [IPControllerApp] engine::Engine Connected: 7
2018-01-31 11:21:00.518 [IPControllerApp] registration::finished registering engine 15:dec8abb6-de5460ec134308d708679f7e
2018-01-31 11:21:00.518 [IPControllerApp] engine::Engine Connected: 15
2018-01-31 11:21:00.521 [IPControllerApp] registration::finished registering engine 21:4a5c669b-c0961cd0bde3321fe0e6627a
2018-01-31 11:21:00.522 [IPControllerApp] engine::Engine Connected: 21
2018-01-31 11:21:00.524 [IPControllerApp] registration::finished registering engine 17:d3922901-adb463e16595be82346d6bb4
2018-01-31 11:21:00.524 [IPControllerApp] engine::Engine Connected: 17
2018-01-31 11:21:00.526 [IPControllerApp] registration::finished registering engine 11:e821959d-d61ebe34116944c96992df53
2018-01-31 11:21:00.526 [IPControllerApp] engine::Engine Connected: 11
2018-01-31 11:21:00.529 [IPControllerApp] registration::finished registering engine 4:dfbec0de-5ef2d3752ef990d441b9d835
2018-01-31 11:21:00.529 [IPControllerApp] engine::Engine Connected: 4
2018-01-31 11:21:00.534 [IPControllerApp] registration::finished registering engine 0:221de75c-042096b89bf178921f102e29
2018-01-31 11:21:00.535 [IPControllerApp] engine::Engine Connected: 0
2018-01-31 11:21:00.538 [IPControllerApp] registration::finished registering engine 30:d29c1668-26d1fa96be2d1d225b63a076
2018-01-31 11:21:00.538 [IPControllerApp] engine::Engine Connected: 30
2018-01-31 11:21:00.542 [IPControllerApp] registration::finished registering engine 24:025e535b-acbfb7ad125d872c801620a9
2018-01-31 11:21:00.542 [IPControllerApp] engine::Engine Connected: 24
2018-01-31 11:21:00.546 [IPControllerApp] registration::finished registering engine 1:1a98bfd4-02b7dc27dd867adf20b92181
2018-01-31 11:21:00.546 [IPControllerApp] engine::Engine Connected: 1
2018-01-31 11:21:00.549 [IPControllerApp] registration::finished registering engine 6:63a26a5f-4f544d9bed69852196885af6
2018-01-31 11:21:00.550 [IPControllerApp] engine::Engine Connected: 6
2018-01-31 11:21:00.553 [IPControllerApp] registration::finished registering engine 12:22de0ac9-6d225c3e7cc01994b865b4c8
2018-01-31 11:21:00.554 [IPControllerApp] engine::Engine Connected: 12
2018-01-31 11:21:00.557 [IPControllerApp] registration::finished registering engine 31:757047ac-5a7faacae130985f0fd617a8
2018-01-31 11:21:00.558 [IPControllerApp] engine::Engine Connected: 31
2018-01-31 11:21:00.561 [IPControllerApp] registration::finished registering engine 14:e96e768d-d1bceaed8580d5776cf01171
2018-01-31 11:21:00.562 [IPControllerApp] engine::Engine Connected: 14
2018-01-31 11:21:00.565 [IPControllerApp] registration::finished registering engine 20:35480e5d-0eef0a0d9e59900433f1a717
2018-01-31 11:21:00.566 [IPControllerApp] engine::Engine Connected: 20
2018-01-31 11:21:00.569 [IPControllerApp] registration::finished registering engine 13:4aca9272-8e85dc33f8b5891a9a1f5c83
2018-01-31 11:21:00.570 [IPControllerApp] engine::Engine Connected: 13
2018-01-31 11:21:00.573 [IPControllerApp] registration::finished registering engine 9:059f8fab-473a7f6dbb7ac638c757ae8b
2018-01-31 11:21:00.574 [IPControllerApp] engine::Engine Connected: 9
2018-01-31 11:21:00.577 [IPControllerApp] registration::finished registering engine 3:06ebd41f-f9c9a93a985c7ee52537930b
2018-01-31 11:21:00.577 [IPControllerApp] engine::Engine Connected: 3
2018-01-31 11:21:00.581 [IPControllerApp] registration::finished registering engine 18:9bb2f7e2-e6e9a2aa3b6f1decd60a5023
2018-01-31 11:21:00.582 [IPControllerApp] engine::Engine Connected: 18
2018-01-31 11:21:00.585 [IPControllerApp] registration::finished registering engine 10:25522ddf-1aed48cd5774b31fef4dee6f
2018-01-31 11:21:00.585 [IPControllerApp] engine::Engine Connected: 10
2018-01-31 11:21:00.589 [IPControllerApp] registration::finished registering engine 16:8fb6efac-2f33413659d5ec891b25e61a
2018-01-31 11:21:00.589 [IPControllerApp] engine::Engine Connected: 16
2018-01-31 11:21:00.593 [IPControllerApp] registration::finished registering engine 25:85c909f5-2c9336f58a91ad9bec2ff470
2018-01-31 11:21:00.593 [IPControllerApp] engine::Engine Connected: 25
2018-01-31 11:21:00.597 [IPControllerApp] registration::finished registering engine 28:66af1237-59a732fa3867f3466ca0ec07
2018-01-31 11:21:00.597 [IPControllerApp] engine::Engine Connected: 28
2018-01-31 11:21:00.600 [IPControllerApp] registration::finished registering engine 29:e68fcf6a-014bd2d87f1c4771185db4e8
2018-01-31 11:21:00.600 [IPControllerApp] engine::Engine Connected: 29
2018-01-31 11:21:00.604 [IPControllerApp] registration::finished registering engine 8:f9c57525-ca556eb391ccaec5ac5ddc45
2018-01-31 11:21:00.604 [IPControllerApp] engine::Engine Connected: 8
2018-01-31 11:21:00.608 [IPControllerApp] registration::finished registering engine 19:93acde55-16764335b26cc694bea1c9bd
2018-01-31 11:21:00.608 [IPControllerApp] engine::Engine Connected: 19
2018-01-31 11:21:00.611 [IPControllerApp] registration::finished registering engine 26:63f0cd2d-d2225129818a82119664ebdf
2018-01-31 11:21:00.612 [IPControllerApp] engine::Engine Connected: 26
2018-01-31 11:21:00.615 [IPControllerApp] registration::finished registering engine 22:de074122-c02623f43923adf72ef7542e
2018-01-31 11:21:00.615 [IPControllerApp] engine::Engine Connected: 22
+ srun -N 1 -n 1 -c 2 --cpu_bind=cores python apply_example.py --cluster-id=test_parallel_interfaces_ipyp_20180131_111933
/usr/common/software/python/2.7-anaconda-4.4/bin/python: can't open file 'apply_example.py': [Errno 2] No such file or directory
srun: error: nid12696: task 0: Exited with exit code 2
srun: Terminating job step 9846754.2
