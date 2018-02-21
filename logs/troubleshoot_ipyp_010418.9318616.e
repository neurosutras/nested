ModuleCmd_Switch.c(179):ERROR:152: Module 'PrgEnv-intel' is currently not loaded
+ cd /global/homes/a/aaronmil/python_modules/nested
++ date +%Y%m%d_%H%M%S
+ export DATE=20180104_112710
+ DATE=20180104_112710
+ cluster_id=troubleshoot_ipyp_20180104_112710
+ sleep 1
+ srun -N 1 -n 1 -c 2 --cpu_bind=cores ipcontroller '--ip=*' --nodb --cluster-id=troubleshoot_ipyp_20180104_112710
+ sleep 45
2018-01-04 11:27:21.734 [IPControllerApp] Hub listening on tcp://*:57071 for registration.
2018-01-04 11:27:21.736 [IPControllerApp] Hub using DB backend: 'NoDB'
2018-01-04 11:27:22.047 [IPControllerApp] hub::created hub
2018-01-04 11:27:22.047 [IPControllerApp] writing connection info to /global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-client.json
2018-01-04 11:27:22.057 [IPControllerApp] writing connection info to /global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json
2018-01-04 11:27:22.065 [IPControllerApp] task::using Python leastload Task scheduler
2018-01-04 11:27:22.065 [IPControllerApp] Heartmonitor started
2018-01-04 11:27:22.104 [scheduler] Scheduler started [leastload]
2018-01-04 11:27:22.106 [IPControllerApp] Creating pid file: /global/u1/a/aaronmil/.ipython/profile_default/pid/ipcontroller-troubleshoot_ipyp_20180104_112710.pid
2018-01-04 11:27:22.110 [IPControllerApp] client::client '\x00k\x8bEg' requested u'connection_request'
2018-01-04 11:27:22.110 [IPControllerApp] client::client ['\x00k\x8bEg'] connected
+ sleep 1
+ srun -N 1 -n 32 -c 2 --cpu_bind=cores ipengine --mpi=mpi4py --cluster-id=troubleshoot_ipyp_20180104_112710
+ sleep 180
2018-01-04 11:28:33.079 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.079 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.079 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.079 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.080 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.080 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.080 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.080 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.080 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.080 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.081 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.081 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.081 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.081 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.081 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.082 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.082 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.082 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.082 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.082 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.082 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.082 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.147 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.147 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.210 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.210 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.211 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.211 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.338 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.338 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.377 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.377 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.384 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.384 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.402 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.402 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.440 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.440 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.442 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.443 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.460 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.460 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.465 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.465 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.469 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.469 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.470 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.470 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.474 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.474 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.484 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.484 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.486 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.486 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.488 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.488 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.492 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.492 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.493 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.493 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.494 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.494 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.495 [IPEngineApp] Initializing MPI:
2018-01-04 11:28:33.495 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.803 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.804 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.804 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.804 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.804 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.804 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.804 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.804 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.804 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.804 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.804 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.804 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.804 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:33.804 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_112710-engine.json'
2018-01-04 11:28:34.038 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.038 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.038 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.038 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.038 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.038 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.039 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.039 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.040 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.040 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.040 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.040 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.040 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.041 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.041 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.041 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.041 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.041 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.042 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.042 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.043 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.043 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.043 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.044 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.044 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.044 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.044 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.044 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.044 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.044 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.045 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.045 [IPEngineApp] Registering with controller at tcp://10.128.3.246:57071
2018-01-04 11:28:34.055 [IPControllerApp] client::client 'dac70a67-4a9d1d9307ce9477d82a9046' requested u'registration_request'
2018-01-04 11:28:34.056 [IPControllerApp] client::client '45dd6c74-3a0ae5099b5b5e2b738b5c9c' requested u'registration_request'
2018-01-04 11:28:34.057 [IPControllerApp] client::client '938e0978-024f4e035ebca0bf993282e8' requested u'registration_request'
2018-01-04 11:28:34.058 [IPControllerApp] client::client '6a9b56c7-34faba7b696bfc9b663681b8' requested u'registration_request'
2018-01-04 11:28:34.059 [IPControllerApp] client::client '3aec11f3-f37ba14f9b76b295d3ac81e1' requested u'registration_request'
2018-01-04 11:28:34.060 [IPControllerApp] client::client '668e4a3d-3fe3eee3e58875c52737d65b' requested u'registration_request'
2018-01-04 11:28:34.061 [IPControllerApp] client::client 'cc2af094-c88dd083cd2219a45aa64208' requested u'registration_request'
2018-01-04 11:28:34.062 [IPControllerApp] client::client '2f537e30-17b1ffdb77c361d872590bc7' requested u'registration_request'
2018-01-04 11:28:34.064 [IPControllerApp] client::client 'abbfbb5a-fb5a2974b31f488a3c392ff9' requested u'registration_request'
2018-01-04 11:28:34.065 [IPControllerApp] client::client 'd826f976-d076a10a6901342e800eeb28' requested u'registration_request'
2018-01-04 11:28:34.067 [IPControllerApp] client::client '877ac831-a8e227fb8c616a424d1e2060' requested u'registration_request'
2018-01-04 11:28:34.068 [IPControllerApp] client::client '6a53094b-9fc923d5403e925df06b3e3b' requested u'registration_request'
2018-01-04 11:28:34.069 [IPControllerApp] client::client '28876ddb-ada49085f3ec2e332d66af0f' requested u'registration_request'
2018-01-04 11:28:34.070 [IPControllerApp] client::client '50427d43-2127826c2c4929cbbed4d20b' requested u'registration_request'
2018-01-04 11:28:34.071 [IPControllerApp] client::client '0cec73ae-a5f53e60186ccbb20ca74b12' requested u'registration_request'
2018-01-04 11:28:34.072 [IPControllerApp] client::client '1d00bd3a-6c51c399dddab8d16fa61fbf' requested u'registration_request'
2018-01-04 11:28:34.073 [IPControllerApp] client::client 'a9086739-1632ce4debcbf2ccabea203a' requested u'registration_request'
2018-01-04 11:28:34.073 [IPControllerApp] client::client 'b16ffed7-959a2764d287fd838de766fd' requested u'registration_request'
2018-01-04 11:28:34.074 [IPControllerApp] client::client '5d51b7d4-71f55015fa1c551381463f47' requested u'registration_request'
2018-01-04 11:28:34.075 [IPControllerApp] client::client 'ae41e6de-9693c1384e8c3bdb51967fdc' requested u'registration_request'
2018-01-04 11:28:34.076 [IPControllerApp] client::client '89191cbc-2a3a727e660b29209e799200' requested u'registration_request'
2018-01-04 11:28:34.077 [IPControllerApp] client::client 'ed429cf0-7e8d424362bb0078ad93a338' requested u'registration_request'
2018-01-04 11:28:34.077 [IPControllerApp] client::client '13854826-744fac0db7d981665836d59e' requested u'registration_request'
2018-01-04 11:28:34.078 [IPControllerApp] client::client '597a863c-c2e396c04446549117beff13' requested u'registration_request'
2018-01-04 11:28:34.079 [IPControllerApp] client::client '303e3d11-40fb8336b3454c882c7901c8' requested u'registration_request'
2018-01-04 11:28:34.080 [IPControllerApp] client::client '8b8a89dd-d852c5e2212879d4e296b025' requested u'registration_request'
2018-01-04 11:28:34.085 [IPControllerApp] client::client 'e8dcc2df-f170d249319499fc99ddb199' requested u'registration_request'
2018-01-04 11:28:34.086 [IPControllerApp] client::client '75a33c2d-7c87ba9a211e5c4d203113e8' requested u'registration_request'
2018-01-04 11:28:34.087 [IPControllerApp] client::client '062af323-3e374022fc8ab87c132a391b' requested u'registration_request'
2018-01-04 11:28:34.088 [IPControllerApp] client::client 'a98335e1-05d146e95c41fb9b6256e047' requested u'registration_request'
2018-01-04 11:28:34.089 [IPControllerApp] client::client '4b7ae68f-5d0325087e68012f8455944e' requested u'registration_request'
2018-01-04 11:28:34.091 [IPControllerApp] client::client '661504b6-ecd1bda157c1c199b1dcafc3' requested u'registration_request'
2018-01-04 11:28:34.254 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.255 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.257 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.257 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.257 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.257 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.257 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.257 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.257 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.257 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.257 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.257 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.258 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.258 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.258 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.258 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.258 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.258 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.258 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.258 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.259 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.259 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.259 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.259 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.259 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.259 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.259 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.259 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.259 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.259 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.259 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.260 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 11:28:34.272 [IPEngineApp] Completed registration with id 31
2018-01-04 11:28:34.272 [IPEngineApp] Completed registration with id 8
2018-01-04 11:28:34.279 [IPEngineApp] Completed registration with id 27
2018-01-04 11:28:34.280 [IPEngineApp] Completed registration with id 9
2018-01-04 11:28:34.287 [IPEngineApp] Completed registration with id 25
2018-01-04 11:28:34.288 [IPEngineApp] Completed registration with id 18
2018-01-04 11:28:34.290 [IPEngineApp] Completed registration with id 29
2018-01-04 11:28:34.290 [IPEngineApp] Completed registration with id 19
2018-01-04 11:28:34.290 [IPEngineApp] Completed registration with id 11
2018-01-04 11:28:34.290 [IPEngineApp] Completed registration with id 3
2018-01-04 11:28:34.292 [IPEngineApp] Completed registration with id 10
2018-01-04 11:28:34.292 [IPEngineApp] Completed registration with id 26
2018-01-04 11:28:34.293 [IPEngineApp] Completed registration with id 13
2018-01-04 11:28:34.293 [IPEngineApp] Completed registration with id 0
2018-01-04 11:28:34.293 [IPEngineApp] Completed registration with id 6
2018-01-04 11:28:34.294 [IPEngineApp] Completed registration with id 15
2018-01-04 11:28:34.294 [IPEngineApp] Completed registration with id 12
2018-01-04 11:28:34.295 [IPEngineApp] Completed registration with id 4
2018-01-04 11:28:34.296 [IPEngineApp] Completed registration with id 30
2018-01-04 11:28:34.296 [IPEngineApp] Completed registration with id 16
2018-01-04 11:28:34.296 [IPEngineApp] Completed registration with id 5
2018-01-04 11:28:34.296 [IPEngineApp] Completed registration with id 1
2018-01-04 11:28:34.297 [IPEngineApp] Completed registration with id 7
2018-01-04 11:28:34.297 [IPEngineApp] Completed registration with id 24
2018-01-04 11:28:34.298 [IPEngineApp] Completed registration with id 23
2018-01-04 11:28:34.298 [IPEngineApp] Completed registration with id 20
2018-01-04 11:28:34.298 [IPEngineApp] Completed registration with id 14
2018-01-04 11:28:34.298 [IPEngineApp] Completed registration with id 28
2018-01-04 11:28:34.298 [IPEngineApp] Completed registration with id 21
2018-01-04 11:28:34.298 [IPEngineApp] Completed registration with id 17
2018-01-04 11:28:34.298 [IPEngineApp] Completed registration with id 22
2018-01-04 11:28:34.298 [IPEngineApp] Completed registration with id 2
2018-01-04 11:28:37.068 [IPControllerApp] registration::finished registering engine 5:668e4a3d-3fe3eee3e58875c52737d65b
2018-01-04 11:28:37.069 [IPControllerApp] engine::Engine Connected: 5
2018-01-04 11:28:37.103 [IPControllerApp] registration::finished registering engine 6:cc2af094-c88dd083cd2219a45aa64208
2018-01-04 11:28:37.103 [IPControllerApp] engine::Engine Connected: 6
2018-01-04 11:28:37.105 [IPControllerApp] registration::finished registering engine 2:938e0978-024f4e035ebca0bf993282e8
2018-01-04 11:28:37.105 [IPControllerApp] engine::Engine Connected: 2
2018-01-04 11:28:37.107 [IPControllerApp] registration::finished registering engine 3:6a9b56c7-34faba7b696bfc9b663681b8
2018-01-04 11:28:37.108 [IPControllerApp] engine::Engine Connected: 3
2018-01-04 11:28:37.110 [IPControllerApp] registration::finished registering engine 0:dac70a67-4a9d1d9307ce9477d82a9046
2018-01-04 11:28:37.110 [IPControllerApp] engine::Engine Connected: 0
2018-01-04 11:28:37.112 [IPControllerApp] registration::finished registering engine 1:45dd6c74-3a0ae5099b5b5e2b738b5c9c
2018-01-04 11:28:37.112 [IPControllerApp] engine::Engine Connected: 1
2018-01-04 11:28:37.114 [IPControllerApp] registration::finished registering engine 4:3aec11f3-f37ba14f9b76b295d3ac81e1
2018-01-04 11:28:37.114 [IPControllerApp] engine::Engine Connected: 4
2018-01-04 11:28:40.069 [IPControllerApp] registration::finished registering engine 27:75a33c2d-7c87ba9a211e5c4d203113e8
2018-01-04 11:28:40.069 [IPControllerApp] engine::Engine Connected: 27
2018-01-04 11:28:40.071 [IPControllerApp] registration::finished registering engine 12:28876ddb-ada49085f3ec2e332d66af0f
2018-01-04 11:28:40.072 [IPControllerApp] engine::Engine Connected: 12
2018-01-04 11:28:40.074 [IPControllerApp] registration::finished registering engine 26:e8dcc2df-f170d249319499fc99ddb199
2018-01-04 11:28:40.075 [IPControllerApp] engine::Engine Connected: 26
2018-01-04 11:28:40.078 [IPControllerApp] registration::finished registering engine 9:d826f976-d076a10a6901342e800eeb28
2018-01-04 11:28:40.079 [IPControllerApp] engine::Engine Connected: 9
2018-01-04 11:28:40.082 [IPControllerApp] registration::finished registering engine 10:877ac831-a8e227fb8c616a424d1e2060
2018-01-04 11:28:40.082 [IPControllerApp] engine::Engine Connected: 10
2018-01-04 11:28:40.085 [IPControllerApp] registration::finished registering engine 13:50427d43-2127826c2c4929cbbed4d20b
2018-01-04 11:28:40.086 [IPControllerApp] engine::Engine Connected: 13
2018-01-04 11:28:40.088 [IPControllerApp] registration::finished registering engine 18:5d51b7d4-71f55015fa1c551381463f47
2018-01-04 11:28:40.088 [IPControllerApp] engine::Engine Connected: 18
2018-01-04 11:28:40.092 [IPControllerApp] registration::finished registering engine 28:062af323-3e374022fc8ab87c132a391b
2018-01-04 11:28:40.092 [IPControllerApp] engine::Engine Connected: 28
2018-01-04 11:28:40.097 [IPControllerApp] registration::finished registering engine 30:4b7ae68f-5d0325087e68012f8455944e
2018-01-04 11:28:40.097 [IPControllerApp] engine::Engine Connected: 30
2018-01-04 11:28:40.101 [IPControllerApp] registration::finished registering engine 23:597a863c-c2e396c04446549117beff13
2018-01-04 11:28:40.101 [IPControllerApp] engine::Engine Connected: 23
2018-01-04 11:28:40.105 [IPControllerApp] registration::finished registering engine 7:2f537e30-17b1ffdb77c361d872590bc7
2018-01-04 11:28:40.105 [IPControllerApp] engine::Engine Connected: 7
2018-01-04 11:28:40.109 [IPControllerApp] registration::finished registering engine 14:0cec73ae-a5f53e60186ccbb20ca74b12
2018-01-04 11:28:40.109 [IPControllerApp] engine::Engine Connected: 14
2018-01-04 11:28:40.113 [IPControllerApp] registration::finished registering engine 8:abbfbb5a-fb5a2974b31f488a3c392ff9
2018-01-04 11:28:40.113 [IPControllerApp] engine::Engine Connected: 8
2018-01-04 11:28:40.116 [IPControllerApp] registration::finished registering engine 31:661504b6-ecd1bda157c1c199b1dcafc3
2018-01-04 11:28:40.116 [IPControllerApp] engine::Engine Connected: 31
2018-01-04 11:28:40.119 [IPControllerApp] registration::finished registering engine 19:ae41e6de-9693c1384e8c3bdb51967fdc
2018-01-04 11:28:40.120 [IPControllerApp] engine::Engine Connected: 19
2018-01-04 11:28:40.123 [IPControllerApp] registration::finished registering engine 20:89191cbc-2a3a727e660b29209e799200
2018-01-04 11:28:40.124 [IPControllerApp] engine::Engine Connected: 20
2018-01-04 11:28:40.127 [IPControllerApp] registration::finished registering engine 25:8b8a89dd-d852c5e2212879d4e296b025
2018-01-04 11:28:40.127 [IPControllerApp] engine::Engine Connected: 25
2018-01-04 11:28:40.130 [IPControllerApp] registration::finished registering engine 15:1d00bd3a-6c51c399dddab8d16fa61fbf
2018-01-04 11:28:40.131 [IPControllerApp] engine::Engine Connected: 15
2018-01-04 11:28:40.134 [IPControllerApp] registration::finished registering engine 17:b16ffed7-959a2764d287fd838de766fd
2018-01-04 11:28:40.135 [IPControllerApp] engine::Engine Connected: 17
2018-01-04 11:28:40.138 [IPControllerApp] registration::finished registering engine 16:a9086739-1632ce4debcbf2ccabea203a
2018-01-04 11:28:40.138 [IPControllerApp] engine::Engine Connected: 16
2018-01-04 11:28:40.142 [IPControllerApp] registration::finished registering engine 24:303e3d11-40fb8336b3454c882c7901c8
2018-01-04 11:28:40.142 [IPControllerApp] engine::Engine Connected: 24
2018-01-04 11:28:40.145 [IPControllerApp] registration::finished registering engine 22:13854826-744fac0db7d981665836d59e
2018-01-04 11:28:40.146 [IPControllerApp] engine::Engine Connected: 22
2018-01-04 11:28:40.149 [IPControllerApp] registration::finished registering engine 21:ed429cf0-7e8d424362bb0078ad93a338
2018-01-04 11:28:40.150 [IPControllerApp] engine::Engine Connected: 21
2018-01-04 11:28:40.153 [IPControllerApp] registration::finished registering engine 29:a98335e1-05d146e95c41fb9b6256e047
2018-01-04 11:28:40.153 [IPControllerApp] engine::Engine Connected: 29
2018-01-04 11:28:40.157 [IPControllerApp] registration::finished registering engine 11:6a53094b-9fc923d5403e925df06b3e3b
2018-01-04 11:28:40.157 [IPControllerApp] engine::Engine Connected: 11
+ srun -N 1 -n 1 -c 2 --cpu_bind=cores python apply_example.py --cluster-id=troubleshoot_ipyp_20180104_112710
2018-01-04 11:31:16.481 [IPControllerApp] client::client '\x00k\x8bEh' requested u'connection_request'
2018-01-04 11:31:16.481 [IPControllerApp] client::client ['\x00k\x8bEh'] connected
2018-01-04 11:31:16.491 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'fbfb2b1e-4f97b64ddb803f7378669890' to 0
2018-01-04 11:31:16.492 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'3a3bdb4c-60c8c07418e96bfecceb093a' to 1
2018-01-04 11:31:16.492 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'9bacec3e-b3cfe1327e07eefb68192645' to 2
2018-01-04 11:31:16.493 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'04d63961-63ffe7bbec97aeae35e9cd97' to 3
2018-01-04 11:31:16.493 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'7bd1e4af-20ba3307e09c865adea9d6e5' to 4
2018-01-04 11:31:16.494 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a52d7b90-6b3589f4ee130e5c5e0f9f36' to 5
2018-01-04 11:31:16.494 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'5e1d87ee-a0e87298dc793f9ef97d799d' to 6
2018-01-04 11:31:16.495 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'8b948e9a-f71dbf967469e851be290964' to 7
2018-01-04 11:31:16.495 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'8f601506-89266c3caf1884847bb44950' to 8
2018-01-04 11:31:16.496 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'1fd5cc57-603a17eff744da6ab62f074b' to 9
2018-01-04 11:31:16.497 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'895b390b-558f8121432eecb87d7a6938' to 10
2018-01-04 11:31:16.497 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'b0043c28-d0e946301f88e1e10ca8bfd4' to 11
2018-01-04 11:31:16.498 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'0bc20f92-af679bea0ac6fb059cbcf18d' to 12
2018-01-04 11:31:16.498 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'd4e4b57d-33760b3a9550af74fb8ece65' to 13
2018-01-04 11:31:16.499 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'95ddf241-4c83e783bdf2066b640d1e51' to 14
2018-01-04 11:31:16.499 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'042cc9d4-98d91b82b50f1f11c4527d77' to 15
2018-01-04 11:31:16.500 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'52f8502f-861b55b1ae106732f40b27c6' to 16
2018-01-04 11:31:16.500 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'e1f0772c-edab5e3c80146f4b6f0bf2b6' to 17
2018-01-04 11:31:16.501 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'fc3f53f0-5f7f709099fb2817953e4509' to 18
2018-01-04 11:31:16.501 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'45e46691-7acb1f7cb0d07bd5ca8a0f84' to 19
2018-01-04 11:31:16.502 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'7424f7c6-6635cfbaaa17bf4f13595723' to 20
2018-01-04 11:31:16.502 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'86987d90-36d94023f8aa45dd316484c9' to 21
2018-01-04 11:31:16.503 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c47466ba-d7d03cb2bee054cc12f80a6a' to 22
2018-01-04 11:31:16.503 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'92b2235b-090af502bcdad074cd19fa6c' to 23
2018-01-04 11:31:16.504 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'0e88df52-87a35256b6b1e003fe928cf3' to 24
2018-01-04 11:31:16.504 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'6d0ea5fa-29030c788d8aa46e70fa731d' to 25
2018-01-04 11:31:16.505 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'b898ed0b-51a072af35155ee199432b74' to 26
2018-01-04 11:31:16.505 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'fb638e22-0bbcefbd2c4443f37057a0a8' to 27
2018-01-04 11:31:16.506 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'51e7b1cb-044a6ab19fec7c417ee7e651' to 28
2018-01-04 11:31:16.506 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c86cb82b-363a1228cccd9486e8c9c15e' to 29
2018-01-04 11:31:16.507 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'ab8dc234-8990da1d30f8f43cac5d649b' to 30
2018-01-04 11:31:16.507 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'3d1e53b5-b871ce6ab7aebfb9184d72f3' to 31
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7462f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.594 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 11:31:28.597 [IPControllerApp] queue::request u'042cc9d4-98d91b82b50f1f11c4527d77' completed on 15
2018-01-04 11:31:28.598 [IPControllerApp] queue::request u'd4e4b57d-33760b3a9550af74fb8ece65' completed on 13
2018-01-04 11:31:28.600 [IPControllerApp] queue::request u'b0043c28-d0e946301f88e1e10ca8bfd4' completed on 11
2018-01-04 11:31:28.601 [IPControllerApp] queue::request u'6d0ea5fa-29030c788d8aa46e70fa731d' completed on 25
2018-01-04 11:31:28.602 [IPControllerApp] queue::request u'45e46691-7acb1f7cb0d07bd5ca8a0f84' completed on 19
2018-01-04 11:31:28.603 [IPControllerApp] queue::request u'ab8dc234-8990da1d30f8f43cac5d649b' completed on 30
2018-01-04 11:31:28.604 [IPControllerApp] queue::request u'1fd5cc57-603a17eff744da6ab62f074b' completed on 9
2018-01-04 11:31:28.606 [IPControllerApp] queue::request u'92b2235b-090af502bcdad074cd19fa6c' completed on 23
2018-01-04 11:31:28.607 [IPControllerApp] queue::request u'fc3f53f0-5f7f709099fb2817953e4509' completed on 18
2018-01-04 11:31:28.608 [IPControllerApp] queue::request u'b898ed0b-51a072af35155ee199432b74' completed on 26
2018-01-04 11:31:28.609 [IPControllerApp] queue::request u'895b390b-558f8121432eecb87d7a6938' completed on 10
2018-01-04 11:31:28.610 [IPControllerApp] queue::request u'a52d7b90-6b3589f4ee130e5c5e0f9f36' completed on 5
2018-01-04 11:31:28.612 [IPControllerApp] queue::request u'e1f0772c-edab5e3c80146f4b6f0bf2b6' completed on 17
2018-01-04 11:31:28.613 [IPControllerApp] queue::request u'51e7b1cb-044a6ab19fec7c417ee7e651' completed on 28
2018-01-04 11:31:28.614 [IPControllerApp] queue::request u'fb638e22-0bbcefbd2c4443f37057a0a8' completed on 27
2018-01-04 11:31:28.615 [IPControllerApp] queue::request u'0bc20f92-af679bea0ac6fb059cbcf18d' completed on 12
2018-01-04 11:31:28.616 [IPControllerApp] queue::request u'5e1d87ee-a0e87298dc793f9ef97d799d' completed on 6
2018-01-04 11:31:28.617 [IPControllerApp] queue::request u'86987d90-36d94023f8aa45dd316484c9' completed on 21
2018-01-04 11:31:28.619 [IPControllerApp] queue::request u'8b948e9a-f71dbf967469e851be290964' completed on 7
2018-01-04 11:31:28.620 [IPControllerApp] queue::request u'c86cb82b-363a1228cccd9486e8c9c15e' completed on 29
2018-01-04 11:31:28.621 [IPControllerApp] queue::request u'0e88df52-87a35256b6b1e003fe928cf3' completed on 24
2018-01-04 11:31:28.622 [IPControllerApp] queue::request u'3a3bdb4c-60c8c07418e96bfecceb093a' completed on 1
2018-01-04 11:31:28.623 [IPControllerApp] queue::request u'8f601506-89266c3caf1884847bb44950' completed on 8
2018-01-04 11:31:28.625 [IPControllerApp] queue::request u'fbfb2b1e-4f97b64ddb803f7378669890' completed on 0
2018-01-04 11:31:28.626 [IPControllerApp] queue::request u'7424f7c6-6635cfbaaa17bf4f13595723' completed on 20
2018-01-04 11:31:28.627 [IPControllerApp] queue::request u'9bacec3e-b3cfe1327e07eefb68192645' completed on 2
2018-01-04 11:31:28.628 [IPControllerApp] queue::request u'3d1e53b5-b871ce6ab7aebfb9184d72f3' completed on 31
2018-01-04 11:31:28.629 [IPControllerApp] queue::request u'52f8502f-861b55b1ae106732f40b27c6' completed on 16
2018-01-04 11:31:28.630 [IPControllerApp] queue::request u'7bd1e4af-20ba3307e09c865adea9d6e5' completed on 4
2018-01-04 11:31:28.632 [IPControllerApp] queue::request u'c47466ba-d7d03cb2bee054cc12f80a6a' completed on 22
2018-01-04 11:31:28.633 [IPControllerApp] queue::request u'95ddf241-4c83e783bdf2066b640d1e51' completed on 14
2018-01-04 11:31:28.634 [IPControllerApp] queue::request u'04d63961-63ffe7bbec97aeae35e9cd97' completed on 3
2018-01-04 11:31:28.634 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'82c3717e-a8bb11253b40d11418ac7cd5' to 0
2018-01-04 11:31:28.635 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'1d8ae328-d63b030aba4a451c5d3e728b' to 1
2018-01-04 11:31:28.637 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'092d071c-fdffe265cbe7760459b25181' to 2
2018-01-04 11:31:28.637 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'13307973-e795b094593d3bcb5c6bf1dd' to 3
2018-01-04 11:31:28.638 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'ff7d6a36-380780b8862e45e646968a48' to 4
2018-01-04 11:31:28.638 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'13d194a5-f59e56845c9041f7fb29fe4d' to 5
2018-01-04 11:31:28.639 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'87677695-770ecfa76928c3e2c67e7c3b' to 6
2018-01-04 11:31:28.640 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'eae58f12-32c5841ed5c1d90f141c6697' to 7
2018-01-04 11:31:28.640 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'b467342a-65aa4d3f56ad275e84ba2382' to 8
2018-01-04 11:31:28.641 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'59e5f928-356d05435f0fd2dc057e6067' to 9
2018-01-04 11:31:28.641 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'2539cb23-d2de13d61d42f0d207de456f' to 10
2018-01-04 11:31:28.642 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'be7e06ec-3ba1c8fd1b2c5007799687a1' to 11
2018-01-04 11:31:28.643 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'0b15853d-9a17888e92c3b9a336d74dd7' to 12
2018-01-04 11:31:28.644 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'bf2642ec-0e2cc995ebc551b361d30d50' to 13
2018-01-04 11:31:28.644 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'0fe40640-c6397fb2d0a53b592ca7058a' to 14
2018-01-04 11:31:28.645 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'bd1feb10-b02bcc2ae9e04869b9bafcfa' to 15
2018-01-04 11:31:28.645 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'6a39dc68-3146523726a9ee7ad43e9eff' to 16
2018-01-04 11:31:28.645 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'7c09e908-7e4f5d74e88f5e91a53dae61' to 17
2018-01-04 11:31:28.646 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'8a3a1368-4f7e170223c4f66dd725320b' to 18
2018-01-04 11:31:28.646 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'e4025eac-1deef4c3502e729ab543179f' to 19
2018-01-04 11:31:28.647 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'61991022-f97939f49c00a8599684b7bb' to 20
2018-01-04 11:31:28.647 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'be834e93-6c1da9880356739a2c08f57c' to 21
2018-01-04 11:31:28.648 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'53fb4108-dd2dbb03a92843e6b9b19bce' to 22
2018-01-04 11:31:28.648 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'117ce563-b6aa81794373da84e13f5b77' to 23
2018-01-04 11:31:28.648 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a157cf51-7c898696434df6a470ba75aa' to 24
2018-01-04 11:31:28.649 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'88ec45cb-b332dd4c846d40e25f3fa368' to 25
2018-01-04 11:31:28.649 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'3ff099e8-611a4f74dc1ed9802c112e6d' to 26
2018-01-04 11:31:28.650 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'9614d1dd-873e38927b8451963f9014ba' to 27
2018-01-04 11:31:28.650 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'9803a56b-4ef605c96ac9234430d64e3d' to 28
2018-01-04 11:31:28.651 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'09ae8386-1dca044078acd4b895c63599' to 29
2018-01-04 11:31:28.651 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'2f695e9c-691c492bcfbcc77fa89e880e' to 30
2018-01-04 11:31:28.651 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'ee2bec4d-a9593af10c3ee1a641e12fee' to 31
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
This application failed to start because it could not find or load the Qt platform plugin "xcb"
in "".

Reinstalling the application may fix this problem.
srun: error: nid01008: tasks 0-31: Aborted
srun: Terminating job step 9318616.1
2018-01-04 11:31:34.068 [IPControllerApp] heartbeat::missed 75a33c2d-7c87ba9a211e5c4d203113e8 : 1
2018-01-04 11:31:34.068 [IPControllerApp] heartbeat::missed 28876ddb-ada49085f3ec2e332d66af0f : 1
2018-01-04 11:31:34.069 [IPControllerApp] heartbeat::missed e8dcc2df-f170d249319499fc99ddb199 : 1
2018-01-04 11:31:34.069 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 1
2018-01-04 11:31:34.069 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 1
2018-01-04 11:31:34.069 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 1
2018-01-04 11:31:34.069 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 1
2018-01-04 11:31:34.069 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 1
2018-01-04 11:31:34.069 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 1
2018-01-04 11:31:34.069 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 1
2018-01-04 11:31:34.069 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 1
2018-01-04 11:31:34.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 1
2018-01-04 11:31:34.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 1
2018-01-04 11:31:34.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 1
2018-01-04 11:31:34.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 1
2018-01-04 11:31:34.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 1
2018-01-04 11:31:34.071 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 1
2018-01-04 11:31:34.071 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 1
2018-01-04 11:31:37.068 [IPControllerApp] heartbeat::missed 75a33c2d-7c87ba9a211e5c4d203113e8 : 2
2018-01-04 11:31:37.068 [IPControllerApp] heartbeat::missed 28876ddb-ada49085f3ec2e332d66af0f : 2
2018-01-04 11:31:37.068 [IPControllerApp] heartbeat::missed e8dcc2df-f170d249319499fc99ddb199 : 2
2018-01-04 11:31:37.069 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 2
2018-01-04 11:31:37.069 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 2
2018-01-04 11:31:37.069 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 2
2018-01-04 11:31:37.069 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 2
2018-01-04 11:31:37.069 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 2
2018-01-04 11:31:37.069 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 2
2018-01-04 11:31:37.069 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 2
2018-01-04 11:31:37.069 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 2
2018-01-04 11:31:37.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 2
2018-01-04 11:31:37.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 2
2018-01-04 11:31:37.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 2
2018-01-04 11:31:37.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 2
2018-01-04 11:31:37.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 2
2018-01-04 11:31:37.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 2
2018-01-04 11:31:40.068 [IPControllerApp] heartbeat::missed 75a33c2d-7c87ba9a211e5c4d203113e8 : 3
2018-01-04 11:31:40.068 [IPControllerApp] heartbeat::missed 28876ddb-ada49085f3ec2e332d66af0f : 3
2018-01-04 11:31:40.068 [IPControllerApp] heartbeat::missed e8dcc2df-f170d249319499fc99ddb199 : 3
2018-01-04 11:31:40.068 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 3
2018-01-04 11:31:40.068 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 3
2018-01-04 11:31:40.068 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 3
2018-01-04 11:31:40.068 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 3
2018-01-04 11:31:40.068 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 3
2018-01-04 11:31:40.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 3
2018-01-04 11:31:40.070 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 3
2018-01-04 11:31:40.070 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 3
2018-01-04 11:31:40.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 3
2018-01-04 11:31:40.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 3
2018-01-04 11:31:40.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 3
2018-01-04 11:31:40.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 3
2018-01-04 11:31:40.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 3
2018-01-04 11:31:40.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 3
2018-01-04 11:31:40.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 3
2018-01-04 11:31:40.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 3
2018-01-04 11:31:43.068 [IPControllerApp] heartbeat::missed 75a33c2d-7c87ba9a211e5c4d203113e8 : 4
2018-01-04 11:31:43.068 [IPControllerApp] heartbeat::missed 28876ddb-ada49085f3ec2e332d66af0f : 4
2018-01-04 11:31:43.068 [IPControllerApp] heartbeat::missed e8dcc2df-f170d249319499fc99ddb199 : 4
2018-01-04 11:31:43.068 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 4
2018-01-04 11:31:43.068 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 4
2018-01-04 11:31:43.068 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 4
2018-01-04 11:31:43.068 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 4
2018-01-04 11:31:43.068 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 4
2018-01-04 11:31:43.068 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 4
2018-01-04 11:31:43.068 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 4
2018-01-04 11:31:43.068 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 4
2018-01-04 11:31:43.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 4
2018-01-04 11:31:43.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 4
2018-01-04 11:31:43.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 4
2018-01-04 11:31:43.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 4
2018-01-04 11:31:43.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 4
2018-01-04 11:31:43.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 4
2018-01-04 11:31:43.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 4
2018-01-04 11:31:46.068 [IPControllerApp] heartbeat::missed 75a33c2d-7c87ba9a211e5c4d203113e8 : 5
2018-01-04 11:31:46.068 [IPControllerApp] heartbeat::missed 28876ddb-ada49085f3ec2e332d66af0f : 5
2018-01-04 11:31:46.068 [IPControllerApp] heartbeat::missed e8dcc2df-f170d249319499fc99ddb199 : 5
2018-01-04 11:31:46.069 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 5
2018-01-04 11:31:46.069 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 5
2018-01-04 11:31:46.069 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 5
2018-01-04 11:31:46.069 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 5
2018-01-04 11:31:46.069 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 5
2018-01-04 11:31:46.069 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 5
2018-01-04 11:31:46.069 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 5
2018-01-04 11:31:46.069 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 5
2018-01-04 11:31:46.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 5
2018-01-04 11:31:46.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 5
2018-01-04 11:31:46.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 5
2018-01-04 11:31:46.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 5
2018-01-04 11:31:46.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 5
2018-01-04 11:31:46.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 5
2018-01-04 11:31:49.068 [IPControllerApp] heartbeat::missed 75a33c2d-7c87ba9a211e5c4d203113e8 : 6
2018-01-04 11:31:49.068 [IPControllerApp] heartbeat::missed 28876ddb-ada49085f3ec2e332d66af0f : 6
2018-01-04 11:31:49.068 [IPControllerApp] heartbeat::missed e8dcc2df-f170d249319499fc99ddb199 : 6
2018-01-04 11:31:49.068 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 6
2018-01-04 11:31:49.068 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 6
2018-01-04 11:31:49.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 6
2018-01-04 11:31:49.070 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 6
2018-01-04 11:31:49.070 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 6
2018-01-04 11:31:49.070 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 6
2018-01-04 11:31:49.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 6
2018-01-04 11:31:49.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 6
2018-01-04 11:31:49.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 6
2018-01-04 11:31:49.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 6
2018-01-04 11:31:49.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 6
2018-01-04 11:31:49.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 6
2018-01-04 11:31:49.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 6
2018-01-04 11:31:49.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 6
2018-01-04 11:31:52.068 [IPControllerApp] heartbeat::missed 75a33c2d-7c87ba9a211e5c4d203113e8 : 7
2018-01-04 11:31:52.068 [IPControllerApp] heartbeat::missed 28876ddb-ada49085f3ec2e332d66af0f : 7
2018-01-04 11:31:52.068 [IPControllerApp] heartbeat::missed e8dcc2df-f170d249319499fc99ddb199 : 7
2018-01-04 11:31:52.068 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 7
2018-01-04 11:31:52.068 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 7
2018-01-04 11:31:52.068 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 7
2018-01-04 11:31:52.068 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 7
2018-01-04 11:31:52.068 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 7
2018-01-04 11:31:52.068 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 7
2018-01-04 11:31:52.068 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 7
2018-01-04 11:31:52.068 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 7
2018-01-04 11:31:52.068 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 7
2018-01-04 11:31:52.069 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 7
2018-01-04 11:31:52.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 7
2018-01-04 11:31:52.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 7
2018-01-04 11:31:52.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 7
2018-01-04 11:31:52.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 7
2018-01-04 11:31:55.068 [IPControllerApp] heartbeat::missed 75a33c2d-7c87ba9a211e5c4d203113e8 : 8
2018-01-04 11:31:55.068 [IPControllerApp] heartbeat::missed 28876ddb-ada49085f3ec2e332d66af0f : 8
2018-01-04 11:31:55.068 [IPControllerApp] heartbeat::missed e8dcc2df-f170d249319499fc99ddb199 : 8
2018-01-04 11:31:55.068 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 8
2018-01-04 11:31:55.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 8
2018-01-04 11:31:55.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 8
2018-01-04 11:31:58.068 [IPControllerApp] heartbeat::missed 75a33c2d-7c87ba9a211e5c4d203113e8 : 9
2018-01-04 11:31:58.068 [IPControllerApp] heartbeat::missed 28876ddb-ada49085f3ec2e332d66af0f : 9
2018-01-04 11:31:58.068 [IPControllerApp] heartbeat::missed e8dcc2df-f170d249319499fc99ddb199 : 9
2018-01-04 11:31:58.068 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 9
2018-01-04 11:31:58.068 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 9
2018-01-04 11:31:58.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 9
2018-01-04 11:31:58.070 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 9
2018-01-04 11:31:58.070 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 9
2018-01-04 11:31:58.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 9
2018-01-04 11:31:58.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 9
2018-01-04 11:31:58.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 9
2018-01-04 11:31:58.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 9
2018-01-04 11:31:58.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 9
2018-01-04 11:31:58.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 9
2018-01-04 11:31:58.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 9
2018-01-04 11:31:58.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 9
2018-01-04 11:32:01.068 [IPControllerApp] heartbeat::missed 75a33c2d-7c87ba9a211e5c4d203113e8 : 10
2018-01-04 11:32:01.068 [IPControllerApp] heartbeat::missed 28876ddb-ada49085f3ec2e332d66af0f : 10
2018-01-04 11:32:01.068 [IPControllerApp] heartbeat::missed e8dcc2df-f170d249319499fc99ddb199 : 10
2018-01-04 11:32:01.068 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 10
2018-01-04 11:32:01.068 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 10
2018-01-04 11:32:01.068 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 10
2018-01-04 11:32:01.068 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 10
2018-01-04 11:32:01.068 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 10
2018-01-04 11:32:01.068 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 10
2018-01-04 11:32:01.068 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 10
2018-01-04 11:32:01.068 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 10
2018-01-04 11:32:01.068 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 10
2018-01-04 11:32:01.068 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 10
2018-01-04 11:32:01.069 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 10
2018-01-04 11:32:01.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 10
2018-01-04 11:32:01.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 10
2018-01-04 11:32:01.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 10
2018-01-04 11:32:01.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 10
2018-01-04 11:32:01.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 10
2018-01-04 11:32:04.068 [IPControllerApp] heartbeat::missed 75a33c2d-7c87ba9a211e5c4d203113e8 : 11
2018-01-04 11:32:04.068 [IPControllerApp] heartbeat::missed 28876ddb-ada49085f3ec2e332d66af0f : 11
2018-01-04 11:32:04.069 [IPControllerApp] heartbeat::missed e8dcc2df-f170d249319499fc99ddb199 : 11
2018-01-04 11:32:04.069 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 11
2018-01-04 11:32:04.069 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 11
2018-01-04 11:32:04.069 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 11
2018-01-04 11:32:04.069 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 11
2018-01-04 11:32:04.069 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 11
2018-01-04 11:32:04.069 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 11
2018-01-04 11:32:04.069 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 11
2018-01-04 11:32:04.069 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 11
2018-01-04 11:32:04.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:04.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:04.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:04.070 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:04.070 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:04.070 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:04.070 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 11
2018-01-04 11:32:04.070 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:04.070 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:04.070 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:04.070 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:04.070 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:04.070 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:04.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:04.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:04.071 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:04.071 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:04.071 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:04.071 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:04.071 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:04.071 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 11
2018-01-04 11:32:04.071 [IPControllerApp] registration::unregister_engine(27)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '75a33c2d-7c87ba9a211e5c4d203113e8'
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed 28876ddb-ada49085f3ec2e332d66af0f : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed e8dcc2df-f170d249319499fc99ddb199 : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:07.068 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:07.069 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:07.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:07.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 11
2018-01-04 11:32:07.070 [IPControllerApp] registration::unregister_engine(12)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '28876ddb-ada49085f3ec2e332d66af0f'
2018-01-04 11:32:10.068 [IPControllerApp] heartbeat::missed e8dcc2df-f170d249319499fc99ddb199 : 11
2018-01-04 11:32:10.068 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 11
2018-01-04 11:32:10.068 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 11
2018-01-04 11:32:10.068 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 11
2018-01-04 11:32:10.068 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 11
2018-01-04 11:32:10.068 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 11
2018-01-04 11:32:10.068 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 11
2018-01-04 11:32:10.068 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 11
2018-01-04 11:32:10.068 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:10.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:10.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:10.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:10.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 11
2018-01-04 11:32:10.070 [IPControllerApp] registration::unregister_engine(26)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'e8dcc2df-f170d249319499fc99ddb199'
2018-01-04 11:32:13.069 [IPControllerApp] heartbeat::missed d826f976-d076a10a6901342e800eeb28 : 11
2018-01-04 11:32:13.069 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 11
2018-01-04 11:32:13.069 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 11
2018-01-04 11:32:13.069 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 11
2018-01-04 11:32:13.069 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 11
2018-01-04 11:32:13.069 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 11
2018-01-04 11:32:13.069 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 11
2018-01-04 11:32:13.069 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 11
2018-01-04 11:32:13.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:13.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:13.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:13.071 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:13.071 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:13.071 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:13.071 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 11
2018-01-04 11:32:13.071 [IPControllerApp] registration::unregister_engine(9)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'd826f976-d076a10a6901342e800eeb28'
2018-01-04 11:32:16.068 [IPControllerApp] heartbeat::missed 877ac831-a8e227fb8c616a424d1e2060 : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:16.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:16.070 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:16.070 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:16.070 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:16.070 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:16.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:16.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:16.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:16.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:16.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:16.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:16.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:16.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 11
2018-01-04 11:32:16.070 [IPControllerApp] registration::unregister_engine(10)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '877ac831-a8e227fb8c616a424d1e2060'
2018-01-04 11:32:19.068 [IPControllerApp] heartbeat::missed 50427d43-2127826c2c4929cbbed4d20b : 11
2018-01-04 11:32:19.068 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 11
2018-01-04 11:32:19.068 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:19.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:19.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:19.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:19.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:19.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:19.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:19.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:19.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:19.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 11
2018-01-04 11:32:19.070 [IPControllerApp] registration::unregister_engine(13)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '50427d43-2127826c2c4929cbbed4d20b'
2018-01-04 11:32:22.068 [IPControllerApp] heartbeat::missed 5d51b7d4-71f55015fa1c551381463f47 : 11
2018-01-04 11:32:22.068 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 11
2018-01-04 11:32:22.068 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 11
2018-01-04 11:32:22.068 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 11
2018-01-04 11:32:22.068 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 11
2018-01-04 11:32:22.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:22.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:22.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:22.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:22.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:22.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:22.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 11
2018-01-04 11:32:22.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:22.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:22.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:22.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:22.069 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:22.070 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:22.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:22.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:22.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:22.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:22.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:22.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:22.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:22.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 11
2018-01-04 11:32:22.070 [IPControllerApp] registration::unregister_engine(18)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '5d51b7d4-71f55015fa1c551381463f47'
2018-01-04 11:32:25.068 [IPControllerApp] heartbeat::missed dac70a67-4a9d1d9307ce9477d82a9046 : 11
2018-01-04 11:32:25.068 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 11
2018-01-04 11:32:25.069 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 11
2018-01-04 11:32:25.069 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 11
2018-01-04 11:32:25.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:25.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:25.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:25.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:25.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:25.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:25.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 11
2018-01-04 11:32:25.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:25.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:25.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:25.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:25.070 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:25.070 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:25.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:25.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:25.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:25.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:25.070 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:25.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:25.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:25.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 11
2018-01-04 11:32:25.070 [IPControllerApp] registration::unregister_engine(0)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'dac70a67-4a9d1d9307ce9477d82a9046'
2018-01-04 11:32:28.068 [IPControllerApp] heartbeat::missed 062af323-3e374022fc8ab87c132a391b : 11
2018-01-04 11:32:28.068 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 11
2018-01-04 11:32:28.068 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 11
2018-01-04 11:32:28.068 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:28.068 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:28.068 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:28.068 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:28.068 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:28.068 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:28.069 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 11
2018-01-04 11:32:28.070 [IPControllerApp] registration::unregister_engine(28)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '062af323-3e374022fc8ab87c132a391b'
2018-01-04 11:32:31.068 [IPControllerApp] heartbeat::missed 4b7ae68f-5d0325087e68012f8455944e : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:31.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:31.070 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:31.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:31.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:31.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:31.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:31.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:31.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 11
2018-01-04 11:32:31.070 [IPControllerApp] registration::unregister_engine(30)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '4b7ae68f-5d0325087e68012f8455944e'
2018-01-04 11:32:34.068 [IPControllerApp] heartbeat::missed 597a863c-c2e396c04446549117beff13 : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:34.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:34.070 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:34.070 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:34.070 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:34.070 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:34.070 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 11
2018-01-04 11:32:34.070 [IPControllerApp] registration::unregister_engine(23)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '597a863c-c2e396c04446549117beff13'
2018-01-04 11:32:37.068 [IPControllerApp] heartbeat::missed 89191cbc-2a3a727e660b29209e799200 : 11
2018-01-04 11:32:37.068 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 11
2018-01-04 11:32:37.068 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:37.068 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:37.068 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:37.068 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:37.068 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:37.068 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:37.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:37.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:37.069 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:37.069 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:37.069 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:37.069 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:37.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:37.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:37.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:37.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:37.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:37.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:37.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:37.069 [IPControllerApp] registration::unregister_engine(20)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '89191cbc-2a3a727e660b29209e799200'
2018-01-04 11:32:40.068 [IPControllerApp] heartbeat::missed 3aec11f3-f37ba14f9b76b295d3ac81e1 : 11
2018-01-04 11:32:40.068 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:40.068 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:40.068 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:40.068 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:40.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:40.070 [IPControllerApp] registration::unregister_engine(4)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '3aec11f3-f37ba14f9b76b295d3ac81e1'
2018-01-04 11:32:43.068 [IPControllerApp] heartbeat::missed 2f537e30-17b1ffdb77c361d872590bc7 : 11
2018-01-04 11:32:43.068 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:43.068 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:43.068 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:43.068 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:43.068 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:43.068 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:43.068 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:43.068 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:43.068 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:43.068 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:43.069 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:43.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:43.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:43.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:43.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:43.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:43.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:43.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:43.069 [IPControllerApp] registration::unregister_engine(7)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '2f537e30-17b1ffdb77c361d872590bc7'
2018-01-04 11:32:46.068 [IPControllerApp] heartbeat::missed 6a9b56c7-34faba7b696bfc9b663681b8 : 11
2018-01-04 11:32:46.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:46.069 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:46.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:46.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:46.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:46.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:46.069 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:46.069 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:46.069 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:46.069 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:46.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:46.069 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:46.070 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:46.070 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:46.070 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:46.070 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:46.070 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:46.070 [IPControllerApp] registration::unregister_engine(3)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '6a9b56c7-34faba7b696bfc9b663681b8'
2018-01-04 11:32:49.068 [IPControllerApp] heartbeat::missed 8b8a89dd-d852c5e2212879d4e296b025 : 11
2018-01-04 11:32:49.068 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:49.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:49.070 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:49.070 [IPControllerApp] registration::unregister_engine(25)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '8b8a89dd-d852c5e2212879d4e296b025'
2018-01-04 11:32:52.068 [IPControllerApp] heartbeat::missed 6a53094b-9fc923d5403e925df06b3e3b : 11
2018-01-04 11:32:52.068 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:52.068 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:52.068 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:52.068 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:52.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:52.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:52.069 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:52.069 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:52.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:52.069 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:52.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:52.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:52.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:52.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:52.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:52.069 [IPControllerApp] registration::unregister_engine(11)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '6a53094b-9fc923d5403e925df06b3e3b'
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed 1d00bd3a-6c51c399dddab8d16fa61fbf : 11
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:55.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:55.070 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:55.070 [IPControllerApp] registration::unregister_engine(15)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '1d00bd3a-6c51c399dddab8d16fa61fbf'
2018-01-04 11:32:58.068 [IPControllerApp] heartbeat::missed 303e3d11-40fb8336b3454c882c7901c8 : 11
2018-01-04 11:32:58.068 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:32:58.068 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:32:58.068 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:32:58.068 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:32:58.069 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:32:58.069 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:32:58.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:32:58.069 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:32:58.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:32:58.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:32:58.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:32:58.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:32:58.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:32:58.069 [IPControllerApp] registration::unregister_engine(24)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '303e3d11-40fb8336b3454c882c7901c8'
2018-01-04 11:33:01.068 [IPControllerApp] heartbeat::missed 0cec73ae-a5f53e60186ccbb20ca74b12 : 11
2018-01-04 11:33:01.068 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:33:01.068 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:33:01.068 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:33:01.068 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:33:01.068 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:33:01.068 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:33:01.068 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:33:01.068 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:33:01.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:33:01.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:33:01.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:33:01.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:33:01.069 [IPControllerApp] registration::unregister_engine(14)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '0cec73ae-a5f53e60186ccbb20ca74b12'
2018-01-04 11:33:04.068 [IPControllerApp] heartbeat::missed 668e4a3d-3fe3eee3e58875c52737d65b : 11
2018-01-04 11:33:04.068 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:33:04.068 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:33:04.068 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:33:04.068 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:33:04.068 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:33:04.068 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:33:04.068 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:33:04.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:33:04.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:33:04.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:33:04.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:33:04.069 [IPControllerApp] registration::unregister_engine(5)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '668e4a3d-3fe3eee3e58875c52737d65b'
2018-01-04 11:33:07.068 [IPControllerApp] heartbeat::missed 45dd6c74-3a0ae5099b5b5e2b738b5c9c : 11
2018-01-04 11:33:07.068 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:33:07.068 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:33:07.068 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:33:07.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:33:07.069 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:33:07.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:33:07.069 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:33:07.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:33:07.069 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:33:07.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:33:07.069 [IPControllerApp] registration::unregister_engine(1)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '45dd6c74-3a0ae5099b5b5e2b738b5c9c'
2018-01-04 11:33:10.068 [IPControllerApp] heartbeat::missed abbfbb5a-fb5a2974b31f488a3c392ff9 : 11
2018-01-04 11:33:10.068 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:33:10.068 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:33:10.068 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:33:10.068 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:33:10.068 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:33:10.068 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:33:10.068 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:33:10.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:33:10.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:33:10.069 [IPControllerApp] registration::unregister_engine(8)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'abbfbb5a-fb5a2974b31f488a3c392ff9'
2018-01-04 11:33:13.068 [IPControllerApp] heartbeat::missed cc2af094-c88dd083cd2219a45aa64208 : 11
2018-01-04 11:33:13.068 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:33:13.068 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:33:13.068 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:33:13.068 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:33:13.068 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:33:13.068 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:33:13.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:33:13.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:33:13.069 [IPControllerApp] registration::unregister_engine(6)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'cc2af094-c88dd083cd2219a45aa64208'
2018-01-04 11:33:16.068 [IPControllerApp] heartbeat::missed 938e0978-024f4e035ebca0bf993282e8 : 11
2018-01-04 11:33:16.068 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:33:16.068 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:33:16.068 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:33:16.068 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:33:16.068 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:33:16.068 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:33:16.068 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:33:16.068 [IPControllerApp] registration::unregister_engine(2)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '938e0978-024f4e035ebca0bf993282e8'
2018-01-04 11:33:19.068 [IPControllerApp] heartbeat::missed b16ffed7-959a2764d287fd838de766fd : 11
2018-01-04 11:33:19.068 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:33:19.068 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:33:19.069 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:33:19.069 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:33:19.069 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:33:19.069 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:33:19.069 [IPControllerApp] registration::unregister_engine(17)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'b16ffed7-959a2764d287fd838de766fd'
2018-01-04 11:33:22.068 [IPControllerApp] heartbeat::missed 13854826-744fac0db7d981665836d59e : 11
2018-01-04 11:33:22.068 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:33:22.068 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:33:22.068 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:33:22.068 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:33:22.068 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:33:22.068 [IPControllerApp] registration::unregister_engine(22)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '13854826-744fac0db7d981665836d59e'
2018-01-04 11:33:25.068 [IPControllerApp] heartbeat::missed a98335e1-05d146e95c41fb9b6256e047 : 11
2018-01-04 11:33:25.068 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:33:25.068 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:33:25.068 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:33:25.068 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:33:25.068 [IPControllerApp] registration::unregister_engine(29)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'a98335e1-05d146e95c41fb9b6256e047'
2018-01-04 11:33:28.068 [IPControllerApp] heartbeat::missed 661504b6-ecd1bda157c1c199b1dcafc3 : 11
2018-01-04 11:33:28.068 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:33:28.068 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:33:28.068 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:33:28.069 [IPControllerApp] registration::unregister_engine(31)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '661504b6-ecd1bda157c1c199b1dcafc3'
2018-01-04 11:33:31.068 [IPControllerApp] heartbeat::missed a9086739-1632ce4debcbf2ccabea203a : 11
2018-01-04 11:33:31.068 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:33:31.068 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:33:31.068 [IPControllerApp] registration::unregister_engine(16)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'a9086739-1632ce4debcbf2ccabea203a'
2018-01-04 11:33:34.068 [IPControllerApp] heartbeat::missed ae41e6de-9693c1384e8c3bdb51967fdc : 11
2018-01-04 11:33:34.068 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:33:34.068 [IPControllerApp] registration::unregister_engine(19)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'ae41e6de-9693c1384e8c3bdb51967fdc'
2018-01-04 11:33:37.069 [IPControllerApp] heartbeat::missed ed429cf0-7e8d424362bb0078ad93a338 : 11
2018-01-04 11:33:37.069 [IPControllerApp] registration::unregister_engine(21)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababf5dd0>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'ed429cf0-7e8d424362bb0078ad93a338'
Traceback (most recent call last):
  File "apply_example.py", line 62, in <module>
    main()
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/click/core.py", line 722, in __call__
    return self.main(*args, **kwargs)
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/click/core.py", line 697, in main
    rv = self.invoke(ctx)
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/click/core.py", line 895, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/click/core.py", line 535, in invoke
    return callback(*args, **kwargs)
  File "apply_example.py", line 48, in main
    print context_monkeys.interface_monkeys.apply(init_worker)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 85, in <lambda>
    self._sync_wrapper(self.AsyncResultWrapper(self.direct_view[:].apply_async(func, *args, **kwargs)))
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/client/client.py", line 1001, in __getitem__
    return self.direct_view(key)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/client/client.py", line 1514, in direct_view
    targets = self._build_targets(targets)[1]
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/client/client.py", line 566, in _build_targets
    raise error.NoEnginesRegistered("Can't build targets without any engines")
ipyparallel.error.NoEnginesRegistered: Can't build targets without any engines
srun: error: nid01857: task 0: Exited with exit code 1
srun: Terminating job step 9318616.2
