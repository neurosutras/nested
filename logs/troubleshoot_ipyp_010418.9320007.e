ModuleCmd_Switch.c(179):ERROR:152: Module 'PrgEnv-intel' is currently not loaded
+ cd /global/homes/a/aaronmil/python_modules/nested
++ date +%Y%m%d_%H%M%S
+ export DATE=20180104_121046
+ DATE=20180104_121046
+ cluster_id=troubleshoot_ipyp_20180104_121046
+ sleep 1
+ srun -N 1 -n 1 -c 2 --cpu_bind=cores ipcontroller '--ip=*' --nodb --cluster-id=troubleshoot_ipyp_20180104_121046
+ sleep 45
2018-01-04 12:11:22.395 [IPControllerApp] Hub listening on tcp://*:37996 for registration.
2018-01-04 12:11:22.398 [IPControllerApp] Hub using DB backend: 'NoDB'
2018-01-04 12:11:22.784 [IPControllerApp] hub::created hub
2018-01-04 12:11:22.785 [IPControllerApp] writing connection info to /global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-client.json
2018-01-04 12:11:22.799 [IPControllerApp] writing connection info to /global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json
2018-01-04 12:11:22.813 [IPControllerApp] task::using Python leastload Task scheduler
2018-01-04 12:11:22.813 [IPControllerApp] Heartmonitor started
2018-01-04 12:11:22.851 [IPControllerApp] Creating pid file: /global/u1/a/aaronmil/.ipython/profile_default/pid/ipcontroller-troubleshoot_ipyp_20180104_121046.pid
2018-01-04 12:11:22.852 [scheduler] Scheduler started [leastload]
2018-01-04 12:11:22.855 [IPControllerApp] client::client '\x00k\x8bEg' requested u'connection_request'
2018-01-04 12:11:22.855 [IPControllerApp] client::client ['\x00k\x8bEg'] connected
+ sleep 1
+ srun -N 1 -n 32 -c 2 --cpu_bind=cores ipengine --mpi=mpi4py --cluster-id=troubleshoot_ipyp_20180104_121046
+ sleep 180
2018-01-04 12:14:08.091 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.091 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.092 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.092 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.093 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.093 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.110 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.110 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.111 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.111 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.113 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.113 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.115 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.115 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.119 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.119 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.125 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.125 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.127 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.127 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.127 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.127 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.132 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.132 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.336 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.336 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.430 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.431 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.746 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.747 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.776 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.776 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.788 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.788 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.837 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.837 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.893 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.893 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.905 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.905 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.936 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.936 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.936 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.936 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.937 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.937 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.957 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.957 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.981 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.981 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.989 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.989 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:08.993 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:08.993 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:09.003 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:09.004 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:09.010 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:09.010 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:09.010 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:09.010 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:09.010 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:09.010 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:09.011 [IPEngineApp] Initializing MPI:
2018-01-04 12:14:09.011 [IPEngineApp] from mpi4py import MPI as mpi
mpi.size = mpi.COMM_WORLD.Get_size()
mpi.rank = mpi.COMM_WORLD.Get_rank()

2018-01-04 12:14:09.347 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.347 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.347 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.347 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.347 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.347 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.347 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.347 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.347 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.347 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.347 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.348 [IPEngineApp] Loading url_file u'/global/u1/a/aaronmil/.ipython/profile_default/security/ipcontroller-troubleshoot_ipyp_20180104_121046-engine.json'
2018-01-04 12:14:09.598 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.598 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.598 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.598 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.598 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.602 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.603 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.603 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.603 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.604 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.605 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.605 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.606 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.606 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.606 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.606 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.607 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.607 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.607 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.607 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.608 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.608 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.609 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.609 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.609 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.610 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.610 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.610 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.610 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.610 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.610 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.611 [IPEngineApp] Registering with controller at tcp://10.128.3.235:37996
2018-01-04 12:14:09.623 [IPControllerApp] client::client 'e7deed74-0ff2978a255660d693e09c3b' requested u'registration_request'
2018-01-04 12:14:09.624 [IPControllerApp] client::client 'e44240a2-b3fc155e9ccb22aa2c660bf5' requested u'registration_request'
2018-01-04 12:14:09.625 [IPControllerApp] client::client '14f1caf3-5b9dd361ed1986f1b1e175d1' requested u'registration_request'
2018-01-04 12:14:09.626 [IPControllerApp] client::client 'e90c3022-c43dd23147ac258928771a60' requested u'registration_request'
2018-01-04 12:14:09.627 [IPControllerApp] client::client 'fd216f94-e9cc397f8c6065c2c90339fe' requested u'registration_request'
2018-01-04 12:14:09.627 [IPControllerApp] client::client '5161769a-5fe47396e022aae8b7951fcf' requested u'registration_request'
2018-01-04 12:14:09.628 [IPControllerApp] client::client 'acc072c8-755c592fd57bb2145362f0ca' requested u'registration_request'
2018-01-04 12:14:09.629 [IPControllerApp] client::client '2ad7eaaf-6241cd4d51d00cda77c1f77f' requested u'registration_request'
2018-01-04 12:14:09.630 [IPControllerApp] client::client '66399edb-ca3ff6346973aafcd1685152' requested u'registration_request'
2018-01-04 12:14:09.631 [IPControllerApp] client::client '6e19df5b-a601af6b09e1f2d5692492ee' requested u'registration_request'
2018-01-04 12:14:09.632 [IPControllerApp] client::client '4eee1008-988784ff1cdee9a7b4af2303' requested u'registration_request'
2018-01-04 12:14:09.634 [IPControllerApp] client::client '2d598dd2-e5a30c8612b864b70098f90f' requested u'registration_request'
2018-01-04 12:14:09.635 [IPControllerApp] client::client '82ab8cb6-d75bca944f9388e20d4151cf' requested u'registration_request'
2018-01-04 12:14:09.636 [IPControllerApp] client::client 'f707aaef-8c155feac7006ea9214c4a70' requested u'registration_request'
2018-01-04 12:14:09.638 [IPControllerApp] client::client '2b0393e0-34904b8e5dfa6abb8e3cbe54' requested u'registration_request'
2018-01-04 12:14:09.639 [IPControllerApp] client::client '1f754c2b-0124e075ff5aa5507cebb6a3' requested u'registration_request'
2018-01-04 12:14:09.640 [IPControllerApp] client::client 'e32f72f2-3b6d2927567029c9fce8415d' requested u'registration_request'
2018-01-04 12:14:09.642 [IPControllerApp] client::client '62c9a487-e05d3f87fb5b42bf1ddc5c4d' requested u'registration_request'
2018-01-04 12:14:09.643 [IPControllerApp] client::client '8fdcefdf-6e29a1d4c0c75dda97d55002' requested u'registration_request'
2018-01-04 12:14:09.644 [IPControllerApp] client::client 'c0b47cdc-8d871db57661c94ed26af39b' requested u'registration_request'
2018-01-04 12:14:09.645 [IPControllerApp] client::client '4b6f15c8-df04e637f90e03cf3b611777' requested u'registration_request'
2018-01-04 12:14:09.646 [IPControllerApp] client::client '53182aa0-6d11fdf0aea26933608fbc49' requested u'registration_request'
2018-01-04 12:14:09.647 [IPControllerApp] client::client '80330e9a-882d59a5521a4d763578276a' requested u'registration_request'
2018-01-04 12:14:09.649 [IPControllerApp] client::client '3302d274-3bf27449bb70e33f43e72c57' requested u'registration_request'
2018-01-04 12:14:09.650 [IPControllerApp] client::client '160cd1ca-1add0d947e80586fba6a76b5' requested u'registration_request'
2018-01-04 12:14:09.651 [IPControllerApp] client::client '26a952f5-49a0e8cf68efa4a063940b5a' requested u'registration_request'
2018-01-04 12:14:09.652 [IPControllerApp] client::client 'bb3d4789-8c83bc13aac5204b069225d8' requested u'registration_request'
2018-01-04 12:14:09.653 [IPControllerApp] client::client 'bb7fdac7-c67298f88ce92956ade790a4' requested u'registration_request'
2018-01-04 12:14:09.654 [IPControllerApp] client::client '0bf769b7-a9df55e1c8d0aa7339f88170' requested u'registration_request'
2018-01-04 12:14:09.655 [IPControllerApp] client::client '28d00e5f-26ff055a06b08244322060f4' requested u'registration_request'
2018-01-04 12:14:09.657 [IPControllerApp] client::client '92b21546-c1b7015ac82f06e7290ee9b0' requested u'registration_request'
2018-01-04 12:14:09.658 [IPControllerApp] client::client 'd69c000d-3a08a244b72cfca20288512c' requested u'registration_request'
2018-01-04 12:14:09.816 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.819 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.819 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.820 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.820 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.820 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.820 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.820 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.820 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.820 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.820 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.820 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.820 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.820 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.820 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.820 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.821 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.821 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.821 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.821 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.822 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.822 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.823 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.823 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.823 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.823 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.823 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.824 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.824 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.824 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.824 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.825 [IPEngineApp] Starting to monitor the heartbeat signal from the hub every 3010 ms.
2018-01-04 12:14:09.834 [IPEngineApp] Completed registration with id 28
2018-01-04 12:14:09.839 [IPEngineApp] Completed registration with id 0
2018-01-04 12:14:09.851 [IPEngineApp] Completed registration with id 10
2018-01-04 12:14:09.852 [IPEngineApp] Completed registration with id 8
2018-01-04 12:14:09.852 [IPEngineApp] Completed registration with id 25
2018-01-04 12:14:09.852 [IPEngineApp] Completed registration with id 6
2018-01-04 12:14:09.854 [IPEngineApp] Completed registration with id 24
2018-01-04 12:14:09.854 [IPEngineApp] Completed registration with id 16
2018-01-04 12:14:09.855 [IPEngineApp] Completed registration with id 17
2018-01-04 12:14:09.856 [IPEngineApp] Completed registration with id 21
2018-01-04 12:14:09.857 [IPEngineApp] Completed registration with id 7
2018-01-04 12:14:09.858 [IPEngineApp] Completed registration with id 19
2018-01-04 12:14:09.858 [IPEngineApp] Completed registration with id 13
2018-01-04 12:14:09.859 [IPEngineApp] Completed registration with id 14
2018-01-04 12:14:09.859 [IPEngineApp] Completed registration with id 3
2018-01-04 12:14:09.859 [IPEngineApp] Completed registration with id 27
2018-01-04 12:14:09.859 [IPEngineApp] Completed registration with id 4
2018-01-04 12:14:09.859 [IPEngineApp] Completed registration with id 23
2018-01-04 12:14:09.860 [IPEngineApp] Completed registration with id 11
2018-01-04 12:14:09.861 [IPEngineApp] Completed registration with id 30
2018-01-04 12:14:09.861 [IPEngineApp] Completed registration with id 29
2018-01-04 12:14:09.861 [IPEngineApp] Completed registration with id 5
2018-01-04 12:14:09.861 [IPEngineApp] Completed registration with id 20
2018-01-04 12:14:09.861 [IPEngineApp] Completed registration with id 12
2018-01-04 12:14:09.861 [IPEngineApp] Completed registration with id 18
2018-01-04 12:14:09.861 [IPEngineApp] Completed registration with id 2
2018-01-04 12:14:09.861 [IPEngineApp] Completed registration with id 26
2018-01-04 12:14:09.862 [IPEngineApp] Completed registration with id 1
2018-01-04 12:14:09.862 [IPEngineApp] Completed registration with id 31
2018-01-04 12:14:09.862 [IPEngineApp] Completed registration with id 15
2018-01-04 12:14:09.862 [IPEngineApp] Completed registration with id 22
2018-01-04 12:14:09.862 [IPEngineApp] Completed registration with id 9
2018-01-04 12:14:13.816 [IPControllerApp] registration::finished registering engine 21:53182aa0-6d11fdf0aea26933608fbc49
2018-01-04 12:14:13.816 [IPControllerApp] engine::Engine Connected: 21
2018-01-04 12:14:13.831 [IPControllerApp] registration::finished registering engine 14:2b0393e0-34904b8e5dfa6abb8e3cbe54
2018-01-04 12:14:13.831 [IPControllerApp] engine::Engine Connected: 14
2018-01-04 12:14:13.834 [IPControllerApp] registration::finished registering engine 0:e7deed74-0ff2978a255660d693e09c3b
2018-01-04 12:14:13.834 [IPControllerApp] engine::Engine Connected: 0
2018-01-04 12:14:13.838 [IPControllerApp] registration::finished registering engine 27:bb7fdac7-c67298f88ce92956ade790a4
2018-01-04 12:14:13.838 [IPControllerApp] engine::Engine Connected: 27
2018-01-04 12:14:13.841 [IPControllerApp] registration::finished registering engine 20:4b6f15c8-df04e637f90e03cf3b611777
2018-01-04 12:14:13.842 [IPControllerApp] engine::Engine Connected: 20
2018-01-04 12:14:13.845 [IPControllerApp] registration::finished registering engine 26:bb3d4789-8c83bc13aac5204b069225d8
2018-01-04 12:14:13.846 [IPControllerApp] engine::Engine Connected: 26
2018-01-04 12:14:13.848 [IPControllerApp] registration::finished registering engine 5:5161769a-5fe47396e022aae8b7951fcf
2018-01-04 12:14:13.849 [IPControllerApp] engine::Engine Connected: 5
2018-01-04 12:14:13.851 [IPControllerApp] registration::finished registering engine 17:62c9a487-e05d3f87fb5b42bf1ddc5c4d
2018-01-04 12:14:13.852 [IPControllerApp] engine::Engine Connected: 17
2018-01-04 12:14:13.854 [IPControllerApp] registration::finished registering engine 18:8fdcefdf-6e29a1d4c0c75dda97d55002
2018-01-04 12:14:13.855 [IPControllerApp] engine::Engine Connected: 18
2018-01-04 12:14:13.858 [IPControllerApp] registration::finished registering engine 3:e90c3022-c43dd23147ac258928771a60
2018-01-04 12:14:13.858 [IPControllerApp] engine::Engine Connected: 3
2018-01-04 12:14:13.862 [IPControllerApp] registration::finished registering engine 10:4eee1008-988784ff1cdee9a7b4af2303
2018-01-04 12:14:13.862 [IPControllerApp] engine::Engine Connected: 10
2018-01-04 12:14:13.866 [IPControllerApp] registration::finished registering engine 25:26a952f5-49a0e8cf68efa4a063940b5a
2018-01-04 12:14:13.866 [IPControllerApp] engine::Engine Connected: 25
2018-01-04 12:14:13.869 [IPControllerApp] registration::finished registering engine 4:fd216f94-e9cc397f8c6065c2c90339fe
2018-01-04 12:14:13.869 [IPControllerApp] engine::Engine Connected: 4
2018-01-04 12:14:13.873 [IPControllerApp] registration::finished registering engine 1:e44240a2-b3fc155e9ccb22aa2c660bf5
2018-01-04 12:14:13.873 [IPControllerApp] engine::Engine Connected: 1
2018-01-04 12:14:13.877 [IPControllerApp] registration::finished registering engine 22:80330e9a-882d59a5521a4d763578276a
2018-01-04 12:14:13.877 [IPControllerApp] engine::Engine Connected: 22
2018-01-04 12:14:13.883 [IPControllerApp] registration::finished registering engine 31:d69c000d-3a08a244b72cfca20288512c
2018-01-04 12:14:13.883 [IPControllerApp] engine::Engine Connected: 31
2018-01-04 12:14:13.888 [IPControllerApp] registration::finished registering engine 16:e32f72f2-3b6d2927567029c9fce8415d
2018-01-04 12:14:13.888 [IPControllerApp] engine::Engine Connected: 16
2018-01-04 12:14:13.893 [IPControllerApp] registration::finished registering engine 29:28d00e5f-26ff055a06b08244322060f4
2018-01-04 12:14:13.894 [IPControllerApp] engine::Engine Connected: 29
2018-01-04 12:14:13.897 [IPControllerApp] registration::finished registering engine 6:acc072c8-755c592fd57bb2145362f0ca
2018-01-04 12:14:13.898 [IPControllerApp] engine::Engine Connected: 6
2018-01-04 12:14:13.902 [IPControllerApp] registration::finished registering engine 8:66399edb-ca3ff6346973aafcd1685152
2018-01-04 12:14:13.902 [IPControllerApp] engine::Engine Connected: 8
2018-01-04 12:14:13.906 [IPControllerApp] registration::finished registering engine 23:3302d274-3bf27449bb70e33f43e72c57
2018-01-04 12:14:13.907 [IPControllerApp] engine::Engine Connected: 23
2018-01-04 12:14:13.911 [IPControllerApp] registration::finished registering engine 2:14f1caf3-5b9dd361ed1986f1b1e175d1
2018-01-04 12:14:13.912 [IPControllerApp] engine::Engine Connected: 2
2018-01-04 12:14:13.916 [IPControllerApp] registration::finished registering engine 19:c0b47cdc-8d871db57661c94ed26af39b
2018-01-04 12:14:13.916 [IPControllerApp] engine::Engine Connected: 19
2018-01-04 12:14:13.920 [IPControllerApp] registration::finished registering engine 12:82ab8cb6-d75bca944f9388e20d4151cf
2018-01-04 12:14:13.920 [IPControllerApp] engine::Engine Connected: 12
2018-01-04 12:14:13.924 [IPControllerApp] registration::finished registering engine 24:160cd1ca-1add0d947e80586fba6a76b5
2018-01-04 12:14:13.924 [IPControllerApp] engine::Engine Connected: 24
2018-01-04 12:14:13.928 [IPControllerApp] registration::finished registering engine 7:2ad7eaaf-6241cd4d51d00cda77c1f77f
2018-01-04 12:14:13.928 [IPControllerApp] engine::Engine Connected: 7
2018-01-04 12:14:13.932 [IPControllerApp] registration::finished registering engine 28:0bf769b7-a9df55e1c8d0aa7339f88170
2018-01-04 12:14:13.932 [IPControllerApp] engine::Engine Connected: 28
2018-01-04 12:14:13.937 [IPControllerApp] registration::finished registering engine 30:92b21546-c1b7015ac82f06e7290ee9b0
2018-01-04 12:14:13.937 [IPControllerApp] engine::Engine Connected: 30
2018-01-04 12:14:13.941 [IPControllerApp] registration::finished registering engine 11:2d598dd2-e5a30c8612b864b70098f90f
2018-01-04 12:14:13.942 [IPControllerApp] engine::Engine Connected: 11
2018-01-04 12:14:13.946 [IPControllerApp] registration::finished registering engine 13:f707aaef-8c155feac7006ea9214c4a70
2018-01-04 12:14:13.946 [IPControllerApp] engine::Engine Connected: 13
2018-01-04 12:14:13.949 [IPControllerApp] registration::finished registering engine 9:6e19df5b-a601af6b09e1f2d5692492ee
2018-01-04 12:14:13.950 [IPControllerApp] engine::Engine Connected: 9
2018-01-04 12:14:13.955 [IPControllerApp] registration::finished registering engine 15:1f754c2b-0124e075ff5aa5507cebb6a3
2018-01-04 12:14:13.955 [IPControllerApp] engine::Engine Connected: 15
+ srun -N 1 -n 1 -c 2 --cpu_bind=cores python apply_example.py --cluster-id=troubleshoot_ipyp_20180104_121046
2018-01-04 12:14:45.317 [IPControllerApp] client::client '\x00k\x8bEh' requested u'connection_request'
2018-01-04 12:14:45.317 [IPControllerApp] client::client ['\x00k\x8bEh'] connected
2018-01-04 12:14:45.324 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'64ea1702-5806697124ea4108f9c83c50' to 0
2018-01-04 12:14:45.324 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a3dd67b6-c471d267ca0fe7c5e5f839a7' to 1
2018-01-04 12:14:45.325 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'69cd7eab-c6c1ca05f5d57952960080a8' to 2
2018-01-04 12:14:45.325 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'b187b2db-b89ff87c630c9ad38d0e11d6' to 3
2018-01-04 12:14:45.326 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'ddd79bac-78aaa4ec3d665327d09468d8' to 4
2018-01-04 12:14:45.327 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'85ab329b-08e35041a0b5ebaa530412f7' to 5
2018-01-04 12:14:45.327 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'39f7565d-7b4faf23043196d95b9bb635' to 6
2018-01-04 12:14:45.327 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'86d0453b-5df4f957cde234c7a89ba84b' to 7
2018-01-04 12:14:45.328 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c78a4181-e7291d067649a1d6a4f05f56' to 8
2018-01-04 12:14:45.328 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'7ece7877-dc6adace005d3785b493a4c9' to 9
2018-01-04 12:14:45.329 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'55c2b311-82f2c2391bfe57812b7c0e8f' to 10
2018-01-04 12:14:45.330 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'3df99a22-202313ab9d49190dfc4df1ec' to 11
2018-01-04 12:14:45.330 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'b0a914eb-5dbe196856ccf070b76e27ba' to 12
2018-01-04 12:14:45.331 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'cd33630a-460a5325a04a17ad4901e159' to 13
2018-01-04 12:14:45.331 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'ed922d49-39eda89214a7b69fed916145' to 14
2018-01-04 12:14:45.332 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'71a9d0b6-acbe2414afc60868506bcf83' to 15
2018-01-04 12:14:45.332 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'3661f537-758548aa53f42d59d85e7c3a' to 16
2018-01-04 12:14:45.333 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'1f4a7d7f-70812d0f9227cac4fa26e399' to 17
2018-01-04 12:14:45.333 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'05b1666e-0b5676117306c31340f2d0b6' to 18
2018-01-04 12:14:45.334 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'd50ae220-0545b84f138de27190c13ed5' to 19
2018-01-04 12:14:45.334 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'2459b54a-8cd7f3b30b982907e7455bb7' to 20
2018-01-04 12:14:45.335 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'8ccdb462-fd30bbfc913cbc2003fb41c6' to 21
2018-01-04 12:14:45.335 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'44ebf016-0ec6c8efefd45639769e5b19' to 22
2018-01-04 12:14:45.336 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'5b11a89c-528fd2ef05c2b6a8d64920df' to 23
2018-01-04 12:14:45.336 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'16dc40e7-1a756f5c12a58e9775ad1c58' to 24
2018-01-04 12:14:45.336 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'15528594-619ef34a7fb3062512051593' to 25
2018-01-04 12:14:45.337 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'7f97275f-1ad51b6647b01200d4f0f170' to 26
2018-01-04 12:14:45.338 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'af58d5c8-67ac1151a6cd2ab13b2f2cb9' to 27
2018-01-04 12:14:45.338 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'7df2d0f8-a57c4fa3df473bb2c2f70f4f' to 28
2018-01-04 12:14:45.338 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'20040a29-a1bf7e5099f9515bf497668c' to 29
2018-01-04 12:14:45.339 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'40a5b123-aa8dd5867241bd74d1537a58' to 30
2018-01-04 12:14:45.339 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'b2997a88-4224b8f9dfaaf86ca9aa13d8' to 31
2018-01-04 12:14:56.241 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.241 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.241 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.241 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.241 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.241 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.241 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.241 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7462f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.242 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.243 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7462f50>
2018-01-04 12:14:56.243 [IPEngineApp] entering eventloop <function loop_qt5 at 0x2aaae7461f50>
2018-01-04 12:14:56.247 [IPControllerApp] queue::request u'85ab329b-08e35041a0b5ebaa530412f7' completed on 5
2018-01-04 12:14:56.248 [IPControllerApp] queue::request u'a3dd67b6-c471d267ca0fe7c5e5f839a7' completed on 1
2018-01-04 12:14:56.250 [IPControllerApp] queue::request u'05b1666e-0b5676117306c31340f2d0b6' completed on 18
2018-01-04 12:14:56.251 [IPControllerApp] queue::request u'b2997a88-4224b8f9dfaaf86ca9aa13d8' completed on 31
2018-01-04 12:14:56.252 [IPControllerApp] queue::request u'7df2d0f8-a57c4fa3df473bb2c2f70f4f' completed on 28
2018-01-04 12:14:56.253 [IPControllerApp] queue::request u'ddd79bac-78aaa4ec3d665327d09468d8' completed on 4
2018-01-04 12:14:56.254 [IPControllerApp] queue::request u'3661f537-758548aa53f42d59d85e7c3a' completed on 16
2018-01-04 12:14:56.256 [IPControllerApp] queue::request u'20040a29-a1bf7e5099f9515bf497668c' completed on 29
2018-01-04 12:14:56.257 [IPControllerApp] queue::request u'86d0453b-5df4f957cde234c7a89ba84b' completed on 7
2018-01-04 12:14:56.258 [IPControllerApp] queue::request u'b0a914eb-5dbe196856ccf070b76e27ba' completed on 12
2018-01-04 12:14:56.259 [IPControllerApp] queue::request u'40a5b123-aa8dd5867241bd74d1537a58' completed on 30
2018-01-04 12:14:56.260 [IPControllerApp] queue::request u'ed922d49-39eda89214a7b69fed916145' completed on 14
2018-01-04 12:14:56.262 [IPControllerApp] queue::request u'15528594-619ef34a7fb3062512051593' completed on 25
2018-01-04 12:14:56.263 [IPControllerApp] queue::request u'2459b54a-8cd7f3b30b982907e7455bb7' completed on 20
2018-01-04 12:14:56.264 [IPControllerApp] queue::request u'55c2b311-82f2c2391bfe57812b7c0e8f' completed on 10
2018-01-04 12:14:56.265 [IPControllerApp] queue::request u'7f97275f-1ad51b6647b01200d4f0f170' completed on 26
2018-01-04 12:14:56.266 [IPControllerApp] queue::request u'5b11a89c-528fd2ef05c2b6a8d64920df' completed on 23
2018-01-04 12:14:56.267 [IPControllerApp] queue::request u'16dc40e7-1a756f5c12a58e9775ad1c58' completed on 24
2018-01-04 12:14:56.268 [IPControllerApp] queue::request u'8ccdb462-fd30bbfc913cbc2003fb41c6' completed on 21
2018-01-04 12:14:56.270 [IPControllerApp] queue::request u'c78a4181-e7291d067649a1d6a4f05f56' completed on 8
2018-01-04 12:14:56.271 [IPControllerApp] queue::request u'1f4a7d7f-70812d0f9227cac4fa26e399' completed on 17
2018-01-04 12:14:56.272 [IPControllerApp] queue::request u'cd33630a-460a5325a04a17ad4901e159' completed on 13
2018-01-04 12:14:56.273 [IPControllerApp] queue::request u'69cd7eab-c6c1ca05f5d57952960080a8' completed on 2
2018-01-04 12:14:56.274 [IPControllerApp] queue::request u'af58d5c8-67ac1151a6cd2ab13b2f2cb9' completed on 27
2018-01-04 12:14:56.275 [IPControllerApp] queue::request u'39f7565d-7b4faf23043196d95b9bb635' completed on 6
2018-01-04 12:14:56.276 [IPControllerApp] queue::request u'71a9d0b6-acbe2414afc60868506bcf83' completed on 15
2018-01-04 12:14:56.278 [IPControllerApp] queue::request u'44ebf016-0ec6c8efefd45639769e5b19' completed on 22
2018-01-04 12:14:56.279 [IPControllerApp] queue::request u'b187b2db-b89ff87c630c9ad38d0e11d6' completed on 3
2018-01-04 12:14:56.280 [IPControllerApp] queue::request u'3df99a22-202313ab9d49190dfc4df1ec' completed on 11
2018-01-04 12:14:56.281 [IPControllerApp] queue::request u'64ea1702-5806697124ea4108f9c83c50' completed on 0
2018-01-04 12:14:56.282 [IPControllerApp] queue::request u'd50ae220-0545b84f138de27190c13ed5' completed on 19
2018-01-04 12:14:56.283 [IPControllerApp] queue::request u'7ece7877-dc6adace005d3785b493a4c9' completed on 9
2018-01-04 12:14:56.283 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'afa0fdbf-b219fd29c955c47fa1105e32' to 0
2018-01-04 12:14:56.284 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'0410327b-e72698cca846e110d0ac4b3b' to 1
2018-01-04 12:14:56.284 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'4c364aa1-ccb9b4acdec1b95be245dfc9' to 2
2018-01-04 12:14:56.285 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'816e6dc9-24be03a1243cc2849247daf1' to 3
2018-01-04 12:14:56.285 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'ad178fcd-588a630d0f62c52f6832640e' to 4
2018-01-04 12:14:56.285 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'8a55677f-0b998f8d7ea1dd63cf29692b' to 5
2018-01-04 12:14:56.286 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'dd02a788-c4a72b3e6b1e28dc5fa8292e' to 6
2018-01-04 12:14:56.286 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'd96ec1dd-e40f6688e7e367677eece25f' to 7
2018-01-04 12:14:56.287 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'5c06f3a7-ca8c5ba14c938756ca301a50' to 8
2018-01-04 12:14:56.287 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a59074b7-47cddb3362fe9b3efd0dc4f3' to 9
2018-01-04 12:14:56.288 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'2d4d1d74-e5a88976e06a121ec745fb28' to 10
2018-01-04 12:14:56.288 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'eaaf366e-69b65321de76854aa6fc6384' to 11
2018-01-04 12:14:56.289 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'c36d4622-8e8d63c4a0615d10388e5698' to 12
2018-01-04 12:14:56.289 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'f8bf5a17-a279a5f4e722752a5aef08f1' to 13
2018-01-04 12:14:56.289 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'14a54c2b-fa75817c571b99da0de602b9' to 14
2018-01-04 12:14:56.290 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'77843f1b-8d4e6aea061adec068632aa4' to 15
2018-01-04 12:14:56.291 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'bc3d7b42-2ff55ed3d491bac5714105e2' to 16
2018-01-04 12:14:56.291 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'7cf19b89-a769e27ca4a92121245c1a64' to 17
2018-01-04 12:14:56.292 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a6fea5a9-e113c6f33d7a19df24eea9fd' to 18
2018-01-04 12:14:56.293 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'1397ec41-da7f5355e6bb16d2766c6572' to 19
2018-01-04 12:14:56.293 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'7e673403-6e0b3c74262e296f45865aed' to 20
2018-01-04 12:14:56.294 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'78d9259f-cfc54d0397d88fdbb1ef5384' to 21
2018-01-04 12:14:56.295 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'cce659d8-949270f438db4551d7f83e55' to 22
2018-01-04 12:14:56.295 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'92000d59-886639653102e6df51e94d71' to 23
2018-01-04 12:14:56.295 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'b2fff128-6ebe96d44e58fbf009e59b39' to 24
2018-01-04 12:14:56.296 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'25f995cf-312f90384373bb06bdb3b4b4' to 25
2018-01-04 12:14:56.296 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'6d3693fc-577572c6c6c168468833f6f4' to 26
2018-01-04 12:14:56.297 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'647f6326-319835b2409d1ef6655702b3' to 27
2018-01-04 12:14:56.297 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'a1206039-93d8571a2a86c5a9e7e77b63' to 28
2018-01-04 12:14:56.298 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'0d038091-7c5e79f63d6a804784e58707' to 29
2018-01-04 12:14:56.298 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'e2a370cb-8206e3c31346c7c91ab383d3' to 30
2018-01-04 12:14:56.298 [IPControllerApp] queue::client '\x00*\xe8\x94J' submitted request u'dae23776-b559bae199c3cd52be20711e' to 31
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
srun: error: nid00997: tasks 0-31: Aborted
srun: Terminating job step 9320007.1
2018-01-04 12:15:01.816 [IPControllerApp] heartbeat::missed 53182aa0-6d11fdf0aea26933608fbc49 : 1
2018-01-04 12:15:01.816 [IPControllerApp] heartbeat::missed 2b0393e0-34904b8e5dfa6abb8e3cbe54 : 1
2018-01-04 12:15:01.816 [IPControllerApp] heartbeat::missed e7deed74-0ff2978a255660d693e09c3b : 1
2018-01-04 12:15:01.816 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 1
2018-01-04 12:15:01.816 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 1
2018-01-04 12:15:01.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 1
2018-01-04 12:15:01.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 1
2018-01-04 12:15:01.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 1
2018-01-04 12:15:01.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 1
2018-01-04 12:15:01.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 1
2018-01-04 12:15:01.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 1
2018-01-04 12:15:01.818 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 1
2018-01-04 12:15:01.818 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 1
2018-01-04 12:15:01.818 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 1
2018-01-04 12:15:01.818 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 1
2018-01-04 12:15:01.818 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 1
2018-01-04 12:15:01.818 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 1
2018-01-04 12:15:01.818 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 1
2018-01-04 12:15:01.818 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 1
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed 53182aa0-6d11fdf0aea26933608fbc49 : 2
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed 2b0393e0-34904b8e5dfa6abb8e3cbe54 : 2
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed e7deed74-0ff2978a255660d693e09c3b : 2
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 2
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 2
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 2
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 2
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 2
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 2
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 2
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 2
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 2
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 2
2018-01-04 12:15:04.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 2
2018-01-04 12:15:04.817 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 2
2018-01-04 12:15:04.817 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 2
2018-01-04 12:15:04.817 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 2
2018-01-04 12:15:04.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 2
2018-01-04 12:15:04.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 2
2018-01-04 12:15:04.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 2
2018-01-04 12:15:04.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 2
2018-01-04 12:15:04.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 2
2018-01-04 12:15:04.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 2
2018-01-04 12:15:04.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 2
2018-01-04 12:15:04.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 2
2018-01-04 12:15:04.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 2
2018-01-04 12:15:04.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 2
2018-01-04 12:15:04.818 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 2
2018-01-04 12:15:04.818 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 2
2018-01-04 12:15:04.818 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 2
2018-01-04 12:15:04.818 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 2
2018-01-04 12:15:04.818 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 2
2018-01-04 12:15:07.815 [IPControllerApp] heartbeat::missed 53182aa0-6d11fdf0aea26933608fbc49 : 3
2018-01-04 12:15:07.815 [IPControllerApp] heartbeat::missed 2b0393e0-34904b8e5dfa6abb8e3cbe54 : 3
2018-01-04 12:15:07.815 [IPControllerApp] heartbeat::missed e7deed74-0ff2978a255660d693e09c3b : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 3
2018-01-04 12:15:07.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 3
2018-01-04 12:15:07.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 3
2018-01-04 12:15:07.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 3
2018-01-04 12:15:07.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 3
2018-01-04 12:15:07.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 3
2018-01-04 12:15:07.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 3
2018-01-04 12:15:07.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 3
2018-01-04 12:15:07.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 3
2018-01-04 12:15:07.817 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 3
2018-01-04 12:15:07.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 3
2018-01-04 12:15:07.817 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 3
2018-01-04 12:15:07.817 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 3
2018-01-04 12:15:07.817 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 3
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed 53182aa0-6d11fdf0aea26933608fbc49 : 4
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed 2b0393e0-34904b8e5dfa6abb8e3cbe54 : 4
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed e7deed74-0ff2978a255660d693e09c3b : 4
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 4
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 4
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 4
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 4
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 4
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 4
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 4
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 4
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 4
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 4
2018-01-04 12:15:10.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 4
2018-01-04 12:15:10.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 4
2018-01-04 12:15:10.818 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 4
2018-01-04 12:15:10.818 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 4
2018-01-04 12:15:10.818 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 4
2018-01-04 12:15:13.815 [IPControllerApp] heartbeat::missed 53182aa0-6d11fdf0aea26933608fbc49 : 5
2018-01-04 12:15:13.815 [IPControllerApp] heartbeat::missed 2b0393e0-34904b8e5dfa6abb8e3cbe54 : 5
2018-01-04 12:15:13.815 [IPControllerApp] heartbeat::missed e7deed74-0ff2978a255660d693e09c3b : 5
2018-01-04 12:15:13.815 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 5
2018-01-04 12:15:13.815 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 5
2018-01-04 12:15:13.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 5
2018-01-04 12:15:13.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 5
2018-01-04 12:15:13.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 5
2018-01-04 12:15:13.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 5
2018-01-04 12:15:13.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 5
2018-01-04 12:15:13.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 5
2018-01-04 12:15:13.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 5
2018-01-04 12:15:13.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 5
2018-01-04 12:15:13.817 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 5
2018-01-04 12:15:13.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 5
2018-01-04 12:15:13.817 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 5
2018-01-04 12:15:13.817 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 5
2018-01-04 12:15:13.817 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 5
2018-01-04 12:15:16.816 [IPControllerApp] heartbeat::missed 53182aa0-6d11fdf0aea26933608fbc49 : 6
2018-01-04 12:15:16.816 [IPControllerApp] heartbeat::missed 2b0393e0-34904b8e5dfa6abb8e3cbe54 : 6
2018-01-04 12:15:16.816 [IPControllerApp] heartbeat::missed e7deed74-0ff2978a255660d693e09c3b : 6
2018-01-04 12:15:16.816 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 6
2018-01-04 12:15:16.816 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 6
2018-01-04 12:15:16.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 6
2018-01-04 12:15:16.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 6
2018-01-04 12:15:16.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 6
2018-01-04 12:15:16.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 6
2018-01-04 12:15:16.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 6
2018-01-04 12:15:16.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 6
2018-01-04 12:15:16.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 6
2018-01-04 12:15:16.818 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 6
2018-01-04 12:15:16.818 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 6
2018-01-04 12:15:16.818 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 6
2018-01-04 12:15:16.818 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 6
2018-01-04 12:15:16.818 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 6
2018-01-04 12:15:16.818 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 6
2018-01-04 12:15:16.818 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 6
2018-01-04 12:15:19.816 [IPControllerApp] heartbeat::missed 53182aa0-6d11fdf0aea26933608fbc49 : 7
2018-01-04 12:15:19.816 [IPControllerApp] heartbeat::missed 2b0393e0-34904b8e5dfa6abb8e3cbe54 : 7
2018-01-04 12:15:19.816 [IPControllerApp] heartbeat::missed e7deed74-0ff2978a255660d693e09c3b : 7
2018-01-04 12:15:19.816 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 7
2018-01-04 12:15:19.816 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 7
2018-01-04 12:15:19.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 7
2018-01-04 12:15:19.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 7
2018-01-04 12:15:19.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 7
2018-01-04 12:15:19.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 7
2018-01-04 12:15:19.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 7
2018-01-04 12:15:19.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 7
2018-01-04 12:15:19.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 7
2018-01-04 12:15:19.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 7
2018-01-04 12:15:19.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 7
2018-01-04 12:15:19.818 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 7
2018-01-04 12:15:19.818 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 7
2018-01-04 12:15:19.818 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 7
2018-01-04 12:15:19.818 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 7
2018-01-04 12:15:19.818 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 7
2018-01-04 12:15:22.815 [IPControllerApp] heartbeat::missed 53182aa0-6d11fdf0aea26933608fbc49 : 8
2018-01-04 12:15:22.815 [IPControllerApp] heartbeat::missed 2b0393e0-34904b8e5dfa6abb8e3cbe54 : 8
2018-01-04 12:15:22.815 [IPControllerApp] heartbeat::missed e7deed74-0ff2978a255660d693e09c3b : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 8
2018-01-04 12:15:22.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 8
2018-01-04 12:15:22.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 8
2018-01-04 12:15:22.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 8
2018-01-04 12:15:22.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 8
2018-01-04 12:15:22.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 8
2018-01-04 12:15:22.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 8
2018-01-04 12:15:22.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 8
2018-01-04 12:15:22.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 8
2018-01-04 12:15:22.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 8
2018-01-04 12:15:22.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 8
2018-01-04 12:15:22.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 8
2018-01-04 12:15:22.817 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 8
2018-01-04 12:15:22.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 8
2018-01-04 12:15:22.817 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 8
2018-01-04 12:15:22.818 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 8
2018-01-04 12:15:22.818 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 8
2018-01-04 12:15:25.815 [IPControllerApp] heartbeat::missed 53182aa0-6d11fdf0aea26933608fbc49 : 9
2018-01-04 12:15:25.815 [IPControllerApp] heartbeat::missed 2b0393e0-34904b8e5dfa6abb8e3cbe54 : 9
2018-01-04 12:15:25.815 [IPControllerApp] heartbeat::missed e7deed74-0ff2978a255660d693e09c3b : 9
2018-01-04 12:15:25.815 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 9
2018-01-04 12:15:25.816 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 9
2018-01-04 12:15:25.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 9
2018-01-04 12:15:25.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 9
2018-01-04 12:15:25.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 9
2018-01-04 12:15:25.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 9
2018-01-04 12:15:25.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 9
2018-01-04 12:15:25.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 9
2018-01-04 12:15:25.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 9
2018-01-04 12:15:25.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 9
2018-01-04 12:15:25.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 9
2018-01-04 12:15:25.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 9
2018-01-04 12:15:25.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 9
2018-01-04 12:15:25.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 9
2018-01-04 12:15:25.817 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 9
2018-01-04 12:15:25.818 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 9
2018-01-04 12:15:28.816 [IPControllerApp] heartbeat::missed 53182aa0-6d11fdf0aea26933608fbc49 : 10
2018-01-04 12:15:28.816 [IPControllerApp] heartbeat::missed 2b0393e0-34904b8e5dfa6abb8e3cbe54 : 10
2018-01-04 12:15:28.816 [IPControllerApp] heartbeat::missed e7deed74-0ff2978a255660d693e09c3b : 10
2018-01-04 12:15:28.816 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 10
2018-01-04 12:15:28.816 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 10
2018-01-04 12:15:28.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 10
2018-01-04 12:15:28.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 10
2018-01-04 12:15:28.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 10
2018-01-04 12:15:28.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 10
2018-01-04 12:15:28.818 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 10
2018-01-04 12:15:28.818 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 10
2018-01-04 12:15:28.818 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 10
2018-01-04 12:15:28.818 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 10
2018-01-04 12:15:28.818 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 10
2018-01-04 12:15:28.818 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 10
2018-01-04 12:15:28.818 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 10
2018-01-04 12:15:31.815 [IPControllerApp] heartbeat::missed 53182aa0-6d11fdf0aea26933608fbc49 : 11
2018-01-04 12:15:31.815 [IPControllerApp] heartbeat::missed 2b0393e0-34904b8e5dfa6abb8e3cbe54 : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed e7deed74-0ff2978a255660d693e09c3b : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:15:31.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:15:31.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:15:31.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:15:31.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:15:31.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:15:31.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:15:31.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:15:31.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:15:31.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:15:31.817 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:15:31.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:15:31.817 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 11
2018-01-04 12:15:31.817 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:15:31.817 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:15:31.817 [IPControllerApp] registration::unregister_engine(21)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '53182aa0-6d11fdf0aea26933608fbc49'
2018-01-04 12:15:34.816 [IPControllerApp] heartbeat::missed 2b0393e0-34904b8e5dfa6abb8e3cbe54 : 11
2018-01-04 12:15:34.816 [IPControllerApp] heartbeat::missed e7deed74-0ff2978a255660d693e09c3b : 11
2018-01-04 12:15:34.816 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 11
2018-01-04 12:15:34.816 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 11
2018-01-04 12:15:34.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 11
2018-01-04 12:15:34.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 11
2018-01-04 12:15:34.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 11
2018-01-04 12:15:34.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 11
2018-01-04 12:15:34.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 11
2018-01-04 12:15:34.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 11
2018-01-04 12:15:34.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:15:34.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:15:34.817 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 11
2018-01-04 12:15:34.817 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:15:34.817 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:15:34.817 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:15:34.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:15:34.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:15:34.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:15:34.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:15:34.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:15:34.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:15:34.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:15:34.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:15:34.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:15:34.818 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:15:34.818 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:15:34.818 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:15:34.818 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 11
2018-01-04 12:15:34.818 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:15:34.818 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:15:34.818 [IPControllerApp] registration::unregister_engine(14)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '2b0393e0-34904b8e5dfa6abb8e3cbe54'
2018-01-04 12:15:37.816 [IPControllerApp] heartbeat::missed e7deed74-0ff2978a255660d693e09c3b : 11
2018-01-04 12:15:37.816 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 11
2018-01-04 12:15:37.816 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 11
2018-01-04 12:15:37.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 11
2018-01-04 12:15:37.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 11
2018-01-04 12:15:37.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 11
2018-01-04 12:15:37.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 11
2018-01-04 12:15:37.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 11
2018-01-04 12:15:37.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:15:37.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:15:37.818 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:15:37.818 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:15:37.818 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:15:37.818 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:15:37.818 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 11
2018-01-04 12:15:37.818 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:15:37.818 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:15:37.818 [IPControllerApp] registration::unregister_engine(0)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'e7deed74-0ff2978a255660d693e09c3b'
2018-01-04 12:15:40.816 [IPControllerApp] heartbeat::missed bb7fdac7-c67298f88ce92956ade790a4 : 11
2018-01-04 12:15:40.816 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 11
2018-01-04 12:15:40.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 11
2018-01-04 12:15:40.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 11
2018-01-04 12:15:40.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 11
2018-01-04 12:15:40.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 11
2018-01-04 12:15:40.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 11
2018-01-04 12:15:40.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:15:40.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:15:40.818 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 11
2018-01-04 12:15:40.818 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:15:40.818 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:15:40.818 [IPControllerApp] registration::unregister_engine(27)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'bb7fdac7-c67298f88ce92956ade790a4'
2018-01-04 12:15:43.815 [IPControllerApp] heartbeat::missed 4b6f15c8-df04e637f90e03cf3b611777 : 11
2018-01-04 12:15:43.815 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 11
2018-01-04 12:15:43.815 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:15:43.816 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:15:43.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:15:43.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:15:43.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:15:43.817 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:15:43.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:15:43.817 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 11
2018-01-04 12:15:43.817 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:15:43.817 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:15:43.817 [IPControllerApp] registration::unregister_engine(20)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '4b6f15c8-df04e637f90e03cf3b611777'
2018-01-04 12:15:46.816 [IPControllerApp] heartbeat::missed bb3d4789-8c83bc13aac5204b069225d8 : 11
2018-01-04 12:15:46.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 11
2018-01-04 12:15:46.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 11
2018-01-04 12:15:46.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 11
2018-01-04 12:15:46.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 11
2018-01-04 12:15:46.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 11
2018-01-04 12:15:46.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:15:46.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:15:46.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 11
2018-01-04 12:15:46.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 11
2018-01-04 12:15:46.817 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:15:46.818 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:15:46.818 [IPControllerApp] registration::unregister_engine(26)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'bb3d4789-8c83bc13aac5204b069225d8'
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed 5161769a-5fe47396e022aae8b7951fcf : 11
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 11
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 11
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 11
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 11
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 11
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:15:49.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:15:49.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:15:49.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:15:49.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:15:49.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:15:49.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:15:49.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:15:49.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:15:49.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:15:49.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:15:49.817 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 11
2018-01-04 12:15:49.817 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:15:49.817 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:15:49.817 [IPControllerApp] registration::unregister_engine(5)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '5161769a-5fe47396e022aae8b7951fcf'
2018-01-04 12:15:52.816 [IPControllerApp] heartbeat::missed 62c9a487-e05d3f87fb5b42bf1ddc5c4d : 11
2018-01-04 12:15:52.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 11
2018-01-04 12:15:52.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 11
2018-01-04 12:15:52.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 11
2018-01-04 12:15:52.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:15:52.816 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:15:52.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 11
2018-01-04 12:15:52.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:15:52.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:15:52.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:15:52.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:15:52.817 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:15:52.817 [IPControllerApp] registration::unregister_engine(17)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '62c9a487-e05d3f87fb5b42bf1ddc5c4d'
2018-01-04 12:15:55.816 [IPControllerApp] heartbeat::missed 8fdcefdf-6e29a1d4c0c75dda97d55002 : 11
2018-01-04 12:15:55.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 11
2018-01-04 12:15:55.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 11
2018-01-04 12:15:55.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:15:55.816 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:15:55.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 11
2018-01-04 12:15:55.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:15:55.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:15:55.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:15:55.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:15:55.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:15:55.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:15:55.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:15:55.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:15:55.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:15:55.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:15:55.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:15:55.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:15:55.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:15:55.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:15:55.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:15:55.817 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 11
2018-01-04 12:15:55.817 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:15:55.817 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:15:55.817 [IPControllerApp] registration::unregister_engine(18)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '8fdcefdf-6e29a1d4c0c75dda97d55002'
2018-01-04 12:15:58.816 [IPControllerApp] heartbeat::missed e90c3022-c43dd23147ac258928771a60 : 11
2018-01-04 12:15:58.816 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 11
2018-01-04 12:15:58.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:15:58.816 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:15:58.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 11
2018-01-04 12:15:58.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:15:58.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:15:58.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:15:58.817 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:15:58.817 [IPControllerApp] registration::unregister_engine(3)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'e90c3022-c43dd23147ac258928771a60'
2018-01-04 12:16:01.815 [IPControllerApp] heartbeat::missed 4eee1008-988784ff1cdee9a7b4af2303 : 11
2018-01-04 12:16:01.815 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:01.815 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:16:01.815 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 11
2018-01-04 12:16:01.815 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:16:01.815 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:16:01.816 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 11
2018-01-04 12:16:01.816 [IPControllerApp] registration::unregister_engine(10)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '4eee1008-988784ff1cdee9a7b4af2303'
2018-01-04 12:16:04.816 [IPControllerApp] heartbeat::missed f707aaef-8c155feac7006ea9214c4a70 : 11
2018-01-04 12:16:04.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 11
2018-01-04 12:16:04.816 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:16:04.816 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:16:04.816 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:16:04.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:16:04.816 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:16:04.816 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:16:04.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:16:04.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:04.816 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:04.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:04.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:16:04.817 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:04.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:04.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:04.817 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:16:04.817 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:04.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:04.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:04.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:16:04.817 [IPControllerApp] registration::unregister_engine(13)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'f707aaef-8c155feac7006ea9214c4a70'
2018-01-04 12:16:07.816 [IPControllerApp] heartbeat::missed e44240a2-b3fc155e9ccb22aa2c660bf5 : 11
2018-01-04 12:16:07.816 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:16:07.816 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:16:07.816 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:16:07.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:16:07.816 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:16:07.816 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:16:07.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:16:07.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:07.817 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:07.817 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:07.817 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:16:07.817 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:07.817 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:07.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:07.817 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:16:07.817 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:07.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:07.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:07.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:16:07.817 [IPControllerApp] registration::unregister_engine(1)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'e44240a2-b3fc155e9ccb22aa2c660bf5'
2018-01-04 12:16:10.815 [IPControllerApp] heartbeat::missed c0b47cdc-8d871db57661c94ed26af39b : 11
2018-01-04 12:16:10.815 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:16:10.815 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:16:10.815 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:16:10.815 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:16:10.815 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:16:10.815 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:16:10.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:10.816 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:10.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:10.816 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:16:10.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:10.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:10.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:10.816 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:16:10.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:10.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:10.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:10.816 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:16:10.816 [IPControllerApp] registration::unregister_engine(19)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'c0b47cdc-8d871db57661c94ed26af39b'
2018-01-04 12:16:13.815 [IPControllerApp] heartbeat::missed 82ab8cb6-d75bca944f9388e20d4151cf : 11
2018-01-04 12:16:13.815 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:13.816 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:16:13.817 [IPControllerApp] registration::unregister_engine(12)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '82ab8cb6-d75bca944f9388e20d4151cf'
2018-01-04 12:16:16.815 [IPControllerApp] heartbeat::missed 6e19df5b-a601af6b09e1f2d5692492ee : 11
2018-01-04 12:16:16.815 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:16:16.815 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:16:16.815 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:16:16.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:16:16.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:16.816 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:16.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:16.816 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:16:16.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:16.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:16.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:16.816 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:16:16.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:16.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:16.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:16.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:16:16.817 [IPControllerApp] registration::unregister_engine(9)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '6e19df5b-a601af6b09e1f2d5692492ee'
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed 92b21546-c1b7015ac82f06e7290ee9b0 : 11
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:19.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:19.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:19.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:16:19.817 [IPControllerApp] registration::unregister_engine(30)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '92b21546-c1b7015ac82f06e7290ee9b0'
2018-01-04 12:16:22.815 [IPControllerApp] heartbeat::missed 160cd1ca-1add0d947e80586fba6a76b5 : 11
2018-01-04 12:16:22.815 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:16:22.815 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:16:22.815 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:16:22.815 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:16:22.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:22.816 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:22.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:22.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:22.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:22.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:22.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:22.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:22.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:22.816 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:16:22.816 [IPControllerApp] registration::unregister_engine(24)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '160cd1ca-1add0d947e80586fba6a76b5'
2018-01-04 12:16:25.816 [IPControllerApp] heartbeat::missed 2ad7eaaf-6241cd4d51d00cda77c1f77f : 11
2018-01-04 12:16:25.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:16:25.816 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:16:25.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:16:25.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:25.816 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:25.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:25.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:25.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:25.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:25.817 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:25.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:25.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:25.817 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:16:25.817 [IPControllerApp] registration::unregister_engine(7)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '2ad7eaaf-6241cd4d51d00cda77c1f77f'
2018-01-04 12:16:28.816 [IPControllerApp] heartbeat::missed 14f1caf3-5b9dd361ed1986f1b1e175d1 : 11
2018-01-04 12:16:28.816 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:16:28.816 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:16:28.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:16:28.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:28.816 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:28.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:28.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:28.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:28.817 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:28.817 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:28.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:28.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:28.817 [IPControllerApp] registration::unregister_engine(2)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '14f1caf3-5b9dd361ed1986f1b1e175d1'
2018-01-04 12:16:31.815 [IPControllerApp] heartbeat::missed 80330e9a-882d59a5521a4d763578276a : 11
2018-01-04 12:16:31.816 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:16:31.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:16:31.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:31.816 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:31.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:31.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:31.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:31.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:31.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:31.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:31.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:31.816 [IPControllerApp] registration::unregister_engine(22)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '80330e9a-882d59a5521a4d763578276a'
2018-01-04 12:16:34.815 [IPControllerApp] heartbeat::missed 0bf769b7-a9df55e1c8d0aa7339f88170 : 11
2018-01-04 12:16:34.815 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:16:34.815 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:34.816 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:34.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:34.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:34.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:34.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:34.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:34.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:34.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:34.816 [IPControllerApp] registration::unregister_engine(28)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '0bf769b7-a9df55e1c8d0aa7339f88170'
2018-01-04 12:16:37.816 [IPControllerApp] heartbeat::missed 66399edb-ca3ff6346973aafcd1685152 : 11
2018-01-04 12:16:37.816 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:37.816 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:37.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:37.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:37.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:37.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:37.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:37.817 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:37.817 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:37.817 [IPControllerApp] registration::unregister_engine(8)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '66399edb-ca3ff6346973aafcd1685152'
2018-01-04 12:16:40.815 [IPControllerApp] heartbeat::missed d69c000d-3a08a244b72cfca20288512c : 11
2018-01-04 12:16:40.815 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:40.815 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:40.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:40.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:40.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:40.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:40.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:40.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:40.816 [IPControllerApp] registration::unregister_engine(31)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'd69c000d-3a08a244b72cfca20288512c'
2018-01-04 12:16:43.816 [IPControllerApp] heartbeat::missed 2d598dd2-e5a30c8612b864b70098f90f : 11
2018-01-04 12:16:43.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:43.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:43.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:43.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:43.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:43.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:43.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:43.816 [IPControllerApp] registration::unregister_engine(11)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '2d598dd2-e5a30c8612b864b70098f90f'
2018-01-04 12:16:46.816 [IPControllerApp] heartbeat::missed 1f754c2b-0124e075ff5aa5507cebb6a3 : 11
2018-01-04 12:16:46.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:46.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:46.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:46.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:46.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:46.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:46.816 [IPControllerApp] registration::unregister_engine(15)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '1f754c2b-0124e075ff5aa5507cebb6a3'
2018-01-04 12:16:49.816 [IPControllerApp] heartbeat::missed e32f72f2-3b6d2927567029c9fce8415d : 11
2018-01-04 12:16:49.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:49.816 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:49.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:49.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:49.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:49.816 [IPControllerApp] registration::unregister_engine(16)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'e32f72f2-3b6d2927567029c9fce8415d'
2018-01-04 12:16:52.815 [IPControllerApp] heartbeat::missed acc072c8-755c592fd57bb2145362f0ca : 11
2018-01-04 12:16:52.816 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:52.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:52.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:52.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:52.816 [IPControllerApp] registration::unregister_engine(6)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'acc072c8-755c592fd57bb2145362f0ca'
2018-01-04 12:16:55.815 [IPControllerApp] heartbeat::missed 26a952f5-49a0e8cf68efa4a063940b5a : 11
2018-01-04 12:16:55.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:55.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:55.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:55.816 [IPControllerApp] registration::unregister_engine(25)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '26a952f5-49a0e8cf68efa4a063940b5a'
2018-01-04 12:16:58.816 [IPControllerApp] heartbeat::missed fd216f94-e9cc397f8c6065c2c90339fe : 11
2018-01-04 12:16:58.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:16:58.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:16:58.816 [IPControllerApp] registration::unregister_engine(4)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: 'fd216f94-e9cc397f8c6065c2c90339fe'
2018-01-04 12:17:01.816 [IPControllerApp] heartbeat::missed 3302d274-3bf27449bb70e33f43e72c57 : 11
2018-01-04 12:17:01.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:17:01.816 [IPControllerApp] registration::unregister_engine(23)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '3302d274-3bf27449bb70e33f43e72c57'
2018-01-04 12:17:04.816 [IPControllerApp] heartbeat::missed 28d00e5f-26ff055a06b08244322060f4 : 11
2018-01-04 12:17:04.816 [IPControllerApp] registration::unregister_engine(29)
ERROR:tornado.application:Exception in callback <bound method HeartMonitor.beat of <ipyparallel.controller.heartmonitor.HeartMonitor object at 0x2aaababfcd90>>
Traceback (most recent call last):
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/tornado/ioloop.py", line 1026, in _run
    return self.callback()
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 133, in beat
    self.handle_heart_failure(failure)
  File "/global/homes/a/aaronmil/.local/cori/2.7-anaconda/lib/python2.7/site-packages/ipyparallel/controller/heartmonitor.py", line 174, in handle_heart_failure
    self.hearts.remove(heart)
KeyError: '28d00e5f-26ff055a06b08244322060f4'
Traceback (most recent call last):
  File "apply_example.py", line 61, in <module>
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
srun: error: nid00998: task 0: Exited with exit code 1
srun: Terminating job step 9320007.2
