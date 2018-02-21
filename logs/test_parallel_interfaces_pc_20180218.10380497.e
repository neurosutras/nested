ModuleCmd_Switch.c(179):ERROR:152: Module 'PrgEnv-intel' is currently not loaded
+ cd /global/homes/a/aaronmil/python_modules/nested
++ date +%Y%m%d_%H%M%S
+ export DATE=20180219_114846
+ DATE=20180219_114846
+ ulimit -n 20000
+ ulimit -u 20000
+ srun -N 32 -n 1024 -c 2 --cpu_bind=cores python test_parallel_interfaces.py --framework=pc
NEURON -- VERSION 7.5 master (d9be927) 2018-02-11
Duke, Yale, and the BlueBrain Project -- Copyright 1984-2016
See http://neuron.yale.edu/neuron/credits

Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 393, in pc_apply_wrapper
    result = func(*args, **kwargs)
  File "test_parallel_interfaces.py", line 11, in test
    print context_monkeys()
RuntimeError: maximum recursion depth exceeded while calling a Python object
0 NEURON: PyObject method call failed:
0  near line 0
0  objref hoc_obj_[2]
                   ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].take("0")
and others
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].take("0")
  0 ParallelContext[0].working()
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].take("0")
    0 ParallelContext[0].working()
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].take("0")
      0 ParallelContext[0].working()
Traceback (most recent call last):
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 396, in pc_apply_wrapper
    interface.wait_for_all_workers(key)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 237, in wait_for_all_workers
    self.pc.take(key)
RuntimeError: hoc error
0 NEURON: PyObject method call failed:
0  near line 0
0  ^
        0 ParallelContext[0].working()
Traceback (most recent call last):
  File "test_parallel_interfaces.py", line 72, in <module>
    main(standalone_mode=False)
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/click/core.py", line 722, in __call__
    return self.main(*args, **kwargs)
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/click/core.py", line 697, in main
    rv = self.invoke(ctx)
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/click/core.py", line 895, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/usr/common/software/python/2.7-anaconda-4.4/lib/python2.7/site-packages/click/core.py", line 535, in invoke
    return callback(*args, **kwargs)
  File "test_parallel_interfaces.py", line 55, in main
    print context_monkeys.interface_monkeys.apply(test, 1, 2, third=3)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 270, in apply_sync
    results = self.collect_results(keys)
  File "/global/homes/a/aaronmil/python_modules/nested/parallel.py", line 306, in collect_results
    while self.pc.working():
RuntimeError: hoc error
srun: got SIGCONT
slurmstepd: error: *** STEP 10380497.0 ON nid00884 CANCELLED AT 2018-02-19T11:50:45 ***
srun: forcing job termination
