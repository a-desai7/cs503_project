## AttributeError: 'int' object has no attribute 'type'
If the error trace looks like this:
```Traceback (most recent call last):
  File "/home/beleznai/git/VI_Project/custom-tools/train.py", line 262, in <module>
    main()
  File "/home/beleznai/git/VI_Project/custom-tools/train.py", line 250, in main
    train_segmentor(
  File "/home/beleznai/git/VI_Project/mmseg/apis/train.py", line 194, in train_segmentor
    runner.run(data_loaders, cfg.workflow)
  File "/home/beleznai/miniconda3/envs/VIP/lib/python3.10/site-packages/mmcv/runner/iter_based_runner.py", line 144, in run
    iter_runner(iter_loaders[i], **kwargs)
  File "/home/beleznai/miniconda3/envs/VIP/lib/python3.10/site-packages/mmcv/runner/iter_based_runner.py", line 64, in train
    outputs = self.model.train_step(data_batch, self.optimizer, **kwargs)
  File "/home/beleznai/miniconda3/envs/VIP/lib/python3.10/site-packages/mmcv/parallel/data_parallel.py", line 76, in train_step
    inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
  File "/home/beleznai/miniconda3/envs/VIP/lib/python3.10/site-packages/mmcv/parallel/data_parallel.py", line 55, in scatter
    return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
  File "/home/beleznai/miniconda3/envs/VIP/lib/python3.10/site-packages/mmcv/parallel/scatter_gather.py", line 60, in scatter_kwargs
    inputs = scatter(inputs, target_gpus, dim) if inputs else []
  File "/home/beleznai/miniconda3/envs/VIP/lib/python3.10/site-packages/mmcv/parallel/scatter_gather.py", line 50, in scatter
    return scatter_map(inputs)
  File "/home/beleznai/miniconda3/envs/VIP/lib/python3.10/site-packages/mmcv/parallel/scatter_gather.py", line 35, in scatter_map
    return list(zip(*map(scatter_map, obj)))
  File "/home/beleznai/miniconda3/envs/VIP/lib/python3.10/site-packages/mmcv/parallel/scatter_gather.py", line 40, in scatter_map
    out = list(map(type(obj), zip(*map(scatter_map, obj.items()))))
  File "/home/beleznai/miniconda3/envs/VIP/lib/python3.10/site-packages/mmcv/parallel/scatter_gather.py", line 35, in scatter_map
    return list(zip(*map(scatter_map, obj)))
  File "/home/beleznai/miniconda3/envs/VIP/lib/python3.10/site-packages/mmcv/parallel/scatter_gather.py", line 33, in scatter_map
    return Scatter.forward(target_gpus, obj.data)
  File "/home/beleznai/miniconda3/envs/VIP/lib/python3.10/site-packages/mmcv/parallel/_functions.py", line 75, in forward
    streams = [_get_stream(device) for device in target_gpus]
  File "/home/beleznai/miniconda3/envs/VIP/lib/python3.10/site-packages/mmcv/parallel/_functions.py", line 75, in <listcomp>
    streams = [_get_stream(device) for device in target_gpus]
  File "/home/beleznai/miniconda3/envs/VIP/lib/python3.10/site-packages/torch/nn/parallel/_functions.py", line 126, in _get_stream
    if device.type == "cpu":
AttributeError: 'int' object has no attribute 'type'
```

Go into the second to last file of the trace, `<conda-env>/lib/python3.10/site-packages/mmcv/parallel/_functions.py` (ctrl + click in the terminal to open the file in editor) and change line 75 from:
```python
	streams = [_get_stream(device) for device in target_gpus]
```
to:
```python
	streams = [_get_stream(torch.device("cuda", device)) for device in target_gpus]
```

## AttributeError: 'MMDistributedDataParallel' object has no attribute '_use_replicated_tensor_module'
If during a distributed training, the trace looks something like this:
```
[rank0]: Traceback (most recent call last):
[rank0]:   File "/home/beleznai/git/VI_Project/custom-tools/train.py", line 261, in <module>
[rank0]:     main()
[rank0]:   File "/home/beleznai/git/VI_Project/custom-tools/train.py", line 249, in main
[rank0]:     train_segmentor(
[rank0]:   File "/home/beleznai/git/VI_Project/mmseg/apis/train.py", line 194, in train_segmentor
[rank0]:     runner.run(data_loaders, cfg.workflow)
[rank0]:   File "/home/beleznai/miniconda3/envs/VIP2/lib/python3.10/site-packages/mmcv/runner/iter_based_runner.py", line 144, in run
[rank0]:     iter_runner(iter_loaders[i], **kwargs)
[rank0]:   File "/home/beleznai/miniconda3/envs/VIP2/lib/python3.10/site-packages/mmcv/runner/iter_based_runner.py", line 70, in train
[rank0]:     self.call_hook('after_train_iter')
[rank0]:   File "/home/beleznai/miniconda3/envs/VIP2/lib/python3.10/site-packages/mmcv/runner/base_runner.py", line 317, in call_hook
[rank0]:     getattr(hook, fn_name)(self)
[rank0]:   File "/home/beleznai/miniconda3/envs/VIP2/lib/python3.10/site-packages/mmcv/runner/hooks/evaluation.py", line 266, in after_train_iter
[rank0]:     self._do_evaluate(runner)
[rank0]:   File "/home/beleznai/git/VI_Project/mmseg/core/evaluation/eval_hooks.py", line 117, in _do_evaluate
[rank0]:     results = multi_gpu_test(
[rank0]:   File "/home/beleznai/git/VI_Project/mmseg/apis/test.py", line 208, in multi_gpu_test
[rank0]:     result = model(return_loss=False, rescale=True, **data)
[rank0]:   File "/home/beleznai/miniconda3/envs/VIP2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1736, in _wrapped_call_impl
[rank0]:     return self._call_impl(*args, **kwargs)
[rank0]:   File "/home/beleznai/miniconda3/envs/VIP2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1747, in _call_impl
[rank0]:     return forward_call(*args, **kwargs)
[rank0]:   File "/home/beleznai/miniconda3/envs/VIP2/lib/python3.10/site-packages/torch/nn/parallel/distributed.py", line 1643, in forward
[rank0]:     else self._run_ddp_forward(*inputs, **kwargs)
[rank0]:   File "/home/beleznai/miniconda3/envs/VIP2/lib/python3.10/site-packages/mmcv/parallel/distributed.py", line 160, in _run_ddp_forward
[rank0]:     self._use_replicated_tensor_module else self.module
[rank0]:   File "/home/beleznai/miniconda3/envs/VIP2/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1931, in __getattr__
[rank0]:     raise AttributeError(
[rank0]: AttributeError: 'MMDistributedDataParallel' object has no attribute '_use_replicated_tensor_module'
```

Go into `<conda-env>/lib/python3.10/site-packages/mmcv/parallel/distributed.py` and change line 160 from:
```python
module_to_run = self._replicated_tensor_module if \
    self._use_replicated_tensor_module else self.module
```

to:
```python
module_to_run = self.module
```