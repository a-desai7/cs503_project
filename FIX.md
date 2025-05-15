# AttributeError: 'int' object has no attribute 'type'
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