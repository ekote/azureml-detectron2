# azureml-detectron2

`python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`


## troubleshooting
if error similar to
```shell script
    ERROR: Command errored out with exit status 1:
     command: 'C:\dev\azureml-detectron2\env\Scripts\python.exe' -c 'import io, os, sys, setuptools, tokenize; sys.argv[0] = '"'"'C:\\Users\\eskot\\AppData\\Local\\Temp\\pip-req-build-sacfj2ru\\setup.py'"'"'; __file__='"'"'C:\\Users
\\eskot\\AppData\\Local\\Temp\\pip-req-build-sacfj2ru\\setup.py'"'"';f = getattr(tokenize, '"'"'open'"'"', open)(__file__) if os.path.exists(__file__) else io.StringIO('"'"'from setuptools import setup; setup()'"'"');code = f.read()
.replace('"'"'\r\n'"'"', '"'"'\n'"'"');f.close();exec(compile(code, __file__, '"'"'exec'"'"'))' egg_info --egg-base 'C:\Users\eskot\AppData\Local\Temp\pip-pip-egg-info-tzqgti7x'
         cwd: C:\Users\eskot\AppData\Local\Temp\pip-req-build-sacfj2ru\
    Complete output (5 lines):
    Traceback (most recent call last):
      File "<string>", line 1, in <module>
      File "C:\Users\eskot\AppData\Local\Temp\pip-req-build-sacfj2ru\setup.py", line 10, in <module>
        import torch
    ModuleNotFoundError: No module named 'torch'
    ----------------------------------------
WARNING: Discarding git+https://github.com/facebookresearch/detectron2.git. Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.
ERROR: Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.

```
install locally torch
`pip3 install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html`



--output

(env) C:\dev\azureml-detectron2\azureml_detectron2>python training/train.py --config-file mask_rcnn_R_50_FPN_3x.yaml --input input.png --output out.png --opts MODEL.WEIGHTS model_final_f10217.pkl MODEL.D
EVICE cpu


(env) C:\dev\azureml-detectron2\azureml_detectron2>python training/train.py --config-file mask_rcnn_R_50_FPN_3x.yaml --input input4.png --output out4.png --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl MODEL.DEVICE cpu


tODO:
- zmiejszyc naklad plikow tutaj do uruchomienia
