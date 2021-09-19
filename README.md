# Lightning SSD
Pytorch Lightning for SSD models 

```
__author__ = Le Trung Phong
__email__ = letrungphong95@gmail.com
__version__ = 1.0.0
__status__ = Developing
```

## 1. Building
### 1.1 Environment  

```
$ python3 -m venv venv 
$ source venv/bin/activate
(venv)$ pip install -r requirements.txt
(venv)$ deactivate 
$ ls 
```

## 2. Excution 
### 2.1. Dataset 

VOC Format 
```
train_data_dir
     ├── Annotations 
     │  ├── img_01.xml
     │  ├── img_02.xml
     │  ├── ...
     │  └── img_n.xml
     ├── ImageSets
     │  └── Main    
     │     ├── trainval.txt
     │     └── test.txt
     └──  JPEGImages 
        ├── img_01.jpg     
        ├── img_02.jpg     
        ├── ...    
        └── img_n.jpg
``` 

### 2.2. Training SSD model 

```
(venv)$ python train_ssd.py --config_file config/base_model.yaml
```

## 3. Task List 

- [x] Configuration 
- [ ] Dataset Loader 
- [ ] Dataset Analysis 
- [ ] Build backbone block 
- [ ] Build util module 
- [ ] Buil SSD model 
- [ ] Build trainer 
- [ ] Build evaluator 
- [ ] Build tester 
- [ ] Build inference script





