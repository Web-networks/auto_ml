# Сервис по поиску и оптимизации нейронных сетей auto_ml

Режим поиска:
```shell script
bash run.sh SearchModel --model-path ./model/ 
  --data-type image --task-type classification 
  --epochs 10 --data-path ./data/  
```

Режим оптимизации:
```shell script
bash run.sh OptimizeModel --model-path ./model/ 
  --data-type image --task-type classification 
  --epochs 10 --data-path ./data/ 
```
Здесь 

--model_path - путь до директории куда сохранить модель, где лежит base_model.py

--data_type - тип данных (image, text или csv)

--task_type - тип решаемой задачи (classification или regression)

--epochs - количетсво эпох

--data_path - путь до данных