# MVTecDestilado

Detección de anomalías industriales mediante destilado de modelos y autoencoders.

Proyecto II del curso de Inteligencia Artificial, Escuela de Ingeniería en Computación, Instituto Tecnológico de Costa Rica.

El proyecto compara tres modelos sobre 10 categorías de MVTec AD. El modelo A es una CNN entrenada desde cero con las primeras etapas de ResNet-18. El modelo B tiene la misma arquitectura pero se entrena por destilado a partir de un ResNet-18 completo. El modelo C es un autoencoder basado en U-Net. Todos se entrenan solo con imágenes sin defectos, y las anomalías se detectan midiendo la distancia de Mahalanobis de cada embedding a la distribución de lo normal de su categoría. Al final se analiza el mejor modelo con PCA, t-SNE y DBSCAN.

El enunciado está en [docs/enunciado_proyecto_02.pdf](docs/enunciado_proyecto_02.pdf).

## Estructura

```
conf/        configuración de Hydra (data, model, trainer, logger)
notebooks/   main.ipynb, con todo el proyecto organizado por secciones
report/      informe y figuras
docs/        enunciado del proyecto
data/        MVTec AD (no se versiona)
outputs/     checkpoints y salidas (no se versiona)
```

## Instalación

Se usa Python 3.11 y una GPU NVIDIA con CUDA.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
wandb login
```

## Datos

MVTec AD se descarga desde https://www.mvtec.com/company/research/datasets/mvtec-ad (licencia CC BY-NC-SA 4.0). Se usan 10 categorías: carpet, leather, tile, wood, bottle, cable, capsule, hazelnut, metal_nut y pill. Cada archivo `<categoria>.tar.xz` se extrae en `data/mvtec_ad/`, de forma que quede así:

```
data/mvtec_ad/<categoria>/train/good/
data/mvtec_ad/<categoria>/test/<tipo_defecto>/
data/mvtec_ad/<categoria>/ground_truth/<tipo_defecto>/
```

## Ejecución

Abrir `notebooks/main.ipynb` con el kernel del entorno y ejecutarlo en orden. La primera ejecución construye un cache de las imágenes redimensionadas en `outputs/cache/`, lo que tarda un par de minutos.

```powershell
python -m ipykernel install --user --name mvtec --display-name "Python (mvtec)"
```

## Configuración

El notebook compone la configuración con la API de Hydra:

```python
from hydra import initialize, compose

with initialize(version_base=None, config_path="../conf"):
    cfg = compose(config_name="config", overrides=["model=cnn_scratch", "data.batch_size=64"])
```
