# MVTecDestilado

Detección de anomalías industriales mediante destilado de modelos y autoencoders.

Proyecto II del curso de Inteligencia Artificial, Escuela de Ingeniería en Computación, Instituto Tecnológico de Costa Rica.

El proyecto compara tres modelos sobre 10 categorías de MVTec AD. El modelo A es una CNN entrenada desde cero con las primeras etapas de ResNet-18. El modelo B tiene la misma arquitectura pero se entrena por destilado a partir de un ResNet-18 completo. El modelo C es un autoencoder basado en U-Net. Todos se entrenan solo con imágenes sin defectos, y las anomalías se detectan midiendo la distancia de Mahalanobis de cada embedding a la distribución de lo normal de su categoría. Al final se analiza el mejor modelo con PCA, t-SNE y DBSCAN.

El enunciado está en [docs/enunciado_proyecto_02.pdf](docs/enunciado_proyecto_02.pdf).

## Estructura

```
conf/                      configuración de Hydra
  config.yaml              raíz: una opción de cada grupo, semilla, nombre de la corrida y rutas
  data/mvtec.yaml          MVTecDataModule: categorías, tamaño de imagen, batch, split de validación
  model/                   un archivo por modelo (argumentos de su LightningModule)
    cnn_scratch.yaml       modelo A
    cnn_distill.yaml       modelo B (hereda de cnn_scratch y agrega el teacher)
    unet_ae.yaml           modelo C
  trainer/default.yaml     argumentos de lightning.Trainer
  callbacks/default.yaml   EarlyStopping, ModelCheckpoint, LearningRateMonitor
  logger/wandb.yaml        WandbLogger
  experiment/<modelo>/     combinaciones de hiperparámetros con nombre (al menos 3 por modelo)
src/                       código auxiliar que importa el notebook
  data.py                  MVTecDataset y MVTecDataModule
  models/                  un LightningModule por modelo
  training.py              construye Trainer, callbacks y logger desde la configuración
  utils.py                 raíz del repositorio y composición de la configuración
notebooks/                 ejecución del entrenamiento y del análisis
  datos.ipynb           exploración de MVTec AD y verificación del DataModule
  modelos_ab.ipynb      modelo A (CNN desde cero) y modelo B (CNN destilada)
  modelo_c.ipynb        modelo C (autoencoder U-Net)
  anomalias.ipynb       detección de anomalías con embeddings y DBSCAN
report/                    informe y figuras
docs/                      enunciado del proyecto
data/                      MVTec AD (no se versiona)
outputs/                   cache, checkpoints, resultados y W&B (no se versiona)
```

Las clases viven en `src/` como scripts auxiliares, y el entrenamiento se ejecuta desde el notebook, como pide el enunciado.

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

### Dataset

Se usa **MVTec AD** (MVTec Anomaly Detection), el dataset original de 2019:

> Paul Bergmann, Michael Fauser, David Sattlegger y Carsten Steger. *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

No debe confundirse con MVTec AD 2 (2025), que es un dataset distinto con otras categorías. MVTec AD tiene 15 categorías (5 texturas y 10 objetos) con fotografías industriales en alta resolución. El conjunto de entrenamiento solo contiene imágenes sin defectos (`train/good`) y el conjunto de prueba mezcla imágenes normales con imágenes que tienen defectos de distintos tipos, acompañadas de máscaras a nivel de píxel. Se descarga desde https://www.mvtec.com/company/research/datasets/mvtec-ad y se distribuye con licencia CC BY-NC-SA 4.0, por lo que no se incluye en el repositorio.

### Clases seleccionadas

El enunciado pide 10 de las 15 categorías. Se eligieron 4 texturas y 6 objetos:

| Tipo | Categoría | Resolución | train/good | test/good | test con defecto | Tipos de defecto |
|---|---|---|---:|---:|---:|---|
| Textura | carpet | 1024 | 280 | 28 | 89 | color, cut, hole, metal_contamination, thread |
| Textura | leather | 1024 | 245 | 32 | 92 | color, cut, fold, glue, poke |
| Textura | tile | 840 | 230 | 33 | 84 | crack, glue_strip, gray_stroke, oil, rough |
| Textura | wood | 1024 | 247 | 19 | 60 | color, combined, hole, liquid, scratch |
| Objeto | bottle | 900 | 209 | 20 | 63 | broken_large, broken_small, contamination |
| Objeto | cable | 1024 | 224 | 58 | 92 | bent_wire, cable_swap, combined, cut_inner_insulation, cut_outer_insulation, missing_cable, missing_wire, poke_insulation |
| Objeto | capsule | 1000 | 219 | 23 | 109 | crack, faulty_imprint, poke, scratch, squeeze |
| Objeto | hazelnut | 1024 | 391 | 40 | 70 | crack, cut, hole, print |
| Objeto | metal_nut | 700 | 220 | 22 | 93 | bent, color, flip, scratch |
| Objeto | pill | 800 | 267 | 26 | 141 | color, combined, contamination, crack, faulty_imprint, pill_type, scratch |
| | **Total** | | **2532** | **301** | **893** | |

Todas las imágenes son PNG cuadradas en RGB. Las categorías que no se usan son grid (textura) y screw, toothbrush, transistor y zipper (objetos).

La selección mantiene las dos familias del dataset (texturas y objetos), porque el detector tiene que funcionar tanto con patrones repetitivos como con piezas que tienen una forma definida. Se descartaron grid, screw y zipper porque sus imágenes vienen en escala de grises, mientras que el teacher es un ResNet-18 preentrenado en ImageNet con imágenes a color. Se descartó toothbrush porque solo tiene 60 imágenes de entrenamiento, muy pocas para separar una validación y estimar una covarianza por categoría. Con eso quedaban 11 categorías a color y se dejó fuera transistor para completar las 10. Las máscaras de `ground_truth/` no se usan, porque el proyecto detecta anomalías a nivel de imagen y no de píxel.

### Descarga y ubicación

Cada archivo `<categoria>.tar.xz` se extrae en `data/mvtec_ad/`, de forma que quede así:

```
data/mvtec_ad/<categoria>/train/good/
data/mvtec_ad/<categoria>/test/<tipo_defecto>/
data/mvtec_ad/<categoria>/ground_truth/<tipo_defecto>/
```

## Ejecución

Los notebooks se abren con el kernel del entorno y se ejecutan en orden, desde la carpeta `notebooks/`. La primera ejecución de cualquiera de ellos construye un cache de las imágenes redimensionadas en `outputs/cache/`, lo que tarda un par de minutos. `anomalias.ipynb` necesita los checkpoints que dejan `modelos_ab.ipynb` y `modelo_c.ipynb` en `outputs/checkpoints/`.

```powershell
python -m ipykernel install --user --name mvtec --display-name "Python (mvtec)"
```

## Trabajo en paralelo

Los modelos A y B y el modelo C se desarrollan en paralelo. Cada parte tiene sus propios archivos, así que nadie edita los del otro:

| Parte | Notebook | Código | Configuración |
|---|---|---|---|
| Modelos A y B | `modelos_ab.ipynb` | `src/models/cnn.py`, `src/models/distill.py` | `conf/model/cnn_scratch.yaml`, `conf/model/cnn_distill.yaml`, `conf/experiment/cnn_scratch/`, `conf/experiment/cnn_distill/` |
| Modelo C | `modelo_c.ipynb` | `src/models/unet_ae.py` | `conf/model/unet_ae.yaml`, `conf/experiment/unet_ae/` |

Lo compartido (`src/data.py`, `src/training.py`, `src/utils.py`, `conf/config.yaml`, `conf/data/`, `conf/trainer/`, `conf/callbacks/`, `conf/logger/` y `datos.ipynb`) solo se cambia después de avisar al otro, porque afecta a los tres modelos. Un notebook nunca lo editan dos personas a la vez, porque Git no puede combinar los cambios de un `.ipynb`.

Para que los tres modelos se puedan evaluar igual en `anomalias.ipynb`, cada `LightningModule` cumple esta interfaz (descrita también en `src/models/unet_ae.py`):

- Se construye con `instantiate(cfg.model)`, así que su `__init__` recibe exactamente las llaves de su YAML, y llama `self.save_hyperparameters(logger=False)`.
- Expone `get_embedding(x)`, que devuelve un tensor `[B, d]`.
- Registra `train/loss`, `val/loss` y `test/loss`. `EarlyStopping` y `ModelCheckpoint` vigilan `val/loss`.
- Cada combinación de hiperparámetros es un archivo en `conf/experiment/<modelo>/` y se entrena con `run_experiment(load_config(["+experiment=<modelo>/<nombre>"]))`.
- Cada experimento define `name` con el formato `<letra>-<modelo>-<hiperparámetros>` (por ejemplo `A-cnn_scratch-lr3e-4_bs32` o `C-unet_ae-z64`), una `description` y `tags` con la letra, el modelo y un tag por hiperparámetro. Ese nombre se usa en W&B, en la carpeta del checkpoint y en el archivo de resultados.

## Configuración

Hydra compone una configuración tomando un archivo de cada grupo de `conf/`. Cada bloque tiene un `_target_` con la clase que construye, y `hydra.utils.instantiate` crea los objetos de Lightning a partir de él: `cfg.data` crea el `LightningDataModule`, `cfg.model` el `LightningModule`, `cfg.callbacks` los callbacks, `cfg.logger` el `WandbLogger` y `cfg.trainer` el `Trainer`.

Una corrida se elige con overrides. Los experimentos de `conf/experiment/` fijan el modelo y sus hiperparámetros con un nombre, y también se puede cambiar cualquier valor suelto:

```python
from src.utils import load_config
from src.training import run_experiment

cfg = load_config(["+experiment=cnn_scratch/lr3e-4_bs32"])     # experimento con nombre
cfg = load_config(["model=unet_ae", "model.z_dim=32", "data.batch_size=64"])  # valores sueltos
results = run_experiment(cfg)
```

`load_config` usa la API de composición de Hydra (`initialize_config_dir` y `compose`), porque el decorador `@hydra.main` no funciona dentro de Jupyter.
