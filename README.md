# TP4 – Redes convolucionales para clasificar lesiones de piel (ISIC)

**Redes Neuronales / Deep Learning – ITBA** · Alumno: Santiago · Profesor: Carlos Selmo

Clasificación de 9 clases de lesiones de piel (dataset ISIC, 878 imágenes; 844 después de eliminar duplicados) comparando una MLP con redes convolucionales. Se analizan arquitectura, campo receptivo, resolución de entrada y data augmentation.

Notebook: [`TP4_CNN_Skin_ISIC.ipynb`](TP4_CNN_Skin_ISIC.ipynb) (ejecutado, con outputs y análisis).

## Resultados principales

En validación (3 semillas), la CNN a 64 px llega a 0,615 ± 0,015 contra 0,525 ± 0,043 de la MLP, con 11,6 veces menos parámetros (549.273 contra 6.358.793). Entre las variantes de arquitectura, 8 filtros (0,681) y 4 bloques (0,669) superan a la base (0,608) por poco más que el ruido, y el resto queda dentro del ruido. La data augmentation no dio una mejora clara: el mejor nivel, flips y rotaciones de 90°, llegó a 0,655 ± 0,011.

En test, usado una sola vez y con 5 semillas, la CNN final (8 filtros con flips y rotaciones) llega a **0,681 ± 0,035** de accuracy y 0,667 de F1 macro, contra 0,504 ± 0,053 de la MLP. El error estándar del test es de ±0,036. Las limitaciones más importantes son el bajo recall de melanoma (0,40, se confunde con nevo) y un posible sesgo por el origen de las imágenes, porque varias clases tienen una resolución o un tipo de foto propios.

## Cómo reproducirlo

El dataset **no está en este repo**: está en el repo del profesor ([cselmo/CNN-en-Skin-Dataset](https://github.com/cselmo/CNN-en-Skin-Dataset)). La primera celda del notebook lo descarga con `git clone` y se para en su carpeta.

**Opción A – Google Colab (como se ejecutó este notebook):**
1. Abrir el notebook en Colab (File → Upload notebook).
2. Runtime → Change runtime type → T4 GPU.
3. Runtime → Run all.

**Opción B – local:**
```
pip install -r requirements.txt
```
Abrir el notebook con Jupyter o VS Code y ejecutar todo. Hace falta tener `git` instalado para la primera celda. Si no lo tenés, descargá el repo del profesor y ejecutá el notebook desde la carpeta que contiene `data/`, salteando esa celda.

En Windows, descomprimir el dataset en una ruta corta (por ejemplo `C:\tp4`), porque algunos nombres de archivo superan el largo máximo de ruta.

## Entorno de la ejecución

Google Colab con GPU (`cuda`), Python 3.13.16 y torch 2.14.1 (build CUDA 13.0). Las versiones exactas están en `requirements.txt`. No se cambió ninguna constante del notebook original: el detalle está en la sección 0 del notebook.

`torch==2.14.1` instala la versión estándar para tu sistema. Para usar GPU fuera de Colab, instalar la build CUDA correspondiente desde https://pytorch.org.
