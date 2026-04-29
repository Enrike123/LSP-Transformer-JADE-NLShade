# LSP-Transformer-JADE-NLShade

**Bioinspirada optimization of Transformer models for Peruvian Sign Language (LSP) recognition using JADE and NL-SHADE algorithms.**

> Tesis de Maestría en Ciencias de la Computación  
> Universidad Nacional de San Agustín de Arequipa (UNSA) — 2025  
> Autor: Jaime Enrique Villegas Medina

---

## Descripción

Este repositorio contiene el código fuente, el notebook de optimización y los resultados del sistema de reconocimiento automático del **Lenguaje de Señas Peruano (LSP)** propuesto en la tesis de maestría. El sistema integra:

- Un modelo **Transformer Encoder** para clasificación de secuencias gestuales
- Extracción de puntos clave con **MediaPipe Holistic**
- Optimización automática de hiperparámetros con **JADE** y **NL-SHADE**
- Validación estadística inferencial (pruebas *t* y ANOVA)
- Interfaz de predicción en tiempo real con **Tkinter + OpenCV**

---

## Resultados principales

| Modelo | Parámetros | F1-macro (Val) | F1-macro (Test) | Latencia |
|---|---|---|---|---|
| Transformer baseline | 410,508 | 0.953 | 0.942 | 48.39 ms |
| Transformer + JADE | 884,623 | **0.982** | **0.970** | 68.17 ms |
| Transformer + NL-SHADE | 1,356,876 | 0.976 | 0.963 | 89.46 ms |

Las mejoras sobre el modelo base son estadísticamente significativas (p < 0.001, prueba *t* pareada y ANOVA de una vía).

---

## Estructura del repositorio

```
LSP-Transformer-JADE-NLShade/
├── README.md
├── LICENSE
├── code/
│   ├── preprocesamiento.py              # Extracción de puntos clave con MediaPipe
│   ├── grabar_gestos2.py                # Captura de videos por clase
│   ├── modelo1_mejorado.py              # Modelo Transformer base
│   └── Transformer_JADE_NLSHADE_Colab.ipynb  # Notebook principal de optimización
└── results/
    ├── jade_best_config.json            # Hiperparámetros óptimos JADE
    └── nlshade_best_config.json         # Hiperparámetros óptimos NL-SHADE
```

---

## Dataset

El dataset no está incluido directamente en este repositorio por su tamaño (316.5 MB). Está disponible en Google Drive:

- **Dataset preprocesado** (`.npy`, 316.5 MB):  
  🔗 🔗 [gesture_dataset12clases_5aum.npy — Google Drive](https://drive.google.com/file/d/1i1JNUQqFJNJw6vbN_gukhMkrN2XxtO6_/view?usp=drive_link)

- **Videos originales** (12 clases, ~100 videos/clase):  
  🔗 🔗 [LSP_Dataset_Videos — Google Drive](https://drive.google.com/drive/folders/15Y34iFOEeOvshLIS-N4Z76hrDr1CzCmb?usp=sharing)

### Descripción del dataset

| Característica | Detalle |
|---|---|
| Clases gestuales | 12 (hola, gracias, mamá, cómo estás, papá, por favor, familia, adiós, jugar, comer, caminar, agua) |
| Muestras originales | 1,200 videos (~100 por clase) |
| Muestras aumentadas | ~7,200 secuencias |
| Duración por muestra | 2 segundos a 30 fps (60 frames) |
| Características por frame | 96 (coordenadas x,y de 48 puntos clave) |
| Formato | NumPy array (.npy), shape: (N, 60, 96) |

---

## Cómo usar

### 1. Requisitos

```bash
pip install tensorflow mediapipe opencv-python numpy pandas matplotlib scikit-learn
```

### 2. Preprocesar videos

```bash
python code/preprocesamiento.py
```

### 3. Ejecutar optimización en Google Colab

Abre `code/Transformer_JADE_NLSHADE_Colab.ipynb` en Google Colab y sigue las instrucciones de cada celda. El notebook está estructurado en secciones numeradas.

### 4. Inferencia en tiempo real

```bash
python code/modelo1_mejorado.py
```

---

## Arquitectura del sistema

```
Video (cámara) → MediaPipe → Vector (60×96) → Transformer Encoder → Softmax → Clase LSP
                                                        ↑
                                              JADE / NL-SHADE
                                          (optimización de hiperparámetros)
```

---

## Hiperparámetros óptimos

### JADE (mejor configuración)
```json
{
  "d_model": 160,
  "n_heads": 4,
  "n_layers": 3,
  "ffw_mult": 3.61,
  "dropout": 0.175,
  "lr": 0.000865,
  "batch_size": 64
}
```

### NL-SHADE (mejor configuración)
```json
{
  "d_model": 192,
  "n_heads": 6,
  "n_layers": 4,
  "ffw_mult": 2.50,
  "dropout": 0.200,
  "lr": 0.001600,
  "batch_size": 32
}
```

---

## Referencia

Si usas este código o dataset en tu investigación, por favor cita:

```bibtex
@mastersthesis{VillegasMedina2025,
  author  = {Enrique Villegas Medina},
  title   = {Optimización bioinspirada de modelos Transformer aplicados al 
             reconocimiento del Lenguaje de Señas Peruano mediante JADE y NL-SHADE},
  school  = {Universidad Nacional de San Agustín de Arequipa},
  year    = {2025},
  type    = {Tesis de Maestría en Ciencias de la Computación}
}
```

---

## Contacto

**Jaime Enrique Villegas Medina**  
email: jvillegasm@unsa.edu.pe
Universidad Nacional de San Agustín de Arequipa  
GitHub: [@Enrike123](https://github.com/Enrike123)

---

## Licencia

Este proyecto está bajo la licencia MIT. Ver archivo [LICENSE](LICENSE) para más detalles.
