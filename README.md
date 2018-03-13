
**********************************
CI396 / CI852

Topics in Computer Vision

Second semester of 2016

### 3rd Practical Activity - Finger Id. ###
**********************************

**Implementado:**

- Leitura base de dados Rindex28

- fingerId_Type.py 
  - Para cada imagem realiza Orientation Computation
    - Para cada imagem realiza Region of Interest Detection
    - Para cada imagem realiza Singular Point Detection
    - Classifica cada label das imagens baseado no consenso (maioria) de cada label

- fingerId_Classification.py 
    - Salvo parte do algoritmo fingerId_Type.py
      - Para cada imagem realiza Orientation Computation
      - Para cada imagem realiza Region of Interest Detection
      - Para cada imagem realiza Singular Point Detection
    - Para cada imagem realiza Image Binarization
    - Para cada imagem realiza Smoothing (lento)
    - Para cada imagem realiza Thinning
    - Para cada imagem realiza Minutiae Detection/Extraction


