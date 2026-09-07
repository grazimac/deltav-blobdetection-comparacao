-Detecção e Comparação de Hexágono (Vértices vs. Blob)

-Descrição do Projeto

Ambiente autônomo e desacoplado para testes comparativos de algoritmos de visão computacional em drone. O objetivo é comparar a robustez e eficiência da detecção de hexágonos usando Aproximação de Vértices (⁠approxPolyDP⁠) e Detecção de Massa (⁠SimpleBlobDetector⁠).

-Arquitetura e Parâmetros

O projeto utiliza um arquivo centralizado ⁠config/parameters.yaml⁠ para ajuste fino de parâmetros sem alteração de código:
 HSV Mask: Limites inferior e superior para isolamento da cor vermelha.
 Blob Filters: Métrica de área, circularidade e convexidade para eliminação de ruídos.

-Estratégia de Versionamento (Git Flow)
 ⁠
main⁠: Versão de referência estável utilizando a lógica de detecção por Vértices.
feature/blob-detector⁠: Branch experimental desenvolvendo e validando o SimpleBlobDetector.

-Como Executar
# Instalar dependências
pip install opencv-python pyyaml numpy

# Executar o script de teste da câmera
python teste_camera.py

