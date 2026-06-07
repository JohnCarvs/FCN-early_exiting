# Setup e execução — FCN-early_exiting

Guia para configurar o ambiente, posicionar os dados e rodar treino/inferência.

> O treino real é feito na máquina com **GPU NVIDIA** (ex.: VM do laboratório). O código também roda em **CPU** (detecta o device automaticamente), mas treinar em CPU é inviável na prática — localmente use só para *smoke-test*.

---

## 1. Pré-requisitos

- **Python 3.10+** (o código usa `match/case`).
- **git**.
- Para treino: **GPU NVIDIA com CUDA** (a CPU serve só para testes rápidos).

---

## 2. Ambiente

Crie e ative um ambiente virtual e instale as dependências:

```bash
# na raiz do repositório
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### Atenção ao `torch` / `torchvision`
O `requirements.txt` lista os pacotes sem versão. A versão correta do `torch`/`torchvision` **depende do CUDA** do ambiente:

- **Na GPU (VM do lab):** instale o wheel que casa com o CUDA da máquina, seguindo o seletor oficial em https://pytorch.org/get-started/locally/ (ex.: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`).
- **Localmente (CPU):** o wheel padrão de CPU já basta (`pip install torch torchvision`).

Verifique a instalação:
```bash
python -c "import torch; print(torch.__version__, 'cuda:', torch.cuda.is_available())"
```

---

## 3. Dados

### 3.1 Treino — dataset SBD

O `--phase train` usa a classe `SBDClassSeg`, que espera **exatamente** esta estrutura, relativa ao caminho passado em `--data`:

```
<--data>/VOC/VOCdevkit/SBDD/dataset/
├── train.txt          # lista de IDs de treino (vem com a SBD)
├── seg11valid.txt     # lista de IDs de validação (NÃO vem com a SBD — obter à parte)
├── img/
│   └── <id>.jpg       # imagens RGB
└── cls/
    └── <id>.mat       # rótulos (campo GTcls.Segmentation, classes 0–20)
```

**Como montar:**

1. **Baixar a SBD** (~1.4 GB):
   `http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz`
   Extraia e copie o conteúdo de `benchmark_RELEASE/dataset/` (as pastas `img/`, `cls/` e o `train.txt`) para `<--data>/VOC/VOCdevkit/SBDD/dataset/`.

2. **Obter o `seg11valid.txt`** (não está no repo nem na SBD):
   `https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/data/pascal/seg11valid.txt`
   Coloque-o dentro de `<--data>/VOC/VOCdevkit/SBDD/dataset/`.

> **Caveat:** o `seg11valid.txt` lista IDs do PASCAL VOC 2011. Cada ID precisa ter o `img/<id>.jpg` **e** o `cls/<id>.mat` presentes na SBD; caso contrário o carregamento daquele item falha (`scipy.io.loadmat`). A maioria existe na SBD, mas pode haver IDs faltantes.

> **Dica:** o default `--data=./train` é relativo ao diretório onde você roda (`src/`). Prefira **caminho absoluto** (ex.: `--data D:\datasets` ou `--data /home/user/datasets`) para evitar confusão.

### 3.2 Teste/inferência

O `--phase test` usa `MyTestData`, que espera apenas uma **pasta plana com arquivos `.jpg`** em `--data` (sem subpastas, sem `.mat`):

```
<--data>/
├── img1.jpg
├── img2.jpg
└── ...
```

As máscaras previstas (`.png` coloridas) são salvas em `--out`.

---

## 4. Rodar

> Rode sempre **de dentro de `src/`** (os imports são relativos: `from data.data import ...`).

### Treino (na GPU)
```bash
cd src
python main.py --phase train --model FCN8 --data /caminho/datasets --out /caminho/out --epochs 90
```

### Retomar treino a partir de um checkpoint
```bash
python main.py --phase train --model FCN8 --data /caminho/datasets --out /caminho/out --param /caminho/out/FCN-epoch-10.pth
```

### Inferência (precisa de um checkpoint treinado)
```bash
python main.py --phase test --model FCN8 --data /caminho/imgs_teste --out /caminho/preds --param /caminho/out/FCN-epoch-XX.pth
```
> Sem um `--param` treinado, a cabeça do FCN está aleatória e as máscaras não fazem sentido.

### Argumentos
| Arg | Default | Função |
|-----|---------|--------|
| `--phase` | `train` | `train` ou `test` |
| `--model` | `FCN8` | `FCN8` / `FCN16` / `FCN32` |
| `--data` | `./train` | raiz dos dados |
| `--out` | `./out` | saída |
| `--epochs` | `90` | nº de épocas |
| `--param` | `None` | checkpoint `.pth` (retomar/inferir) |

---

## 5. Saídas (em `--out`)

- `FCN-epoch-<N>.pth` — checkpoint por época (`model_state_dict`, `optimizer_state_dict`, `best_loss`, `best_epoch`).
- `metrics.csv` — uma linha por época: `epoch,train_loss,val_loss,mean_pixel_acc,miou`.
- `best_epoch.txt` — melhor época e respectiva loss de validação.

---

## 6. Notas e limitações

- **Rode de dentro de `src/`** (ou adicione `src/` ao `PYTHONPATH`), por causa dos imports relativos.
- **CPU vs GPU:** o código detecta automaticamente (`Using device: cuda` ou `cpu`). Treinar em CPU é inviável — use a GPU do lab; localmente, a CPU serve só para validar que o pipeline roda (1 época, poucas imagens).
- Os três modelos (FCN8/16/32) diferem apenas nas *skip connections* / upsampling; o backbone VGG16 é o mesmo.
