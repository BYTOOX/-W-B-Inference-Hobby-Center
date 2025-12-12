# Installation de ROCm sur Debian 13 (Trixie)

Ce guide détaille l'installation de ROCm 6.2+ pour AMD Ryzen AI MAX+ avec GPU Radeon 8060S (RDNA 3.5).

---

## 📋 Prérequis

- **OS** : Debian 13 (Trixie) avec kernel 6.12+
- **GPU** : AMD Radeon (architecture RDNA 3.5)
- **Droits** : Accès sudo

### Vérifier votre GPU

```bash
lspci | grep -i amd
# Devrait afficher quelque chose comme : VGA compatible controller: AMD/ATI [Radeon ...]
```

---

## 🚀 Installation Automatique

Un script d'installation est fourni :

```bash
chmod +x install_rocm.sh
sudo ./install_rocm.sh
```

---

## 📖 Installation Manuelle

### 1. Mise à jour du système

```bash
sudo apt update && sudo apt upgrade -y
```

### 2. Installation des dépendances

```bash
sudo apt install -y \
    wget \
    gnupg2 \
    software-properties-common \
    linux-headers-$(uname -r) \
    build-essential \
    dkms
```

### 3. Ajout du dépôt AMD ROCm

```bash
# Télécharger et installer la clé GPG AMD
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/rocm.gpg

# Ajouter le dépôt ROCm
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/6.2 jammy main" | \
    sudo tee /etc/apt/sources.list.d/rocm.list

# Pour Debian, on utilise le repo Ubuntu jammy qui est compatible
sudo apt update
```

> **Note** : ROCm n'a pas de dépôt officiel Debian 13, on utilise le dépôt Ubuntu 22.04 (jammy) qui est compatible.

### 4. Installation de ROCm

```bash
# Installation du meta-package ROCm
sudo apt install -y rocm-hip-runtime rocm-hip-sdk

# Ou installation complète (plus lourd)
# sudo apt install -y rocm
```

### 5. Configuration de l'utilisateur

```bash
# Ajouter l'utilisateur aux groupes nécessaires
sudo usermod -aG video $USER
sudo usermod -aG render $USER

# Recharger les groupes (ou déconnexion/reconnexion)
newgrp video
newgrp render
```

### 6. Variables d'environnement

Ajoutez à votre `~/.bashrc` ou `~/.zshrc` :

```bash
# ROCm
export ROCM_PATH=/opt/rocm
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Pour PyTorch ROCm
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # Ajuster selon votre GPU
```

Rechargez :

```bash
source ~/.bashrc
```

### 7. Redémarrage

```bash
sudo reboot
```

---

## ✅ Vérification

### Vérifier ROCm

```bash
# Liste des GPU détectés
rocm-smi

# Version ROCm
rocminfo | head -20

# Test HIP
hipcc --version
```

Exemple de sortie `rocm-smi` :

```
========================= ROCm System Management Interface =========================
================================= Concise Info =====================================
GPU  Temp   AvgPwr  SCLK    MCLK    Fan   Perf  PwrCap  VRAM%  GPU%
0    45c    15W     500Mhz  1600Mhz 0%    auto  150W    5%     0%
====================================================================================
```

### Vérifier avec Python

```bash
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA/ROCm available: {torch.cuda.is_available()}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"
```

---

## 🐍 Installation de PyTorch ROCm

### Via pip (recommandé)

```bash
# Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer PyTorch pour ROCm 6.2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Si ROCm 6.1
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.1
```

### Vérification

```bash
python -c "import torch; print(torch.cuda.is_available())"
# Doit afficher: True
```

---

## 🔧 Dépannage

### GPU non détecté

```bash
# Vérifier que le driver est chargé
lsmod | grep amdgpu

# Vérifier les permissions
ls -la /dev/dri/
# Vous devez avoir accès à renderD128

# Si problème de permissions
sudo chmod 666 /dev/dri/renderD128
```

### Erreur HSA

Si vous avez des erreurs HSA, ajustez la version GFX :

```bash
# Pour RDNA 3.5 (gfx1150)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Pour RDNA 3 (gfx1100)
export HSA_OVERRIDE_GFX_VERSION=11.0.0

# Pour RDNA 2 (gfx1030)
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

### PyTorch ne détecte pas le GPU

```bash
# Vérifier HIP
hipconfig --full

# Tester avec un calcul simple
python -c "
import torch
x = torch.randn(1000, 1000, device='cuda')
y = torch.matmul(x, x)
print('GPU computation OK!')
print(f'Result shape: {y.shape}')
"
```

---

## 📊 Optimisations pour RyzenAI-LocalLab

### Unified Memory

Le Ryzen AI MAX+ utilise une mémoire unifiée (shared entre CPU et GPU). Pour en tirer parti :

```python
# Dans le code d'inférence
import torch

# Utiliser toute la mémoire disponible
torch.cuda.set_per_process_memory_fraction(0.95)

# Pour les gros modèles, permettre le offload automatique
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",  # Utilise GPU + CPU automatiquement
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)
```

### Variables d'environnement recommandées

```bash
# Performance
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# Mémoire
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# Debug (si nécessaire)
export AMD_LOG_LEVEL=3
```

---

## 📚 Ressources

- [Documentation ROCm officielle](https://rocm.docs.amd.com/)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)
- [ROCm GitHub](https://github.com/RadeonOpenCompute/ROCm)

---

## ⚠️ Notes Importantes

1. **Kernel 6.12+** : Requis pour le support RDNA 3.5
2. **Debian 13** : Utilise le dépôt Ubuntu jammy pour ROCm
3. **Mémoire partagée** : Le GPU utilise la RAM système, pas de VRAM dédiée
4. **HSA_OVERRIDE_GFX_VERSION** : Peut être nécessaire pour les GPU récents
