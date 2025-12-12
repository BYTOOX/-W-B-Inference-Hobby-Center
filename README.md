# RyzenAI-LocalLab 🚀

**Interface d'Inférence HomeLab pour AMD Ryzen AI MAX+**

Une interface web moderne pour gérer, télécharger et exécuter des modèles d'IA (LLM & Code) en local, optimisée pour l'architecture AMD Ryzen AI avec GPU intégré Radeon (ROCm).

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-red?logo=streamlit)
![ROCm](https://img.shields.io/badge/ROCm-6.2+-orange?logo=amd)

---

## ✨ Fonctionnalités

- **🧠 Gestion des Modèles** : Téléchargement depuis HuggingFace avec progression visuelle
- **⚡ Inférence Optimisée** : Support ROCm/HIP pour GPU AMD Radeon
- **💬 Interface Chat** : Rendu Markdown, syntax highlighting, streaming temps réel
- **📊 Monitoring HomeLab** : Jauges CPU/GPU/RAM, tokens/sec, TTFT
- **🔌 API OpenAI-Compatible** : `/v1/chat/completions` pour intégration externe
- **👥 Multi-Utilisateurs** : Authentification simple avec SQLite
- **📦 Détection Intelligente** : Analyse automatique de la compatibilité modèle/hardware

---

## 🔧 Hardware Cible

| Composant | Spécification |
|-----------|---------------|
| CPU | AMD Ryzen AI MAX+ 395 (16-core) |
| GPU | Radeon 8060S (RDNA 3.5) |
| RAM | ~124 GiB (Unified Memory) |
| OS | Debian 13+ (Linux Kernel 6.12+) |

---

## 🚀 Installation Rapide

### Prérequis

1. **ROCm 6.2+** installé ([Guide d'installation](docs/INSTALL_ROCM.md))
2. **Python 3.11+**
3. **Git**

### Installation

```bash
# Cloner le repo
git clone https://github.com/BYTOOX/RyzenAI-LocalLab.git
cd RyzenAI-LocalLab

# Lancer le script d'installation
chmod +x install.sh
./install.sh

# Activer l'environnement
source venv/bin/activate

# Lancer l'application
./run.sh
```

---

## 📁 Structure du Projet

```
RyzenAI-LocalLab/
├── backend/                 # API FastAPI
│   ├── api/                 # Routes (auth, models, chat, openai)
│   ├── core/                # Config, database, auth
│   ├── services/            # Model manager, inference, monitoring
│   └── main.py              # Entry point API
├── ui/                      # Interface Streamlit
│   ├── pages/               # Chat, Models, Dashboard
│   ├── components/          # Composants réutilisables
│   └── app.py               # Entry point UI
├── docs/                    # Documentation
│   └── INSTALL_ROCM.md      # Guide ROCm Debian 13
├── data/                    # SQLite database
├── requirements.txt         # Dépendances Python
├── install.sh               # Script d'installation
└── run.sh                   # Script de lancement
```

---

## 🔌 API OpenAI-Compatible

L'API est compatible avec les clients OpenAI standard :

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="votre-api-key"
)

response = client.chat.completions.create(
    model="Devstral-Small-2505",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | Liste des modèles disponibles |
| `/v1/chat/completions` | POST | Chat completion (streaming) |

---

## 🎨 Interface HomeLab

L'interface Streamlit propose un design moderne type "HomeLab" :

- **Theme Dark** avec accents cyan/purple
- **Jauges temps réel** pour CPU, GPU, RAM
- **Stats d'inférence** : tokens/sec, TTFT
- **Gestion des modèles** : téléchargement, suppression, info

---

## 📋 Modèles Supportés

| Modèle | Taille | Format | Recommandé |
|--------|--------|--------|------------|
| `mistralai/Devstral-Small-2505` | ~16GB | safetensors | ✅ Code |
| `Qwen/Qwen3-30B-A3B` | ~17GB | safetensors | ✅ Général |
| `Qwen/Qwen3-235B-A22B` | ~140GB | safetensors | ⚠️ Quantification requise |

---

## 🛠️ Configuration

Créez un fichier `.env` à la racine :

```env
# Paths
MODELS_PATH=/srv/models
DATA_PATH=./data

# Server
API_HOST=0.0.0.0
API_PORT=8000
UI_PORT=8501

# Security
SECRET_KEY=your-secret-key-here
FIRST_ADMIN_USERNAME=admin
FIRST_ADMIN_PASSWORD=changeme
```

---

## 📖 Documentation

- [Guide d'installation ROCm](docs/INSTALL_ROCM.md)
- [Configuration avancée](docs/CONFIGURATION.md) *(à venir)*
- [API Reference](docs/API.md) *(à venir)*

---

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir des issues ou PR.

---

## 📄 Licence

MIT License - Voir [LICENSE](LICENSE)
