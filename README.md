# AICA
AI card generator


# Install

## conda env
- python 3.8.10
- conda create -n -y aica python=3.8.10 && conda activate aica
- pip install -r requirements.txt

## Google Colab

https://colab.research.google.com/drive/1xsGjyfeuuWWK8m7CRIXFz2FQBX4O66Zy?usp=sharing

```
!pip install -q openai
!pip install gradio -q
!pip install -q git+https://github.com/huggingface/diffusers
!pip install -q git+https://github.com/huggingface/transformers
!pip install -q pynvml
!pip install -q diffusers["torch"]
!pip install accelerate
!pip install -q gTTS==2.2.1

```

# start
- python aica.py