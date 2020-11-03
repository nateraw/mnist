# mnist

Set up environment:

```
conda create -n mnist-env python=3.7
conda activate mnist-env
pip install -r requirements.txt
```

Run it:

```
python pl_mnist.py --max_epochs 5 --lr 1e-4
```
