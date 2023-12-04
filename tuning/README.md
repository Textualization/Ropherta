# Fine-tuning RoBERTa-base and exporting to ONNX to use with Ropherta

Install the requirements on a python virtual environment:

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Given a large amount of in-domain text in `train.txt` with a held-out sample in `test.txt` fine-tune RoBERTa-base (this will take many hours, possibly days and might not be feasible without a GPU):

```bash
python fine_tune.py
```

Once the model has been fine-tuned, it can be exported to ONNX format:

```bash
python onnx_export.py
```

## Learn more

* https://towardsdatascience.com/fine-tuning-for-domain-adaptation-in-nlp-c47def356fd6
* https://towardsdatascience.com/perplexity-in-language-models-87a196019a94
