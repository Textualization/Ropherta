# Compute RoBERTa embeddings in PHP using ONNX framework.

This brings the power of Transformers to the PHP world.

## Installation

Add this project to your dependencies

```
composer require textualization/ropherta
composer update
```

Before using it, you will need to install the ONNX framework:

```
composer exec -- php -r "require 'vendor/autoload.php'; OnnxRuntime\Vendor::check();"
```

and download the RoBERTa ONNX model (this takes a while, the model is 477Mb in size):

```
composer exec -- php -r "require 'vendor/autoload.php'; Textualization\Ropherta\Vendor::check();"
```

## Computing embeddings

```php

$model = new RophertaModel();

$emb = $model->embeddings("Text");
```

Check `\Textualization\Ropherta\Distances` to check whether two embeddings are closer to each other.

## Using custom embeddings

```php
$model = new RophertaModel("/path/to/model.onnx");
$emb = $model->embeddings("Text");
```

To fine-tune a model you will need a large amount of in-domain text and use Python in a machine with a GPU. See [tuning](tuning/) for details.

## Sponsors

We thank our sponsor:

<a href="https://evoludata.com/"><img src="https://evoludata.com/display208"></a>

