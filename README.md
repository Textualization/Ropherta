# Compute RoBERTa embeddings using ONNX framework.

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

