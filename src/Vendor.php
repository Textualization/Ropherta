<?php

namespace Textualization\Ropherta;

require 'vendor/autoload.php';

class Vendor {

    public static function model() {
        return self::libDir() . '/' . "roberta-base-11.onnx";
    }
    
    public static function check($event=null) {
        $dest = self::model();
        
        if(file_exists($dest)) {
            echo "✔ RoBERTa ONNX Model found\n";
            return;
        }
        
        $dir = self::libDir();
        if (!file_exists($dir)) {
            mkdir($dir);
        }

        echo "Downloading RoBERTa ONNX Model...\n";

        $url = "https://github.com/onnx/models/raw/main/text/machine_comprehension/roberta/model/roberta-base-11.onnx";
        $contents = file_get_contents($url);

        $checksum = hash('sha256', $contents);
        if($checksum != "ad476a33a4b227f6e6b2e1c7192df1b61640657f26b390857ab943de66236c0b") {
            throw new Exception("Bad checksum: $checksum");
        }
        file_put_contents($dest, $contents);
        
        echo "✔ Success\n";        
    }

    private static function libDir() {
        return __DIR__ . '/../lib';
    }
}
    
