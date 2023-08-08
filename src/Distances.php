<?php

namespace Textualization\Ropherta;

require 'vendor/autoload.php';

class Distances {

    protected static function norm(array $arr) : float {
        $result = 0.0;
        foreach($arr as $v) {
            $result += $v * $v;
        }
        return \sqrt($result);
    }

    public static function cosine(array $emb1, array $emb2) : float {
        $n1 = self::norm($emb1);
        $n2 = self::norm($emb2);
        $cross = 0.0;
        $len = count($emb1);
        for($i=0; $i<$len; $i++){
            $cross += $emb1[$i] * $emb2[$i];
        }
        return 1.0 - $cross / ($n1*$n2);
    }
}
    
