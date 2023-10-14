<?php

namespace Textualization\Ropherta;

require 'vendor/autoload.php';

use \OnnxRuntime\Model;
use \Textualization\Ropherta\Tokenizer;

class RophertaModel {

    private \OnnxRuntime\Model $model;
    protected Tokenizer $tokenizer;
    protected int $input_size;

    function __construct($model=null, $input_size=512)
    {
        if(! $model) {
            $model = Vendor::model();
        }
        $this->model = new \OnnxRuntime\Model($model);
        $this->input_size = $input_size;

        $this->tokenizer = new Tokenizer();
    }

    function embeddings(string|array $text_or_tokens) : array
    {
        $output = $this->_encode($text_or_tokens);
        
        if(isset($output["output_2"])){
            $output = $output["output_2"][0];
        }
        if(isset($output["last_hidden_state"])){
            $output = $output["last_hidden_state"][0][0];
        }
        return $output;
    }
    
    function _encode(string|array $text_or_tokens) : array {
        if(is_array($text_or_tokens)) {
            $tokens = $text_or_tokens;
        }else{
            $tokens = $this->tokenizer->encode($text_or_tokens);
        }
        if(count($tokens) > $this->input_size) {
            $tokens = \array_slice($tokens, 0, $this->input_size);
        }
        
        $input = [ 'input_ids'=>[ $tokens ] ];
        if(count($this->model->inputs()) > 1) {
            // has mask
            $mask = [];
            foreach($tokens as $tok){
                $mask[] = 1.0;
            }
            $input['attention_mask'] = [ $mask ];
        }
        
        return $this->model->predict($input);
    }
}
