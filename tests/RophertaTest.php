<?php

namespace Textualization\Ropherta\Tests;

use Textualization\Ropherta\Tokenizer;
use Textualization\Ropherta\RophertaModel;
use Textualization\Ropherta\Vendor;
use Textualization\Ropherta\Distances;

use PHPUnit\Framework\TestCase;

class RophertaTest extends TestCase {
    private RophertaModel $model;

    protected function setUp() : void
    {
        parent::setUp();
        Vendor::check();
        
        $this->model = new RophertaModel();
    }

    public function test_long_text() : void
    {
        $longText = file_get_contents(__DIR__ . '/../vendor/gioni06/gpt3-tokenizer/tests/__fixtures__/long_text.txt');
        $embedding = $this->model->embeddings($longText);

        $this->assertEquals(768, count($embedding));
        $expected = [ 66.0, -2042.0, -1620.0, -684.0, 1238.0, 1990.0, 2390.0, -947.0, -608.0, -1621.0 ];
        $got = array_slice($embedding, 0, 10);
        for($i=0; $i<10; $i++){
            $expected[$i] = \floor($expected[$i]);
            $got[$i]      = \floor($got[$i]      * 10000);
        }
        $this->assertEquals($got, $expected);
    }

    public function test_long_text_lines() : void
    {

        $embedding1 = $this->model->embeddings('One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin. He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections. The bedding was hardly able to cover it and seemed ready to slide off any moment. His many legs, pitifully thin compared with the size of the rest of him, waved about helplessly as he looked. "What\'s happened to me? " he thought. It wasn\'t a dream.');

        $embedding2 = $this->model->embeddings('" He felt a slight itch up on his belly; pushed himself slowly up on his back towards the headboard so that he could lift his head better; found where the itch was, and saw that it was covered with lots of little white spots which he didn\'t know what to make of; and when he tried to feel the place with one of his legs he drew it quickly back because as soon as he touched it he was overcome by a cold shudder. He slid back into his former position. "Getting up early all the time", he thought, "it makes you stupid. You\'ve got to get enough sleep. Other travelling salesmen live a life of luxury.');

        $embedding3 = $this->model->embeddings('Vancouver is a major city in western Canada, located in the Lower Mainland region of British Columbia. As the most populous city in the province, the 2021 Canadian census recorded 662,248 people in the city, up from 631,486 in 2016. The Greater Vancouver area had a population of 2.6 million in 2021, making it the third-largest metropolitan area in Canada. Greater Vancouver, along with the Fraser Valley, comprises the Lower Mainland with a regional population of over 3 million. Vancouver has the highest population density in Canada, with over 5,700 people per square kilometre, and fourth highest in North America (after New York City, San Francisco, and Mexico City). ');

        $d12 = Distances::cosine($embedding1, $embedding2);
        $d13 = Distances::cosine($embedding1, $embedding3);
        $d23 = Distances::cosine($embedding2, $embedding3);

        #echo "d12 = $d12\nd13 = $d13\nd23 = $d23\n";

        $this->assertTrue($d12 < $d13);
        $this->assertTrue($d12 < $d23);
    }

    public function test_short_texts() : void
    {

        $embedding1 = $this->model->embeddings('A dog is happy.');

        $embedding2 = $this->model->embeddings('The cat is playing.');

        $embedding3 = $this->model->embeddings('A caterpillar is pupping.');

        $d12 = Distances::cosine($embedding1, $embedding2);
        $d13 = Distances::cosine($embedding1, $embedding3);
        $d23 = Distances::cosine($embedding2, $embedding3);

        #echo "d12 = $d12\nd13 = $d13\nd23 = $d23\n";

        $this->assertTrue($d12 < $d13);
        $this->assertTrue($d12 < $d23);
    }        
}
