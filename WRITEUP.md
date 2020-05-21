# Project Write-Up

You can use this document as a template for providing your project write-up. However, if you
have a different format you prefer, feel free to use it as long as you answer all required
questions.

## Explaining Custom Layers

Custom layers are layers that are not included in a list of known layers. If topology contains any layers that are not in the list of known layers, the Model Optimizer classifies them as custom.

For example in case of tensorflow there are couple of options:

1: Register those layers as extensions to Model Optimizer(MO). In this case MO generates a valid and optimized intermediate representation
2: Subgraph replacement

## Comparing Model Performance

I have noticed the accuracy was dropped as once person is in the frame but model failed to identfy.

The size of the model pre- and post-conversion was...
Newly converted:
-rw-r--r-- 1 root   root  67272876 May 19 09:29 frozen_inference_graph.bin
-rw-r--r-- 1 root   root    112028 May 19 09:29 frozen_inference_graph.xml

From OpenZoo(already converted):
-rw-r--r-- 1 root root 1445640 May 11 09:04 person-detection-retail-0013.bin
-rw-r--r-- 1 root root  155730 May 11 09:04 person-detection-retail-0013.xml

The inference time of the model pre- and post-conversion was...
The inference time is increased from 44ms to 68ms

## Assess Model Use Cases

Padestrian crossing to manage the traffic lights
People entering leaving supermarket

Each of these use cases would be useful because...
The first one could be used to manage the traffic lights
The second could be used for availability of stock levels as how many people are in supermarket at one time

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows...

Lighting is an important factor as poor lighting can reduce the accuracy of the model
Camera image size should be compatible with the model.



## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: ssd_mobilenet_v1_coco
- [Model Source]
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz
tar -xvf ssd_mobilenet_v1_coco_2018_01_28.tar.gz
- I converted the model to an Intermediate Representation with the following arguments...
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

- The model was insufficient for the app because...
  
False detections e.g. poor accuracy 11 detections vs 6  
  
Output from Mosca server (FP32)
Published to person <- {"total counted": 1}
Published to person/duration <- {"duration": 2.4302010536193848}
Published to person <- {"total counted": 2}
Published to person/duration <- {"duration": 1.7882623672485352}
Published to person <- {"total counted": 3}
Published to person/duration <- {"duration": 3.115046501159668}
Published to person <- {"total counted": 4}
Published to person/duration <- {"duration": 1.9842798709869385}
Published to person <- {"total counted": 5}
Published to person/duration <- {"duration": 21.58997654914856}
Published to person <- {"total counted": 6}
Published to person/duration <- {"duration": 13.462673425674438}
Published to person <- {"total counted": 7}
Published to person/duration <- {"duration": 1.215794563293457}
Published to person <- {"total counted": 8}
Published to person/duration <- {"duration": 1.2253837585449219}
Published to person <- {"total counted": 9}
Published to person/duration <- {"duration": 2.102912187576294}
Published to person <- {"total counted": 10}
Published to person/duration <- {"duration": 13.721031665802002}



- Model 2: ssdlite_mobilenet_v2_coco
  - [Model Source]
wget http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz
tar -xvf ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz

- I converted the model to an Intermediate Representation with the following arguments...

python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

  - The model was insufficient for the app because...

False detections e.g. poor accuracy 15 detections vs 6

Output from Mosca server(FP32)
Published to person <- {"total counted": 1}
Published to person/duration <- {"duration": 6.03876805305481}
Published to person <- {"total counted": 2}
Published to person/duration <- {"duration": 6.6065754890441895}
Published to person <- {"total counted": 3}
Published to person/duration <- {"duration": 2.8322677612304688}
Published to person <- {"total counted": 4}
Published to person/duration <- {"duration": 1.6399800777435303}
Published to person <- {"total counted": 5}
Published to person/duration <- {"duration": 2.932816743850708}
Published to person <- {"total counted": 6}
Published to person/duration <- {"duration": 3.0917751789093018}
Published to person <- {"total counted": 7}
Published to person/duration <- {"duration": 2.047184705734253}
Published to person <- {"total counted": 8}
Published to person/duration <- {"duration": 3.5894429683685303}
Published to person <- {"total counted": 9}
Published to person/duration <- {"duration": 12.070576190948486}
Published to person <- {"total counted": 10}
Published to person/duration <- {"duration": 2.5189623832702637}
Published to person <- {"total counted": 11}
Published to person/duration <- {"duration": 1.364839792251587}
Published to person <- {"total counted": 12}
Published to person/duration <- {"duration": 12.772639989852905}
Published to person <- {"total counted": 13}
Published to person/duration <- {"duration": 1.1689000129699707}
Published to person <- {"total counted": 14}
Published to person/duration <- {"duration": 2.41715407371521}
Published to person <- {"total counted": 15}
Published to person/duration <- {"duration": 2.6075494289398193}
Published to person <- {"total counted": 16}
Published to person/duration <- {"duration": 2.0553598403930664}
Published to person <- {"total counted": 17}
Published to person/duration <- {"duration": 9.923475742340088}


- Model 3: ssd_mobilenet_v2_coco
  - [Model Source]
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz

  - I converted the model to an Intermediate Representation with the following arguments...
python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config /opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

  - The model was insufficient for the app because...

poor inference time around 70 ms + poor accuracy 13 detections vs 6

Output from Mosca server (FP32)
Published to person <- {"total counted": 1}
Published to person/duration <- {"duration": 17.941181182861328}
Published to person <- {"total counted": 2}
Published to person/duration <- {"duration": 3.948986053466797}
Published to person <- {"total counted": 3}
Published to person/duration <- {"duration": 2.5567140579223633}
Published to person <- {"total counted": 4}
Published to person/duration <- {"duration": 4.051571846008301}
Published to person <- {"total counted": 5}
Published to person/duration <- {"duration": 11.923787832260132}
Published to person <- {"total counted": 6}
Published to person/duration <- {"duration": 3.1187174320220947}
Published to person <- {"total counted": 7}
Published to person/duration <- {"duration": 5.056046724319458}
Published to person <- {"total counted": 8}
Published to person/duration <- {"duration": 17.024503469467163}
Published to person <- {"total counted": 9}
Published to person/duration <- {"duration": 3.362593412399292}
Published to person <- {"total counted": 10}
Published to person/duration <- {"duration": 24.258610486984253}
Published to person <- {"total counted": 11}
Published to person/duration <- {"duration": 4.761319637298584}
Published to person <- {"total counted": 12}
Published to person/duration <- {"duration": 3.771207571029663}
Published to person <- {"total counted": 13}
Published to person/duration <- {"duration": 17.33377170562744}

- Model 4: ssd_mobilenet_v2_coco (Intel's model)
cd /opt/intel/openvino/deployment_tools/tools/model_downloader
python downloader.py --name person-detection-retail-0013 -o /home/workspace
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m intel/person-detection-retail-0013/FP16/person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

Output from Mosca server (FP32)
Published to person <- {"total counted": 1}
Published to person/duration <- {"duration": 15.416040897369385}
Published to person <- {"total counted": 2}
Published to person/duration <- {"duration": 24.97369885444641}
Published to person <- {"total counted": 3}
Published to person/duration <- {"duration": 22.165178298950195}
Published to person <- {"total counted": 4}
Published to person/duration <- {"duration": 13.997902870178223}
Published to person <- {"total counted": 5}
Published to person/duration <- {"duration": 30.38458251953125}
Published to person <- {"total counted": 6}
Published to person/duration <- {"duration": 13.618701219558716}

Output from Mosca server (FP16)
Published to person <- {"total counted": 1}
Published to person/duration <- {"duration": 15.346906185150146}
Published to person <- {"total counted": 2}
Published to person/duration <- {"duration": 24.639443159103394}
Published to person <- {"total counted": 3}
Published to person/duration <- {"duration": 21.82383441925049}
Published to person <- {"total counted": 4}
Published to person/duration <- {"duration": 13.797784805297852}
Published to person <- {"total counted": 5}
Published to person/duration <- {"duration": 30.202072620391846}
Published to person <- {"total counted": 6}
Published to person/duration <- {"duration": 13.491873025894165}