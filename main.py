"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 
 
"""


import os
import sys
import time
import socket
import json
import cv2
import math

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.55,
                        help="Probability threshold for detections filtering"
                        "(0.55 by default)")
    return parser


def connect_to_mqtt_server():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client

def draw_bounding_box(coords, frame, initial_w, initial_h):
    draw_bounding_box.euclidean_distance = getattr(draw_bounding_box, 'euclidean_distance', 0)
    draw_bounding_box.empty_frames = getattr(draw_bounding_box, 'empty_frames', 0)
    current_count = 0     
    
    # Draw bounding box for each object when it's probability is more than the specified threshold
    for obj in coords[0][0]:        
        if obj[2] > prob_threshold:
            xmin = int(obj[3] * initial_w)
            ymin = int(obj[4] * initial_h)
            xmax = int(obj[5] * initial_w)
            ymax = int(obj[6] * initial_h)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
            current_count = current_count + 1
            
            # Calculate mid point of the frame
            frame_x = frame.shape[1]/2
            frame_y = frame.shape[0]/2
            
            # Calculate mid point of the bounding box
            bb_mid_x = (xmax + xmin)/2
            bb_mid_y = (ymax + ymin)/2
                
            # Calculate euclidean distance between mid points of the frame and bounding box  
            # and reset empty_frames since bounding box drawn in the frame
            draw_bounding_box.euclidean_distance =  math.sqrt(float(math.pow(bb_mid_x-frame_x, 2) +  math.pow(bb_mid_y-frame_y,2))) 
            draw_bounding_box.empty_frames = 0

    # Increment empty_frames since there was no one in this frame
    if current_count < 1:
        draw_bounding_box.empty_frames = draw_bounding_box.empty_frames + 1
    
    # If there is euclidean distance from previous frames and there are few empty frames 
    # That means previous person hasn't left; just the case that model hasn't detected 
    # the person properly so set the current_count and increment empty_frames
    # draw_bounding_box.empty_frames threshold is 0.5 second assuming frame rate of 30 frames per second
    if draw_bounding_box.euclidean_distance > 0 and draw_bounding_box.empty_frames < 15:
        current_count = 1 
        draw_bounding_box.empty_frames = draw_bounding_box.empty_frames + 1
        
        # Reset the empty_frames after 5 seconds just in case no one enters in frame for long time
        # Assuming the frame rate is 30 frames per second
        if draw_bounding_box.empty_frames > 150:
            draw_bounding_box.empty_frames = 0
    return frame, current_count

def infer_on_stream(args, mqtt_client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    
    cur_request_id = 0
    
    ### TODO: Load the model through `infer_network` ### 
    n, c, h, w = infer_network.load_model(args.model, args.device, 1, 1, cur_request_id, args.cpu_extension)[1]

    ### TODO: Handle the input stream ###
    if args.input == 'CAM':
        input_stream = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp'):
        single_img_flag = True
        input_stream = args.input
    else:
        input_stream = args.input
        assert os.path.isfile(args.input), "Specified input file doesn't exist"
    
    try:
        capture = cv2.VideoCapture(args.input)
    except FileNotFoundError:
        print("Cannot locate video file: "+ args.input)
    except Exception as e:
        print("Something else went wrong with the video file: ", e)
        
    # Various variables to track status
    global initial_w, initial_h, prob_threshold
    global overlay_colour
    overlay_colour = (0,0,255)
    single_img_flag = False
    start_time = 0    
    last_count = 0
    total_count = 0
    duration = 0   
    
    initial_w = capture.get(3)
    initial_h = capture.get(4)
    prob_threshold = args.prob_threshold
    
    ### TODO: Loop until stream is over ###
    while capture.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = capture.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)
        
        ### TODO: Pre-process the image as needed ###
        ### Change layout from HxWxC to CxHxW ###
        image = cv2.resize(frame, (w, h))
        image = image.transpose((2, 0, 1))
        image = image.reshape((n, c, h, w))
        
        ### TODO: Start asynchronous inference for specified request ###
        start_time = time.time()
        infer_network.exec_net(cur_request_id, image)
        
        ### TODO: Wait for the result ###
        if infer_network.wait(cur_request_id) == 0:
            # Note the end time and calculate inference time
            inference_time = time.time() - start_time

            ### TODO: Get the results of the inference request ###
            result = infer_network.get_output(cur_request_id)
            
            ### TODO: Extract any desired stats from the results ###
            frame, current_count = draw_bounding_box(result, frame, initial_w, initial_h)
            
            # Adding overlays for the inference time
            inference_duration = "Inference duration: {:.3f}ms".format(inference_time * 1000)
            cv2.putText(frame, inference_duration, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, overlay_colour, 1)

            # Adding overlays for the current count
            current_count_message = "Current count: %d " %current_count
            cv2.putText(frame, current_count_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, overlay_colour, 1)
            
            # When new person enters the video; publish total counted
            if current_count > last_count:
                person_entry_time = time.time()
                total_count = total_count + current_count - last_count
                mqtt_client.publish("person", json.dumps({"total counted": total_count}))            
            
            # Calculate duration in the video; publish the duration
            if current_count < last_count:
                person_exit_time = time.time()
                duration =  person_exit_time - person_entry_time
                mqtt_client.publish("person/duration", json.dumps({"duration": duration}))
           
            # Adding overlays for the Total count
            total_count_message = "Total count: %d " %total_count
            cv2.putText(frame, total_count_message, (15, 45), cv2.FONT_HERSHEY_COMPLEX, 0.5, overlay_colour, 1)

            last_count = current_count
            
            if key_pressed == 27:
                break

        ### TODO: Send the frame to the FFMPEG server ###
        sys.stdout.buffer.write(frame)  
        sys.stdout.flush()
        
        ### TODO: Write an output image if `single_image_mode` ###
        if single_img_flag:
            cv2.imwrite('output_image.jpg', frame)
       
    capture.release()
    cv2.destroyAllWindows()
    mqtt_client.disconnect()
    infer_network.clean()

def main():
    """
    Load the network and parse the output.

    :return: None
    """
    #log.getLogger().setLevel(log.INFO)
    #log.info('Message: duration: %f'%duration)

    # Grab command line args
    args = build_argparser().parse_args()
    
    # Connect to the MQTT server
    mqtt_client = connect_to_mqtt_server()
    
    # Perform inference on the input stream
    infer_on_stream(args, mqtt_client)


if __name__ == '__main__':
    main()