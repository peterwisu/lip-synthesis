# Gradio app

import gradio as gr
import os
import argparse 
from src.main.inference import Inference 

MODEL_TYPE = ['lstm','attn_lstm']

MODEL_NAME = { 'lstm_mse':('./checkpoints/generator/pretrain003_mse_loss.pth','lstm'),
              'attention_lstm_l1' : ('./checkpoints/generator/attnLSTM_pretrain030_l1.pth','attn_lstm'),
              'attention_lstm_mse': ('./checkpoints/generator/attnLSTM_pretrain030_mse.pth','attn_lstm')}

print(MODEL_NAME.keys())

def func(video,audio,check,drop_down):

    
    path , model_type = MODEL_NAME[drop_down]

    print(path)

    print(model_type)


    parser = argparse.ArgumentParser(description="File for running Inference")

    parser.add_argument('--model_type', help='Type of generator model', default=model_type, type=str)

    parser.add_argument('--generator_checkpoint', type=str ,default=path)

    parser.add_argument('--image2image_checkpoint', type=str, default='./checkpoints/image2image/image2image.pth',required=False)

    parser.add_argument('--input_face', type=str,default=video, required=False)

    parser.add_argument('--input_audio', type=str,  default=audio, required=False)

    # parser.add_argument('--output_path', type=str, help="Path for saving the result", default='result.mp4', required=False)

    parser.add_argument('--fps', type=float, default=25,required=False)

    parser.add_argument('--fl_detector_batchsize',  type=int , default = 32)

    parser.add_argument('--generator_batchsize', type=int, default=16)

    parser.add_argument('--output_name', type=str , default="results.mp4")

    parser.add_argument('--vis_fl', type=bool, default=check)

    parser.add_argument('--test_img2img', type=bool, help="Testing image2image module with no lip generation" , default=False)

    args = parser.parse_args()


    Inference(args=args).start()


    return './results.mp4'



def gui():
    with gr.Blocks() as video_tab:

        with gr.Row():
            
            with gr.Column():
                video = gr.Video().style()

                audio = gr.Audio(source="upload", type="filepath")         

            with gr.Column():
                outputs = gr.PlayableVideo()
                       


        with gr.Row():


            with gr.Column():
            
                check_box = gr.Checkbox(value=False,label="Do you want to visualize reconstructed facial landmark??")


                drop_down = gr.Dropdown(list(MODEL_NAME.keys()), label="Select Model")

        with gr.Row():
            with gr.Column():     

                inputs = [video,audio,check_box,drop_down]    
                gr.Button("Sync").click(

                        fn=func,
                        inputs=inputs,
                        outputs=outputs
                        )




    with gr.Blocks() as image_tab:

        
        with gr.Row():
            
            with gr.Column():
                video = gr.Image(type="filepath")

                audio = gr.Audio(source="upload", type="filepath")
        
                    

            with gr.Column():
                outputs = gr.PlayableVideo()


        with gr.Row():
            
            with gr.Column():

                check_box = gr.Checkbox(value=False,label="Do you want to visualize reconstructed facial landmark??")

                drop_down = gr.Dropdown(list(MODEL_NAME.keys()), label="Select Model")

        with gr.Row():
            with gr.Column():     

                inputs = [video,audio,check_box,drop_down]    
                gr.Button("Sync").click(

                        fn=func,
                        inputs=inputs,
                        outputs=outputs
                        )



    with gr.Blocks() as main:

        gr.Markdown(
            """
        # Audio-Visual Lip Synthesis!

        ### Creator : Wish Suharitdamrong


        Start typing below to see the output.
        """
        )
        gui = gr.TabbedInterface([video_tab,image_tab],['Using Video as input','Using Image as input'])


    main.launch()




if __name__ == "__main__":
    gui()
