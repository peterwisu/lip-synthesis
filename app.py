import gradio as gr
import os
import argparse 
from src.main.inference import Inference 


def func(video,audio,check):

    print(check)


    parser = argparse.ArgumentParser(description="File for running Inference")

    parser.add_argument('--generator_checkpoint', type=str ,default='./checkpoints/generator/checkpoint_lip_fl_epoch000000014.pth')

    parser.add_argument('--image2image_checkpoint', type=str, default='./src/models/ckpt/image2image.pth',required=False)

    parser.add_argument('--input_face', type=str,default=video, required=False)

    parser.add_argument('--input_audio', type=str,  default=audio, required=False)

    # parser.add_argument('--output_path', type=str, help="Path for saving the result", default='result.mp4', required=False)

    parser.add_argument('--fps', type=float, default=25,required=False)

    parser.add_argument('--fl_detector_batchsize',  type=int , default = 32)

    parser.add_argument('--generator_batchsize', type=int, default=16)

    parser.add_argument('--output_name', type=str , default="results.mp4")

    parser.add_argument('--vis_fl', type=bool, default=check)

    args = parser.parse_args()


    Inference(args=args).start()

    #inference.start()

    return './results.mp4'



def gui():
    with gr.Blocks() as video_tab:
        

        title = gr.title="Audio-Visual Lip Synthesis"

        desc = gr.description="Creator :  Wish Suharitdarmong"


        
        with gr.Row():
            
            with gr.Column():
                video = gr.Video().style()

                audio = gr.Audio(source="upload", type="filepath")         

            with gr.Column():
                outputs = gr.PlayableVideo()
                       


        with gr.Row():
            
            check_box = gr.Checkbox(value=False,label="Do you want to visualize reconstructed facial landmark??")

        with gr.Row():
            with gr.Column():     

                inputs = [video,audio,check_box]    
                gr.Button("Sync").click(

                        fn=func,
                        inputs=inputs,
                        outputs=outputs
                        )




    with gr.Blocks() as image_tab:

        
        title = gr.title="Audio-Visual Lip Synthesis"

        desc = gr.description="Creator :  Wish Suharitdarmong"
        
        with gr.Row():
            
            with gr.Column():
                video = gr.Image(type="filepath")

                audio = gr.Audio(source="upload", type="filepath")
        
                    

            with gr.Column():
                outputs = gr.PlayableVideo()


        with gr.Row():
            
            check_box = gr.Checkbox(value=False,label="Do you want to visualize reconstructed facial landmark??")

        with gr.Row():
            with gr.Column():     

                inputs = [video,audio,check_box]    
                gr.Button("Sync").click(

                        fn=func,
                        inputs=inputs,
                        outputs=outputs
                        )

    demo = gr.Interface(func, 
                        gr.Video(), 
                        "playable_video", 
                        cache_examples=True)


    with gr.Blocks() as main:

        gr.Markdown(
            """
        # Audio-Visual Lip Synthesis!

        ### Creator : Wish Suharitdamrong


        Start typing below to see the output.
        """
        )
        gui = gr.TabbedInterface([video_tab,image_tab,demo],['Using Video as input','Using Image as input','demo'])


    main.launch()




if __name__ == "__main__":
    gui()
