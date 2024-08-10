import time
import cv2
from PotraitFace import PotraitFace
import streamlit as st
import tempfile
import io
import zipfile
from PIL import Image


def initialize(cap):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"fps of video {fps}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of frames: {total_frames}")
    frame_list=[]
    for count in range(total_frames):
        ret, frame=cap.read()
        if not ret:
            break
        
        if count%fps==0:
            frame = cv2.resize(frame, (640,640),interpolation=cv2.INTER_CUBIC)
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_list.append(frame)
    print("frame list extracted.")
    print(f"the length of frame list is : {len(frame_list)}")
    
    return frame_list

def get_potraits(frame_list):
    p_list=[]
    for img in frame_list:
        
        emb=pf.get_embeddings(img)
        coords=pf.get_face_coordinates(emb)
        p_list.append(pf.get_faces(img,coords))
        
    
    images_list=[]
    for sub_res in p_list:
        if sub_res==[]:
            continue
        for k in sub_res:
            images_list.append(k)
            
    return images_list
        


if __name__ == '__main__':
    
    pf=PotraitFace(r"C:\Users\snman\Desktop\2024\Learn\Deep_Face\Yolo-Face-detection\yolov8n-face.pt")
    
    
    upload_file=st.file_uploader( "Choose a mp4 file", type=['mp4'],accept_multiple_files=False)
    st.write("please upload a vido in mp4 format")
    if upload_file is not None:
    
       # st.session_state.last_uploaded_file = upload_file.name
        start_time=time.time()

        st.write(f"file name : {upload_file.name}")
        st.write(f"file type is  : {upload_file.type}")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(upload_file.read())
        cap=cv2.VideoCapture(tfile.name)
        
            #vid_path=r"C:\Users\snman\Desktop\2024\Learn\Streamlit\media\v4-mini.mp4"
        frame_list=initialize(cap)
        print("##########################################################################################################################")
        print("extracting faces....")

        images_list=get_potraits(frame_list)
        st.write("images extracted")
        print(f"images extracted {len(images_list)}")
        cap.release()

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i, img in enumerate(images_list, start=1):
                # Convert the NumPy array to a PIL Image
                pil_img = Image.fromarray(img)
                    
                    # Convert the PIL image to bytes
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='JPEG')
                img_byte_arr.seek(0)
                    
                    # Add the image to the zip file with a sequential name
                zipf.writestr(f"{i}.jpg", img_byte_arr.read())
        st.download_button(label="Download Images ZIP", data=zip_buffer, file_name="vid_faces.zip", mime="application/zip")

        end_time=time.time()
        st.write(f"Time taken to extract faces: {(end_time-start_time)/60} minutes")
        st.stop()
    
            
    else:
            st.write("Please upload a new file or a different file.")
