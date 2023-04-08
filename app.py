import streamlit as st
import inference

# @st.cache_resource
def get_model():
  print('\n\n\n\n------start making object---------\n\n')

  return inference.Inference('/content/rtmdet-ins_s_8xb32-300e_coco.py', '/content/drive/MyDrive/epoch_14.pth', './results/result.jpg')


inferer = get_model()
print('\n\n\nfinished\n\n\n')
uploaded_file = st.sidebar.file_uploader("Choose a file")
# print('start making Inference obj')
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    print('\n\n\n\n------making inference-------\n\n\n\n')
    inferer.make_inference(bytes_data, is_binary=True)
    get_model.clear()
    print('\n\n\n\n\nshowing uploaded image\n\n\n\n')
    st.image(bytes_data, caption = "Uploaded Image", width=700)
    print('\n\n\nshowing result image\n\n\n')
    st.image('./results/result.jpg', caption='Segmentation result', width=700)
