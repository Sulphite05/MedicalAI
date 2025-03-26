import streamlit as st
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

st.set_page_config(
    page_title="Diabetic Retinopathy Detector",
    page_icon="👁"
)

col1, col2 = st.columns([0.8, 5])

with col1: st.image('images/logo.png', width=100)
with col2: st.title("Retinopathy Detector")
st.divider()
st.subheader("How it works 💡")
st.markdown("""
This Retinopathy detector uses a pretrained and fine-tuned EfficientNet B3 model to classify diabetes stage 
of a person through their retinal image.<br>
It classifies the stages as follows:
- No DR
- Mild
- Moderate
- Severe
- Proliferative DR
""")


def load_model(model_path, num_classes=5):
    model = models.efficientnet_b3(pretrained=False)

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier[1].in_features, num_classes)
    )

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    new_state_dict = {}
    for key, value in state_dict.items():
        if key == 'classifier.1.1.weight':
            new_state_dict['classifier.1.weight'] = value
        elif key == 'classifier.1.1.bias':
            new_state_dict['classifier.1.bias'] = value
        else:
            new_state_dict[key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image


def classify_image(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
        return predicted_class


dct = {
    0: 'No DR',
    1: 'Mild',
    2: 'Moderate',
    3: 'Severe',
    4: 'Proliferative DR'
}


st.subheader('Input Image')
upload = st.file_uploader('Insert retinal image for classification:', type=['png', 'jpg'])
c1, c2 = st.columns(2)

if upload is not None:
    im = Image.open(upload)
    img = np.asarray(im)
    img = np.expand_dims(img, 0)

    c1.image(im)
    c1.write(img.shape)

    model = load_model('model/fineTunedEfficientnet_b3.pt')
    image_tensor = preprocess_image(upload)
    prediction = classify_image(model, image_tensor)
    c1.header('Prediction')
    st.write(f'The diabetic stage is **{dct[prediction]}**.')
