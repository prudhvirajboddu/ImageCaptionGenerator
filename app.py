import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import random
from torch.autograd import Variable
import torchvision
from model import PositionalEncoding,ImageCaptionModel

max_seq_len = 33
vocab_size = 8360
import pickle
afile = open('I2W.pkl', 'rb')
index_to_word = pickle.load(afile)
afile.close()
bfile = open('W2I.pkl', 'rb')
word_to_index = pickle.load(bfile)
bfile.close()
start_token = word_to_index['<start>']
end_token = word_to_index['<end>']
pad_token = word_to_index['<pad>']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImageCaptionModel(n_head=16, n_decoder_layer=5, vocab_size=vocab_size, embedding_size=512).to(device)
model = torch.load('BestModelResnet.pt', map_location=device)

# Define the function to generate captions

resnet18 = torchvision.models.resnet18(pretrained=True).to('cpu')
resnet18.eval()

resNet18Layer4 = resnet18._modules.get('layer4').to('cpu')

def get_resnet_vector(t_img):
    
    t_img = torch.from_numpy(t_img).float().unsqueeze(0).permute(0,3,1,2)
    t_img = Variable(t_img)

    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    t_img = preprocess(t_img).to('cpu')

    my_embedding = torch.zeros(1, 512, 7, 7)
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    
    h = resNet18Layer4.register_forward_hook(copy_data)
    resnet18(t_img)
    
    h.remove()
    return my_embedding


def generate_caption(image):

    image = get_resnet_vector(image)
    image = image.permute(0,2,3,1)

    image = image.view(image.size(0), -1, image.size(3))
   

    image = image.to(device)
    input_seq = [pad_token]*max_seq_len
    input_seq[0] = start_token

    input_seq = torch.tensor(input_seq).unsqueeze(0).to(device)

    predicted_sentence = []

    K = 2
    model.eval()

    with torch.no_grad():
        for eval_iter in range(0, max_seq_len-1):

            output, _ = model.forward(image, input_seq)

            output = output[eval_iter, 0, :]

            values = torch.topk(output, K).values.tolist()
            indices = torch.topk(output, K).indices.tolist()

            next_word_index = random.choices(indices, values, k = 1)[0]

            next_word = index_to_word[next_word_index]

            input_seq[:, eval_iter+1] = next_word_index

            if next_word == '<end>' :
                break

            predicted_sentence.append(next_word)

    return(" ".join(predicted_sentence+['.']))

# Create the Gradio interface
inputs = gr.Image()
outputs = gr.Textbox()
interface = gr.Interface(fn=generate_caption, inputs=inputs, outputs=outputs)

# Run the Gradio app
interface.launch()
