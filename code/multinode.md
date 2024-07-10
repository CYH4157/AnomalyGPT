# Use multi-node
We use two A30s to complete the multi-node model training.

##

'''
sudo docker run -idt --ipc=host --gpus all --network=host -v /home/nchc/:/workspace/:rw huggingface/transformers-pytorch-gpu
'''
