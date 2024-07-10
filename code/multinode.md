# Use multi-node
We use two A30s to complete the multi-node model training.

## Setp 1: Docker environment
Run the code in a docker environment.

```
sudo docker run -idt --ipc=host --gpus all --network=host -v /home/nchc/:/workspace/:rw huggingface/transformers-pytorch-gpu
```

* Installation of the required library
```
apt-get update
cd anomalygpt
pip install -r requirements.txt
```

## Setp 2: Set up passwordless SSH login on all nodes

Ensure Passwordless SSH Login
* Installation of the required tools
```
apt-get update
apt install openssh-server
apt-get install net-tools

## setting root passwd
passwd root
```

* Generate an SSH key pair (if not already done):
```
cd
mkdir .ssh
ssh-keygen
```

* Add the public key to the `authorized_keys` file:
Master node setup public key for localhost passwordless
```
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
```

* Ensure the permissions of the .ssh directory and authorized_keys file are correct:

```
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

```
sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
```

If you want to change the port of ssh, you can do it in the following way
```
sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config
service ssh restart && netstat -tulpn
```

2. Worker Node Setting Public Key.
Put the public key to the host you want to log in.

```
cd .ssh
scp -P 2222 a30ip31.pub root@10.250.64.21:.ssh
```

The default file for handling public keys is
authorized_keys, so the public keys just sent must be accumulated into this file:

```
cat .ssh/a30ip31.pub >> .ssh/authorized_keys
```

Ensure the permissions of the .ssh directory and authorized_keys file are correct:

```
chmod 700 .ssh/ 
chmod 644 .ssh/authorized_keys 
```

check the authority

```
ll -d .ssh
ll .ssh/authorized_keys 
```

Finish and leave.
```
exit
```

## Setp 3: Create config file with direct ssh name

```
cd .ssh
vim config
```

config file:

```
Host my-30
    HostName 10.250.64.30
    User root
    IdentityFile ~/.ssh/a30ip31
    Port 2222

Host my-21
    HostName 10.250.64.21
    User root
    IdentityFile ~/.ssh/a30ip31
    Port 2222

Host localhost
    HostName 10.250.64.31
    User root
    IdentityFile ~/.ssh/a30ip31
    Port 2222
```

* Restart the SSH service:

```
sudo systemctl restart ssh
```

* Verify passwordless SSH login:
```
ssh localhost
ssh my-21
ssh my-30
```

## Setp 5: Set Up the Hostfile for deepspeed
Create a hostfile that lists all participating nodes and their GPU counts. Assuming host and worker are the hostnames or IP addresses of your master and worker nodes:

However, it is important to make sure that each worker node has the same code and directory as the master node.

The hostfile can be placed in any directory.
hostfile:
```
localhost slots=2
worker slots=2
```



## Setp 7: Start DeepSpeed Multi-Node Training
Run the DeepSpeed command on the master node (host), specifying the location of the hostfile and other relevant parameters:

```
deepspeed --hostfile=./hostfile --master_port=28400 train_mvtec.py \
    --model openllama_peft \
    --stage 1 \
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth \
    --vicuna_ckpt_path ../pretrained_ckpt/vicuna_ckpt/7b_v0/ \
    --delta_ckpt_path ../pretrained_ckpt/pandagpt_ckpt/7b/pytorch_model.pt \
    --max_tgt_len 1024 \
    --data_path  ../data/pandagpt4_visual_instruction_data.json \
    --image_root_path ../data/images/ \
    --save_path  ./ckpt/train_mvtec/ \
    --log_path ./ckpt/train_mvtec/log_rest/
```
