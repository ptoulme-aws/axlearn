#! /bin/bash

set -e

sudo dpkg --configure -a
# Configure Ubuntu for Neuron repository updates
. /etc/os-release
sudo tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main
EOF
wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB \
	    | sudo apt-key add -

# Update OS packages and install OS headers
sudo apt-get update -y && sudo apt-get install -y linux-headers-$(uname -r)
sudo apt-get remove -y aws-neuronx-devtools --allow-change-held-packages || true
sudo apt-get remove -y aws-neuronx-tools aws-neuronx-collectives aws-neuronx-dkms aws-neuronx-runtime-lib --allow-change-held-packages
# Install Neuron OS packages and dependencies
sudo dpkg -i /shared_new/ptoulme/axlearn/aws-neuronx-runtime-lib-2.x.15467.0-8af04688f.deb
sudo dpkg -i /shared_new/ptoulme/axlearn/aws-neuronx-collectives-2.x.16467.0-f08f66d8f.deb
#sudo dpkg -i aws-neuronx-runtime-lib-2.x.9937.0-b6de4480f.deb
sudo apt-get -o Dpkg::Options::="--force-overwrite" install --reinstall --allow-downgrades -y aws-neuronx-dkms aws-neuronx-tools

TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600"` && INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s  http://169.254.169.254/latest/meta-data/instance-id)
echo "instance_id:$INSTANCE_ID hostname:$(hostname)"
echo "runtime versions"
sudo apt list | grep neuron | grep installed

echo "==============================================="
echo "Dependency versions"
echo "==============================================="
apt list | grep neuron
pip freeze | grep neuron
echo "==============================================="