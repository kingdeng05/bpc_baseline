export GCSFUSE_REPO=gcsfuse-$(lsb_release -c -s)
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y gcsfuse
mkdir -p /content/gcs-bucket
gcsfuse --implicit-dirs --key-file static-nomad-452701-n4-42f2015247b3.json bin-picking-challenge-2025 /content/gcs-bucket
