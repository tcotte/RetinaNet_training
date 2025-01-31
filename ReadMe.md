# Train Pytorch RetinaNet model

### Launch train script in Colab without Docker

```
%%shell
export api_token="<api_token>"
export organization_id="<organization_id>"
export experiment_id="<experiment_id>"

git clone <repo>
python training_image/picsellia_folder/main.py
```