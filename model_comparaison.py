import os
import pandas as pd
import numpy as np
from picsellia import Client

if __name__ == '__main__':
    project_id = '01977951-bae6-7873-80ed-7206256fff6b'

    # Picsell.ia connection
    api_token = os.environ["api_token"]
    organization_id = os.environ["organization_id"]
    client = Client(api_token=api_token, organization_id=organization_id)

    project = client.get_project_by_id(project_id)

    experiments = [exp for exp in project.list_experiments() if exp.get_base_model_version().origin_name == 'RetinaNet']

    for i, exp in enumerate(experiments):
        d = {}
        d['experiment_name'] = exp.name
        d['experiment_id'] = exp.id
        d['experiment_version'] = exp.get_base_model_version().name

        best_epoch = np.argmax(exp.get_log('Validation mAP[50]').data)
        d['best_epoch'] = best_epoch
        d['best_map_50'] = exp.get_log('Validation mAP[50]').data[best_epoch]
        d['best_map'] = exp.get_log('Validation map').data[best_epoch]
        d['best_map_75'] = exp.get_log('Validation mAP[75]').data[best_epoch]
        d['best_precision'] = exp.get_log('Validation precision').data[best_epoch]
        d['best_recall'] = exp.get_log('Validation precision').data[best_epoch]

        d.update(exp.get_log('All parameters').data)
        d.update(exp.get_log('parameters').data)

        d['anchor_boxes_sizes'] = str(d['anchor_boxes_sizes'])
        d['anchor_boxes_aspect_ratios'] = str(d['anchor_boxes_aspect_ratios'])
        d['augmentations_normalization_std'] = str(d['augmentations_normalization_std'])
        d['augmentations_normalization_mean'] = str(d['augmentations_normalization_mean'])


        if i == 0:
            df = pd.DataFrame.from_records([d])
        else:
            df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)

    df.to_excel('output.xlsx')




