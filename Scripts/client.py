import os
import json
import time
import requests

import numpy as np
from random import randint

from MLP import MLP
from dataProcessing import create_clients


for _ in range(4):
    i = int(os.environ['N_Client'])
    X_slices, y_slices = create_clients()

    client_model = MLP()
    client_model.build()

    global_weights = []
    for layer in range(len(client_model.get_layers()) * 2):
        r = requests.get(
            'http://server:8000/get_weights/?layer={}'.format(layer))
        global_weights.append(np.array(r.json()[str(layer)]))

    client_model.compile()
    client_model.set_weights(global_weights)
    client_model.fit(X_slices[i], y_slices[i])

    local_weights = client_model.get_weights()

    url = 'http://server:8000/send_weights/'

    for layer in range(len(client_model.get_layers()) * 2):
        myobj = json.dumps({"layer": layer,
                            "data_qtd": X_slices.shape[0],
                            "weights": local_weights[layer].tolist()}
                           ).encode('utf-8')
        x = requests.post(url, data=myobj)

    time.sleep(randint(10, 15))
