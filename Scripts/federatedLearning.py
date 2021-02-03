import numpy as np


def try_to_update(clients_weights, global_weights):
    update = False
    clients = []
    if len(clients_weights) >= 6:
        update = True
        for key in clients_weights:
            if len(clients_weights[key].keys()) == 7:
                update = True
                clients.append(key)
            else:
                update = False
    if update:
        weights = update_global_weights(clients, clients_weights,
                                        global_weights)
        for client in clients:
            del clients_weights[client]
        return weights
    else:
        return False


def update_global_weights(clients, clients_weights, global_weights):
    new_global_weights = [np.zeros(weight.shape) for weight in global_weights]

    global_data_size = sum([clients_weights[client]["data_qtd"]
                            for client in clients])

    for client in clients:
        client_data_size = clients_weights[client]["data_qtd"]
        client_weights = [np.array(clients_weights[client][layer])
                          for layer in clients_weights[client]
                          if layer != "data_qtd"]

        new_global_weights += (client_data_size / global_data_size) * np.array(
                                client_weights, dtype='object')
    return new_global_weights
