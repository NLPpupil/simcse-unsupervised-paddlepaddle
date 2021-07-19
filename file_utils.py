def read_config():
    config = {}
    with open('config.txt') as f:
        for line in f:
            columns = line.strip().split('=')
            config[columns[0]] = columns[1]

    return config


def read_data(filename):
    with open(filename) as f:
        return [l.strip() for l in f]