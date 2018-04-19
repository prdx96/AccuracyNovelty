from configparser import ConfigParser

def read_config(sec,key):
    config=ConfigParser()
    config.read('config.ini')
    return config.get(sec,key)

def set_config(sec,key,value):
    config=ConfigParser()
    config.read('config.ini')
    config.set(sec,key,value)
    with open("config.ini",'w') as fp:
        config.write(fp)

if __name__ == '__main__':
    set_config('basic-settings','data_dir','test')
    print(read_config('basic-settings','data_dir'))
    set_config('basic-settings','data_dir','./ml-100k')
